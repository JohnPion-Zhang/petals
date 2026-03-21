"""
Main ExecutionFeedbackLoop class implementing CodeAct self-correction pattern.

This module provides the ExecutionFeedbackLoop class that manages the
self-correction cycle: Generate -> Execute -> Capture error -> Format feedback -> LLM corrects.

Example:
    >>> from petals.client.feedback import ExecutionFeedbackLoop, FeedbackLoopConfig
    >>> from petals.client.tool_registry import ToolRegistry
    >>>
    >>> config = FeedbackLoopConfig(
    ...     max_retries=3,
    ...     enable_correction=True,
    ...     correction_llm=llm_provider
    ... )
    >>> loop = ExecutionFeedbackLoop(registry, config)
    >>>
    >>> # Execute a node with feedback
    >>> result_node = await loop.execute_with_feedback(node)
"""
import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, AsyncGenerator

import logging

from petals.client.dag.tool_call_node import ToolCallNode
from petals.client.dag.dag import ToolCallDAG
from petals.client.tool_registry import ToolRegistry
from petals.client.feedback.traceback import CapturedTraceback, TracebackCapture, ErrorSeverity
from petals.client.feedback.correction import (
    LLMCorrector,
    CorrectionResult,
    CorrectionStrategy
)
from petals.client.feedback.retry_policy import RetryPolicy, BackoffStrategy, CircuitBreaker
from petals.data_structures import CallStatus

logger = logging.getLogger(__name__)


class FeedbackAction(Enum):
    """Actions the feedback loop can take on error."""
    CONTINUE = "continue"
    RETRY = "retry"
    CORRECT = "correct"
    SKIP = "skip"
    ABORT = "abort"


@dataclass
class FeedbackEntry:
    """A single feedback entry in the loop history.

    Attributes:
        node_id: ID of the node this entry is for.
        action: The action taken.
        error: Captured error information (if any).
        correction: Correction result (if any).
        timestamp: Timestamp of the entry.
    """
    node_id: str
    action: FeedbackAction
    error: Optional[CapturedTraceback] = None
    correction: Optional[CorrectionResult] = None
    timestamp: float = field(default_factory=lambda: time.time())


@dataclass
class FeedbackLoopConfig:
    """Configuration for the feedback loop.

    Attributes:
        max_retries: Maximum number of retry attempts per node.
        enable_correction: Whether to enable LLM-based correction.
        enable_backoff: Whether to apply exponential backoff.
        base_backoff: Base backoff delay in seconds.
        max_backoff: Maximum backoff delay cap.
        retry_on_types: Tuple of exception types that trigger retry.
        correction_llm: LLM provider for correction (if enabled).
    """
    max_retries: int = 3
    enable_correction: bool = True
    enable_backoff: bool = True
    base_backoff: float = 1.0
    max_backoff: float = 30.0
    retry_on_types: tuple = (TimeoutError, ConnectionError, ConnectionResetError)
    correction_llm: Optional[Any] = None


class ExecutionFeedbackLoop:
    """CodeAct-style feedback loop for self-correction.

    Pattern: LLM generates -> Execute -> Capture error -> Format feedback -> LLM corrects

    Features:
    - Automatic retry with backoff
    - LLM-based error correction
    - Feedback history tracking
    - Configurable error handling
    - Abort on uncorrectable errors

    Example:
        >>> loop = ExecutionFeedbackLoop(registry, config)
        >>> result_node = await loop.execute_with_feedback(node)
        >>> print(f"Node {node.id} status: {node.status}")
    """

    def __init__(
        self,
        registry: ToolRegistry,
        config: Optional[FeedbackLoopConfig] = None
    ):
        """Initialize the ExecutionFeedbackLoop.

        Args:
            registry: ToolRegistry for executing tools.
            config: Feedback loop configuration.
        """
        self.registry = registry
        self.config = config or FeedbackLoopConfig()

        self._corrector: Optional[LLMCorrector] = None
        self._retry_policy: Optional[RetryPolicy] = None
        self._circuit_breaker: Optional[CircuitBreaker] = None
        self._history: List[FeedbackEntry] = []
        self._retry_counts: Dict[str, int] = {}
        self._results_cache: Dict[str, Any] = {}

        # Initialize corrector if LLM is provided
        if self.config.enable_correction and self.config.correction_llm:
            self._corrector = LLMCorrector(
                llm_provider=self.config.correction_llm,
                max_retries=self.config.max_retries,
                base_backoff=self.config.base_backoff,
                max_backoff=self.config.max_backoff
            )

        # Initialize retry policy
        self._retry_policy = RetryPolicy(
            max_attempts=self.config.max_retries,
            base_delay=self.config.base_backoff,
            max_delay=self.config.max_backoff,
            backoff_strategy=(
                BackoffStrategy.EXPONENTIAL_WITH_JITTER
                if self.config.enable_backoff
                else BackoffStrategy.FIXED
            ),
            retryable_exceptions=self.config.retry_on_types
        )

        # Initialize circuit breaker
        self._circuit_breaker = CircuitBreaker()

    async def execute_with_feedback(
        self,
        node: ToolCallNode
    ) -> ToolCallNode:
        """Execute a node with feedback loop support.

        Implements the CodeAct pattern:
        1. Execute the tool
        2. On error: capture traceback
        3. If correction enabled: use LLM to correct
        4. Retry with corrected args
        5. Track history for analysis

        Args:
            node: The ToolCallNode to execute.

        Returns:
            The executed node with updated status and result/error.
        """
        node_id = node.id
        self._retry_counts[node_id] = 0

        while True:
            # Check if we should abort
            if self.should_abort(node_id):
                logger.warning(f"Aborting node {node_id}: max retries exceeded")
                node.mark_failed(
                    f"Max retries ({self.config.max_retries}) exceeded",
                    error_feedback=f"Retry history: {self.get_history(node_id)}"
                )
                self.record_feedback(FeedbackEntry(
                    node_id=node_id,
                    action=FeedbackAction.ABORT
                ))
                break

            # Execute a single attempt
            success, error, traceback_info = await self._execute_single_attempt(node)

            if success:
                logger.debug(f"Node {node_id} executed successfully")
                self.record_feedback(FeedbackEntry(
                    node_id=node_id,
                    action=FeedbackAction.CONTINUE
                ))
                break

            # Handle error
            if traceback_info is not None:
                action = await self._handle_error(node, error, traceback_info)

                if action == FeedbackAction.CONTINUE:
                    # Non-critical error, continue
                    break

                elif action == FeedbackAction.RETRY:
                    # Simple retry
                    retry_result = await self._retry(node)
                    if not retry_result:
                        continue  # Will check abort condition in next iteration

                elif action == FeedbackAction.CORRECT:
                    # LLM correction and retry
                    correction_result = await self._correct_and_retry(node, traceback_info)
                    if correction_result.success:
                        continue  # Success, continue
                    elif correction_result.strategy == CorrectionStrategy.ABORT:
                        node.mark_failed(
                            f"Correction failed: {correction_result.original_error}",
                            error_feedback=traceback_info.format_for_llm()
                        )
                        self.record_feedback(FeedbackEntry(
                            node_id=node_id,
                            action=FeedbackAction.ABORT,
                            error=traceback_info,
                            correction=correction_result
                        ))
                        break

                elif action == FeedbackAction.CORRECT:
                    # LLM correction failed but not aborting, continue to retry loop
                    continue

                elif action == FeedbackAction.SKIP:
                    # Skip this node
                    logger.info(f"Skipping node {node_id}")
                    node.mark_failed("Skipped by feedback loop")
                    self.record_feedback(FeedbackEntry(
                        node_id=node_id,
                        action=FeedbackAction.SKIP,
                        error=traceback_info
                    ))
                    break

                elif action == FeedbackAction.ABORT:
                    # Abort immediately
                    logger.error(f"Aborting node {node_id} due to error")
                    node.mark_failed(
                        str(error),
                        error_feedback=traceback_info.format_for_llm() if traceback_info else None
                    )
                    self.record_feedback(FeedbackEntry(
                        node_id=node_id,
                        action=FeedbackAction.ABORT,
                        error=traceback_info
                    ))
                    break

            else:
                # No traceback info, abort
                node.mark_failed(str(error))
                self.record_feedback(FeedbackEntry(
                    node_id=node_id,
                    action=FeedbackAction.ABORT
                ))
                break

        # Cache result if successful
        if node.status == CallStatus.DONE:
            self._results_cache[node_id] = node.result

        return node

    async def _execute_single_attempt(
        self,
        node: ToolCallNode
    ) -> tuple[bool, Optional[Exception], Optional[CapturedTraceback]]:
        """Execute a single attempt and capture result.

        Args:
            node: The node to execute.

        Returns:
            Tuple of (success, error, traceback_info).
        """
        node.mark_running()

        try:
            # Resolve dependencies if any
            resolved_args = self._resolve_dependencies(node)

            # Execute with timeout
            result = await asyncio.wait_for(
                self.registry.execute(node.name, resolved_args),
                timeout=30.0  # Default timeout
            )

            # Check for error in result
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(result["error"])

            node.mark_done(result)
            return True, None, None

        except asyncio.TimeoutError as e:
            traceback_info = TracebackCapture.capture_exception(e)
            traceback_info.severity = ErrorSeverity.ERROR
            return False, e, traceback_info

        except Exception as e:
            traceback_info = TracebackCapture.capture_exception(e)
            return False, e, traceback_info

    def _resolve_dependencies(self, node: ToolCallNode) -> Dict[str, Any]:
        """Resolve node arguments using cached results.

        Args:
            node: The node whose arguments to resolve.

        Returns:
            Resolved arguments dictionary.
        """
        resolved_args = {}

        for key, value in node.arguments.items():
            if isinstance(value, dict) and "from_dep" in value:
                dep_id = value["from_dep"]
                if dep_id in self._results_cache:
                    resolved_args[key] = self._results_cache[dep_id]
                else:
                    resolved_args[key] = value
            elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                dep_id = value[2:-1]
                if dep_id in self._results_cache:
                    resolved_args[key] = self._results_cache[dep_id]
                else:
                    resolved_args[key] = value
            else:
                resolved_args[key] = value

        return resolved_args

    async def _handle_error(
        self,
        node: ToolCallNode,
        error: Exception,
        traceback_info: CapturedTraceback
    ) -> FeedbackAction:
        """Determine what action to take on error.

        Args:
            node: The node that failed.
            error: The exception that was raised.
            traceback_info: Captured traceback information.

        Returns:
            The action to take.
        """
        # Check if error is retryable
        if isinstance(error, self.config.retry_on_types):
            # Can retry this error
            if self._retry_counts[node.id] < self.config.max_retries:
                return FeedbackAction.RETRY

        # Check if LLM correction is available and enabled
        if self.config.enable_correction and self._corrector is not None:
            return FeedbackAction.CORRECT

        # Default: abort
        return FeedbackAction.ABORT

    async def _retry(
        self,
        node: ToolCallNode,
        corrected_args: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Retry execution with optional corrected arguments.

        Args:
            node: The node to retry.
            corrected_args: Optional corrected arguments.

        Returns:
            True if retry was successful.
        """
        node_id = node.id
        self._retry_counts[node_id] = self._retry_counts.get(node_id, 0) + 1
        node.increment_retry()

        logger.info(
            f"Retrying node {node_id} "
            f"(attempt {self._retry_counts[node_id]}/{self.config.max_retries})"
        )

        # Apply backoff if enabled
        if self.config.enable_backoff and self.config.max_backoff > 0:
            delay = min(
                self.config.base_backoff * (2 ** (self._retry_counts[node_id] - 1)),
                self.config.max_backoff
            )
            await asyncio.sleep(delay)

        # Update arguments if corrected
        if corrected_args:
            node.arguments = corrected_args

        self.record_feedback(FeedbackEntry(
            node_id=node_id,
            action=FeedbackAction.RETRY
        ))

        # Check abort condition
        if self.should_abort(node_id):
            return False

        # Execute retry
        success, _, _ = await self._execute_single_attempt(node)
        return success

    async def _correct_and_retry(
        self,
        node: ToolCallNode,
        error: CapturedTraceback
    ) -> CorrectionResult:
        """Attempt LLM correction and retry.

        Args:
            node: The node to correct and retry.
            error: Captured error information.

        Returns:
            CorrectionResult indicating outcome.
        """
        if self._corrector is None:
            return CorrectionResult(
                success=False,
                strategy=CorrectionStrategy.ABORT,
                original_error="Corrector not initialized"
            )

        logger.info(f"Requesting LLM correction for node {node.id}")

        # Get correction from LLM
        correction_result = await self._corrector.correct(
            tool_name=node.name,
            arguments=node.arguments,
            error=error,
            context={"node_id": node.id}
        )

        if not correction_result.success:
            return correction_result

        # Record correction
        self.record_feedback(FeedbackEntry(
            node_id=node.id,
            action=FeedbackAction.CORRECT,
            error=error,
            correction=correction_result
        ))

        # Handle based on strategy
        if correction_result.strategy == CorrectionStrategy.SKIP:
            return correction_result

        if correction_result.strategy == CorrectionStrategy.ABORT:
            return correction_result

        if correction_result.strategy in (
            CorrectionStrategy.RETRY_SAME,
            CorrectionStrategy.RETRY_MODIFIED
        ):
            # Increment retry count
            node_id = node.id
            self._retry_counts[node_id] = self._retry_counts.get(node_id, 0) + 1
            node.increment_retry()

            # Apply correction and retry
            if correction_result.corrected_arguments:
                node.arguments = correction_result.corrected_arguments

            success, _, _ = await self._execute_single_attempt(node)
            if success:
                correction_result.success = True
            else:
                correction_result.success = False

            return correction_result

        if correction_result.strategy == CorrectionStrategy.FALLBACK:
            # Use fallback logic if available
            return correction_result

        return correction_result

    def record_feedback(self, entry: FeedbackEntry) -> None:
        """Record a feedback entry in history.

        Args:
            entry: The feedback entry to record.
        """
        self._history.append(entry)
        logger.debug(f"Recorded feedback for node {entry.node_id}: {entry.action}")

    def get_retry_count(self, node_id: str) -> int:
        """Get current retry count for a node.

        Args:
            node_id: The node ID to query.

        Returns:
            Number of retries attempted.
        """
        return self._retry_counts.get(node_id, 0)

    def get_history(self, node_id: Optional[str] = None) -> List[FeedbackEntry]:
        """Get feedback history, optionally filtered by node.

        Args:
            node_id: Optional node ID to filter by.

        Returns:
            List of feedback entries.
        """
        if node_id is None:
            return list(self._history)
        return [entry for entry in self._history if entry.node_id == node_id]

    def should_abort(self, node_id: str) -> bool:
        """Check if we should abort for this node.

        Args:
            node_id: The node ID to check.

        Returns:
            True if abort threshold reached.
        """
        return self.get_retry_count(node_id) >= self.config.max_retries

    def get_stats(self) -> Dict[str, Any]:
        """Get feedback loop statistics.

        Returns:
            Dictionary with loop statistics.
        """
        total_entries = len(self._history)
        actions = {action.value: 0 for action in FeedbackAction}
        for entry in self._history:
            actions[entry.action.value] += 1

        return {
            "total_entries": total_entries,
            "actions": actions,
            "retry_counts": dict(self._retry_counts),
            "corrector_available": self._corrector is not None,
            "circuit_breaker_state": (
                self._circuit_breaker.state
                if self._circuit_breaker
                else None
            )
        }

    def reset(self) -> None:
        """Reset the feedback loop state."""
        self._history.clear()
        self._retry_counts.clear()
        self._results_cache.clear()
        if self._corrector:
            self._corrector.reset()
        if self._circuit_breaker:
            self._circuit_breaker.reset()
        logger.info("Feedback loop state reset")

    async def execute_dag_with_feedback(
        self,
        dag: ToolCallDAG
    ) -> AsyncGenerator[ToolCallNode, None]:
        """Execute a DAG with feedback loop support.

        Args:
            dag: The ToolCallDAG to execute.

        Yields:
            Completed nodes as they finish.
        """
        # Check for cycles
        cycle = dag.detect_cycle()
        if cycle:
            raise ValueError(f"Cannot execute DAG with cycle: {' -> '.join(cycle)}")

        # Get waves
        waves = dag.get_waves()

        logger.info(f"Executing DAG with feedback: {len(dag)} nodes in {len(waves)} waves")

        for wave in waves:
            # Execute each node in the wave with feedback
            wave_tasks = [
                self.execute_with_feedback(node)
                for node in wave
            ]

            # Wait for wave to complete
            completed_nodes = await asyncio.gather(*wave_tasks)

            # Update results cache
            for node in completed_nodes:
                if node.status == CallStatus.DONE:
                    self._results_cache[node.id] = node.result

            # Yield completed nodes
            for node in completed_nodes:
                yield node
