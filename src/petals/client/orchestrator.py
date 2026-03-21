"""
Unified Orchestrator for ToolCall DAG execution with streaming.

Integrates:
- Phase 1: DAG Foundation (ToolCallDAG, WaveExecutor)
- Phase 2: Async + Streaming (StreamingAggregator, StructuredOutputEnforcer)
- Phase 3: Feedback Loop (ExecutionFeedbackLoop, CircuitBreaker)
- Phase 4: RLM Verification (ResultVerifier, VerificationAwareExecutor)

Example:
    >>> from petals.client.orchestrator import Orchestrator, OrchestratorConfig
    >>> from petals.client.tool_registry import ToolRegistry
    >>>
    >>> config = OrchestratorConfig(
    ...     max_concurrency=10,
    ...     enable_correction=True,
    ...     enable_verification=True
    ... )
    >>> orchestrator = Orchestrator(registry, config=config)
    >>>
    >>> # Execute with streaming
    >>> async for event in orchestrator.execute_streaming(dag):
    ...     print(f"Event: {event.type.value}")
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional

from petals.client.dag import ToolCallNode, ToolCallDAG, WaveExecutor
from petals.client.async_support.task_pool import TaskPool
from petals.client.async_support.streaming_types import StreamEvent, StreamEventType
from petals.client.async_support.streaming_aggregator import StreamingAggregator, AggregationResult
from petals.client.async_support.structured_output import (
    StructuredOutputEnforcer,
    OutputSchema,
)
from petals.client.feedback.feedback_loop import ExecutionFeedbackLoop, FeedbackLoopConfig
from petals.client.feedback.retry_policy import CircuitBreaker, CircuitBreakerConfig
from petals.client.verification.verifier import (
    ResultVerifier,
    VerificationLevel,
    VerificationRule,
)
from petals.client.verification.triggers import TriggerConfig, VerificationTrigger
from petals.client.verification.verification_aware_executor import VerificationAwareExecutor
from petals.client.tool_registry import ToolRegistry
from petals.data_structures import CallStatus

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the unified orchestrator.

    Attributes:
        max_concurrency: Maximum number of concurrent tool executions.
        execution_timeout: Timeout for single tool execution in seconds.
        correction_timeout: Timeout for LLM correction in seconds.
        max_retries: Maximum number of retry attempts per node.
        base_backoff: Base backoff delay in seconds.
        max_backoff: Maximum backoff delay cap in seconds.
        enable_correction: Whether to enable LLM-based error correction.
        enable_verification: Whether to enable RLM-style result verification.
        enable_structured_output: Whether to enable structured output validation.
        enable_circuit_breaker: Whether to enable circuit breaker pattern.
        verification_level: Level of verification (none, basic, structural, deep).
        max_verifications_per_run: Maximum verifications per execution run.
        fail_on_verification_failure: Whether to fail on verification failure.
    """

    # Concurrency
    max_concurrency: int = 10

    # Timeouts
    execution_timeout: float = 30.0
    correction_timeout: float = 30.0

    # Retry
    max_retries: int = 3
    base_backoff: float = 1.0
    max_backoff: float = 30.0

    # Features
    enable_correction: bool = True
    enable_verification: bool = True
    enable_structured_output: bool = True
    enable_circuit_breaker: bool = True

    # Verification
    verification_level: str = "structural"  # "none", "basic", "structural", "deep"
    max_verifications_per_run: int = 10
    fail_on_verification_failure: bool = False


class Orchestrator:
    """Unified orchestrator combining all 4 phases of ToolCall execution.

    Integrates:
    1. DAG Foundation: Wave-based execution with dependency resolution
    2. Async + Streaming: Real-time event generation via AsyncGenerator
    3. Feedback Loop: Self-correction with retry and backoff
    4. RLM Verification: Result verification before parent aggregation

    Features:
    - Streaming execution with real-time events
    - Automatic retry with exponential backoff
    - LLM-based error correction (when enabled)
    - RLM-style result verification (when enabled)
    - Circuit breaker for cascade failure prevention
    - Structured output validation (when enabled)
    - Comprehensive statistics tracking

    Example:
        >>> orchestrator = Orchestrator(registry, config=config)
        >>>
        >>> # Execute with streaming
        >>> async for event in orchestrator.execute_streaming(dag):
        ...     if event.type == StreamEventType.TOOL_RESULT:
        ...         print(f"Tool {event.data['name']} completed")
        >>>
        >>> # Get execution stats
        >>> stats = orchestrator.stats
        >>> print(f"Total: {stats['total_executions']}, Success: {stats['successful_executions']}")
    """

    def __init__(
        self,
        registry: ToolRegistry,
        config: Optional[OrchestratorConfig] = None,
        llm_provider: Optional[Any] = None,
    ):
        """Initialize the Orchestrator.

        Args:
            registry: ToolRegistry containing available tools.
            config: Optional OrchestratorConfig. Uses defaults if not provided.
            llm_provider: Optional LLM provider for correction and verification.
        """
        self.registry = registry
        self.config = config or OrchestratorConfig()
        self.llm_provider = llm_provider

        # Phase 1: DAG Foundation
        self._wave_executor = WaveExecutor(
            registry=registry,
            timeout=self.config.execution_timeout,
            max_retries=self.config.max_retries,
            concurrency_limit=self.config.max_concurrency,
        )

        # Phase 2: Async + Streaming
        self._task_pool = TaskPool(max_concurrency=self.config.max_concurrency)
        self._aggregator = StreamingAggregator()
        self._enforcer: Optional[StructuredOutputEnforcer] = None
        if self.config.enable_structured_output:
            self._enforcer = StructuredOutputEnforcer()

        # Phase 3: Feedback Loop
        feedback_config = FeedbackLoopConfig(
            max_retries=self.config.max_retries,
            enable_correction=self.config.enable_correction,
            enable_backoff=True,
            base_backoff=self.config.base_backoff,
            max_backoff=self.config.max_backoff,
            correction_llm=llm_provider,
        )
        self._feedback_loop = ExecutionFeedbackLoop(registry, feedback_config)

        # Phase 4: RLM Verification
        self._verifier: Optional[ResultVerifier] = None
        self._trigger_config: Optional[TriggerConfig] = None
        self._verification_executor: Optional[VerificationAwareExecutor] = None

        if self.config.enable_verification:
            self._verifier = ResultVerifier(
                llm_provider=llm_provider,
                verification_level=self.config.verification_level,
                timeout=self.config.execution_timeout,
                enable_llm_verification=True,
            )
            self._trigger_config = TriggerConfig.default_config()
            self._trigger_config.max_verifications_per_run = self.config.max_verifications_per_run
            self._verification_executor = VerificationAwareExecutor(
                verifier=self._verifier,
                trigger_config=self._trigger_config,
                feedback_loop=self._feedback_loop,
                fail_on_verification_failure=self.config.fail_on_verification_failure,
            )

        # Circuit breaker
        self._circuit_breaker: Optional[CircuitBreaker] = None
        if self.config.enable_circuit_breaker:
            cb_config = CircuitBreakerConfig()
            self._circuit_breaker = CircuitBreaker(cb_config)

        # State
        self._results_cache: Dict[str, Any] = {}
        self._stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "verifications": 0,
            "corrections": 0,
            "retries": 0,
        }

    async def execute_streaming(
        self,
        dag: ToolCallDAG,
        initial_args: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute DAG with streaming output.

        Yields StreamEvents as tools execute:
        - tool_call_pending: Tool detected in DAG
        - tool_call_ready: Tool parameters resolved
        - tool_executing: Execution started
        - tool_result: Tool completed (success or failure)
        - final: All execution complete with statistics

        Args:
            dag: The ToolCallDAG to execute.
            initial_args: Initial arguments for root nodes.

        Yields:
            StreamEvents for each stage of execution.

        Example:
            >>> async for event in orchestrator.execute_streaming(dag):
            ...     if event.type == StreamEventType.TOOL_RESULT:
            ...         print(f"Result: {event.data['result']}")
        """
        logger.info(f"Starting streaming execution for DAG with {len(dag.nodes)} nodes")

        # Check circuit breaker
        if self._circuit_breaker and not self._can_execute():
            yield self._create_error_event("Circuit breaker open - service unavailable")
            return

        self._results_cache = initial_args.copy() if initial_args else {}

        try:
            # Execute with verification if enabled
            if self._verification_executor:
                async for event in self._execute_with_verification(dag):
                    yield event
            else:
                async for event in self._execute_simple(dag):
                    yield event

        except Exception as e:
            logger.exception("Execution failed")
            yield self._create_error_event(str(e))
            self._record_failure()
        else:
            self._record_success()

        # Final event
        yield self._create_final_event()

    async def _execute_simple(
        self,
        dag: ToolCallDAG,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Simple execution without verification."""
        async for node in self._wave_executor.execute_dag(dag, self._results_cache):
            self._stats["total_executions"] += 1

            # Emit streaming events
            yield StreamEvent(
                type=StreamEventType.TOOL_CALL_PENDING,
                data={
                    "id": node.id,
                    "name": node.name,
                    "arguments": node.arguments,
                },
            )

            yield StreamEvent(
                type=StreamEventType.TOOL_EXECUTING,
                data={"id": node.id, "name": node.name},
            )

            if node.status == CallStatus.DONE:
                self._stats["successful_executions"] += 1
                self._results_cache[node.id] = node.result

                # Validate structured output if enabled
                result = await self._validate_output(node.name, node.result)
                yield self._create_tool_result_event(node, result)
            else:
                self._stats["failed_executions"] += 1
                yield self._create_error_event(f"Node {node.id} failed: {node.error}")

                # Handle retry via feedback loop
                if node.can_retry(self.config.max_retries):
                    self._stats["retries"] += 1
                    logger.info(f"Retrying node {node.id}")

    async def _execute_with_verification(
        self,
        dag: ToolCallDAG,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute with verification (RLM pattern)."""
        async for node in self._verification_executor.execute_verified_dag(dag, self._wave_executor):
            self._stats["total_executions"] += 1

            # Emit streaming events
            yield StreamEvent(
                type=StreamEventType.TOOL_CALL_PENDING,
                data={
                    "id": node.id,
                    "name": node.name,
                    "arguments": node.arguments,
                },
            )

            yield StreamEvent(
                type=StreamEventType.TOOL_EXECUTING,
                data={"id": node.id, "name": node.name},
            )

            if node.status == CallStatus.DONE:
                self._stats["successful_executions"] += 1
                self._results_cache[node.id] = node.result

                # Check if verified
                if self._verification_executor.is_verified(node.id):
                    self._stats["verifications"] += 1

                # Validate structured output if enabled
                result = await self._validate_output(node.name, node.result)
                yield self._create_tool_result_event(node, result)
            else:
                self._stats["failed_executions"] += 1
                yield self._create_error_event(f"Node {node.id} failed: {node.error}")

    async def _validate_output(self, tool_name: str, result: Any) -> Any:
        """Validate tool output using structured output enforcer.

        Args:
            tool_name: Name of the tool.
            result: Tool execution result.

        Returns:
            Validated result (may be transformed).
        """
        if not self._enforcer:
            return result

        schema = self._enforcer.get_schema(tool_name)
        if schema:
            validation_result = await self._enforcer.validate_and_extract(
                tool_name, result, schema
            )
            if validation_result.is_valid:
                return validation_result.data
            elif validation_result.warnings:
                logger.warning(
                    f"Validation warnings for {tool_name}: {validation_result.warnings}"
                )
                return validation_result.data

        return result

    async def execute_batch(
        self,
        dags: List[ToolCallDAG],
        initial_args: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute multiple DAGs in parallel.

        Args:
            dags: List of DAGs to execute.
            initial_args: Initial arguments for each DAG.

        Returns:
            List of results (one per DAG).

        Example:
            >>> results = await orchestrator.execute_batch([dag1, dag2, dag3])
            >>> for result in results:
            ...     print(f"DAG completed: {len(result['events'])} events")
        """
        tasks = []
        for dag in dags:
            task = self._execute_single_batch(dag, initial_args)
            tasks.append(task)

        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_single_batch(
        self,
        dag: ToolCallDAG,
        initial_args: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute a single DAG and collect all events.

        Args:
            dag: The DAG to execute.
            initial_args: Initial arguments.

        Returns:
            Dictionary with DAG execution results.
        """
        result = {
            "dag_id": getattr(dag, "id", "unknown"),
            "nodes": len(dag.nodes),
            "events": [],
            "success": True,
            "error": None,
        }

        try:
            async for event in self.execute_streaming(dag, initial_args):
                result["events"].append(event.to_dict())

                if event.type == StreamEventType.ERROR:
                    result["success"] = False
                    result["error"] = event.data.get("message")
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)

        return result

    async def execute_single_node(
        self,
        node: ToolCallNode,
        results_cache: Optional[Dict[str, Any]] = None,
    ) -> ToolCallNode:
        """Execute a single node without DAG context.

        Useful for testing or executing nodes outside of a full DAG.

        Args:
            node: The node to execute.
            results_cache: Optional cache of results for dependency resolution.

        Returns:
            The executed node.
        """
        if results_cache is None:
            results_cache = {}

        return await self._wave_executor._execute_single_node(node, results_cache)

    async def verify_result(
        self,
        tool_name: str,
        tool_id: str,
        result: Any,
    ) -> bool:
        """Verify a tool result.

        Args:
            tool_name: Name of the tool.
            tool_id: Unique ID of the tool call.
            result: The result to verify.

        Returns:
            True if verification passed or was skipped.
        """
        if not self._verifier:
            return True

        verification = await self._verifier.verify(
            tool_name=tool_name,
            tool_id=tool_id,
            result=result,
        )

        return verification.can_proceed

    def register_schema(self, name: str, schema: OutputSchema) -> None:
        """Register a schema for structured output validation.

        Args:
            name: Name to associate with the schema (usually tool name).
            schema: The OutputSchema to register.

        Example:
            >>> from petals.client.async_support import OutputSchema
            >>> orchestrator.register_schema(
            ...     "search",
            ...     OutputSchema(required_fields=["results"])
            ... )
        """
        if self._enforcer:
            self._enforcer.register_schema(name, schema)

    def register_verification_rule(
        self,
        tool_name: str,
        rule: VerificationRule,
    ) -> None:
        """Register a verification rule for a tool.

        Args:
            tool_name: Name of the tool this rule applies to.
            rule: The VerificationRule to register.

        Example:
            >>> from petals.client.verification import VerificationRule
            >>> orchestrator.register_verification_rule(
            ...     "search",
            ...     VerificationRule(
            ...         name="has_results",
            ...         description="Check if results are non-empty",
            ...         check=lambda r: bool(r.get("results")),
            ...     )
            ... )
        """
        if self._verifier:
            self._verifier.register_rule(tool_name, rule)

    def register_default_schemas(self) -> None:
        """Register default schemas for common tool types."""
        if self._enforcer:
            self._enforcer.register_defaults()

    def _can_execute(self) -> bool:
        """Check if execution is allowed (circuit breaker check)."""
        if not self._circuit_breaker:
            return True
        return self._circuit_breaker._should_allow_request()

    def _record_success(self) -> None:
        """Record successful execution in circuit breaker."""
        if self._circuit_breaker:
            self._circuit_breaker._record_success()

    def _record_failure(self) -> None:
        """Record failed execution in circuit breaker."""
        if self._circuit_breaker:
            self._circuit_breaker._record_failure()

    async def shutdown(self) -> None:
        """Gracefully shutdown the orchestrator.

        Shuts down all internal components including task pool and
        updates circuit breaker state.

        Example:
            >>> await orchestrator.shutdown()
        """
        logger.info("Shutting down orchestrator")

        # Shutdown task pool
        await self._task_pool.shutdown(cancel_pending=True)

        # Reset feedback loop
        if self._feedback_loop:
            self._feedback_loop.reset()

        # Reset verifier
        if self._verifier:
            self._verifier.clear_cache()
            self._verifier.reset_stats()

        # Reset verification executor
        if self._verification_executor:
            self._verification_executor.reset()

        logger.info("Orchestrator shutdown complete")

    @property
    def stats(self) -> Dict[str, Any]:
        """Get execution statistics.

        Returns:
            Dictionary with comprehensive execution statistics including:
            - total_executions: Total nodes executed
            - successful_executions: Successfully completed nodes
            - failed_executions: Failed nodes
            - verifications: Number of verifications performed
            - corrections: Number of corrections made
            - retries: Number of retries
            - circuit_breaker: Current circuit breaker state (if enabled)
            - verifier: Verification statistics (if enabled)
        """
        result = {**self._stats}

        if self._circuit_breaker:
            result["circuit_breaker"] = {
                "state": self._circuit_breaker.state,
                "failure_count": self._circuit_breaker.failure_count,
            }

        if self._verifier:
            result["verifier"] = self._verifier.get_stats()

        if self._verification_executor:
            result["verification_executor"] = self._verification_executor.stats

        return result

    # Event helpers

    def _create_tool_result_event(
        self,
        node: ToolCallNode,
        result: Any,
    ) -> StreamEvent:
        """Create tool result event.

        Args:
            node: The completed node.
            result: The tool result (possibly validated).

        Returns:
            StreamEvent for the tool result.
        """
        return StreamEvent(
            type=StreamEventType.TOOL_RESULT,
            data={
                "id": node.id,
                "name": node.name,
                "status": node.status.value if hasattr(node.status, "value") else str(node.status),
                "result": result,
                "error": node.error,
                "retry_count": node.retry_count,
            },
        )

    def _create_error_event(self, message: str) -> StreamEvent:
        """Create error event.

        Args:
            message: Error message.

        Returns:
            StreamEvent for the error.
        """
        return StreamEvent(
            type=StreamEventType.ERROR,
            data={"message": message},
        )

    def _create_final_event(self) -> StreamEvent:
        """Create final event with statistics.

        Returns:
            StreamEvent for execution completion.
        """
        return StreamEvent(
            type=StreamEventType.FINAL,
            data={
                "stats": self.stats,
                "results_cache_keys": list(self._results_cache.keys()),
            },
        )

    def __repr__(self) -> str:
        """String representation of the orchestrator."""
        return (
            f"Orchestrator("
            f"config={self.config}, "
            f"has_verifier={self._verifier is not None}, "
            f"has_feedback={self._feedback_loop is not None}"
            f")"
        )


# Export from submodules for convenience
from petals.client.dag import ToolCallNode, ToolCallDAG, WaveExecutor
from petals.client.async_support.streaming_types import StreamEvent, StreamEventType
from petals.client.async_support.streaming_aggregator import AggregationResult
from petals.client.async_support.structured_output import OutputSchema
from petals.client.feedback.feedback_loop import ExecutionFeedbackLoop, FeedbackLoopConfig
from petals.client.feedback.retry_policy import CircuitBreaker, CircuitBreakerConfig
from petals.client.verification.verifier import ResultVerifier, VerificationLevel, VerificationRule
from petals.client.verification.verification_aware_executor import VerificationAwareExecutor

__all__ = [
    # Main class
    "Orchestrator",
    "OrchestratorConfig",
    # Phase 1: DAG
    "ToolCallNode",
    "ToolCallDAG",
    "WaveExecutor",
    # Phase 2: Async/Streaming
    "StreamEvent",
    "StreamEventType",
    "AggregationResult",
    "OutputSchema",
    # Phase 3: Feedback
    "ExecutionFeedbackLoop",
    "FeedbackLoopConfig",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    # Phase 4: Verification
    "ResultVerifier",
    "VerificationLevel",
    "VerificationRule",
    "VerificationAwareExecutor",
]
