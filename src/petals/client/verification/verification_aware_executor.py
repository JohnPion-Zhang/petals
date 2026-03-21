"""
VerificationAwareExecutor - DAG Execution with Integrated Verification

Executes DAG with RLM-style verification between waves. Verifies child
results before parent aggregation using the ResultVerifier and TriggerConfig.

Features:
- Verifies child results before parent aggregation
- Triggers verification based on configuration
- Handles verification failures gracefully
- Integrates with feedback loop for correction
"""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Set
import logging

from petals.client.dag import ToolCallNode, ToolCallDAG
from petals.client.verification.verifier import (
    ResultVerifier,
    VerificationResult,
    VerificationStatus,
    VerificationLevel,
)
from petals.client.verification.triggers import TriggerConfig, VerificationTrigger

logger = logging.getLogger(__name__)


class VerificationAwareExecutor:
    """Executes DAG with RLM-style verification.

    This executor extends the base DAG execution with verification between waves.
    It follows the RLM pattern where parent nodes wait for child verification
    before aggregation.

    Features:
    - Verifies child results before parent aggregation
    - Triggers verification based on configuration
    - Handles verification failures gracefully
    - Integrates with feedback loop for correction
    - Provides detailed verification statistics

    Example:
        >>> verifier = ResultVerifier(verification_level=VerificationLevel.STRUCTURAL)
        >>> config = TriggerConfig.default_config()
        >>> executor = VerificationAwareExecutor(verifier, config)
        >>>
        >>> async for node in executor.execute_verified_dag(dag, base_executor):
        ...     print(f"Completed and verified: {node.id}")
    """

    def __init__(
        self,
        verifier: ResultVerifier,
        trigger_config: Optional[TriggerConfig] = None,
        feedback_loop: Optional[Any] = None,
        fail_on_verification_failure: bool = False,
    ):
        """Initialize the VerificationAwareExecutor.

        Args:
            verifier: ResultVerifier instance for performing verification.
            trigger_config: Optional TriggerConfig for when to verify.
            feedback_loop: Optional ExecutionFeedbackLoop for correction.
            fail_on_verification_failure: If True, fail node on verification failure.
        """
        self.verifier = verifier
        self.trigger_config = trigger_config or TriggerConfig.default_config()
        self.feedback_loop = feedback_loop
        self.fail_on_verification_failure = fail_on_verification_failure

        # Statistics
        self._verification_count = 0
        self._failed_verifications = 0
        self._skipped_verifications = 0
        self._verified_results: Dict[str, VerificationResult] = {}

        # Track nodes that passed/failed verification
        self._verified_nodes: Set[str] = set()
        self._failed_nodes: Set[str] = set()

    async def execute_verified_dag(
        self,
        dag: ToolCallDAG,
        executor: Any,
    ) -> AsyncGenerator[ToolCallNode, None]:
        """Execute DAG with verification between waves.

        Algorithm:
        1. Execute wave
        2. For each completed node: check if verification needed
        3. If verification triggered: verify result
        4. If verification fails: handle via feedback loop or fail
        5. Proceed to next wave

        Args:
            dag: The ToolCallDAG to execute.
            executor: Base executor (e.g., WaveExecutor) for node execution.

        Yields:
            Completed ToolCallNodes as they finish verification.
        """
        # Check for cycles
        cycle = dag.detect_cycle()
        if cycle:
            raise ValueError(f"Cannot execute DAG with cycle: {' -> '.join(cycle)}")

        # Get waves
        waves = dag.get_waves()

        logger.info(
            f"Executing verified DAG: {len(dag)} nodes in {len(waves)} waves"
        )

        # Track completed nodes
        completed: set = set()

        for wave_idx, wave in enumerate(waves):
            logger.debug(f"Executing wave {wave_idx + 1}/{len(waves)}")

            # Wait for all children in previous waves to be verified
            for node in wave:
                children_verified = await self._wait_for_children(node, dag, completed)
                if not children_verified:
                    # Children failed verification, mark this node as failed
                    node.mark_failed(
                        "Child verification failed - cannot aggregate"
                    )
                    self._failed_nodes.add(node.id)
                    yield node
                    continue

            # Execute the current wave
            wave_nodes = [n for n in wave if n.id not in self._failed_nodes]

            if not wave_nodes:
                continue

            # Execute nodes in parallel
            wave_tasks = [self._execute_and_verify(node, executor) for node in wave_nodes]
            results = await asyncio.gather(*wave_tasks, return_exceptions=True)

            # Process results
            for node, result in zip(wave_nodes, results):
                if isinstance(result, Exception):
                    logger.error(f"Wave execution error for {node.id}: {result}")
                    node.mark_failed(str(result))
                    self._failed_nodes.add(node.id)
                elif isinstance(result, ToolCallNode):
                    # Mark as completed
                    completed.add(node.id)

                    if node.id in self._verified_nodes:
                        # Verification passed
                        self._verification_count += 1
                    elif node.id in self._failed_nodes:
                        # Verification failed
                        self._failed_verifications += 1
                    else:
                        # Verification skipped
                        self._skipped_verifications += 1

                    yield result

        logger.info(
            f"DAG execution complete: "
            f"{self._verification_count} verified, "
            f"{self._failed_verifications} failed, "
            f"{self._skipped_verifications} skipped"
        )

    async def _execute_and_verify(
        self,
        node: ToolCallNode,
        executor: Any,
    ) -> ToolCallNode:
        """Execute a node and verify its result.

        Args:
            node: The ToolCallNode to execute.
            executor: Base executor for node execution.

        Returns:
            The executed node with updated status.
        """
        # Check if this node needs verification
        if not self._needs_verification(node):
            # Execute without verification
            return await self._execute_node(node, executor)

        # Execute node
        executed_node = await self._execute_node(node, executor)

        if executed_node.status.name != "DONE":
            # Execution failed, no need to verify
            return executed_node

        # Verify the result
        verification = await self._verify_node(node, {})

        # Store verification result
        self._verified_results[node.id] = verification

        # Handle verification result
        if verification.is_verified:
            self._verified_nodes.add(node.id)
            logger.debug(f"Node {node.id} verified successfully")
        elif verification.status == VerificationStatus.SKIPPED:
            self._skipped_verifications += 1
            logger.debug(f"Node {node.id} verification skipped")
        else:
            self._failed_nodes.add(node.id)
            await self._handle_verification_failure(node, verification)

        return node

    async def _execute_node(self, node: ToolCallNode, executor: Any) -> ToolCallNode:
        """Execute a single node using the base executor.

        Args:
            node: The node to execute.
            executor: Base executor.

        Returns:
            The executed node.
        """
        try:
            # Check if executor has execute_dag method
            if hasattr(executor, "execute_dag"):
                # Create a single-node DAG for this node
                single_dag = ToolCallDAG()
                single_dag.add_node(node)

                # Execute
                async for completed in executor.execute_dag(single_dag):
                    pass  # Just execute, result is in node

                return node
            elif hasattr(executor, "execute_node"):
                # Direct node execution
                return await executor.execute_node(node)
            else:
                # Simple execution via registry
                if hasattr(executor, "registry"):
                    result = await executor.registry.execute(node.name, node.arguments)
                    node.mark_done(result)
                return node

        except Exception as e:
            logger.error(f"Execution error for {node.id}: {e}")
            node.mark_failed(str(e))
            return node

    def _needs_verification(self, node: ToolCallNode) -> bool:
        """Check if a node needs verification.

        Args:
            node: The node to check.

        Returns:
            True if verification should be performed.
        """
        # Check if node has verification flag
        if node.requires_verification:
            return True

        # Check trigger config
        return self.trigger_config.should_verify(
            tool_name=node.name,
            result=node.result,
            node=node,
        )

    async def _verify_node(
        self,
        node: ToolCallNode,
        context: Dict[str, Any],
    ) -> VerificationResult:
        """Verify a single node's result.

        Args:
            node: The node to verify.
            context: Additional context for verification.

        Returns:
            VerificationResult with verification status.
        """
        # Check if verifier can handle this tool
        if not self.verifier.should_verify(node.name):
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                tool_name=node.name,
                tool_id=node.id,
                details={"reason": "no_rules_or_llm"},
            )

        # Perform verification
        result = await self.verifier.verify(
            tool_name=node.name,
            tool_id=node.id,
            result=node.result,
            context=context,
        )

        return result

    async def _handle_verification_failure(
        self,
        node: ToolCallNode,
        verification: VerificationResult,
    ) -> None:
        """Handle failed verification.

        Args:
            node: The node that failed verification.
            verification: The verification result with failure details.
        """
        logger.warning(
            f"Verification failed for {node.id}: "
            f"failed_rules={verification.failed_rules}"
        )

        if self.fail_on_verification_failure:
            # Mark node as failed
            node.mark_failed(
                f"Verification failed: {', '.join(verification.failed_rules)}",
                error_feedback=verification.llm_feedback,
            )
        elif self.feedback_loop is not None:
            # Try correction via feedback loop
            try:
                await self.feedback_loop.execute_with_feedback(node)
            except Exception as e:
                logger.error(f"Feedback loop error for {node.id}: {e}")
                node.mark_failed(f"Verification failed: {str(e)}")
        else:
            # Just log and continue
            node.error_feedback = f"Verification issues: {verification.llm_feedback}"

    async def _wait_for_children(
        self,
        node: ToolCallNode,
        dag: ToolCallDAG,
        completed: set,
    ) -> bool:
        """Wait for all child dependencies to complete and verify.

        This implements the RLM pattern where parent waits for child
        verification before aggregation.

        Args:
            node: The parent node waiting for children.
            dag: The DAG containing the node.
            completed: Set of completed node IDs.

        Returns:
            True if all children verified successfully, False otherwise.
        """
        # Get child nodes from dependencies
        for dep_id in node.dependencies:
            child_node = dag.get_node(dep_id)
            if child_node is None:
                continue

            # Check if child is verified
            if dep_id not in completed:
                logger.warning(
                    f"Parent {node.id} waiting for child {dep_id} - not yet completed"
                )
                return False

            # Check if child failed verification
            if dep_id in self._failed_nodes:
                logger.warning(
                    f"Parent {node.id} has failed child {dep_id} - cannot aggregate"
                )
                return False

        return True

    def get_verification_result(self, node_id: str) -> Optional[VerificationResult]:
        """Get verification result for a node.

        Args:
            node_id: ID of the node.

        Returns:
            VerificationResult if available, None otherwise.
        """
        return self._verified_results.get(node_id)

    def is_verified(self, node_id: str) -> bool:
        """Check if a node passed verification.

        Args:
            node_id: ID of the node.

        Returns:
            True if node passed verification.
        """
        return node_id in self._verified_nodes

    @property
    def stats(self) -> Dict[str, Any]:
        """Return verification statistics.

        Returns:
            Dictionary with verification statistics.
        """
        total = self._verification_count + self._failed_verifications + self._skipped_verifications
        success_rate = (
            self._verification_count / max(1, self._verification_count + self._failed_verifications)
            if self._failed_verifications > 0
            else 1.0
        )

        return {
            "total_verifications": total,
            "verification_count": self._verification_count,
            "failed_verifications": self._failed_verifications,
            "skipped_verifications": self._skipped_verifications,
            "success_rate": success_rate,
            "verified_nodes": list(self._verified_nodes),
            "failed_nodes": list(self._failed_nodes),
            "verifier_stats": self.verifier.get_stats(),
            "trigger_stats": self.trigger_config.get_stats(),
        }

    def reset(self) -> None:
        """Reset executor state."""
        self._verification_count = 0
        self._failed_verifications = 0
        self._skipped_verifications = 0
        self._verified_results.clear()
        self._verified_nodes.clear()
        self._failed_nodes.clear()
        self.trigger_config.reset_counts()
        logger.info("VerificationAwareExecutor state reset")

    async def verify_existing_node(
        self,
        node: ToolCallNode,
    ) -> VerificationResult:
        """Verify an already-executed node.

        Useful for on-demand verification of nodes that were executed
        before the verification system was integrated.

        Args:
            node: The node to verify.

        Returns:
            VerificationResult.
        """
        if node.result is None:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                tool_name=node.name,
                tool_id=node.id,
                details={"reason": "no_result"},
            )

        return await self._verify_node(node, {})
