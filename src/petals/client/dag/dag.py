"""
ToolCallDAG - Directed Acyclic Graph for ToolCall Orchestration

A DAG structure for managing tool call dependencies, computing execution
orderings, and detecting cycles.
"""
import logging
from collections import deque
from typing import Any, Dict, List, Optional, Set

from petals.client.dag.tool_call_node import ToolCallNode
from petals.data_structures import CallStatus

logger = logging.getLogger(__name__)


class ToolCallDAG:
    """Directed Acyclic Graph for ToolCall orchestration.

    Features:
    - Topological sort for execution ordering (Kahn's algorithm)
    - Wave-based parallel execution computation
    - Shared execution key deduplication
    - Cycle detection using DFS

    Attributes:
        nodes: Dictionary mapping node IDs to ToolCallNode instances.

    Example:
        >>> dag = ToolCallDAG()
        >>> node_a = ToolCallNode(id="a", name="first", arguments={})
        >>> node_b = ToolCallNode(id="b", name="second", arguments={}, dependencies=["a"])
        >>> dag.add_node(node_a)
        >>> dag.add_node(node_b)
        >>> waves = dag.get_waves()
        >>> # [[node_a], [node_b]] - b waits for a
    """

    def __init__(self):
        """Initialize an empty DAG."""
        self.nodes: Dict[str, ToolCallNode] = {}

    def add_node(self, node: ToolCallNode) -> None:
        """Add a node to the DAG.

        Automatically updates dependent nodes' dependencies and
        this node's dependents lists.

        Args:
            node: The ToolCallNode to add.
        """
        if node.id in self.nodes:
            raise ValueError(f"Node with id '{node.id}' already exists in DAG")

        self.nodes[node.id] = node
        self._update_dependencies(node)
        self._update_dependents(node)

        logger.debug(f"Added node: {node.id}")

    def _update_dependencies(self, node: ToolCallNode) -> None:
        """Update node's dependency references to point to actual nodes.

        Args:
            node: The node whose dependencies to update.
        """
        # Ensure all dependencies exist in the DAG
        for dep_id in node.dependencies:
            if dep_id not in self.nodes:
                logger.warning(f"Dependency '{dep_id}' for node '{node.id}' not found in DAG")

    def _update_dependents(self, node: ToolCallNode) -> None:
        """Update dependent nodes to reference this node.

        When a node is added, this method updates all its dependency
        nodes to include this node in their dependents list.

        Args:
            node: The node whose dependents to update.
        """
        for dep_id in node.dependencies:
            if dep_id in self.nodes:
                dep_node = self.nodes[dep_id]
                if node.id not in dep_node.dependents:
                    dep_node.dependents.append(node.id)

    def add_edge(self, from_id: str, to_id: str) -> None:
        """Add a dependency edge: from_id must complete before to_id.

        Args:
            from_id: ID of the source node (dependency).
            to_id: ID of the target node (dependent).

        Raises:
            ValueError: If either node ID doesn't exist in the DAG.
        """
        if from_id not in self.nodes:
            raise ValueError(f"Source node '{from_id}' not found in DAG")
        if to_id not in self.nodes:
            raise ValueError(f"Target node '{to_id}' not found in DAG")

        from_node = self.nodes[from_id]
        to_node = self.nodes[to_id]

        # Add to_id's dependency on from_id
        if from_id not in to_node.dependencies:
            to_node.dependencies.append(from_id)

        # Add to_node to from_id's dependents
        if to_id not in from_node.dependents:
            from_node.dependents.append(to_id)

        logger.debug(f"Added edge: {from_id} -> {to_id}")

    def remove_edge(self, from_id: str, to_id: str) -> None:
        """Remove a dependency edge.

        Args:
            from_id: ID of the source node.
            to_id: ID of the target node.
        """
        if from_id in self.nodes and to_id in self.nodes:
            from_node = self.nodes[from_id]
            to_node = self.nodes[to_id]

            if from_id in to_node.dependencies:
                to_node.dependencies.remove(from_id)
            if to_id in from_node.dependents:
                from_node.dependents.remove(to_id)

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the DAG.

        Also removes all edges to/from this node.

        Args:
            node_id: ID of the node to remove.
        """
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]

        # Remove from dependents of dependencies
        for dep_id in node.dependencies:
            if dep_id in self.nodes:
                dep_node = self.nodes[dep_id]
                if node_id in dep_node.dependents:
                    dep_node.dependents.remove(node_id)

        # Remove from dependencies of dependents
        for dep_id in node.dependents:
            if dep_id in self.nodes:
                dep_node = self.nodes[dep_id]
                if node_id in dep_node.dependencies:
                    dep_node.dependencies.remove(node_id)

        del self.nodes[node_id]

    def get_node(self, node_id: str) -> Optional[ToolCallNode]:
        """Get a node by ID.

        Args:
            node_id: ID of the node to retrieve.

        Returns:
            The ToolCallNode if found, None otherwise.
        """
        return self.nodes.get(node_id)

    def topological_sort(self) -> List[ToolCallNode]:
        """Compute topological sort using Kahn's algorithm.

        Returns nodes in an order where all dependencies come before dependents.

        Returns:
            List of nodes in topological order.

        Raises:
            ValueError: If a cycle is detected in the DAG.
        """
        # Calculate in-degree for each node
        in_degree: Dict[str, int] = {node_id: len(node.dependencies) for node_id, node in self.nodes.items()}

        # Queue of nodes with no dependencies
        queue: deque = deque()
        for node_id, degree in in_degree.items():
            if degree == 0:
                queue.append(node_id)

        result: List[ToolCallNode] = []

        while queue:
            node_id = queue.popleft()
            node = self.nodes[node_id]
            result.append(node)

            # Reduce in-degree for all dependents
            for dependent_id in node.dependents:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)

        if len(result) != len(self.nodes):
            raise ValueError("Cycle detected in DAG during topological sort")

        return result

    def get_waves(self) -> List[List[ToolCallNode]]:
        """Compute execution waves using topological sort.

        Returns nodes grouped by wave, where each wave contains nodes
        that can execute in parallel (all dependencies in previous waves).

        Returns:
            List of waves, each wave is a list of nodes that can execute in parallel.

        Raises:
            ValueError: If a cycle is detected in the DAG.
        """
        waves: List[List[ToolCallNode]] = []
        remaining: Set[str] = set(self.nodes.keys())
        completed: Set[str] = set()

        while remaining:
            wave: List[ToolCallNode] = []

            for node_id in list(remaining):
                node = self.nodes[node_id]
                # Node is ready if all dependencies are completed
                if all(dep_id in completed for dep_id in node.dependencies):
                    wave.append(node)

            if not wave:
                # No nodes ready but still have remaining - cycle detected
                raise ValueError(
                    f"Cycle detected in DAG. Remaining nodes: {remaining}"
                )

            waves.append(wave)

            # Mark all nodes in this wave as completed
            for node in wave:
                remaining.remove(node.id)
                completed.add(node.id)

        logger.debug(f"Computed {len(waves)} waves for {len(self.nodes)} nodes")
        return waves

    def get_ready_nodes(self, completed: Set[str]) -> List[ToolCallNode]:
        """Get nodes whose dependencies are all satisfied.

        Args:
            completed: Set of node IDs that have completed execution.

        Returns:
            List of nodes ready to execute.
        """
        ready: List[ToolCallNode] = []

        for node in self.nodes.values():
            if node.status == CallStatus.PENDING:
                if all(dep_id in completed for dep_id in node.dependencies):
                    ready.append(node)

        return ready

    def get_failed_nodes(self) -> List[ToolCallNode]:
        """Get all failed nodes in the DAG.

        Returns:
            List of nodes with FAILED status.
        """
        return [node for node in self.nodes.values() if node.status == CallStatus.FAILED]

    def get_completed_nodes(self) -> List[ToolCallNode]:
        """Get all successfully completed nodes in the DAG.

        Returns:
            List of nodes with DONE status.
        """
        return [node for node in self.nodes.values() if node.status == CallStatus.DONE]

    def get_running_nodes(self) -> List[ToolCallNode]:
        """Get all currently running nodes in the DAG.

        Returns:
            List of nodes with RUNNING status.
        """
        return [node for node in self.nodes.values() if node.status == CallStatus.RUNNING]

    def get_pending_nodes(self) -> List[ToolCallNode]:
        """Get all pending nodes in the DAG.

        Returns:
            List of nodes with PENDING status.
        """
        return [node for node in self.nodes.values() if node.status == CallStatus.PENDING]

    def detect_cycle(self) -> Optional[List[str]]:
        """Detect cycles in the DAG using DFS.

        Returns:
            List of node IDs forming the cycle if found, None otherwise.
        """
        # Use three-color DFS for cycle detection
        # WHITE (0) = unvisited, GRAY (1) = in progress, BLACK (2) = done

        color: Dict[str, int] = {node_id: 0 for node_id in self.nodes}
        parent: Dict[str, Optional[str]] = {node_id: None for node_id in self.nodes}

        def dfs(node_id: str) -> Optional[List[str]]:
            """DFS to detect cycle starting from node."""
            color[node_id] = 1  # Mark as GRAY (in progress)

            node = self.nodes[node_id]
            for dep_id in node.dependencies:
                if dep_id not in self.nodes:
                    continue

                if color[dep_id] == 1:  # Found a back edge - cycle!
                    # Reconstruct cycle path
                    cycle_path = [dep_id]
                    current = node_id
                    while current != dep_id:
                        cycle_path.append(current)
                        current = parent[current]  # type: ignore
                    cycle_path.reverse()
                    return cycle_path

                if color[dep_id] == 0:  # WHITE - not visited
                    parent[dep_id] = node_id
                    cycle = dfs(dep_id)
                    if cycle:
                        return cycle

            color[node_id] = 2  # Mark as BLACK (done)
            return None

        # Check each unvisited node
        for node_id in self.nodes:
            if color[node_id] == 0:
                cycle = dfs(node_id)
                if cycle:
                    logger.warning(f"Cycle detected: {' -> '.join(cycle)}")
                    return cycle

        return None

    def get_shared_execution_groups(self) -> Dict[str, List[ToolCallNode]]:
        """Group nodes by execution_key for deduplication.

        Nodes with the same execution_key can be deduplicated since
        they represent identical tool calls that would produce the same result.

        Returns:
            Dictionary mapping execution_key to list of nodes with that key.
            Nodes without execution_key are grouped under None.
        """
        groups: Dict[str, List[ToolCallNode]] = {}

        for node in self.nodes.values():
            key = node.execution_key
            if key not in groups:
                groups[key] = []
            groups[key].append(node)

        return groups

    def get_execution_order(self) -> List[List[str]]:
        """Get execution order as list of node ID waves.

        This is a convenience method that returns wave IDs instead of nodes.

        Returns:
            List of waves, each wave is a list of node IDs.
        """
        waves = self.get_waves()
        return [[node.id for node in wave] for wave in waves]

    def get_root_nodes(self) -> List[ToolCallNode]:
        """Get all root nodes (nodes with no dependencies).

        Returns:
            List of root nodes.
        """
        return [node for node in self.nodes.values() if node.is_root]

    def get_leaf_nodes(self) -> List[ToolCallNode]:
        """Get all leaf nodes (nodes with no dependents).

        Returns:
            List of leaf nodes.
        """
        return [node for node in self.nodes.values() if node.is_leaf]

    def get_node_depth(self, node_id: str) -> int:
        """Get the depth of a node in the DAG (distance from root).

        Args:
            node_id: ID of the node.

        Returns:
            Depth of the node (0 for root nodes).

        Raises:
            KeyError: If node_id not found in DAG.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' not found in DAG")

        node = self.nodes[node_id]
        if node.is_root:
            return 0

        # Depth is max depth of dependencies + 1
        max_dep_depth = 0
        for dep_id in node.dependencies:
            dep_depth = self.get_node_depth(dep_id)
            max_dep_depth = max(max_dep_depth, dep_depth)

        return max_dep_depth + 1

    def get_all_depths(self) -> Dict[str, int]:
        """Get depths for all nodes in the DAG.

        Returns:
            Dictionary mapping node IDs to their depths.
        """
        return {node_id: self.get_node_depth(node_id) for node_id in self.nodes}

    def clear(self) -> None:
        """Remove all nodes from the DAG."""
        self.nodes.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize DAG to dict for debugging/serialization.

        Returns:
            Dictionary representation of the DAG.
        """
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "num_nodes": len(self.nodes),
            "waves": self.get_execution_order(),
        }

    def __len__(self) -> int:
        """Get number of nodes in the DAG."""
        return len(self.nodes)

    def __contains__(self, node_id: str) -> bool:
        """Check if a node exists in the DAG."""
        return node_id in self.nodes

    def __repr__(self) -> str:
        """String representation of the DAG."""
        return f"ToolCallDAG(nodes={len(self.nodes)})"
