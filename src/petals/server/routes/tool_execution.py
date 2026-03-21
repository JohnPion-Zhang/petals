"""
SSE streaming endpoint for tool execution.

Provides:
- POST /api/v1/execute - Execute a DAG with streaming response
- POST /api/v1/execute/batch - Execute multiple DAGs
- GET /api/v1/health - Health check

Example:
    >>> from fastapi import FastAPI
    >>> from petals.server.routes.tool_execution import router, init_orchestrator
    >>> from petals.client.tool_registry import ToolRegistry
    >>>
    >>> app = FastAPI()
    >>> app.include_router(router)
    >>>
    >>> registry = ToolRegistry()
    >>> # ... register tools ...
    >>> init_orchestrator(registry)
"""
import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from petals.client.dag import ToolCallNode, ToolCallDAG
from petals.client.tool_registry import ToolRegistry
from petals.client.orchestrator import Orchestrator, OrchestratorConfig
from petals.client.async_support.streaming_types import StreamEvent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["execution"])

# Global orchestrator instance (initialized on startup)
_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    """Get the global orchestrator instance.

    Returns:
        The initialized Orchestrator instance.

    Raises:
        HTTPException: 503 if orchestrator is not initialized.
    """
    if _orchestrator is None:
        raise HTTPException(503, "Orchestrator not initialized")
    return _orchestrator


# --- Pydantic Models ---

class ToolCallInput(BaseModel):
    """Input for a single tool call.

    Attributes:
        name: Tool name (required).
        arguments: Tool arguments dictionary.
        id: Optional tool call ID (auto-generated if not provided).
        dependencies: List of dependency node IDs.
        execution_key: Optional key for deduplication.
        requires_verification: Whether verification is required.
    """

    name: str = Field(..., description="Tool name")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    id: Optional[str] = Field(None, description="Optional tool call ID")
    dependencies: List[str] = Field(default_factory=list, description="Dependency IDs")
    execution_key: Optional[str] = Field(None, description="For deduplication")
    requires_verification: bool = Field(False, description="Require verification")


class DAGInput(BaseModel):
    """Input for a DAG to execute.

    Attributes:
        nodes: List of tool calls in the DAG.
        edges: List of edges as (from_id, to_id) tuples.
        initial_args: Initial arguments for root nodes.
    """

    nodes: List[ToolCallInput] = Field(..., description="Tool calls in the DAG")
    edges: List[List[str]] = Field(default_factory=list, description="Edges [from_id, to_id]")
    initial_args: Dict[str, Any] = Field(default_factory=dict, description="Initial arguments")


class ExecuteRequest(BaseModel):
    """Request to execute a DAG.

    Attributes:
        dag: The DAG to execute.
        config: Optional orchestrator config overrides.
    """

    dag: DAGInput
    config: Optional[Dict[str, Any]] = Field(None, description="Orchestrator config overrides")


class BatchExecuteRequest(BaseModel):
    """Request to execute multiple DAGs.

    Attributes:
        dags: List of DAGs to execute.
        parallel: Whether to execute in parallel (default True).
    """

    dags: List[DAGInput]
    parallel: bool = Field(True, description="Execute in parallel")


# --- Endpoints ---

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint.

    Returns:
        Health status with service name.
    """
    return {"status": "healthy", "service": "tool-execution"}


@router.post("/execute")
async def execute_dag(request: ExecuteRequest) -> StreamingResponse:
    """Execute a DAG with streaming response via SSE.

    Returns Server-Sent Events stream with tool execution progress.

    Args:
        request: The execution request containing the DAG and optional config.

    Returns:
        StreamingResponse with SSE content type.

    SSE Events:
        - tool_call_pending: Tool detected in DAG
        - tool_executing: Tool execution started
        - tool_result: Tool completed (success or failure)
        - error: Error occurred
        - final: All execution complete
    """
    orchestrator = get_orchestrator()

    # Build DAG from input
    dag = _build_dag(request.dag)

    async def generate_sse() -> AsyncGenerator[str, None]:
        """Generate SSE stream from orchestrator events."""
        try:
            async for event in orchestrator.execute_streaming(dag, request.dag.initial_args):
                yield _format_sse_event(event)
        except asyncio.CancelledError:
            logger.info("SSE connection cancelled")
        except Exception as e:
            logger.exception("SSE stream error")
            yield _format_error_event(str(e))

    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.post("/execute/batch")
async def execute_batch(request: BatchExecuteRequest) -> Dict[str, Any]:
    """Execute multiple DAGs.

    If parallel=True, executes all DAGs concurrently.
    Otherwise, executes sequentially and returns events.

    Args:
        request: Batch execution request with DAGs.

    Returns:
        Dictionary with results list.
    """
    orchestrator = get_orchestrator()

    if request.parallel:
        # Execute all DAGs concurrently
        dags = [_build_dag(dag_input) for dag_input in request.dags]
        results = await orchestrator.execute_batch(dags)
        return {"results": results}
    else:
        # Execute sequentially, collecting events
        results = []
        for dag_input in request.dags:
            dag = _build_dag(dag_input)
            result = {
                "dag_id": getattr(dag, "id", f"dag_{len(results)}"),
                "nodes": len(dag.nodes),
                "events": [],
                "success": True,
                "error": None,
            }

            try:
                async for event in orchestrator.execute_streaming(dag, dag_input.initial_args):
                    result["events"].append(event.to_dict())
            except Exception as e:
                result["success"] = False
                result["error"] = str(e)

            results.append(result)

        return {"results": results}


# --- Helper Functions ---

def _build_dag(input: DAGInput) -> ToolCallDAG:
    """Build a ToolCallDAG from input.

    Args:
        input: DAGInput containing nodes and edges.

    Returns:
        Constructed ToolCallDAG instance.

    Example:
        >>> dag_input = DAGInput(
        ...     nodes=[
        ...         {"id": "a", "name": "add", "arguments": {"a": 1, "b": 2}},
        ...         {"id": "b", "name": "multiply", "arguments": {"a": 0, "b": 0}, "dependencies": ["a"]},
        ...     ],
        ...     edges=[],
        ...     initial_args={}
        ... )
        >>> dag = _build_dag(dag_input)
        >>> len(dag.nodes)
        2
    """
    dag = ToolCallDAG()
    node_map: Dict[str, ToolCallNode] = {}

    # Add nodes
    for idx, node_input in enumerate(input.nodes):
        node_id = node_input.id or f"node_{idx}"
        node = ToolCallNode(
            id=node_id,
            name=node_input.name,
            arguments=node_input.arguments,
            dependencies=list(node_input.dependencies),
            execution_key=node_input.execution_key,
            requires_verification=node_input.requires_verification,
        )
        dag.add_node(node)
        node_map[node_id] = node

    # Add edges
    for edge in input.edges:
        if len(edge) >= 2:
            from_id, to_id = edge[0], edge[1]
            dag.add_edge(from_id, to_id)

    return dag


def _format_sse_event(event: StreamEvent) -> str:
    """Format a StreamEvent as SSE data.

    Args:
        event: The StreamEvent to format.

    Returns:
        SSE-formatted string.
    """
    # Convert event data to JSON, handling non-serializable objects
    try:
        data_json = json.dumps(event.data, default=_json_serializer)
    except (TypeError, ValueError):
        # Fallback for complex objects
        data_json = json.dumps({"type": event.type.value, "message": "Complex object"})

    return f"event: {event.type.value}\ndata: {data_json}\n\n"


def _format_error_event(message: str) -> str:
    """Format an error as SSE event.

    Args:
        message: Error message.

    Returns:
        SSE-formatted error string.
    """
    data = json.dumps({"message": message, "type": "server_error"})
    return f"event: error\ndata: {data}\n\n"


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for non-standard objects.

    Args:
        obj: Object to serialize.

    Returns:
        JSON-serializable representation.
    """
    if hasattr(obj, "__dict__"):
        return {"_type": type(obj).__name__, "data": obj.__dict__}
    elif hasattr(obj, "value"):  # Enum values
        return obj.value
    elif hasattr(obj, "to_dict"):
        return obj.to_dict()
    else:
        return str(obj)


# --- Initialization Functions ---

def init_orchestrator(
    registry: ToolRegistry,
    config: Optional[OrchestratorConfig] = None,
) -> None:
    """Initialize the global orchestrator.

    Args:
        registry: ToolRegistry with registered tools.
        config: Optional OrchestratorConfig.
    """
    global _orchestrator
    _orchestrator = Orchestrator(registry=registry, config=config)
    logger.info("Orchestrator initialized")


async def shutdown_orchestrator() -> None:
    """Shutdown the global orchestrator gracefully."""
    global _orchestrator
    if _orchestrator:
        await _orchestrator.shutdown()
        _orchestrator = None
        logger.info("Orchestrator shutdown complete")
