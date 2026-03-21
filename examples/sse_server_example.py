"""
Example: SSE Tool Execution Server

Demonstrates how to use the FastAPI-based SSE server for streaming tool execution.

Run:
    uvicorn petals.server.routes.tool_execution:app --reload

Or programmatically:
    python examples/sse_server_example.py
"""
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

from petals.server.routes.tool_execution import router, init_orchestrator, shutdown_orchestrator
from petals.client.tool_registry import ToolRegistry


# Create FastAPI app
app = FastAPI(title="Petals Tool Execution Server")
app.include_router(router)


# --- Tool Registry Setup ---

def create_tool_registry() -> ToolRegistry:
    """Create a tool registry with example tools."""
    registry = ToolRegistry()

    # Basic math tools
    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    async def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    async def divide(a: int, b: int) -> dict:
        """Divide two numbers."""
        if b == 0:
            return {"error": "Division by zero"}
        return {"result": a / b}

    # Data processing tools
    async def process_data(data: str, uppercase: bool = False) -> dict:
        """Process data with optional transformation."""
        result = data.upper() if uppercase else data
        return {"original": data, "processed": result}

    async def fetch_data(query: str) -> dict:
        """Simulate fetching data based on query."""
        await asyncio.sleep(0.1)  # Simulate network delay
        return {
            "query": query,
            "results": [f"Result for {query}", f"Another result"],
            "count": 2,
        }

    # Register all tools
    registry.register("add", add)
    registry.register("multiply", multiply)
    registry.register("divide", divide)
    registry.register("process_data", process_data)
    registry.register("fetch_data", fetch_data)

    return registry


# --- Lifecycle Events ---

@app.on_event("startup")
async def startup_event():
    """Initialize orchestrator on startup."""
    registry = create_tool_registry()
    init_orchestrator(registry)
    print("Tool execution server started!")
    print("Available endpoints:")
    print("  - GET  /api/v1/health      Health check")
    print("  - POST /api/v1/execute    Execute DAG with SSE streaming")
    print("  - POST /api/v1/execute/batch  Batch execute DAGs")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown orchestrator on shutdown."""
    await shutdown_orchestrator()
    print("Tool execution server stopped!")


# --- Main ---

if __name__ == "__main__":
    print("Starting SSE Tool Execution Server...")
    print("Docs available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
