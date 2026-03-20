"""
Tests for ToolRegistry - Agent Tool Call Layer

TDD Iteration 2: Red Phase - These tests should fail until implementation is provided.
"""
import pytest

from petals.client.tool_registry import ToolRegistry


# --- Test Fixtures ---

@pytest.fixture
def registry():
    """Create a fresh ToolRegistry instance for each test."""
    return ToolRegistry()


@pytest.fixture
def sample_schema():
    """Sample JSON schema for a calculator tool."""
    return {
        "name": "calculator",
        "description": "Performs basic arithmetic operations",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First operand"},
                "b": {"type": "number", "description": "Second operand"},
                "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]}
            },
            "required": ["a", "b", "operation"]
        }
    }


# --- Registration Tests ---

def test_register_tool_with_schema(registry, sample_schema):
    """Register a tool with a JSON schema."""
    def calculator(a: float, b: float, operation: str) -> float:
        ops = {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b}
        return ops[operation]

    registry.register("calc", calculator, schema=sample_schema)

    # Verify schema is stored
    schema = registry.get_schema()
    assert "calc" in schema
    assert schema["calc"]["name"] == "calculator"
    assert "parameters" in schema["calc"]


def test_register_duplicate_tool_raises(registry):
    """Registering the same tool name twice raises ValueError."""
    def dummy_tool():
        pass

    registry.register("tool", dummy_tool)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("tool", dummy_tool)


# --- Schema Retrieval Tests ---

def test_get_schema_returns_all_tools(registry, sample_schema):
    """get_schema returns combined schema dict for all registered tools."""
    def tool_a():
        pass

    def tool_b():
        pass

    schema_a = {"name": "tool_a", "description": "Tool A"}
    schema_b = {"name": "tool_b", "description": "Tool B"}

    registry.register("a", tool_a, schema=schema_a)
    registry.register("b", tool_b, schema=schema_b)

    schema = registry.get_schema()

    assert "a" in schema
    assert "b" in schema
    assert schema["a"]["name"] == "tool_a"
    assert schema["b"]["name"] == "tool_b"


def test_get_schema_without_schema_returns_empty_dict(registry):
    """Tools registered without schema return empty dict for that tool."""
    def no_schema_tool():
        pass

    registry.register("no_schema", no_schema_tool)

    schema = registry.get_schema()
    assert "no_schema" in schema
    assert schema["no_schema"] == {}


# --- Tool Retrieval Tests ---

def test_get_tool_returns_callable(registry):
    """get_tool returns the registered callable function."""
    def my_tool(x: int) -> int:
        return x * 2

    registry.register("doubler", my_tool)

    retrieved = registry.get_tool("doubler")
    assert retrieved is my_tool
    assert retrieved(5) == 10


def test_tool_not_found_raises(registry):
    """Requesting an unknown tool raises ValueError."""
    with pytest.raises(ValueError, match="Unknown tool"):
        registry.get_tool("nonexistent")


# --- Execution Tests ---

@pytest.mark.asyncio
async def test_tool_execution_returns_result(registry):
    """Async execution of a tool returns the result."""
    def add(a: int, b: int) -> int:
        return a + b

    registry.register("add", add)

    result = await registry.execute("add", {"a": 3, "b": 5})
    assert result == 8


@pytest.mark.asyncio
async def test_async_tool_execution_returns_result(registry):
    """Async tool functions are properly awaited and return results."""
    async def fetch_data(endpoint: str) -> dict:
        return {"endpoint": endpoint, "status": "ok"}

    registry.register("fetch", fetch_data)

    result = await registry.execute("fetch", {"endpoint": "/api/users"})
    assert result == {"endpoint": "/api/users", "status": "ok"}


@pytest.mark.asyncio
async def test_tool_execution_catches_exceptions(registry):
    """Exceptions during tool execution are caught and returned as error dict."""
    def failing_tool():
        raise RuntimeError("Something went wrong")

    registry.register("fail", failing_tool)

    result = await registry.execute("fail", {})

    assert isinstance(result, dict)
    assert "error" in result
    assert "Something went wrong" in result["error"]


@pytest.mark.asyncio
async def test_execute_unknown_tool_raises(registry):
    """Executing an unknown tool raises ValueError."""
    with pytest.raises(ValueError, match="Unknown tool"):
        await registry.execute("nonexistent", {})


# --- Edge Cases ---

def test_register_multiple_tools_with_different_schemas(registry):
    """Can register multiple tools with distinct schemas."""
    def tool1(): pass
    def tool2(): pass
    def tool3(): pass

    schema1 = {"name": "tool1", "version": "1.0"}
    schema2 = {"name": "tool2", "version": "2.0"}
    schema3 = {"name": "tool3", "version": "3.0"}

    registry.register("t1", tool1, schema=schema1)
    registry.register("t2", tool2, schema=schema2)
    registry.register("t3", tool3, schema=schema3)

    schema = registry.get_schema()
    assert len(schema) == 3
    assert schema["t1"]["version"] == "1.0"
    assert schema["t2"]["version"] == "2.0"
    assert schema["t3"]["version"] == "3.0"


@pytest.mark.asyncio
async def test_async_tool_execution_catches_exceptions(registry):
    """Exceptions in async tools are also caught and returned as error dict."""
    async def async_fail():
        raise ValueError("Async error occurred")

    registry.register("async_fail", async_fail)

    result = await registry.execute("async_fail", {})

    assert isinstance(result, dict)
    assert "error" in result
    assert "Async error occurred" in result["error"]
