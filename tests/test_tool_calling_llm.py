"""Tests for ToolCallingLLM."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from petals.client.tool_calling_llm import ToolCallingLLM, create_tool_calling_llm
from petals.client.providers.base import LLMChunk, LLMResponse
from petals.client.tool_parser import ToolParser
from petals.data_structures import ToolCall
from petals.client.dag import ToolCallDAG


class TestToolCallingLLM:
    """Tests for ToolCallingLLM class."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock LLM provider."""
        provider = MagicMock()
        provider.complete = AsyncMock()
        provider.stream = AsyncMock()
        provider.close = AsyncMock()
        return provider

    @pytest.fixture
    def mock_registry(self):
        """Create mock tool registry."""
        registry = MagicMock()
        registry.has_tool = MagicMock(return_value=True)
        registry.list_tools = MagicMock(return_value={})
        return registry

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator."""
        orchestrator = MagicMock()
        orchestrator.execute_streaming = MagicMock()
        return orchestrator

    @pytest.fixture
    def llm(self, mock_provider, mock_registry, mock_orchestrator):
        """Create ToolCallingLLM instance."""
        return ToolCallingLLM(
            provider=mock_provider,
            registry=mock_registry,
            orchestrator=mock_orchestrator,
        )

    def test_initialization(self, llm, mock_provider, mock_registry, mock_orchestrator):
        """Test ToolCallingLLM initialization."""
        assert llm.provider is mock_provider
        assert llm.registry is mock_registry
        assert llm.orchestrator is mock_orchestrator
        assert isinstance(llm.tool_parser, ToolParser)

    def test_initialization_with_custom_parser(self, mock_provider, mock_registry, mock_orchestrator):
        """Test initialization with custom tool parser."""
        parser = ToolParser(strict=False)
        llm = ToolCallingLLM(
            provider=mock_provider,
            registry=mock_registry,
            orchestrator=mock_orchestrator,
            tool_parser=parser,
        )
        assert llm.tool_parser is parser

    @pytest.mark.asyncio
    async def test_run_no_tools(self, llm, mock_provider):
        """Test run without tool calls."""
        mock_provider.complete.return_value = LLMResponse(
            content="Hello, world!",
            usage={"total_tokens": 10},
        )

        result = await llm.run("Hello")

        assert result["content"] == "Hello, world!"
        assert result["tool_results"] is None
        mock_provider.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_tools(self, llm, mock_provider, mock_orchestrator):
        """Test run with tool calls."""
        mock_provider.complete.return_value = LLMResponse(
            content="<tool_call>search({\"query\": \"test\"})</tool_call>",
            usage={"total_tokens": 20},
        )

        # Mock orchestrator streaming
        async def mock_stream():
            yield MagicMock(type="tool_result", data={"id": "call_1", "result": {"found": True}})
            yield MagicMock(type="final", data={})

        mock_orchestrator.execute_streaming.return_value = mock_stream()

        result = await llm.run("Search for something")

        assert "tool_call>" in result["content"]
        assert result["tool_results"] is not None
        mock_orchestrator.execute_streaming.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_streaming_no_tools(self, llm, mock_provider):
        """Test streaming without tool calls."""
        chunks = [
            LLMChunk(content="Hello"),
            LLMChunk(content=", "),
            LLMChunk(content="world!", is_complete=True),
        ]
        mock_provider.stream = MagicMock(return_value=async_iter(chunks))

        events = []
        async for event in llm.run_streaming("Hello"):
            events.append(event)

        assert len(events) == 4  # 3 text chunks + final
        # Verify FINAL event
        final_events = [e for e in events if e.type.value == "final"]
        assert len(final_events) == 1

    @pytest.mark.asyncio
    async def test_run_streaming_with_tool_detection(self, llm, mock_provider, mock_orchestrator):
        """Test streaming with tool detection."""
        chunks = [
            LLMChunk(content="Let me search"),
            LLMChunk(content=" for that", tool_names=["search"]),
            LLMChunk(content="...", is_complete=True),
        ]
        mock_provider.stream = MagicMock(return_value=async_iter(chunks))

        # Mock orchestrator
        async def mock_stream():
            yield MagicMock(type="tool_result", data={"node_id": "call_1", "result": {}})

        mock_orchestrator.execute_streaming.return_value = mock_stream()

        events = []
        async for event in llm.run_streaming("Search"):
            events.append(event)

        # Should have tool pending event
        tool_pending = [e for e in events if e.type.value == "tool_call_pending"]
        assert len(tool_pending) == 1
        assert tool_pending[0].data["tool_name"] == "search"

    def test_build_dag_from_tool_calls(self, llm):
        """Test DAG building from tool calls."""
        tool_calls = [
            ToolCall(name="search", arguments={"query": "test"}, id="call_1"),
            ToolCall(name="calc", arguments={"expr": "1+1"}, id="call_2"),
        ]

        dag = llm._build_dag_from_tool_calls(tool_calls)

        assert len(dag.nodes) == 2
        assert "call_1" in dag.nodes
        assert "call_2" in dag.nodes
        assert dag.nodes["call_1"].name == "search"
        assert dag.nodes["call_2"].name == "calc"

    def test_build_dag_with_dependencies(self, llm):
        """Test DAG building with dependencies."""
        tool_calls = [
            ToolCall(name="search", arguments={"query": "test"}, id="call_1"),
            ToolCall(
                name="calc",
                arguments={"expr": "1+1"},
                id="call_2",
                dependencies=["call_1"],
            ),
        ]

        dag = llm._build_dag_from_tool_calls(tool_calls)

        assert len(dag.nodes) == 2
        assert "call_2" in dag.nodes["call_1"].dependents
        assert "call_1" in dag.nodes["call_2"].dependencies

    def test_get_tool_definitions(self, llm, mock_registry):
        """Test getting tool definitions from registry."""
        mock_tool = MagicMock()
        mock_tool.name = "search"
        mock_tool.description = "Search the web"
        mock_tool.parameters = {"type": "object", "properties": {}}

        mock_registry.list_tools.return_value = {"search": mock_tool}

        tools = llm.get_tool_definitions()

        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "search"

    def test_get_tool_definitions_filtered(self, llm, mock_registry):
        """Test getting filtered tool definitions."""
        mock_registry.has_tool.return_value = False

        mock_tool = MagicMock()
        mock_tool.name = "search"
        mock_tool.description = "Search"
        mock_tool.parameters = {}

        mock_registry.get_tool.return_value = mock_tool
        mock_registry.has_tool.side_effect = lambda n: n == "search"

        tools = llm.get_tool_definitions(tool_names=["search", "calc"])

        # Only "search" should be returned
        assert len(tools) == 1

    def test_get_tool_definitions_empty_registry(self, llm):
        """Test with empty registry."""
        llm.registry = None
        tools = llm.get_tool_definitions()
        assert tools == []

    @pytest.mark.asyncio
    async def test_close(self, llm, mock_provider):
        """Test close method."""
        await llm.close()
        mock_provider.close.assert_called_once()


class TestCreateToolCallingLLM:
    """Tests for create_tool_calling_llm factory."""

    @pytest.fixture
    def mock_registry(self):
        return MagicMock()

    @pytest.fixture
    def mock_orchestrator(self):
        return MagicMock()

    def test_create_openai_llm(self, mock_registry, mock_orchestrator):
        """Test creating OpenAI-based ToolCallingLLM."""
        llm = create_tool_calling_llm(
            "openai",
            registry=mock_registry,
            orchestrator=mock_orchestrator,
            api_key="test-key",
            model="gpt-4",
        )

        assert isinstance(llm, ToolCallingLLM)
        assert llm.provider.model == "gpt-4"

    def test_create_anthropic_llm(self, mock_registry, mock_orchestrator):
        """Test creating Anthropic-based ToolCallingLLM."""
        llm = create_tool_calling_llm(
            "anthropic",
            registry=mock_registry,
            orchestrator=mock_orchestrator,
            api_key="test-key",
        )

        assert isinstance(llm, ToolCallingLLM)
        assert "claude" in llm.provider.model


# Helper for async iteration
def async_iter(items):
    """Create async iterator from list."""
    async def generator():
        for item in items:
            yield item
    return generator()
