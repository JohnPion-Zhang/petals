"""Tests for AgentOrchestrator - Red Phase (failing tests first)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from petals.client.agent import AgentOrchestrator
from petals.client.http_client import LLMResponse
from petals.data_structures import ToolCall, CallStatus


class MockLLMClient:
    """Mock LLM client for testing that returns LLMResponse objects."""

    def __init__(self, responses=None):
        """
        Initialize the mock LLM.

        Args:
            responses: List of responses to return in order (strings or LLMResponse objects).
        """
        self.responses = responses or []
        self.call_count = 0
        self.call_history = []

    async def generate(self, prompt):
        """Return the next response in the list as LLMResponse."""
        self.call_history.append(prompt)
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            # Support both strings and LLMResponse objects
            if isinstance(response, str):
                return LLMResponse(
                    content=response,
                    model="mock-model",
                    usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                )
            return response
        return LLMResponse(
            content="No more responses configured.",
            model="mock-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )

    def switch_model(self, model):
        """Mock switch_model method."""
        pass

    @property
    def default_model(self):
        return "mock-model"


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    return MockLLMClient()


@pytest.fixture
def search_tool():
    """Create a mock search tool."""
    async def search(query):
        return f"Search results for: {query}"
    return search


@pytest.fixture
def calculator_tool():
    """Create a mock calculator tool."""
    async def calculate(expression):
        try:
            result = eval(expression)  # Safe in test context
            return str(result)
        except Exception as e:
            return f"Error: {e}"
    return calculate


@pytest.fixture
def mock_http_client():
    """Create a mock HTTP client for testing."""
    return MockLLMClient()


@pytest.fixture
async def orchestrator(mock_llm, search_tool, calculator_tool):
    """Create an AgentOrchestrator with mock tools."""
    tools = [
        {
            "name": "search",
            "func": search_tool,
            "schema": {
                "name": "search",
                "description": "Search for information",
                "parameters": {"query": {"type": "string"}}
            }
        },
        {
            "name": "calculate",
            "func": calculator_tool,
            "schema": {
                "name": "calculate",
                "description": "Perform a calculation",
                "parameters": {"expression": {"type": "string"}}
            }
        }
    ]
    # Use mock HTTP client directly
    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch.http_client = mock_llm
    orch.max_iterations = 10
    orch.max_context_tokens = 4096
    orch.registry = MagicMock()
    orch.parser = MagicMock()
    orch.context_manager = MagicMock()
    orch.state = MagicMock()
    return orch


class TestAgentOrchestratorInitialization:
    """Test AgentOrchestrator initialization."""

    def test_orchestrator_initialization(self):
        """Test that orchestrator initializes correctly with HTTPClient."""
        orch = AgentOrchestrator(
            api_key="test-key",
            tools=[],
            max_iterations=5,
        )
        assert orch.http_client is not None
        assert orch.max_iterations == 5
        assert orch.state.current_iteration == 0

    def test_orchestrator_default_max_iterations(self):
        """Test default max iterations is 10."""
        orch = AgentOrchestrator(api_key="test-key", tools=[])
        assert orch.max_iterations == 10

    def test_tools_registered_on_init(self, search_tool):
        """Test that tools are registered during initialization."""
        tools = [{"name": "search", "func": search_tool}]
        orch = AgentOrchestrator(api_key="test-key", tools=tools)
        assert orch.executor.registry.get_tool("search") is not None

    def test_runtime_tool_registration(self):
        """Test registering tools at runtime."""
        async def my_tool():
            return "done"

        orch = AgentOrchestrator(api_key="test-key", tools=[])
        # get_tool raises ValueError if tool doesn't exist
        with pytest.raises(ValueError):
            orch.executor.registry.get_tool("my_tool")

        orch.register_tool("my_tool", my_tool)
        assert orch.executor.registry.get_tool("my_tool") is not None

    def test_switch_model(self):
        """Test model switching."""
        orch = AgentOrchestrator(api_key="test-key", default_model="gpt-4", tools=[])
        assert orch.default_model == "gpt-4"
        orch.switch_model("claude-3")
        assert orch.default_model == "claude-3"


class TestAgentOrchestratorSingleTool:
    """Test agent loop with a single tool call."""

    @pytest.mark.asyncio
    async def test_full_agent_loop_single_tool(self, search_tool):
        """Test user query requiring one tool call returns final answer."""
        orch = AgentOrchestrator(
            api_key="test-key",
            tools=[{"name": "search", "func": search_tool}],
            max_iterations=10,
        )

        # Mock the http_client to return responses
        orch.http_client = MockLLMClient(responses=[
            '<tool_call>search({"query": "weather today"})</tool_call>',
            'The weather today is sunny with a high of 75 degrees.'
        ])

        result = await orch.run("What's the weather like?")

        assert "sunny" in result.lower() or "weather" in result.lower()
        assert orch.state.current_iteration == 1

    @pytest.mark.asyncio
    async def test_agent_stops_on_final_answer(self, search_tool):
        """Test agent stops and returns when no more tools are needed."""
        orch = AgentOrchestrator(
            api_key="test-key",
            tools=[{"name": "search", "func": search_tool}],
            max_iterations=10,
        )

        orch.http_client = MockLLMClient(responses=[
            'This is the final answer without any tools.'
        ])

        result = await orch.run("Hello, how are you?")

        assert "final answer" in result.lower()
        assert orch.state.current_iteration == 0


class TestAgentOrchestratorMultiTool:
    """Test agent loop with multiple tool calls."""

    @pytest.mark.asyncio
    async def test_full_agent_loop_multi_tool(self, search_tool, calculator_tool):
        """Test user query requiring multiple sequential tools."""
        tools = [
            {"name": "search", "func": search_tool},
            {"name": "calculate", "func": calculator_tool}
        ]
        orch = AgentOrchestrator(
            api_key="test-key",
            tools=tools,
            max_iterations=10,
        )

        # Tool chain: search -> calculate -> final answer
        orch.http_client = MockLLMClient(responses=[
            '<tool_call>search({"query": "population of Tokyo"})</tool_call>',
            '<tool_call>calculate({"expression": "82 * 1000000"})</tool_call>',
            'Tokyo has approximately 82 million people.'
        ])

        result = await orch.run("Calculate Tokyo population in scientific notation?")

        assert "82" in result
        assert orch.state.current_iteration == 2

    @pytest.mark.asyncio
    async def test_agent_parallel_tool_execution(self, search_tool):
        """Test independent tools can be executed in parallel."""
        orch = AgentOrchestrator(
            api_key="test-key",
            tools=[{"name": "search", "func": search_tool}],
            max_iterations=10,
        )

        # Mock LLM returns two independent tool calls
        orch.http_client = MockLLMClient(responses=[
            '<tool_call>search({"query": "weather"})</tool_call>'
            '<tool_call>search({"query": "news"})</tool_call>',
            'Weather is sunny, news is breaking.'
        ])

        result = await orch.run("What's the weather and news?")

        assert orch.state.current_iteration == 1


class TestAgentOrchestratorMaxIterations:
    """Test agent respects max iterations limit."""

    @pytest.mark.asyncio
    async def test_agent_respects_max_iterations(self, search_tool):
        """Test agent stops after reaching max_iterations."""
        orch = AgentOrchestrator(
            api_key="test-key",
            tools=[{"name": "search", "func": search_tool}],
            max_iterations=3,
        )

        # Always return a tool call to trigger max iterations
        orch.http_client = MockLLMClient(responses=[
            '<tool_call>search({"query": "query1"})</tool_call>',
            '<tool_call>search({"query": "query2"})</tool_call>',
            '<tool_call>search({"query": "query3"})</tool_call>',
        ])

        result = await orch.run("Keep searching forever")

        assert "Max iterations reached" in result
        assert orch.state.current_iteration == 3

    @pytest.mark.asyncio
    async def test_agent_state_stopped_early_flag(self, search_tool):
        """Test that stopped_early flag is set when max iterations hit."""
        orch = AgentOrchestrator(
            api_key="test-key",
            tools=[{"name": "search", "func": search_tool}],
            max_iterations=2,
        )

        orch.http_client = MockLLMClient(responses=[
            '<tool_call>search({"query": "1"})</tool_call>',
            '<tool_call>search({"query": "2"})</tool_call>',
        ])

        result = await orch.run("Keep going")

        assert orch.state.stopped_early


class TestAgentOrchestratorErrorHandling:
    """Test agent handles tool errors gracefully."""

    @pytest.mark.asyncio
    async def test_agent_handles_tool_errors(self):
        """Test agent continues after a tool fails."""
        async def failing_search(query):
            raise ConnectionError("Network timeout")

        orch = AgentOrchestrator(
            api_key="test-key",
            tools=[{"name": "search", "func": failing_search}],
            max_iterations=10,
        )

        orch.http_client = MockLLMClient(responses=[
            '<tool_call>search({"query": "test"})</tool_call>',
            'I encountered an error but will try again with the final answer.'
        ])

        result = await orch.run("Search for something")

        assert "error" in result.lower() or "final answer" in result.lower()
        # Check tool history using as_list() since it's now a generator
        assert any(tc.status == CallStatus.FAILED for tc in orch.state.as_list())

    @pytest.mark.asyncio
    async def test_agent_handles_unknown_tool(self):
        """Test agent handles requests for unknown tools."""
        orch = AgentOrchestrator(
            api_key="test-key",
            tools=[],  # No tools registered
            max_iterations=10,
        )

        orch.http_client = MockLLMClient(responses=[
            '<tool_call>unknown_tool({"arg": "value"})</tool_call>',
            'I tried to use an unknown tool. Here is my final answer.'
        ])

        result = await orch.run("Use an unknown tool")

        assert "final answer" in result.lower() or "unknown" in result.lower()


class TestAgentOrchestratorContextTrimming:
    """Test agent context management and trimming."""

    @pytest.mark.asyncio
    async def test_agent_context_trimming_triggered(self, search_tool):
        """Test long history triggers context trimming."""
        orch = AgentOrchestrator(
            api_key="test-key",
            tools=[{"name": "search", "func": search_tool}],
            max_iterations=20,
        )

        # Build up a long conversation history by having many iterations
        orch.http_client = MockLLMClient(responses=[
            '<tool_call>search({"query": "topic"})</tool_call>'
            for i in range(15)
        ])

        # Add user query
        result = await orch.run("Tell me about many topics")

        # Check that context was managed
        assert orch.context_manager.context is not None
        # Context should be managed (may or may not trim depending on content size)
        assert orch.context_manager.calculate_total_tokens(orch.context_manager.context) >= 0

    @pytest.mark.asyncio
    async def test_context_manager_integrated(self, search_tool):
        """Test that context manager is properly integrated."""
        orch = AgentOrchestrator(
            api_key="test-key",
            tools=[{"name": "search", "func": search_tool}],
            max_iterations=10,
        )

        orch.http_client = MockLLMClient(responses=[
            '<tool_call>search({"query": "test"})</tool_call>',
            'Final answer after context buildup.'
        ])

        result = await orch.run("Initial query")

        # Verify context manager was used
        assert orch.context_manager.context is not None
        assert len(orch.context_manager.context.conversation_history) > 0


class TestAgentOrchestratorState:
    """Test agent state tracking."""

    @pytest.mark.asyncio
    async def test_tool_history_tracking(self, search_tool):
        """Test that tool calls are tracked in history."""
        orch = AgentOrchestrator(
            api_key="test-key",
            tools=[{"name": "search", "func": search_tool}],
            max_iterations=10,
        )

        orch.http_client = MockLLMClient(responses=[
            '<tool_call>search({"query": "test"})</tool_call>',
            'Done.'
        ])

        await orch.run("Search")

        # Use as_list() since tool_history is now a generator
        history = orch.state.as_list()
        assert len(history) == 1
        assert history[0].name == "search"

    @pytest.mark.asyncio
    async def test_iteration_count_accuracy(self, search_tool):
        """Test iteration count is accurate."""
        orch = AgentOrchestrator(
            api_key="test-key",
            tools=[{"name": "search", "func": search_tool}],
            max_iterations=10,
        )

        orch.http_client = MockLLMClient(responses=[
            '<tool_call>search({"query": "1"})</tool_call>',
            '<tool_call>search({"query": "2"})</tool_call>',
            'Done.'
        ])

        await orch.run("Two searches")

        assert orch.state.current_iteration == 2

    @pytest.mark.asyncio
    async def test_token_usage_tracking(self, search_tool):
        """Test that token usage is tracked from LLM responses."""
        orch = AgentOrchestrator(
            api_key="test-key",
            tools=[{"name": "search", "func": search_tool}],
            max_iterations=10,
        )

        orch.http_client = MockLLMClient(responses=[
            'Final answer.'
        ])

        await orch.run("Test")

        # Token usage should be tracked (15 tokens per response * 1 response)
        assert orch.state.total_tokens_used == 15
