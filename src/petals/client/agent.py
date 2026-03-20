"""Agent orchestrator for managing the full agent loop with tool calls."""

from typing import Dict, List, Optional, Any

from petals.data_structures import (
    AgentState,
    CallStatus,
    ToolCall,
)
from petals.client.data_structures import ContextWindow, Message
from petals.client.http_client import HTTPClient
from petals.client.tool_parser import ToolParser
from petals.client.tool_executor import ToolExecutor
from petals.client.tool_registry import ToolRegistry
from petals.client.context_manager import ContextManager


class AgentOrchestrator:
    """
    Orchestrates the full agent loop including:
    - LLM generation (via HTTPClient with litellm)
    - Tool call parsing
    - Tool execution (with generator pattern)
    - Context management
    - Response handling

    The agent loop continues until:
    - The LLM provides a final answer (no tool calls)
    - Max iterations are reached
    - An unrecoverable error occurs
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        default_model: str = "gpt-4o-mini",
        tools: Optional[List[Dict]] = None,
        max_iterations: int = 10,
        max_context_tokens: int = 4096,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """
        Initialize the agent orchestrator with HTTPClient.

        Args:
            api_key: API key for the LLM provider.
            base_url: Optional custom base URL for OpenAI-compatible APIs.
            default_model: Default model to use for generation.
            tools: List of tool definitions with 'name', 'func', and optional 'schema'.
            max_iterations: Maximum number of agent iterations before stopping.
            max_context_tokens: Maximum context length in tokens.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts for failed requests.
        """
        # Initialize HTTP client with litellm
        self.http_client = HTTPClient(
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            timeout=timeout,
            max_retries=max_retries,
        )

        self.max_iterations = max_iterations
        self.max_context_tokens = max_context_tokens

        # Initialize components
        self.registry = ToolRegistry()
        if tools:
            for tool in tools:
                self.registry.register(
                    tool["name"],
                    tool["func"],
                    tool.get("schema"),
                )

        self.parser = ToolParser()
        self.executor = ToolExecutor(self.registry, state=None)  # We'll track manually
        self.context_manager = ContextManager()

        self.state = AgentState(max_iterations=max_iterations)

    def switch_model(self, model: str) -> None:
        """
        Switch the default model at runtime.

        Args:
            model: New model name to use.
        """
        self.http_client.switch_model(model)

    @property
    def default_model(self) -> str:
        """Get the current default model."""
        return self.http_client.default_model

    def register_tool(
        self,
        name: str,
        func: Any,
        schema: Optional[Dict] = None,
    ) -> None:
        """
        Register a new tool at runtime.

        Args:
            name: Name of the tool.
            func: The function implementing the tool.
            schema: Optional JSON schema for the tool.
        """
        self.registry.register(name, func, schema)

    async def run(self, user_input: str) -> str:
        """
        Run the full agent loop until completion or max iterations.

        Args:
            user_input: The user's input/query.

        Returns:
            The final response from the agent.
        """
        # Initialize state for new run
        self.state.clear()

        # Create context window with user input
        self.context_manager.context = ContextWindow(
            max_tokens=self.max_context_tokens,
            system_prompt="",
            active_context=user_input,
        )

        while self.state.current_iteration < self.max_iterations:
            # Generate LLM output and track usage
            response = await self._generate()

            # Parse tool calls from LLM output
            tool_calls = self.parser.parse(response.content)

            if not tool_calls:
                # No more tools - return final answer
                return response.content

            # Execute tools using generator pattern
            results = []
            async for result in self.executor.parallel_execute(tool_calls):
                results.append(result)
                self.state.add_tool(result)

            # Update context with results
            self._update_context(results, response.content)

            # Update state
            self.state.current_iteration += 1

            # Check if context needs trimming
            if not self.context_manager.fits_in_budget(self.context_manager.context):
                self.context_manager.context = self.context_manager.trim(self.context_manager.context)

        # Max iterations reached
        self.state.stopped_early = True
        return "Max iterations reached"

    async def _generate(self) -> Any:
        """
        Call LLM with current context.

        Returns:
            LLMResponse with content and usage info.
        """
        prompt = self.context_manager.build_prompt(self.context_manager.context)
        response = await self.http_client.generate(prompt)

        # Track token usage
        if response.usage:
            self.state.update_token_usage(response.usage)

        return response

    def _update_context(
        self,
        tool_results: List[ToolCall],
        llm_output: str,
    ) -> None:
        """
        Add tool results and LLM output to conversation history.

        Args:
            tool_results: List of executed tool calls with results.
            llm_output: The original LLM output containing tool calls.
        """
        # Add assistant message with tool calls
        self.context_manager.context.conversation_history.append(
            Message(role="assistant", content=llm_output, is_tool_result=False)
        )

        # Add tool result messages
        for tc in tool_results:
            status_str = "SUCCESS" if tc.status == CallStatus.DONE else "FAILED"
            content = f"Tool {tc.name}: {status_str}"
            if tc.result is not None:
                content += f" - {tc.result}"
            if tc.error:
                content += f" - Error: {tc.error}"

            self.context_manager.context.conversation_history.append(
                Message(role="tool", content=content, is_tool_result=True)
            )

    def _has_pending_tools(self) -> bool:
        """
        Check if there are pending tool calls.

        Returns:
            True if there are pending tools, False otherwise.
        """
        # Get last 10 tools from history
        history = self.state.as_list()
        recent = history[-10:] if len(history) >= 10 else history
        return any(tc.status == CallStatus.PENDING for tc in recent)

    def get_tool_history(self) -> List[ToolCall]:
        """
        Get the history of all tool calls.

        Returns:
            List of ToolCall objects.
        """
        return self.state.as_list()

    def get_conversation_history(self) -> List[Message]:
        """
        Get the conversation history.

        Returns:
            List of Message objects.
        """
        if self.context_manager.context is None:
            return []
        return self.context_manager.context.conversation_history
