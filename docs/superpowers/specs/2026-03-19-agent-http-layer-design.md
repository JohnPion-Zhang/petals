# Agent HTTP Layer Refactoring Design

**Date:** 2026-03-19
**Status:** In Review
**Author:** Claude

## Overview

Refactor the P2P Petals agent layer to:
1. Replace mock LLM client with HTTP-based litellm integration
2. Support runtime provider switching (OpenAI, Anthropic, custom)
3. Convert `List[ToolCall]` to generator pattern for memory efficiency

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      AgentOrchestrator                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │ToolParser    │  │ToolExecutor   │  │  HTTPClient       │   │
│  │(tool_call    │  │(async exec,   │  │  (litellm wrapper)│   │
│  │ parsing)     │  │ generators)   │  │                   │   │
│  └──────────────┘  └──────────────┘  └───────────────────┘   │
│         ▲                 ▲                    ▲               │
│         │                 │                    │               │
│  ┌──────┴─────────────────┴────────────────────┴───────────┐  │
│  │              ContextManager                              │  │
│  │  (token budget, smart trimming, generator history)       │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  litellm        │
                    │  (http layer)   │
                    │  - Provider     │
                    │    detection    │
                    │  - Retries      │
                    │  - Format conv │  │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  External APIs  │
                    │  OpenAI/Claude │
                    │  Custom URLs    │
                    └─────────────────┘
```

**Note:** Provider enum removed — litellm handles provider detection internally based on model names.

## HTTP Client Interface

### Protocol Definition

```python
# src/petals/client/http_client.py
from typing import Protocol, Optional, Dict, Any, List, NamedTuple

class LLMResponse(NamedTuple):
    """Structured response from LLM."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None  # prompt_tokens, completion_tokens, total_tokens
    raw_response: Optional[Any] = None

class LLMClient(Protocol):
    """Protocol for LLM client adapters."""

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion from prompt."""
        ...

    @property
    def default_model(self) -> str:
        """Get the default model."""
        ...
```

### HTTPClient Implementation (litellm wrapper)

```python
# src/petals/client/http_client.py
import litellm  # Module-level import for efficiency

from typing import Protocol, Optional, Dict, Any, List, NamedTuple
from dataclasses import dataclass

@dataclass
class LLMResponse:
    """Structured response from LLM."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None  # prompt_tokens, completion_tokens, total_tokens
    raw_response: Optional[Any] = None

class HTTPClient:
    """Litellm wrapper providing unified LLM interface.

    Supports:
    - OpenAI /chat/completions
    - Anthropic /messages
    - Any OpenAI-compatible endpoint
    - Runtime provider switching via model name
    - Token usage tracking
    - Configurable timeouts and retries
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        default_model: str = "gpt-4",
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """Initialize HTTP client.

        Args:
            api_key: API key for the LLM provider.
            base_url: Optional custom base URL for OpenAI-compatible APIs.
            default_model: Default model to use.
            timeout: Request timeout in seconds (default: 60).
            max_retries: Maximum retry attempts for failed requests (default: 3).
        """
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max_retries

        # Configure litellm defaults
        litellm.drop_params = True
        litellm.set_verbose = False

    @property
    def default_model(self) -> str:
        """Get the current default model."""
        return self._default_model

    @default_model.setter
    def default_model(self, value: str) -> None:
        """Set the default model."""
        self._default_model = value

    def _build_litellm_kwargs(self, model: str, **kwargs) -> Dict[str, Any]:
        """Build litellm-compatible kwargs.

        Args:
            model: Model name.
            **kwargs: Additional parameters.

        Returns:
            kwargs dict with api_key and api_base set.
        """
        params = dict(kwargs)
        params["api_key"] = self.api_key

        # Use api_base for custom URLs (litellm parameter name)
        if self.base_url:
            params["api_base"] = self.base_url

        # Set timeout
        if "timeout" not in params:
            params["timeout"] = self.timeout

        # Set max retries
        if "max_retries" not in params:
            params["max_retries"] = self.max_retries

        return params

    async def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> LLMResponse:
        """Generate completion using litellm.

        Args:
            prompt: The prompt to generate from.
            model: Optional model override.
            **kwargs: Additional litellm parameters.

        Returns:
            LLMResponse with content, model, and usage info.
        """
        model = model or self._default_model
        params = self._build_litellm_kwargs(model, **kwargs)

        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **params
        )

        # Extract usage info if available
        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }

        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage=usage,
            raw_response=response,
        )

    async def chat(self, messages: List[Dict], model: Optional[str] = None, **kwargs) -> LLMResponse:
        """Chat completion with message history.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: Optional model override.
            **kwargs: Additional litellm parameters.

        Returns:
            LLMResponse with content, model, and usage info.
        """
        model = model or self._default_model
        params = self._build_litellm_kwargs(model, **kwargs)

        response = await litellm.acompletion(
            model=model,
            messages=messages,
            **params
        )

        # Extract usage info if available
        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }

        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage=usage,
            raw_response=response,
        )

    def switch_model(self, model: str) -> None:
        """Switch default model at runtime.

        Args:
            model: New default model name.
        """
        self._default_model = model
```

## Generator Refactoring

### AgentState Changes

```python
# src/petals/data_structures.py - AgentState
from typing import Iterator, Generator

@dataclasses.dataclass
class AgentState:
    current_iteration: int = 0
    max_iterations: int = 10
    _tool_history: List[ToolCall] = dataclasses.field(default_factory=list)
    total_tokens_used: int = 0
    stopped_early: bool = False

    @property
    def tool_history(self) -> Generator[ToolCall, None, None]:
        """Generator yielding tool history as tools complete.

        Note: Generators are single-use. Iterate once, or use `as_list()`
        for multiple iterations.

        Usage:
            for tool in agent.state.tool_history:
                print(f"Tool: {tool.name}, Status: {tool.status}")
        """
        yield from self._tool_history

    def add_tool(self, tool_call: ToolCall) -> None:
        """Add tool to history (for streaming/generator pattern)."""
        self._tool_history.append(tool_call)

    def clear(self) -> None:
        """Clear tool history. Use instead of accessing _tool_history directly."""
        self._tool_history.clear()
        self.current_iteration = 0
        self.stopped_early = False

    def as_list(self) -> List[ToolCall]:
        """Materialize full history for debugging.

        Returns:
            Complete list of all tool calls.
        """
        return list(self._tool_history)

    def update_token_usage(self, usage: Dict[str, int]) -> None:
        """Update total tokens used from LLM response.

        Args:
            usage: Dict with 'prompt_tokens', 'completion_tokens', 'total_tokens'.
        """
        if usage and "total_tokens" in usage:
            self.total_tokens_used += usage["total_tokens"]
```

### ToolExecutor Changes

```python
# src/petals/client/tool_executor.py
async def parallel_execute(self, tool_calls: List[ToolCall]) -> Generator[ToolCall, None, None]:
    """Execute tools in parallel, yielding results as they complete.

    Yields results as they finish (not in submission order), enabling
    streaming behavior and reducing memory for large tool sets.

    Args:
        tool_calls: List of ToolCalls to execute.

    Yields:
        ToolCall with updated status and result as each completes.
    """
    # Execute all tool calls concurrently
    tasks = [self.execute(tc) for tc in tool_calls]

    # Use as_completed to yield results as they finish
    for coro in asyncio.as_completed(tasks):
        result = await coro
        self.state.add_tool(result)  # Update state
        yield result

async def execute_with_dependencies(
    self,
    tool_calls: List[ToolCall],
    results_cache: dict = None
) -> Generator[ToolCall, None, None]:
    """Execute tool calls respecting their dependency order.

    Tools in each wave execute in parallel. Results are yielded per wave.

    Args:
        tool_calls: List of ToolCalls to execute.
        results_cache: Dictionary to store results for dependency resolution.

    Yields:
        ToolCall with updated status as each wave completes.
    """
    if results_cache is None:
        results_cache = {}

    completed = set()
    remaining = list(tool_calls)

    while remaining:
        # Find tools whose dependencies are all satisfied
        ready = []
        waiting = []

        for tc in remaining:
            deps_satisfied = all(dep in completed for dep in tc.dependencies)
            if deps_satisfied:
                ready.append(tc)
            else:
                waiting.append(tc)

        if not ready:
            # No tools ready but still waiting - cycle dependency
            for tc in waiting:
                tc.status = CallStatus.FAILED
                tc.result = {"error": "Unresolved dependency"}
                yield tc
            break

        # Execute ready tools in parallel
        async for result in self.parallel_execute(ready):
            if result.status == CallStatus.DONE:
                completed.add(result.id)
                results_cache[result.id] = result.result
            yield result

        remaining = waiting
```

## Agent Orchestrator Updates

```python
# src/petals/client/agent.py
from petals.client.http_client import HTTPClient, LLMResponse

class AgentOrchestrator:
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        default_model: str = "gpt-4",
        tools: List[Dict] = None,
        max_iterations: int = 10,
        max_context_tokens: int = 4096,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """Initialize agent orchestrator.

        Args:
            api_key: API key for LLM provider.
            base_url: Optional custom base URL.
            default_model: Default model for generation.
            tools: List of tool definitions.
            max_iterations: Maximum agent iterations.
            max_context_tokens: Token budget for context.
            timeout: LLM request timeout in seconds.
            max_retries: Maximum LLM retry attempts.
        """
        # HTTP Client with litellm
        self.http_client = HTTPClient(
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            timeout=timeout,
            max_retries=max_retries,
        )

        # Existing components unchanged
        self.registry = ToolRegistry()
        self.parser = ToolParser()
        self.executor = ToolExecutor(self.registry)
        self.context_manager = ContextManager()
        self.state = AgentState(max_iterations=max_iterations)

        # Register tools
        for tool in tools or []:
            self.registry.register(tool["name"], tool["func"], tool.get("schema"))

    @property
    def default_model(self) -> str:
        """Get current default model."""
        return self.http_client.default_model

    async def run(self, user_input: str, model: Optional[str] = None) -> str:
        """Run agent with optional model override for runtime switching.

        Args:
            user_input: User's input/query.
            model: Optional model override for this run.

        Returns:
            Final response from the agent.
        """
        model = model or self.http_client.default_model

        # Initialize state for new run
        self.state.clear()
        self.state.total_tokens_used = 0

        # Create context window with user input
        self.context_manager.context = ContextWindow(
            max_tokens=self.max_context_tokens,
            system_prompt="",
            active_context=user_input,
        )

        while self.state.current_iteration < self.max_iterations:
            # Generate using HTTP client
            llm_response = await self._generate(user_input, model)

            # Track token usage
            if llm_response.usage:
                self.state.update_token_usage(llm_response.usage)

            # Parse tool calls from LLM output
            tool_calls = self.parser.parse(llm_response.content)

            if not tool_calls:
                # No more tools - return final answer
                return llm_response.content

            # Execute tools (iterate through generator)
            results = []
            async for result in self.executor.parallel_execute(tool_calls):
                results.append(result)

            # Update context with results
            self._update_context(results, llm_response.content)

            # Update state
            self.state.current_iteration += 1

            # Check if context needs trimming
            if not self.context_manager.fits_in_budget(self.context_manager.context):
                self.context_manager.context = self.context_manager.trim(
                    self.context_manager.context
                )

        # Max iterations reached
        self.state.stopped_early = True
        return "Max iterations reached"

    async def _generate(self, user_input: str, model: str) -> LLMResponse:
        """Call LLM via HTTP client.

        Args:
            user_input: Current user input.
            model: Model to use for generation.

        Returns:
            LLMResponse with content and usage info.
        """
        prompt = self.context_manager.build_prompt(self.context_manager.context)

        return await self.http_client.generate(
            prompt=prompt,
            model=model,
            max_tokens=self.max_context_tokens // 2,
        )

    def switch_model(self, model: str) -> None:
        """Switch default model at runtime.

        Args:
            model: New default model name.
        """
        self.http_client.switch_model(model)
```

## File Changes

| File | Action | Changes |
|------|--------|---------|
| `src/petals/client/http_client.py` | CREATE | HTTPClient wrapper for litellm, LLMResponse dataclass, LLMClient protocol |
| `src/petals/data_structures.py` | MODIFY | `AgentState.tool_history` → generator, `clear()` method, `update_token_usage()` |
| `src/petals/client/tool_executor.py` | MODIFY | Return generators from `parallel_execute`, `execute_with_dependencies` |
| `src/petals/client/agent.py` | MODIFY | Use HTTPClient, delegate default_model, track token usage |
| `tests/client/test_http_client.py` | CREATE | Comprehensive HTTPClient tests |
| `tests/client/test_tool_executor.py` | MODIFY | Update for generator-based return types |
| `tests/client/test_agent_orchestrator.py` | MODIFY | Update for new HTTP client and generator patterns |

## Dependencies

Add to `pyproject.toml`:

```toml
dependencies = [
    # ... existing ...
    "litellm>=1.0.0",
]
```

## Usage Examples

### Basic Usage with OpenAI

```python
from petals.client.agent import AgentOrchestrator

agent = AgentOrchestrator(
    api_key="sk-...",
    default_model="gpt-4",
    tools=[{"name": "search", "func": search_fn}],
)

result = await agent.run("What's the weather in Tokyo?")
```

### Runtime Provider Switching

```python
# Start with OpenAI
result = await agent.run("Explain quantum computing", model="gpt-4")

# Switch to Claude mid-session
result = await agent.run("Now explain it differently", model="claude-3-5-sonnet-20241022")

# Or use switch_model
agent.switch_model("claude-3-5-sonnet-20241022")
result = await agent.run("Another explanation")
```

### Custom OpenAI-Compatible Endpoint

```python
agent = AgentOrchestrator(
    api_key="kala-0719",
    base_url="http://127.0.0.1:3456/v1",
    default_model="claude-4.5-sonnet",
    tools=[...],
)

result = await agent.run("Hello!")
```

## Testing Strategy

### HTTPClient Tests

```python
# tests/client/test_http_client.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

class TestHTTPClient:
    @pytest.fixture
    def mock_litellm_response(self):
        """Create a mock litellm response matching actual API structure."""
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "Test response"
        response.model = "gpt-4"
        response.usage = MagicMock()
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 5
        response.usage.total_tokens = 15
        return response

    @pytest.mark.asyncio
    async def test_generate_returns_llm_response(self, mock_litellm_response):
        with patch("petals.client.http_client.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_litellm_response)

            client = HTTPClient(api_key="test-key", default_model="gpt-4")
            result = await client.generate("Hello")

            assert isinstance(result, LLMResponse)
            assert result.content == "Test response"
            assert result.usage["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_chat_with_messages(self, mock_litellm_response):
        with patch("petals.client.http_client.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_litellm_response)

            client = HTTPClient(api_key="test-key")
            messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"}
            ]
            result = await client.chat(messages)

            assert result.content == "Test response"
            mock_litellm.acompletion.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_custom_url(self, mock_litellm_response):
        with patch("petals.client.http_client.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_litellm_response)

            client = HTTPClient(
                api_key="test-key",
                base_url="http://localhost:3456/v1"
            )
            await client.generate("Hello", model="claude-4")

            # Verify api_base is passed to litellm
            call_kwargs = mock_litellm.acompletion.call_args.kwargs
            assert call_kwargs["api_base"] == "http://localhost:3456/v1"

    @pytest.mark.asyncio
    async def test_switch_model(self):
        client = HTTPClient(api_key="test", default_model="gpt-4")
        assert client.default_model == "gpt-4"

        client.switch_model("claude-3-5-sonnet")
        assert client.default_model == "claude-3-5-sonnet"

    @pytest.mark.asyncio
    async def test_error_propagation(self):
        with patch("petals.client.http_client.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                side_effect=Exception("API Error")
            )

            client = HTTPClient(api_key="test-key")
            with pytest.raises(Exception, match="API Error"):
                await client.generate("Hello")

    def test_default_timeout_and_retries(self):
        client = HTTPClient(api_key="test-key")
        assert client.timeout == 60.0
        assert client.max_retries == 3

    def test_custom_timeout_and_retries(self):
        client = HTTPClient(api_key="test-key", timeout=120.0, max_retries=5)
        assert client.timeout == 120.0
        assert client.max_retries == 5
```

### Error Response Tests

```python
class TestHTTPClientErrors:
    @pytest.mark.asyncio
    async def test_authentication_error(self):
        with patch("petals.client.http_client.litellm") as mock_litellm:
            from litellm import AuthenticationError
            mock_litellm.acompletion = AsyncMock(
                side_effect=AuthenticationError(message="Invalid API key")
            )

            client = HTTPClient(api_key="invalid-key")
            with pytest.raises(AuthenticationError):
                await client.generate("Hello")

    @pytest.mark.asyncio
    async def test_rate_limit_with_retry(self):
        with patch("petels.client.http_client.litellm") as mock_litellm:
            from litellm import RateLimitError
            # First call fails, second succeeds
            mock_litellm.acompletion = AsyncMock(side_effect=[
                RateLimitError(message="Rate limited"),
                MagicMock(
                    choices=[MagicMock(message=MagicMock(content="Success"))],
                    model="gpt-4",
                    usage=None
                )
            ])

            client = HTTPClient(api_key="test-key", max_retries=1)
            result = await client.generate("Hello")
            assert result.content == "Success"
            assert mock_litellm.acompletion.call_count == 2
```

### Agent Orchestrator Tests

```python
class TestAgentOrchestratorIntegration:
    @pytest.mark.asyncio
    async def test_token_usage_tracking(self):
        with patch("petals.client.http_client.litellm") as mock_litellm:
            # Mock response with usage
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Final answer"
            mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)

            agent = AgentOrchestrator(api_key="test")
            result = await agent.run("Simple question")

            assert result == "Final answer"
            assert agent.state.total_tokens_used == 150

    @pytest.mark.asyncio
    async def test_runtime_model_switch(self):
        with patch("petals.client.http_client.litellm") as mock_litellm:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Done"
            mock_response.usage = None
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)

            agent = AgentOrchestrator(api_key="test", default_model="gpt-4")

            # Use different model for this call
            await agent.run("Test", model="claude-3-5-sonnet")

            # Verify model was passed to litellm
            call_kwargs = mock_litellm.acompletion.call_args.kwargs
            assert call_kwargs["model"] == "claude-3-5-sonnet"
```

## Error Handling

| Scenario | Behavior |
|----------|----------|
| API key invalid | litellm raises `AuthenticationError`, propagated |
| Rate limit | litellm retries with exponential backoff (up to max_retries) |
| Model not found | litellm raises `BadRequestError`, propagated |
| Timeout | `httpx.TimeoutException` after timeout seconds, propagated |
| Custom URL unreachable | `httpx.ConnectError`, propagated |
| Max retries exceeded | Final exception after all retries fail |

## Performance Considerations

1. **Generator pattern** — Reduces memory for large tool sets by yielding results incrementally
2. **Module-level litellm import** — Avoids repeated import overhead
3. **Async throughout** — All I/O operations are non-blocking
4. **Connection pooling** — litellm/httpx handles connection reuse internally
5. **Token usage tracking** — Minimal overhead, enables cost monitoring

## Limitations

1. **Generator exhaustion** — `tool_history` property returns a generator that can only be iterated once. For multiple iterations, use `as_list()`.

2. **litellm as required dependency** — The design assumes litellm handles all provider differences. If litellm has bugs with a specific provider, we inherit those issues.

3. **No streaming** — Current implementation uses `litellm.acompletion` (non-streaming). Streaming support would require separate implementation.
