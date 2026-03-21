"""Base LLM provider interface."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Awaitable
import logging

from petals.client.feedback.retry_policy import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
)
from .resilience import TimeoutConfig, LLMToolError

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized LLM response.

    Attributes:
        content: The text content of the response.
        tool_calls: Optional list of tool calls returned by the LLM.
        usage: Token usage information (prompt_tokens, completion_tokens, total_tokens).
        model: The model that generated the response.
        finish_reason: Why the generation stopped (stop, length, tool_calls, etc.).
        raw: Raw response data from the provider for debugging.
    """

    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    raw: Any = None


@dataclass
class LLMChunk:
    """Streaming chunk from LLM.

    Attributes:
        content: Text content of the chunk.
        tool_names: Detected tool names when <tool_call> patterns appear.
        is_complete: Whether this is the final chunk.
        raw: Raw chunk data from the provider.
    """

    content: str = ""
    tool_names: Optional[List[str]] = None
    is_complete: bool = False
    raw: Any = None


class BaseLLMProvider(ABC):
    """Abstract base class for HTTP-based LLM providers.

    All HTTP-based LLM providers should implement this interface. This provides
    a unified API for completion, streaming, and token counting across
    different LLM providers (OpenAI, Anthropic, etc.).

    Attributes:
        api_key: API key for authentication.
        base_url: Base URL for the API endpoint.
        timeout: Request timeout in seconds (legacy, use timeout_config instead).
        max_retries: Maximum number of retry attempts.
        timeout_config: Detailed timeout configuration for different operations.

    Example:
        >>> class MyProvider(BaseLLMProvider):
        ...     async def complete(self, prompt, **kwargs):
        ...         # Implementation
        ...         return LLMResponse(content="...")
        ...
        ...     async def stream(self, prompt, **kwargs):
        ...         # Implementation
        ...         yield LLMChunk(content="...")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        timeout_config: Optional[TimeoutConfig] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        """Initialize the LLM provider.

        Args:
            api_key: API key for authentication.
            base_url: Base URL for the API endpoint.
            timeout: Request timeout in seconds (used for backward compatibility).
            max_retries: Maximum number of retry attempts.
            timeout_config: Detailed timeout configuration. If not provided,
                a default TimeoutConfig will be created with sensible defaults.
            circuit_breaker: Optional circuit breaker for resilience. If provided,
                requests will be protected by the circuit breaker pattern.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._timeout_config = timeout_config
        self._circuit_breaker = circuit_breaker

    @property
    def timeout_config(self) -> TimeoutConfig:
        """Get the timeout configuration.

        Returns:
            TimeoutConfig instance with timeout settings.
        """
        if self._timeout_config is None:
            self._timeout_config = TimeoutConfig(
                connect_timeout=self.timeout,
                read_timeout=self.timeout,
                total_timeout=self.timeout,
                chat_timeout=self.timeout,
                stream_timeout=self.timeout,
            )
        return self._timeout_config

    @timeout_config.setter
    def timeout_config(self, value: TimeoutConfig) -> None:
        """Set the timeout configuration.

        Args:
            value: TimeoutConfig instance.
        """
        self._timeout_config = value

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion.

        Args:
            prompt: User prompt.
            system: Optional system prompt to prepend.
            tools: Optional tool definitions for function calling.
            **kwargs: Provider-specific options (e.g., temperature, max_tokens).

        Returns:
            LLMResponse with content and optional tool_calls.

        Raises:
            httpx.HTTPStatusError: On HTTP errors (401, 429, 500, etc.).
        """
        pass

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMChunk, None]:
        """Generate a streaming completion.

        Yields LLMChunk as content is generated. Tool names are detected
        when <tool_call> patterns appear in the stream.

        Args:
            prompt: User prompt.
            system: Optional system prompt.
            tools: Optional tool definitions.
            **kwargs: Provider-specific options.

        Yields:
            LLMChunk with content and tool names.

        Raises:
            httpx.HTTPStatusError: On HTTP errors.
        """
        pass

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: The text to count tokens for.

        Returns:
            Approximate token count.
        """
        pass

    def supports_tools(self) -> bool:
        """Check if provider supports tool/function calling.

        Returns:
            True by default. Override to return False for providers
            that don't support function calling.
        """
        return True

    async def _call_with_timeout(
        self,
        coro: Any,
        operation: str,
    ) -> Any:
        """Execute an async coroutine with timeout.

        This method wraps async operations with configurable timeouts
        and provides consistent error handling.

        Args:
            coro: The coroutine to execute.
            operation: Name of the operation for error messages (e.g., "complete", "stream").

        Returns:
            Result of the coroutine.

        Raises:
            LLMToolError: If the operation times out.
        """
        timeout = self.timeout_config.get_timeout_for_operation(operation)
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise LLMToolError(
                f"Timeout after {timeout}s during {operation}",
                recoverable=True,
                operation=operation,
                duration=timeout,
            )

    @property
    def circuit_breaker(self) -> Optional[CircuitBreaker]:
        """Get the circuit breaker instance.

        Returns:
            CircuitBreaker instance if configured, None otherwise.
        """
        return self._circuit_breaker

    @circuit_breaker.setter
    def circuit_breaker(self, value: CircuitBreaker) -> None:
        """Set the circuit breaker instance.

        Args:
            value: CircuitBreaker instance to use for this provider.
        """
        self._circuit_breaker = value

    async def _call_with_circuit_breaker(
        self,
        coro_func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute an async coroutine with circuit breaker protection.

        If a circuit breaker is configured, this method wraps the coroutine
        to prevent cascade failures when the provider is experiencing issues.

        Args:
            coro_func: The async function to execute.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            Result of the coroutine.

        Raises:
            CircuitOpenError: If the circuit breaker is open.
        """
        if self._circuit_breaker is None:
            return await coro_func(*args, **kwargs)

        return await self._circuit_breaker.call(coro_func, *args, **kwargs)

    async def close(self) -> None:
        """Close the provider and release resources.

        Override this method to clean up HTTP clients, connection pools, etc.
        """
        pass
