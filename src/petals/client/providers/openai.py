"""OpenAI-compatible provider."""

import asyncio
import httpx
from typing import Any, AsyncGenerator, Dict, List, Optional
import json
import logging

from .base import BaseLLMProvider, LLMResponse, LLMChunk
from .resilience import TimeoutConfig, LLMToolError

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI-compatible API provider.

    Supports:
    - OpenAI Chat Completions API
    - Azure OpenAI
    - Any OpenAI-compatible API (vLLM, Ollama, LM Studio, etc.)

    Attributes:
        model: The model to use for completions.
        _client: The underlying HTTP client (lazy initialization).

    Example:
        >>> provider = OpenAIProvider(
        ...     api_key="sk-...",
        ...     model="gpt-4",
        ...     base_url="https://api.openai.com/v1",
        ... )
        >>> response = await provider.complete("Hello, world!")
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        max_retries: int = 3,
        timeout_config: Optional[TimeoutConfig] = None,
        circuit_breaker: Optional[Any] = None,
        **kwargs: Any,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key.
            model: Model name (e.g., gpt-4, gpt-3.5-turbo).
            base_url: Base URL for the API. Change for Azure or custom endpoints.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries.
            timeout_config: Detailed timeout configuration for different operations.
            circuit_breaker: Optional circuit breaker for resilience.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            timeout_config=timeout_config,
            circuit_breaker=circuit_breaker,
        )
        self.model = model
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client.

        Returns:
            Configured AsyncClient instance.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url.rstrip("/"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        return self._client

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate completion via OpenAI API.

        Args:
            prompt: User message.
            system: Optional system prompt.
            tools: Optional list of tool definitions.
            **kwargs: Additional OpenAI parameters (temperature, max_tokens, etc.).

        Returns:
            LLMResponse with content and optional tool_calls.

        Raises:
            httpx.HTTPStatusError: On API errors.
            LLMToolError: On timeout.
        """
        client = await self._get_client()

        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        request_data: Dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "messages": messages,
            **kwargs,
        }

        if tools:
            request_data["tools"] = tools
            request_data["tool_choice"] = "auto"

        async def _do_request():
            response = await client.post("/chat/completions", json=request_data)
            response.raise_for_status()
            return response

        # Apply both circuit breaker and timeout protection
        async def _protected_request():
            return await self._call_with_timeout(_do_request(), "complete")

        response = await self._call_with_circuit_breaker(_protected_request)

        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]

        return LLMResponse(
            content=message.get("content", ""),
            tool_calls=message.get("tool_calls"),
            usage=data.get("usage"),
            model=data.get("model"),
            finish_reason=choice.get("finish_reason"),
            raw=data,
        )

    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMChunk, None]:
        """Generate streaming completion.

        Args:
            prompt: User message.
            system: Optional system prompt.
            tools: Optional list of tool definitions.
            **kwargs: Additional parameters.

        Yields:
            LLMChunk for each token and tool call.

        Raises:
            httpx.HTTPStatusError: On API errors.
            LLMToolError: On timeout.
        """
        client = await self._get_client()

        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        request_data: Dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "messages": messages,
            "stream": True,
            **kwargs,
        }

        if tools:
            request_data["tools"] = tools
            request_data["tool_choice"] = "auto"

        try:
            async with client.stream("POST", "/chat/completions", json=request_data) as response:
                response.raise_for_status()

                content_buffer = ""
                tool_names_detected: List[str] = []

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    if line == "data: [DONE]":
                        yield LLMChunk(
                            content="",
                            tool_names=tool_names_detected,
                            is_complete=True,
                        )
                        break

                    data_str = line[6:]  # Remove "data: " prefix
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    delta = data.get("choices", [{}])[0].get("delta", {})

                    # Content chunks
                    if "content" in delta:
                        content_buffer += delta["content"]
                        yield LLMChunk(content=delta["content"])

                    # Tool call detection
                    if "tool_calls" in delta:
                        for tc in delta["tool_calls"]:
                            if tc.get("function"):
                                name = tc["function"]["name"]
                                if name not in tool_names_detected:
                                    tool_names_detected.append(name)
                                    yield LLMChunk(
                                        content="",
                                        tool_names=[name],
                                        is_complete=False,
                                    )
        except asyncio.TimeoutError:
            raise LLMToolError(
                f"Timeout after {self.timeout_config.stream_timeout}s during stream",
                recoverable=True,
                operation="stream",
                duration=self.timeout_config.stream_timeout,
            )

    async def count_tokens(self, text: str) -> int:
        """Estimate token count using simple approximation.

        Uses ~4 characters per token on average for English text.

        Args:
            text: Text to count tokens for.

        Returns:
            Estimated token count.
        """
        # Simple approximation: ~4 characters per token
        # OpenAI uses tiktoken for accurate counting
        return len(text) // 4

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
