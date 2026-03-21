"""Anthropic Claude provider."""

import asyncio
import httpx
from typing import Any, AsyncGenerator, Dict, List, Optional
import json
import logging

from .base import BaseLLMProvider, LLMResponse, LLMChunk
from .resilience import TimeoutConfig, LLMToolError

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider.

    Supports:
    - Claude 3.5 Sonnet
    - Claude 3 Opus
    - Claude 3 Haiku
    - Other Anthropic models

    Attributes:
        model: The model to use (defaults to claude-3-5-sonnet-20241022).
        _client: The underlying HTTP client (lazy initialization).

    Example:
        >>> provider = AnthropicProvider(
        ...     api_key="sk-ant-...",
        ...     model="claude-3-5-sonnet-20241022",
        ... )
        >>> response = await provider.complete("Hello, Claude!")
    """

    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = "https://api.anthropic.com",
        timeout: float = 60.0,
        max_retries: int = 3,
        timeout_config: Optional[TimeoutConfig] = None,
        circuit_breaker: Optional[Any] = None,
        **kwargs: Any,
    ):
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key.
            model: Model name.
            base_url: Base URL for the API.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries.
            timeout_config: Detailed timeout configuration for different operations.
            circuit_breaker: Optional circuit breaker for resilience.
            **kwargs: Additional arguments.
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
            Configured AsyncClient instance with Anthropic headers.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url.rstrip("/"),
                headers={
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": self.API_VERSION,
                    "anthropic-dangerous-direct-browser-access": "true",
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
        """Generate completion via Anthropic API.

        Args:
            prompt: User message.
            system: Optional system prompt (max_tokens required for Claude).
            tools: Optional tool definitions (Anthropic-specific format).
            **kwargs: Additional parameters (temperature, max_tokens, etc.).

        Returns:
            LLMResponse with content.

        Raises:
            httpx.HTTPStatusError: On API errors.
            LLMToolError: On timeout.
        """
        client = await self._get_client()

        # Build messages array
        messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]

        # Build request body
        request_data: Dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "messages": messages,
            **kwargs,
        }

        # Add system prompt if provided
        if system:
            request_data["system"] = system

        # Convert OpenAI-style tools to Anthropic format if needed
        if tools:
            request_data["tools"] = self._convert_tools(tools)

        async def _do_request():
            response = await client.post("/v1/messages", json=request_data)
            response.raise_for_status()
            return response

        response = await self._call_with_timeout(_do_request(), "complete")

        data = response.json()

        # Extract content
        content_text = ""
        for block in data.get("content", []):
            block_type = block.get("type")
            if block_type == "text":
                content_text += block.get("text", "")
            elif block_type == "thinking":
                # Handle thinking blocks (e.g., from MiniMax proxy)
                # Extract thinking content but truncate for brevity
                thinking = block.get("thinking", "")
                if thinking:
                    # Truncate long thinking content
                    truncated = thinking[:500] + "..." if len(thinking) > 500 else thinking
                    content_text += f"[Thinking: {truncated}]"

        # Extract tool uses (Anthropic calls them "tool_use")
        tool_calls = None
        for block in data.get("content", []):
            if block.get("type") == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    {
                        "id": block.get("id"),
                        "type": "function",
                        "function": {
                            "name": block.get("name"),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    }
                )

        # Extract usage
        usage = None
        if "usage" in data:
            usage = {
                "prompt_tokens": data["usage"].get("input_tokens", 0),
                "completion_tokens": data["usage"].get("output_tokens", 0),
                "total_tokens": data["usage"].get("input_tokens", 0)
                + data["usage"].get("output_tokens", 0),
            }

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            usage=usage,
            model=data.get("model"),
            finish_reason=data.get("stop_reason"),
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
            tools: Optional tool definitions.
            **kwargs: Additional parameters.

        Yields:
            LLMChunk for each token and event.

        Raises:
            httpx.HTTPStatusError: On API errors.
            LLMToolError: On timeout.
        """
        client = await self._get_client()

        messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]

        request_data: Dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "messages": messages,
            "stream": True,
            **kwargs,
        }

        if system:
            request_data["system"] = system

        if tools:
            request_data["tools"] = self._convert_tools(tools)

        try:
            async with client.stream("POST", "/v1/messages", json=request_data) as response:
                response.raise_for_status()

                content_buffer = ""
                tool_names_detected: List[str] = []

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    if line == "data: [DONE]":
                        break

                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    event_type = data.get("type")

                    if event_type == "content_block_delta":
                        delta = data.get("delta", {})
                        delta_type = delta.get("type")

                        if delta_type == "text_delta":
                            content_buffer += delta.get("text", "")
                            yield LLMChunk(content=delta.get("text", ""))

                        elif delta_type == "thinking_delta":
                            # Claude's extended thinking
                            yield LLMChunk(content=delta.get("thinking", ""))

                        elif delta_type == "input_json_delta":
                            # Tool input streaming
                            pass

                    elif event_type == "content_block_start":
                        block = data.get("content_block", {})
                        if block.get("type") == "tool_use":
                            name = block.get("name")
                            if name and name not in tool_names_detected:
                                tool_names_detected.append(name)
                                yield LLMChunk(
                                    content="",
                                    tool_names=[name],
                                    is_complete=False,
                                )

                    elif event_type == "message_delta":
                        # Final message state
                        if data.get("delta", {}).get("stop_reason"):
                            yield LLMChunk(
                                content="",
                                tool_names=tool_names_detected,
                                is_complete=True,
                            )
        except asyncio.TimeoutError:
            raise LLMToolError(
                f"Timeout after {self.timeout_config.stream_timeout}s during stream",
                recoverable=True,
                operation="stream",
                duration=self.timeout_config.stream_timeout,
            )

    async def count_tokens(self, text: str) -> int:
        """Estimate token count.

        Uses ~4 characters per token approximation.

        Args:
            text: Text to count tokens for.

        Returns:
            Estimated token count.
        """
        return len(text) // 4

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style tools to Anthropic format.

        Args:
            tools: List of OpenAI-style tool definitions.

        Returns:
            List of Anthropic-style tool definitions.
        """
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                anthropic_tools.append(
                    {
                        "name": func.get("name"),
                        "description": func.get("description"),
                        "input_schema": func.get("parameters", {"type": "object"}),
                    }
                )
            else:
                # Pass through non-function tools
                anthropic_tools.append(tool)
        return anthropic_tools
