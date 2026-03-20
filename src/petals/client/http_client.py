"""
HTTP Client for LLM API calls using litellm.

Provides a unified interface for OpenAI, Anthropic, and custom OpenAI-compatible endpoints.
Supports runtime provider switching via model name.
"""
import litellm
from typing import Protocol, Optional, Dict, Any, List, Union
from dataclasses import dataclass


@dataclass
class LLMResponse:
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

    async def chat(self, messages: List[Dict], **kwargs) -> LLMResponse:
        """Chat completion with message history."""
        ...

    @property
    def default_model(self) -> str:
        """Get the default model."""
        ...


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
        default_model: str = "gpt-4o-mini",
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
        self._default_model = default_model
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

    def _is_anthropic_model(self, model: str) -> bool:
        """Check if the model is Anthropic-based."""
        return (
            model.startswith("anthropic/") or
            "claude" in model.lower() or
            model.startswith("claude-")
        )

    def _is_responses_model(self, model: str) -> bool:
        """Check if the model uses the OpenAI Responses API format."""
        # Responses API is typically used with specific models or custom endpoints
        # that don't support chat completions. We detect it by checking if the
        # base_url ends with /responses or is the Responses API port.
        if self.base_url:
            base = self.base_url.rstrip("/")
            # Strip /v1 suffix if present
            if base.endswith("/v1"):
                base = base[:-3]
            # Check for /responses in URL or Responses API port (18002)
            if base.endswith("/responses"):
                return True
            # Check if using Responses API port
            from urllib.parse import urlparse
            parsed = urlparse(base)
            if parsed.port == 18002:
                return True
        return False

    def _build_openai_kwargs(self, model: str, **kwargs) -> Dict[str, Any]:
        """Build kwargs for OpenAI-compatible endpoints."""
        params = dict(kwargs)
        params["api_key"] = self.api_key

        if self.base_url:
            # Ensure api_base ends with /v1 for proper endpoint routing
            api_base = self.base_url.rstrip("/")
            if not api_base.endswith("/v1"):
                api_base = api_base + "/v1"
            params["api_base"] = api_base

        # For custom endpoints, litellm needs the openai/ provider prefix
        # to correctly route to /v1/chat/completions endpoint
        if self.base_url and not model.startswith(("openai/", "azure/", "anthropic/", "huggingface/", "custom/")):
            model = f"openai/{model}"

        params["model"] = model

        if "timeout" not in params:
            params["timeout"] = self.timeout
        if "max_retries" not in params:
            params["max_retries"] = self.max_retries

        return params

    def _build_anthropic_kwargs(self, model: str, **kwargs) -> Dict[str, Any]:
        """Build kwargs for Anthropic-compatible endpoints."""
        params = dict(kwargs)
        params["api_key"] = self.api_key

        if self.base_url:
            # Strip trailing /v1 since litellm adds it for Anthropic /messages endpoint
            api_base = self.base_url.rstrip("/")
            if api_base.endswith("/v1"):
                api_base = api_base[:-3]  # Remove "/v1"
            params["api_base"] = api_base

        # Use anthropic/ prefix for litellm to recognize it as Anthropic
        if not model.startswith("anthropic/"):
            model = f"anthropic/{model}"
        params["model"] = model

        if "timeout" not in params:
            params["timeout"] = self.timeout
        if "max_retries" not in params:
            params["max_retries"] = self.max_retries

        return params

    def _extract_anthropic_content(self, response) -> str:
        """Extract content from Anthropic response format."""
        # Check if it's in Anthropic format (content is a list of blocks)
        if hasattr(response, "_response_format"):
            format_type = getattr(response, "_response_format", None)
            if format_type == "anthropic":
                return self._extract_anthropic_content(response)

        # Try OpenAI-style response with choices
        if hasattr(response, "choices") and response.choices:
            message = response.choices[0].message
            # Check main content first
            if hasattr(message, "content") and message.content:
                return message.content
            # Fallback to reasoning_content (for thinking/reasoning models)
            if hasattr(message, "reasoning_content") and message.reasoning_content:
                return message.reasoning_content

        # Anthropic-style response
        if hasattr(response, "content") and response.content:
            content_list = response.content
            if isinstance(content_list, list) and len(content_list) > 0:
                first_block = content_list[0]
                if hasattr(first_block, "text"):
                    return first_block.text
                elif isinstance(first_block, dict):
                    return first_block.get("text", str(first_block))

        return str(response)

    def _extract_anthropic_usage(self, response) -> Optional[Dict[str, int]]:
        """Extract usage info from Anthropic response."""
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            return {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            }
        return None

    def _convert_to_anthropic_format(self, messages: List[Dict]) -> List[Dict]:
        """Convert messages to Anthropic API format.

        Anthropic uses 'role' and 'content' fields directly,
        which is compatible with OpenAI format, but we ensure
        proper typing and structure.
        """
        anthropic_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Ensure content is a string
            if not isinstance(content, str):
                content = str(content)

            anthropic_messages.append({
                "role": role,
                "content": content
            })

        return anthropic_messages

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
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, model=model, **kwargs)

    async def chat(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        system: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Chat completion with message history.

        Supports OpenAI (chat/completions), OpenAI (responses), and Anthropic formats.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: Optional model override.
            system: Optional system prompt (for Anthropic compatibility).
            **kwargs: Additional litellm parameters.

        Returns:
            LLMResponse with content, model, and usage info.
        """
        model = model or self._default_model

        # Route to appropriate handler based on endpoint type
        if self._is_anthropic_model(model):
            return await self._anthropic_chat(messages, model, system, **kwargs)
        elif self._is_responses_model(model):
            return await self._openai_responses(messages, model, **kwargs)
        else:
            return await self._openai_chat(messages, model, **kwargs)

    async def _openai_chat(self, messages: List[Dict], model: str, **kwargs) -> LLMResponse:
        """Handle OpenAI-compatible chat completion."""
        params = self._build_openai_kwargs(model, **kwargs)

        response = await litellm.acompletion(
            messages=messages,
            **params
        )

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

    async def _openai_responses(self, messages: List[Dict], model: str, **kwargs) -> LLMResponse:
        """Handle OpenAI Responses API format (/v1/responses).

        The Responses API uses 'input' instead of 'messages' and has a different
        response structure with 'output' containing message blocks.
        """
        params = self._build_openai_kwargs(model, **kwargs)

        # Convert messages to input format for Responses API
        input_data = [{"role": msg.get("role", "user"), "content": msg.get("content", "")} for msg in messages]

        # Use litellm.aresponses() for Responses API endpoint
        response = await litellm.aresponses(
            input=input_data,
            **params
        )

        # Extract content from Responses API format
        content = ""
        if hasattr(response, "output") and response.output:
            for block in response.output:
                if hasattr(block, "content"):
                    for content_block in block.content:
                        if hasattr(content_block, "text"):
                            content = content_block.text
                            break
                        elif isinstance(content_block, dict) and content_block.get("type") == "output_text":
                            content = content_block.get("text", "")
                            break
                elif isinstance(block, dict):
                    if block.get("type") == "message":
                        for content_block in block.get("content", []):
                            if content_block.get("type") == "output_text":
                                content = content_block.get("text", "")
                                break

        # Fallback to string representation if extraction failed
        if not content:
            content = str(response)

        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": getattr(response.usage, "input_tokens", 0),
                "completion_tokens": getattr(response.usage, "output_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }

        return LLMResponse(
            content=content,
            model=getattr(response, "model", model),
            usage=usage,
            raw_response=response,
        )

    async def _anthropic_chat(
        self,
        messages: List[Dict],
        model: str,
        system: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Handle Anthropic-compatible chat completion."""
        # Build params with anthropic/ prefix (pass only api_key and api_base)
        params = self._build_anthropic_kwargs(model)

        # Convert messages to Anthropic format
        anthropic_messages = self._convert_to_anthropic_format(messages)

        # Add system prompt if provided directly
        system_prompt = system
        if not system_prompt and kwargs.get("system"):
            system_prompt = kwargs.pop("system")

        # Ensure max_tokens is set (required by Anthropic)
        max_tokens = kwargs.pop("max_tokens", 1024)

        response = await litellm.acompletion(
            messages=anthropic_messages,
            system=system_prompt,
            max_tokens=max_tokens,
            **params,
            **kwargs  # Pass remaining kwargs like timeout, max_retries
        )

        usage = self._extract_anthropic_usage(response)
        content = self._extract_anthropic_content(response)

        return LLMResponse(
            content=content,
            model=getattr(response, "model", model),
            usage=usage,
            raw_response=response,
        )

    def switch_model(self, model: str) -> None:
        """Switch default model at runtime.

        Args:
            model: New default model name.
        """
        self._default_model = model
