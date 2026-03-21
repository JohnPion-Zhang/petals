"""LLM Provider factory and exports."""

from typing import Optional, Dict, Any

from .base import BaseLLMProvider, LLMResponse, LLMChunk
from .resilience import TimeoutConfig, LLMToolError
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .fallback import (
    FallbackChain,
    FallbackConfig,
    LLMProviderRegistry,
    AllProvidersFailedError,
)


def create_provider(
    provider_type: str,
    **kwargs: Any,
) -> BaseLLMProvider:
    """Create an LLM provider by type.

    Args:
        provider_type: Type of provider to create.
            Supported types: "openai", "anthropic".
        **kwargs: Arguments to pass to the provider constructor.

    Returns:
        Configured provider instance.

    Raises:
        ValueError: If provider_type is not supported.

    Example:
        >>> provider = create_provider(
        ...     "openai",
        ...     api_key="sk-...",
        ...     model="gpt-4",
        ... )
        >>> response = await provider.complete("Hello")
    """
    providers: Dict[str, type] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }

    provider_class = providers.get(provider_type.lower())
    if not provider_class:
        supported = ", ".join(providers.keys())
        raise ValueError(
            f"Unknown provider type: {provider_type!r}. "
            f"Supported types: {supported}"
        )

    return provider_class(**kwargs)


__all__ = [
    # Base classes
    "BaseLLMProvider",
    "LLMResponse",
    "LLMChunk",
    # Resilience
    "TimeoutConfig",
    "LLMToolError",
    # Providers
    "OpenAIProvider",
    "AnthropicProvider",
    # Factory
    "create_provider",
    # Fallback
    "FallbackChain",
    "FallbackConfig",
    "LLMProviderRegistry",
    "AllProvidersFailedError",
]
