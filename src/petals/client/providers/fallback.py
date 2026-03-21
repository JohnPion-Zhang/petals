"""
Fallback chain logic for graceful degradation across multiple LLM providers.

This module provides fallback mechanisms to gracefully handle provider failures
by automatically switching to alternative providers based on configured priorities.

Example:
    >>> from petals.client.providers.fallback import (
    ...     FallbackChain,
    ...     FallbackConfig,
    ...     LLMProviderRegistry,
    ...     AllProvidersFailedError,
    ... )
    >>>
    >>> # Create registry and register providers
    >>> registry = LLMProviderRegistry()
    >>> registry.register("openai", openai_provider, CircuitBreakerConfig())
    >>> registry.register("anthropic", anthropic_provider, CircuitBreakerConfig())
    >>>
    >>> # Create fallback chain
    >>> config = FallbackConfig(providers=["openai", "anthropic"])
    >>> chain = FallbackChain(config, registry)
    >>>
    >>> # Execute with automatic fallback
    >>> result = await chain.execute("complete", prompt="Hello")
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from petals.client.feedback.retry_policy import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerManager,
    CircuitOpenError,
)
from petals.client.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class AllProvidersFailedError(Exception):
    """Exception raised when all providers in the fallback chain fail.

    Attributes:
        attempted: List of provider names that were attempted.
        provider_count: Total number of providers that failed.
        last_exception: The last exception that was raised.
    """

    def __init__(
        self,
        message: Optional[str] = None,
        attempted: Optional[List[str]] = None,
        provider_count: Optional[int] = None,
        last_exception: Optional[Exception] = None,
    ):
        if message is None:
            attempted = attempted or []
            provider_count = provider_count or len(attempted)
            message = f"All {provider_count} providers failed: {', '.join(attempted)}"

        super().__init__(message)
        self.attempted = attempted or []
        self.provider_count = provider_count or len(self.attempted)
        self.last_exception = last_exception


@dataclass
class FallbackConfig:
    """Configuration for provider fallback chain.

    Attributes:
        providers: List of provider names in priority order (first is highest).
        retry_on_fallback: Whether to retry failed providers on subsequent calls.
        fallback_delay: Delay in seconds before trying the next fallback provider.

    Example:
        >>> config = FallbackConfig(
        ...     providers=["openai", "anthropic", "cohere"],
        ...     retry_on_fallback=True,
        ...     fallback_delay=0.5,
        ... )
    """

    providers: List[str] = field(default_factory=list)
    retry_on_fallback: bool = True
    fallback_delay: float = 0.0

    def __post_init__(self):
        """Validate configuration."""
        if not self.providers:
            logger.warning("FallbackConfig created with empty providers list")


class LLMProviderRegistry:
    """Registry of all LLM providers with their circuit breakers.

    This class manages provider registration, retrieval, and lifecycle.
    Each provider is associated with its own circuit breaker for
    independent failure tracking.

    Example:
        >>> registry = LLMProviderRegistry()
        >>> registry.register(
        ...     "openai",
        ...     OpenAIProvider(api_key="..."),
        ...     CircuitBreakerConfig(failure_threshold=3),
        ... )
        >>>
        >>> provider = registry.get_provider("openai")
        >>> breaker = registry.get_breaker("openai")
    """

    def __init__(self):
        """Initialize the provider registry."""
        self._providers: Dict[str, BaseLLMProvider] = {}
        self._circuit_manager = CircuitBreakerManager()

    def register(
        self,
        name: str,
        provider: BaseLLMProvider,
        breaker_config: Optional[CircuitBreakerConfig] = None,
    ) -> None:
        """Register a provider with its circuit breaker configuration.

        Args:
            name: Unique identifier for the provider.
            provider: The LLM provider instance.
            breaker_config: Optional circuit breaker configuration.
                Uses default if not provided.

        Raises:
            ValueError: If a provider with the given name is already registered.

        Example:
            >>> registry.register(
            ...     "openai",
            ...     OpenAIProvider(api_key="..."),
            ...     CircuitBreakerConfig(failure_threshold=3),
            ... )
        """
        if name in self._providers:
            raise ValueError(f"Provider '{name}' is already registered")

        self._providers[name] = provider
        self._circuit_manager.get_breaker(name, breaker_config)
        logger.info(f"Registered provider: {name}")

    def deregister(self, name: str) -> None:
        """Deregister a provider.

        Args:
            name: Provider name to deregister.

        Raises:
            KeyError: If provider is not found.
        """
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' not found")

        del self._providers[name]
        self._circuit_manager.remove_breaker(name)
        logger.info(f"Deregistered provider: {name}")

    def get_provider(self, name: str) -> BaseLLMProvider:
        """Get a registered provider by name.

        Args:
            name: Provider name.

        Returns:
            The provider instance.

        Raises:
            KeyError: If provider is not found.
        """
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' not found in registry")
        return self._providers[name]

    def get_breaker(self, name: str) -> CircuitBreaker:
        """Get the circuit breaker for a provider.

        Args:
            name: Provider name.

        Returns:
            The circuit breaker instance.

        Raises:
            KeyError: If provider is not registered.
        """
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' not found in registry")
        return self._circuit_manager.get_breaker(name)

    def list_providers(self) -> List[str]:
        """List all registered provider names.

        Returns:
            List of provider names.
        """
        return list(self._providers.keys())

    def get_breaker_stats(self, name: str) -> Dict[str, Any]:
        """Get circuit breaker statistics for a provider.

        Args:
            name: Provider name.

        Returns:
            Dictionary with breaker statistics.

        Raises:
            KeyError: If provider is not registered.
        """
        breaker = self.get_breaker(name)
        return breaker.get_stats()

    def get_all_states(self) -> Dict[str, str]:
        """Get the state of all circuit breakers.

        Returns:
            Dictionary mapping provider names to circuit states.
        """
        return self._circuit_manager.get_all_states()

    def reset_all_breakers(self) -> None:
        """Reset all circuit breakers to closed state."""
        self._circuit_manager.reset_all()

    @property
    def circuit_manager(self) -> CircuitBreakerManager:
        """Get the circuit breaker manager.

        Returns:
            The CircuitBreakerManager instance.
        """
        return self._circuit_manager


class FallbackChain:
    """Manages fallback between multiple LLM providers.

    The FallbackChain attempts to execute operations using a priority-ordered
    list of providers. When a provider fails (either by raising an exception
    or having an open circuit breaker), the chain automatically tries the
    next provider in the configuration.

    The chain respects circuit breaker states, skipping providers with open
    circuits to avoid unnecessary failed requests.

    Example:
        >>> config = FallbackConfig(
        ...     providers=["openai", "anthropic", "cohere"],
        ...     retry_on_fallback=True,
        ... )
        >>> chain = FallbackChain(config, registry)
        >>>
        >>> # Automatically falls back on failure
        >>> result = await chain.execute("complete", prompt="Hello")
        >>>
        >>> # Check which providers were attempted
        >>> print(f"Attempted: {chain.attempted}")

    Attributes:
        config: The fallback configuration.
        registry: The LLM provider registry.
        attempted: Set of provider names that were attempted in the last execution.
    """

    def __init__(
        self,
        config: FallbackConfig,
        registry: LLMProviderRegistry,
    ):
        """Initialize the fallback chain.

        Args:
            config: Fallback configuration with provider priority order.
            registry: The LLM provider registry containing provider instances.
        """
        self.config = config
        self.registry = registry
        self._attempted: Set[str] = set()

    @property
    def attempted(self) -> Set[str]:
        """Get the set of providers that were attempted.

        Returns:
            Set of provider names attempted in the last execute call.
        """
        return self._attempted.copy()

    async def execute(
        self,
        operation: str,
        **kwargs: Any,
    ) -> Any:
        """Execute an operation with fallback logic.

        Attempts to execute the operation on each provider in priority order
        until one succeeds. Providers with open circuit breakers are skipped.

        Args:
            operation: The operation to execute on each provider.
                Supported operations: "complete", "stream", "count_tokens".
            **kwargs: Arguments to pass to the operation.

        Returns:
            The result from the first successful provider.

        Raises:
            AllProvidersFailedError: When all providers fail.
            KeyError: If an operation is not found on a provider.
            CircuitOpenError: If a provider has an open circuit breaker.
        """
        self._attempted.clear()
        last_exception: Optional[Exception] = None

        for provider_name in self.config.providers:
            # Check circuit breaker state
            breaker = self.registry.get_breaker(provider_name)

            if breaker.state == "open":
                logger.warning(
                    f"Skipping provider '{provider_name}': circuit breaker is open "
                    f"(failures: {breaker.failure_count})"
                )
                continue

            provider = self.registry.get_provider(provider_name)

            # Check if operation exists
            if not hasattr(provider, operation):
                raise KeyError(
                    f"Provider '{provider_name}' does not support operation '{operation}'"
                )

            try:
                # Apply circuit breaker protection to the call
                op_func = getattr(provider, operation)

                if breaker.state == "closed":
                    # Use circuit breaker to protect the call
                    result = await breaker.call(op_func, **kwargs)
                else:
                    # In half-open state, allow the call without extra protection
                    result = await op_func(**kwargs)

                self._attempted.add(provider_name)
                logger.info(
                    f"Provider '{provider_name}' succeeded for operation '{operation}'"
                )
                return result

            except CircuitOpenError as e:
                logger.warning(
                    f"Provider '{provider_name}' circuit breaker opened during "
                    f"operation '{operation}': {e}"
                )
                self._attempted.add(provider_name)
                last_exception = e
                # Continue to next provider

            except Exception as e:
                logger.warning(
                    f"Provider '{provider_name}' failed for operation "
                    f"'{operation}': {type(e).__name__}: {e}"
                )
                self._attempted.add(provider_name)
                last_exception = e

                # Apply fallback delay if configured
                if self.config.fallback_delay > 0:
                    await asyncio.sleep(self.config.fallback_delay)

                # If retry_on_fallback is False, raise immediately
                if not self.config.retry_on_fallback:
                    raise AllProvidersFailedError(
                        attempted=list(self._attempted),
                        provider_count=len(self._attempted),
                        last_exception=last_exception,
                    )

        # All providers failed
        raise AllProvidersFailedError(
            attempted=list(self._attempted),
            provider_count=len(self._attempted),
            last_exception=last_exception,
        )

    async def complete_with_fallback(
        self,
        prompt: str,
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Any:
        """Convenience method for completing with fallback.

        This is a wrapper around execute("complete", ...) with
        LLM-specific parameters.

        Args:
            prompt: User prompt.
            system: Optional system prompt.
            tools: Optional tool definitions.
            **kwargs: Additional provider-specific options.

        Returns:
            LLMResponse from the first successful provider.

        Raises:
            AllProvidersFailedError: When all providers fail.
        """
        return await self.execute(
            "complete",
            prompt=prompt,
            system=system,
            tools=tools,
            **kwargs,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all providers in the fallback chain.

        Returns:
            Dictionary with provider names mapped to their circuit breaker stats.
        """
        stats = {}
        for provider_name in self.config.providers:
            try:
                stats[provider_name] = self.registry.get_breaker_stats(provider_name)
            except KeyError:
                stats[provider_name] = {"error": "Provider not registered"}
        return stats


__all__ = [
    "AllProvidersFailedError",
    "FallbackChain",
    "FallbackConfig",
    "LLMProviderRegistry",
]
