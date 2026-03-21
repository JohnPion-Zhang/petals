"""
Tests for resilience features including per-provider circuit breakers.

This module tests the CircuitBreakerManager and its ability to maintain
independent circuit breaker state for each provider.
"""

import pytest
import asyncio
from typing import Optional

from petals.client.feedback.retry_policy import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerManager,
    CircuitOpenError,
)
from petals.client.providers.base import BaseLLMProvider, LLMResponse
from petals.client.providers.openai import OpenAIProvider
from petals.client.providers.anthropic import AnthropicProvider
from petals.client.providers.resilience import TimeoutConfig, LLMToolError


class MockProvider(BaseLLMProvider):
    """Mock provider for testing circuit breaker integration."""

    def __init__(
        self,
        should_fail: bool = False,
        fail_count: int = 0,
        circuit_breaker: Optional[CircuitBreaker] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.should_fail = should_fail
        self.fail_count = fail_count
        self.call_count = 0
        self._circuit_breaker = circuit_breaker

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        tools=None,
        **kwargs,
    ) -> LLMResponse:
        """Mock completion that can fail."""
        self.call_count += 1

        if self._circuit_breaker:
            async def _do_request():
                return await self._execute_request()

            return await self._circuit_breaker.call(_do_request)

        return await self._execute_request()

    async def _execute_request(self) -> LLMResponse:
        """Execute the actual request logic."""
        if self.should_fail and self.call_count <= self.fail_count:
            raise ValueError("Simulated provider failure")

        return LLMResponse(content=f"Response for call {self.call_count}")

    async def stream(self, prompt, system=None, tools=None, **kwargs):
        """Mock streaming."""
        response = await self.complete(prompt, system, tools, **kwargs)
        yield response

    async def count_tokens(self, text: str) -> int:
        """Mock token counting."""
        return len(text) // 4


class TestCircuitBreakerManager:
    """Tests for CircuitBreakerManager class."""

    def test_create_manager(self):
        """Test that manager initializes with empty breakers."""
        manager = CircuitBreakerManager()
        states = manager.get_all_states()
        assert states == {}

    def test_get_breaker_creates_new(self):
        """Test that get_breaker creates a new breaker if none exists."""
        manager = CircuitBreakerManager()
        config = CircuitBreakerConfig(failure_threshold=3)

        breaker = manager.get_breaker("openai", config)

        assert breaker is not None
        assert breaker.state == CircuitBreaker.STATE_CLOSED

    def test_get_breaker_returns_same_instance(self):
        """Test that get_breaker returns the same instance for same provider."""
        manager = CircuitBreakerManager()
        config = CircuitBreakerConfig(failure_threshold=3)

        breaker1 = manager.get_breaker("openai", config)
        breaker2 = manager.get_breaker("openai", config)

        assert breaker1 is breaker2

    def test_get_breaker_different_providers(self):
        """Test that different providers get different breakers."""
        manager = CircuitBreakerManager()
        config = CircuitBreakerConfig(failure_threshold=3)

        openai_breaker = manager.get_breaker("openai", config)
        anthropic_breaker = manager.get_breaker("anthropic", config)

        assert openai_breaker is not anthropic_breaker

    def test_get_all_states(self):
        """Test that get_all_states returns all breaker states."""
        manager = CircuitBreakerManager()
        config = CircuitBreakerConfig(failure_threshold=3)

        manager.get_breaker("openai", config)
        manager.get_breaker("anthropic", config)

        states = manager.get_all_states()

        assert "openai" in states
        assert "anthropic" in states
        assert states["openai"] == CircuitBreaker.STATE_CLOSED
        assert states["anthropic"] == CircuitBreaker.STATE_CLOSED


class TestIndependentCircuitBreakers:
    """Tests that verify provider circuit breakers operate independently."""

    @pytest.mark.asyncio
    async def test_openai_failure_does_not_affect_anthropic(self):
        """Test that OpenAI failures don't affect Anthropic circuit breaker."""
        manager = CircuitBreakerManager()
        config = CircuitBreakerConfig(failure_threshold=2)

        # Get breakers for each provider
        openai_breaker = manager.get_breaker("openai", config)
        anthropic_breaker = manager.get_breaker("anthropic", config)

        # Create providers with circuit breakers
        openai_provider = MockProvider(
            should_fail=True,
            fail_count=3,
            circuit_breaker=openai_breaker,
        )
        anthropic_provider = MockProvider(
            should_fail=False,
            circuit_breaker=anthropic_breaker,
        )

        # OpenAI should fail and trip its breaker
        for _ in range(3):
            try:
                await openai_provider.complete("test")
            except (ValueError, CircuitOpenError):
                pass

        # OpenAI breaker should be open
        assert openai_breaker.state == CircuitBreaker.STATE_OPEN

        # Anthropic breaker should still be closed
        assert anthropic_breaker.state == CircuitBreaker.STATE_CLOSED

        # Anthropic should still work
        response = await anthropic_provider.complete("test")
        assert response.content == "Response for call 1"

    @pytest.mark.asyncio
    async def test_anthropic_failure_does_not_affect_openai(self):
        """Test that Anthropic failures don't affect OpenAI circuit breaker."""
        manager = CircuitBreakerManager()
        config = CircuitBreakerConfig(failure_threshold=2)

        # Get breakers for each provider
        openai_breaker = manager.get_breaker("openai", config)
        anthropic_breaker = manager.get_breaker("anthropic", config)

        # Create providers with circuit breakers
        anthropic_provider = MockProvider(
            should_fail=True,
            fail_count=3,
            circuit_breaker=anthropic_breaker,
        )
        openai_provider = MockProvider(
            should_fail=False,
            circuit_breaker=openai_breaker,
        )

        # Anthropic should fail and trip its breaker
        for _ in range(3):
            try:
                await anthropic_provider.complete("test")
            except (ValueError, CircuitOpenError):
                pass

        # Anthropic breaker should be open
        assert anthropic_breaker.state == CircuitBreaker.STATE_OPEN

        # OpenAI breaker should still be closed
        assert openai_breaker.state == CircuitBreaker.STATE_CLOSED

        # OpenAI should still work
        response = await openai_provider.complete("test")
        assert response.content == "Response for call 1"

    @pytest.mark.asyncio
    async def test_circuit_breaker_rejects_requests_when_open(self):
        """Test that open circuit breaker rejects requests."""
        manager = CircuitBreakerManager()
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=60.0)

        breaker = manager.get_breaker("openai", config)

        provider = MockProvider(
            should_fail=True,
            fail_count=5,
            circuit_breaker=breaker,
        )

        # First call should fail
        with pytest.raises(ValueError):
            await provider.complete("test")

        # Circuit should now be open
        assert breaker.state == CircuitBreaker.STATE_OPEN

        # Subsequent calls should raise CircuitOpenError
        with pytest.raises(CircuitOpenError):
            await provider.complete("test")

    @pytest.mark.asyncio
    async def test_different_config_per_provider(self):
        """Test that providers can have different circuit breaker configs."""
        manager = CircuitBreakerManager()

        # OpenAI with stricter settings
        openai_config = CircuitBreakerConfig(failure_threshold=2)
        # Anthropic with more lenient settings
        anthropic_config = CircuitBreakerConfig(failure_threshold=5)

        openai_breaker = manager.get_breaker("openai", openai_config)
        anthropic_breaker = manager.get_breaker("anthropic", anthropic_config)

        # OpenAI breaker should trip after 2 failures
        openai_provider = MockProvider(
            should_fail=True,
            fail_count=5,
            circuit_breaker=openai_breaker,
        )

        anthropic_provider = MockProvider(
            should_fail=True,
            fail_count=5,
            circuit_breaker=anthropic_breaker,
        )

        # Fail 2 requests on OpenAI
        for i in range(2):
            try:
                await openai_provider.complete(f"test {i}")
            except ValueError:
                pass

        # OpenAI should be open
        assert openai_breaker.state == CircuitBreaker.STATE_OPEN

        # Fail 2 requests on Anthropic (should not trip yet)
        for i in range(2):
            try:
                await anthropic_provider.complete(f"test {i}")
            except ValueError:
                pass

        # Anthropic should still be closed (needs 5 failures)
        assert anthropic_breaker.state == CircuitBreaker.STATE_CLOSED


class TestProviderCircuitBreakerIntegration:
    """Integration tests for providers with circuit breakers."""

    def test_provider_accepts_circuit_breaker(self):
        """Test that providers accept circuit breaker via constructor."""
        breaker = CircuitBreaker(CircuitBreakerConfig())

        # Should not raise
        provider = MockProvider(circuit_breaker=breaker)
        assert provider._circuit_breaker is breaker

    def test_openai_provider_with_circuit_breaker(self):
        """Test OpenAIProvider accepts circuit breaker parameter."""
        breaker = CircuitBreaker(CircuitBreakerConfig())

        # The provider should accept circuit_breaker in kwargs
        # and store it appropriately
        provider = OpenAIProvider(
            api_key="test-key",
            model="gpt-4",
            circuit_breaker=breaker,
        )

        # Verify the provider is created successfully
        assert provider.model == "gpt-4"
        assert provider.api_key == "test-key"

    def test_anthropic_provider_with_circuit_breaker(self):
        """Test AnthropicProvider accepts circuit breaker parameter."""
        breaker = CircuitBreaker(CircuitBreakerConfig())

        provider = AnthropicProvider(
            api_key="test-key",
            model="claude-3-5-sonnet-20241022",
            circuit_breaker=breaker,
        )

        # Verify the provider is created successfully
        assert provider.model == "claude-3-5-sonnet-20241022"
        assert provider.api_key == "test-key"


class TestCircuitBreakerManagerStats:
    """Tests for CircuitBreakerManager statistics methods."""

    def test_get_stats(self):
        """Test that get_stats returns all breaker statistics."""
        manager = CircuitBreakerManager()
        config = CircuitBreakerConfig(failure_threshold=3)

        openai_breaker = manager.get_breaker("openai", config)
        anthropic_breaker = manager.get_breaker("anthropic", config)

        stats = manager.get_stats()

        assert "openai" in stats
        assert "anthropic" in stats
        assert stats["openai"]["failure_threshold"] == 3
        assert stats["anthropic"]["failure_threshold"] == 3


# Imports for fallback chain tests
from petals.client.providers.fallback import (
    FallbackConfig,
    FallbackChain,
    LLMProviderRegistry,
    AllProvidersFailedError,
)


class MockFallbackProvider(BaseLLMProvider):
    """Mock LLM provider for testing fallback logic."""

    def __init__(
        self,
        name: str,
        should_fail: bool = False,
        response_content: str = "default response",
        failure_exception: Optional[Exception] = None,
    ):
        super().__init__(api_key="test")
        self.name = name
        self.should_fail = should_fail
        self.response_content = response_content
        self.failure_exception = failure_exception or ConnectionError(f"{name} failed")
        self.call_count = 0

    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        self.call_count += 1
        if self.should_fail:
            raise self.failure_exception
        return LLMResponse(content=self.response_content)

    async def stream(self, prompt: str, **kwargs):
        if self.should_fail:
            raise self.failure_exception
        yield LLMResponse(content=self.response_content)

    async def count_tokens(self, text: str) -> int:
        return len(text.split())


class TestFallbackConfig:
    """Tests for FallbackConfig dataclass."""

    def test_default_config(self):
        """Test default fallback configuration."""
        config = FallbackConfig(providers=["openai", "anthropic"])
        assert config.providers == ["openai", "anthropic"]
        assert config.retry_on_fallback is True
        assert config.fallback_delay == 0.0

    def test_custom_config(self):
        """Test custom fallback configuration."""
        config = FallbackConfig(
            providers=["primary", "secondary", "tertiary"],
            retry_on_fallback=False,
            fallback_delay=1.5,
        )
        assert config.providers == ["primary", "secondary", "tertiary"]
        assert config.retry_on_fallback is False
        assert config.fallback_delay == 1.5


class TestLLMProviderRegistry:
    """Tests for LLMProviderRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry."""
        return LLMProviderRegistry()

    def test_registry_empty_init(self, registry):
        """Test registry initializes empty."""
        assert len(registry.list_providers()) == 0

    def test_register_provider(self, registry):
        """Test registering a provider."""
        provider = MockFallbackProvider("test_provider")
        breaker_config = CircuitBreakerConfig(failure_threshold=3)

        registry.register("test_provider", provider, breaker_config)

        assert "test_provider" in registry.list_providers()
        assert registry.get_provider("test_provider") is provider
        assert registry.get_breaker("test_provider") is not None

    def test_register_multiple_providers(self, registry):
        """Test registering multiple providers."""
        providers = {
            "openai": MockFallbackProvider("openai"),
            "anthropic": MockFallbackProvider("anthropic"),
            "cohere": MockFallbackProvider("cohere"),
        }

        for name, provider in providers.items():
            registry.register(name, provider, CircuitBreakerConfig())

        assert len(registry.list_providers()) == 3
        for name in providers.keys():
            assert registry.get_provider(name) is not None

    def test_register_duplicate_raises(self, registry):
        """Test registering duplicate provider raises error."""
        provider = MockFallbackProvider("test")
        registry.register("test", provider, CircuitBreakerConfig())

        with pytest.raises(ValueError, match="already registered"):
            registry.register("test", provider, CircuitBreakerConfig())

    def test_get_nonexistent_provider_raises(self, registry):
        """Test getting non-existent provider raises error."""
        with pytest.raises(KeyError, match="not found"):
            registry.get_provider("nonexistent")

    def test_get_nonexistent_breaker_raises(self, registry):
        """Test getting non-existent breaker raises error."""
        with pytest.raises(KeyError, match="not found"):
            registry.get_breaker("nonexistent")

    def test_deregister_provider(self, registry):
        """Test deregistering a provider."""
        provider = MockFallbackProvider("test")
        registry.register("test", provider, CircuitBreakerConfig())
        registry.deregister("test")

        assert "test" not in registry.list_providers()

    def test_get_breaker_stats(self, registry):
        """Test getting breaker statistics."""
        provider = MockFallbackProvider("test")
        registry.register("test", provider, CircuitBreakerConfig(failure_threshold=3))

        stats = registry.get_breaker_stats("test")
        assert "state" in stats
        assert "failure_count" in stats
        assert stats["failure_threshold"] == 3


class TestFallbackChain:
    """Tests for FallbackChain class."""

    @pytest.fixture
    def registry(self):
        """Create a registry with multiple providers."""
        reg = LLMProviderRegistry()
        reg.register("primary", MockFallbackProvider("primary"), CircuitBreakerConfig())
        reg.register("secondary", MockFallbackProvider("secondary"), CircuitBreakerConfig())
        reg.register("tertiary", MockFallbackProvider("tertiary"), CircuitBreakerConfig())
        return reg

    @pytest.mark.asyncio
    async def test_successful_execution_first_provider(self, registry):
        """Test successful execution on first provider."""
        config = FallbackConfig(providers=["primary", "secondary", "tertiary"])
        chain = FallbackChain(config, registry)

        result = await chain.execute("complete", prompt="Hello")

        assert result.content == "default response"
        assert registry.get_provider("primary").call_count == 1

    @pytest.mark.asyncio
    async def test_fallback_to_second_provider(self, registry):
        """Test fallback to second provider when first fails."""
        registry.get_provider("primary").should_fail = True

        config = FallbackConfig(providers=["primary", "secondary", "tertiary"])
        chain = FallbackChain(config, registry)

        result = await chain.execute("complete", prompt="Hello")

        assert result.content == "default response"
        assert registry.get_provider("primary").call_count == 1
        assert registry.get_provider("secondary").call_count == 1

    @pytest.mark.asyncio
    async def test_fallback_through_all_providers(self, registry):
        """Test fallback through all providers until one succeeds."""
        # All providers fail except tertiary
        registry.get_provider("primary").should_fail = True
        registry.get_provider("secondary").should_fail = True

        config = FallbackConfig(providers=["primary", "secondary", "tertiary"])
        chain = FallbackChain(config, registry)

        result = await chain.execute("complete", prompt="Hello")

        assert result.content == "default response"
        assert registry.get_provider("primary").call_count == 1
        assert registry.get_provider("secondary").call_count == 1
        assert registry.get_provider("tertiary").call_count == 1

    @pytest.mark.asyncio
    async def test_all_providers_fail_raises(self, registry):
        """Test all providers fail raises AllProvidersFailedError."""
        # All providers fail
        registry.get_provider("primary").should_fail = True
        registry.get_provider("secondary").should_fail = True
        registry.get_provider("tertiary").should_fail = True

        config = FallbackConfig(providers=["primary", "secondary", "tertiary"])
        chain = FallbackChain(config, registry)

        with pytest.raises(AllProvidersFailedError) as exc_info:
            await chain.execute("complete", prompt="Hello")

        assert exc_info.value.provider_count == 3
        assert set(exc_info.value.attempted) == {"primary", "secondary", "tertiary"}

    @pytest.mark.asyncio
    async def test_skip_open_circuit_breaker(self, registry):
        """Test skipping provider with open circuit breaker."""
        # Open the circuit breaker for primary
        primary_breaker = registry.get_breaker("primary")
        primary_breaker._state = "open"
        primary_breaker._failure_count = 5

        config = FallbackConfig(providers=["primary", "secondary"])
        chain = FallbackChain(config, registry)

        result = await chain.execute("complete", prompt="Hello")

        # Should succeed with secondary
        assert result.content == "default response"
        # Primary should not be called due to open circuit
        assert registry.get_provider("primary").call_count == 0
        assert registry.get_provider("secondary").call_count == 1

    @pytest.mark.asyncio
    async def test_no_retry_on_fallback(self, registry):
        """Test retry_on_fallback=False raises immediately."""
        registry.get_provider("primary").should_fail = True

        config = FallbackConfig(providers=["primary", "secondary"], retry_on_fallback=False)
        chain = FallbackChain(config, registry)

        with pytest.raises(AllProvidersFailedError):
            await chain.execute("complete", prompt="Hello")

    @pytest.mark.asyncio
    async def test_attempted_providers_tracking(self, registry):
        """Test that attempted providers are tracked correctly."""
        registry.get_provider("primary").should_fail = True
        registry.get_provider("secondary").should_fail = True
        registry.get_provider("tertiary").should_fail = True

        config = FallbackConfig(providers=["primary", "secondary", "tertiary"])
        chain = FallbackChain(config, registry)

        with pytest.raises(AllProvidersFailedError):
            await chain.execute("complete", prompt="Hello")

        assert chain.attempted == {"primary", "secondary", "tertiary"}

    @pytest.mark.asyncio
    async def test_fallback_with_delay(self, registry):
        """Test fallback with configured delay."""
        registry.get_provider("primary").should_fail = True

        config = FallbackConfig(providers=["primary", "secondary"], fallback_delay=0.05)
        chain = FallbackChain(config, registry)

        start = asyncio.get_event_loop().time()
        result = await chain.execute("complete", prompt="Hello")
        elapsed = asyncio.get_event_loop().time() - start

        # Should have some delay
        assert elapsed >= 0.03  # Allow some tolerance
        assert result.content == "default response"


class TestAllProvidersFailedError:
    """Tests for AllProvidersFailedError exception."""

    def test_error_creation(self):
        """Test error creation with all attributes."""
        error = AllProvidersFailedError(
            attempted=["provider1", "provider2"],
            provider_count=2,
            last_exception=ConnectionError("Test error"),
        )

        assert error.provider_count == 2
        assert error.attempted == ["provider1", "provider2"]
        assert error.last_exception is not None
        assert "provider1" in str(error)
        assert "provider2" in str(error)

    def test_error_string_representation(self):
        """Test error string representation."""
        error = AllProvidersFailedError(
            attempted=["a", "b", "c"],
            provider_count=3,
        )

        error_str = str(error)
        assert "All 3 providers failed" in error_str
        assert "a" in error_str
        assert "b" in error_str
        assert "c" in error_str


class TestIntegration:
    """Integration tests combining registry, fallback, and circuit breakers."""

    @pytest.mark.asyncio
    async def test_full_integration_flow(self):
        """Test full integration of registry, circuit breakers, and fallback."""
        registry = LLMProviderRegistry()

        # Register providers with different configurations
        registry.register(
            "openai",
            MockFallbackProvider("openai"),
            CircuitBreakerConfig(failure_threshold=3),
        )
        registry.register(
            "anthropic",
            MockFallbackProvider("anthropic"),
            CircuitBreakerConfig(failure_threshold=5),
        )
        registry.register(
            "cohere",
            MockFallbackProvider("cohere"),
            CircuitBreakerConfig(failure_threshold=2),
        )

        # Simulate cascading failures
        registry.get_provider("openai").should_fail = True
        registry.get_provider("anthropic").should_fail = True

        config = FallbackConfig(
            providers=["openai", "anthropic", "cohere"],
            retry_on_fallback=True,
            fallback_delay=0.0,
        )
        chain = FallbackChain(config, registry)

        result = await chain.execute("complete", prompt="Integration test")

        assert result.content == "default response"
        assert registry.get_provider("openai").call_count == 1
        assert registry.get_provider("anthropic").call_count == 1
        assert registry.get_provider("cohere").call_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker opens and fallback still works."""
        registry = LLMProviderRegistry()

        # Create a failing provider
        failing_provider = MockFallbackProvider("failing")
        failing_provider.should_fail = True

        registry.register(
            "failing",
            failing_provider,
            CircuitBreakerConfig(failure_threshold=1),
        )
        registry.register(
            "healthy",
            MockFallbackProvider("healthy"),
            CircuitBreakerConfig(),
        )

        config = FallbackConfig(providers=["failing", "healthy"])
        chain = FallbackChain(config, registry)

        # First attempt - should fail and open circuit
        await chain.execute("complete", prompt="test")

        # Verify circuit opened for failing provider
        breaker = registry.get_breaker("failing")
        assert breaker.state == "open"
        assert breaker.failure_count >= 1

        # Healthy provider should have been called
        assert registry.get_provider("healthy").call_count == 1

    @pytest.mark.asyncio
    async def test_priority_order_respected(self):
        """Test that provider priority order is respected."""
        registry = LLMProviderRegistry()

        # Register providers in different order than fallback config
        registry.register("second", MockFallbackProvider("second"), CircuitBreakerConfig())
        registry.register("first", MockFallbackProvider("first"), CircuitBreakerConfig())

        config = FallbackConfig(providers=["first", "second"])
        chain = FallbackChain(config, registry)

        result = await chain.execute("complete", prompt="test")

        # First in config should be tried first
        assert registry.get_provider("first").call_count == 1
        assert registry.get_provider("second").call_count == 0


# ============= Timeout Configuration Tests =============

from unittest.mock import AsyncMock, MagicMock, patch


class TestTimeoutConfig:
    """Tests for TimeoutConfig dataclass."""

    def test_default_values(self):
        """Test TimeoutConfig has correct defaults."""
        config = TimeoutConfig()
        assert config.connect_timeout == 10.0
        assert config.read_timeout == 120.0
        assert config.total_timeout == 180.0
        assert config.chat_timeout == 120.0
        assert config.embedding_timeout == 30.0
        assert config.stream_timeout == 120.0

    def test_custom_values(self):
        """Test TimeoutConfig with custom values."""
        config = TimeoutConfig(
            connect_timeout=5.0,
            read_timeout=60.0,
            total_timeout=90.0,
            chat_timeout=60.0,
            embedding_timeout=15.0,
        )
        assert config.connect_timeout == 5.0
        assert config.read_timeout == 60.0
        assert config.total_timeout == 90.0
        assert config.chat_timeout == 60.0
        assert config.embedding_timeout == 15.0

    def test_get_timeout_for_operation(self):
        """Test getting timeout for specific operations."""
        config = TimeoutConfig(chat_timeout=30.0, embedding_timeout=10.0, stream_timeout=45.0)

        # Default should be used for unknown operations
        assert config.get_timeout_for_operation("unknown") == config.total_timeout
        # Known operations should return their specific timeout
        assert config.get_timeout_for_operation("complete") == 30.0
        assert config.get_timeout_for_operation("chat") == 30.0
        assert config.get_timeout_for_operation("embedding") == 10.0
        assert config.get_timeout_for_operation("stream") == 45.0


class TestLLMToolError:
    """Tests for LLMToolError exception."""

    def test_error_creation(self):
        """Test LLMToolError creation."""
        error = LLMToolError("Test error", recoverable=True)
        assert str(error) == "Test error"
        assert error.recoverable is True

    def test_error_with_duration(self):
        """Test LLMToolError with duration info."""
        error = LLMToolError(
            "Timeout after 30s during chat",
            recoverable=True,
            operation="chat",
            duration=30.0,
        )
        assert "30s" in str(error)
        assert error.operation == "chat"
        assert error.duration == 30.0

    def test_non_recoverable_error(self):
        """Test non-recoverable error."""
        error = LLMToolError("Fatal error", recoverable=False)
        assert error.recoverable is False

    def test_error_inheritance(self):
        """Test LLMToolError inherits from Exception."""
        error = LLMToolError("Test")
        assert isinstance(error, Exception)


class TestTimeoutBehavior:
    """Tests for timeout behavior in providers."""

    @pytest.fixture
    def timeout_config(self):
        """Create a short timeout config for testing."""
        return TimeoutConfig(
            connect_timeout=1.0,
            read_timeout=1.0,
            total_timeout=1.0,
            chat_timeout=1.0,
            stream_timeout=1.0,
        )

    @pytest.fixture
    def openai_provider(self):
        """Create OpenAIProvider instance."""
        return OpenAIProvider(
            api_key="test-key",
            model="gpt-4",
        )

    @pytest.fixture
    def anthropic_provider(self):
        """Create AnthropicProvider instance."""
        return AnthropicProvider(
            api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_complete_timeout_raises_llm_tool_error(self, openai_provider, timeout_config):
        """Test that complete() raises LLMToolError on timeout."""
        openai_provider.timeout_config = timeout_config

        # Mock client that takes too long
        async def slow_request(*args, **kwargs):
            await asyncio.sleep(10)  # Simulate slow request
            return MagicMock()

        with patch.object(openai_provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = slow_request
            mock_get_client.return_value = mock_client

            with pytest.raises(LLMToolError) as exc_info:
                await openai_provider.complete("Hello")

            assert "Timeout" in str(exc_info.value)
            assert exc_info.value.recoverable is True

    @pytest.mark.asyncio
    async def test_anthropic_complete_timeout(self, anthropic_provider, timeout_config):
        """Test Anthropic provider timeout behavior."""
        anthropic_provider.timeout_config = timeout_config

        async def slow_request(*args, **kwargs):
            await asyncio.sleep(10)
            return MagicMock()

        with patch.object(anthropic_provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = slow_request
            mock_get_client.return_value = mock_client

            with pytest.raises(LLMToolError) as exc_info:
                await anthropic_provider.complete("Hello")

            assert "Timeout" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_complete_success_within_timeout(self, openai_provider):
        """Test that complete() succeeds when response is fast."""
        config = TimeoutConfig(
            chat_timeout=10.0,
            total_timeout=10.0,
        )
        openai_provider.timeout_config = config

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Fast response!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(openai_provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            response = await openai_provider.complete("Hello")
            assert response.content == "Fast response!"

    @pytest.mark.asyncio
    async def test_operation_timeout_included_in_error(self, openai_provider):
        """Test that timeout error includes operation name."""
        config = TimeoutConfig(chat_timeout=5.0, total_timeout=5.0)
        openai_provider.timeout_config = config

        async def slow_request(*args, **kwargs):
            await asyncio.sleep(10)
            return MagicMock()

        with patch.object(openai_provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = slow_request
            mock_get_client.return_value = mock_client

            with pytest.raises(LLMToolError) as exc_info:
                await openai_provider.complete("Hello")

            assert exc_info.value.operation == "complete"
            assert exc_info.value.duration == 5.0


class TestProviderWithTimeoutConfig:
    """Tests for providers accepting TimeoutConfig."""

    def test_openai_accepts_timeout_config(self):
        """Test OpenAIProvider accepts TimeoutConfig."""
        config = TimeoutConfig(chat_timeout=60.0)
        provider = OpenAIProvider(
            api_key="key",
            timeout_config=config,
        )
        assert provider.timeout_config is config

    def test_anthropic_accepts_timeout_config(self):
        """Test AnthropicProvider accepts TimeoutConfig."""
        config = TimeoutConfig(chat_timeout=90.0)
        provider = AnthropicProvider(
            api_key="key",
            timeout_config=config,
        )
        assert provider.timeout_config is config

    def test_default_timeout_config_created(self):
        """Test that default TimeoutConfig is created if not provided."""
        provider = OpenAIProvider(api_key="key")
        assert provider.timeout_config is not None
        assert isinstance(provider.timeout_config, TimeoutConfig)
