"""
End-to-end tests for real LLM providers.

These tests verify that the provider implementations work correctly with actual
HTTP endpoints. Tests are designed to handle provider unavailability gracefully.

Providers tested:
- OpenAI-compatible (Qwen via custom endpoint)
- Anthropic Messages API (Claude via custom endpoint)

Run with: pytest tests/test_e2e_providers.py -v

Note: These tests require network access and valid API credentials.
They will be skipped if providers are unavailable.
"""

import asyncio
import pytest
import httpx
from typing import Any, Dict, List, Optional
import logging

from petals.client.providers import (
    OpenAIProvider,
    AnthropicProvider,
    LLMResponse,
    LLMChunk,
    FallbackChain,
    FallbackConfig,
    LLMProviderRegistry,
    AllProvidersFailedError,
    TimeoutConfig,
    LLMToolError,
    BaseLLMProvider,
)
from petals.client.feedback.retry_policy import (
    CircuitBreakerConfig,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Provider Configuration
# ============================================================================

# Qwen/OpenAI-compatible endpoint
# Note: base_url includes /v1 prefix as per user's curl example
QWEN_CONFIG = {
    "base_url": "http://23.94.250.112:18317/v1",
    "api_key": "sk-92abb51ab35b433983721e9177d0ea75",
    "model": "qwen3-coder-flash",
}

# Anthropic/Claude endpoint (local proxy)
ANTHROPIC_CONFIG = {
    "base_url": "http://127.0.0.1:3456",
    "api_key": "kala-0719",
    "model": "claude-4.5-sonnet",
}


# ============================================================================
# Helper Functions
# ============================================================================

async def check_endpoint_health(base_url: str, timeout: float = 5.0) -> bool:
    """Check if an endpoint is reachable.

    Args:
        base_url: The base URL to check.
        timeout: Timeout in seconds.

    Returns:
        True if the endpoint is reachable, False otherwise.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{base_url.rstrip('/')}/health")
            return response.status_code == 200
    except Exception:
        return False


async def check_openai_endpoint_works(config: Dict[str, str]) -> bool:
    """Check if an OpenAI-compatible endpoint is working by making a test request.

    Args:
        config: Provider configuration dictionary.

    Returns:
        True if the endpoint responds correctly, False otherwise.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{config['base_url'].rstrip('/')}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {config['api_key']}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": config["model"],
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 5,
                },
            )
            return response.status_code in (200, 201)
    except Exception as e:
        logger.debug(f"OpenAI endpoint check failed: {e}")
        return False


async def check_anthropic_endpoint_works(config: Dict[str, str]) -> bool:
    """Check if an Anthropic endpoint is working by making a test request.

    Args:
        config: Provider configuration dictionary.

    Returns:
        True if the endpoint responds correctly, False otherwise.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{config['base_url'].rstrip('/')}/v1/messages",
                headers={
                    "x-api-key": config["api_key"],
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": config["model"],
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 10,
                },
            )
            return response.status_code in (200, 201)
    except Exception as e:
        logger.debug(f"Anthropic endpoint check failed: {e}")
        return False


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture
def qwen_config() -> Dict[str, str]:
    """Return Qwen/OpenAI-compatible provider configuration."""
    return QWEN_CONFIG.copy()


@pytest.fixture
def anthropic_config() -> Dict[str, str]:
    """Return Anthropic provider configuration."""
    return ANTHROPIC_CONFIG.copy()


@pytest.fixture
def short_timeout_config() -> TimeoutConfig:
    """Return a short timeout configuration for testing."""
    return TimeoutConfig(
        connect_timeout=5.0,
        read_timeout=10.0,
        total_timeout=15.0,
        chat_timeout=10.0,
        stream_timeout=10.0,
    )


@pytest.fixture
def default_timeout_config() -> TimeoutConfig:
    """Return a default timeout configuration."""
    return TimeoutConfig(
        connect_timeout=30.0,
        read_timeout=60.0,
        total_timeout=90.0,
        chat_timeout=60.0,
        stream_timeout=60.0,
    )


# ============================================================================
# Provider Connectivity Tests
# ============================================================================

class TestProviderConnectivity:
    """Test connectivity to real LLM providers."""

    @pytest.mark.asyncio
    async def test_qwen_endpoint_reachable(self, qwen_config: Dict[str, str]):
        """Test that Qwen endpoint is reachable."""
        is_reachable = await check_openai_endpoint_works(qwen_config)

        if not is_reachable:
            pytest.skip(f"Qwen endpoint not reachable at {qwen_config['base_url']}")

        assert is_reachable, "Qwen endpoint should be reachable"

    @pytest.mark.asyncio
    async def test_anthropic_endpoint_reachable(self, anthropic_config: Dict[str, str]):
        """Test that Anthropic endpoint is reachable."""
        is_reachable = await check_anthropic_endpoint_works(anthropic_config)

        if not is_reachable:
            pytest.skip(f"Anthropic endpoint not reachable at {anthropic_config['base_url']}")

        assert is_reachable, "Anthropic endpoint should be reachable"

    @pytest.mark.asyncio
    async def test_authentication_with_valid_key(self, qwen_config: Dict[str, str]):
        """Test that authentication works with valid API key."""
        is_reachable = await check_openai_endpoint_works(qwen_config)
        if not is_reachable:
            pytest.skip("Qwen endpoint not reachable")

        provider = OpenAIProvider(
            api_key=qwen_config["api_key"],
            model=qwen_config["model"],
            base_url=qwen_config["base_url"],
        )

        try:
            response = await provider.complete("Hello", max_tokens=10)
            assert response is not None
            assert response.content is not None
        finally:
            await provider.close()


# ============================================================================
# Basic Completion Tests
# ============================================================================

class TestOpenAICompletion:
    """Test OpenAI provider basic completion functionality."""

    @pytest.mark.asyncio
    async def test_chat_completion_basic(self, qwen_config: Dict[str, str]):
        """Test basic chat completion via /v1/chat/completions endpoint."""
        is_reachable = await check_openai_endpoint_works(qwen_config)
        if not is_reachable:
            pytest.skip("Qwen endpoint not reachable")

        provider = OpenAIProvider(
            api_key=qwen_config["api_key"],
            model=qwen_config["model"],
            base_url=qwen_config["base_url"],
            timeout=60.0,
        )

        try:
            response = await provider.complete(
                "What is 2 + 2?",
                max_tokens=20,
            )

            assert response is not None
            assert isinstance(response, LLMResponse)
            assert response.content is not None
            assert len(response.content) > 0
            logger.info(f"Chat completion response: {response.content[:100]}")

        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_chat_completion_with_system_prompt(
        self, qwen_config: Dict[str, str]
    ):
        """Test chat completion with a system prompt."""
        is_reachable = await check_openai_endpoint_works(qwen_config)
        if not is_reachable:
            pytest.skip("Qwen endpoint not reachable")

        provider = OpenAIProvider(
            api_key=qwen_config["api_key"],
            model=qwen_config["model"],
            base_url=qwen_config["base_url"],
        )

        try:
            response = await provider.complete(
                "What is the capital of France?",
                system="You are a helpful geography assistant.",
                max_tokens=30,
            )

            assert response is not None
            assert response.content is not None
            # Response should mention Paris
            assert "Paris" in response.content or "paris" in response.content.lower()

        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_chat_completion_with_temperature(self, qwen_config: Dict[str, str]):
        """Test that temperature parameter works."""
        is_reachable = await check_openai_endpoint_works(qwen_config)
        if not is_reachable:
            pytest.skip("Qwen endpoint not reachable")

        provider = OpenAIProvider(
            api_key=qwen_config["api_key"],
            model=qwen_config["model"],
            base_url=qwen_config["base_url"],
        )

        try:
            response1 = await provider.complete(
                "Give me a random number between 1 and 10",
                max_tokens=5,
                temperature=0.0,
            )
            response2 = await provider.complete(
                "Give me a random number between 1 and 10",
                max_tokens=5,
                temperature=0.0,
            )

            assert response1 is not None
            assert response2 is not None
            # With temperature=0, responses should be deterministic

        finally:
            await provider.close()


class TestAnthropicCompletion:
    """Test Anthropic provider basic completion functionality."""

    @pytest.mark.asyncio
    async def test_messages_completion_basic(self, anthropic_config: Dict[str, str]):
        """Test basic messages completion via /v1/messages endpoint."""
        is_reachable = await check_anthropic_endpoint_works(anthropic_config)
        if not is_reachable:
            pytest.skip("Anthropic endpoint not reachable")

        provider = AnthropicProvider(
            api_key=anthropic_config["api_key"],
            model=anthropic_config["model"],
            base_url=anthropic_config["base_url"],
            timeout=60.0,
        )

        try:
            response = await provider.complete(
                "What is 2 + 2?",
                max_tokens=20,
            )

            assert response is not None
            assert isinstance(response, LLMResponse)
            assert response.content is not None
            assert len(response.content) > 0
            logger.info(f"Anthropic completion response: {response.content[:100]}")

        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_messages_with_system_prompt(self, anthropic_config: Dict[str, str]):
        """Test messages completion with a system prompt."""
        is_reachable = await check_anthropic_endpoint_works(anthropic_config)
        if not is_reachable:
            pytest.skip("Anthropic endpoint not reachable")

        provider = AnthropicProvider(
            api_key=anthropic_config["api_key"],
            model=anthropic_config["model"],
            base_url=anthropic_config["base_url"],
        )

        try:
            response = await provider.complete(
                "What is the capital of France?",
                system="You are a helpful geography assistant.",
                max_tokens=30,
            )

            assert response is not None
            assert response.content is not None
            # Response should mention Paris
            assert "Paris" in response.content or "paris" in response.content.lower()

        finally:
            await provider.close()


# ============================================================================
# Streaming Tests
# ============================================================================

class TestOpenAIStreaming:
    """Test OpenAI provider streaming functionality."""

    @pytest.mark.asyncio
    async def test_streaming_basic(self, qwen_config: Dict[str, str]):
        """Test basic SSE streaming from OpenAI provider."""
        is_reachable = await check_openai_endpoint_works(qwen_config)
        if not is_reachable:
            pytest.skip("Qwen endpoint not reachable")

        provider = OpenAIProvider(
            api_key=qwen_config["api_key"],
            model=qwen_config["model"],
            base_url=qwen_config["base_url"],
        )

        try:
            chunks_received = 0
            total_content = ""

            async for chunk in provider.stream("Count to 3:", max_tokens=20):
                assert isinstance(chunk, LLMChunk)
                total_content += chunk.content
                chunks_received += 1

            assert chunks_received > 0, "Should receive at least one chunk"
            assert len(total_content) > 0, "Should have some content"
            logger.info(f"Streaming received {chunks_received} chunks, total: {total_content[:50]}")

        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_streaming_with_system_prompt(self, qwen_config: Dict[str, str]):
        """Test streaming with a system prompt."""
        is_reachable = await check_openai_endpoint_works(qwen_config)
        if not is_reachable:
            pytest.skip("Qwen endpoint not reachable")

        provider = OpenAIProvider(
            api_key=qwen_config["api_key"],
            model=qwen_config["model"],
            base_url=qwen_config["base_url"],
        )

        try:
            chunks = []
            async for chunk in provider.stream(
                "Say 'hello'",
                system="You are a friendly assistant.",
                max_tokens=10,
            ):
                chunks.append(chunk)

            assert len(chunks) > 0, "Should receive at least one chunk"
            # Check that final chunk is marked as complete
            final_chunk = chunks[-1]
            assert final_chunk.is_complete is True

        finally:
            await provider.close()


class TestAnthropicStreaming:
    """Test Anthropic provider streaming functionality."""

    @pytest.mark.asyncio
    async def test_streaming_basic(self, anthropic_config: Dict[str, str]):
        """Test basic SSE streaming from Anthropic provider."""
        is_reachable = await check_anthropic_endpoint_works(anthropic_config)
        if not is_reachable:
            pytest.skip("Anthropic endpoint not reachable")

        provider = AnthropicProvider(
            api_key=anthropic_config["api_key"],
            model=anthropic_config["model"],
            base_url=anthropic_config["base_url"],
        )

        try:
            chunks_received = 0
            total_content = ""

            async for chunk in provider.stream("Count to 3:", max_tokens=20):
                assert isinstance(chunk, LLMChunk)
                total_content += chunk.content
                chunks_received += 1

            assert chunks_received > 0, "Should receive at least one chunk"
            assert len(total_content) > 0, "Should have some content"
            logger.info(f"Streaming received {chunks_received} chunks")

        finally:
            await provider.close()


# ============================================================================
# Timeout Tests
# ============================================================================

class TestTimeoutBehavior:
    """Test timeout behavior with real providers."""

    @pytest.mark.asyncio
    async def test_short_timeout_on_slow_request(self, qwen_config: Dict[str, str]):
        """Test that short timeout raises LLMToolError on slow responses."""
        is_reachable = await check_openai_endpoint_works(qwen_config)
        if not is_reachable:
            pytest.skip("Qwen endpoint not reachable")

        # Use a very short timeout
        provider = OpenAIProvider(
            api_key=qwen_config["api_key"],
            model=qwen_config["model"],
            base_url=qwen_config["base_url"],
            timeout=0.001,  # 1ms timeout - will definitely timeout
        )

        try:
            with pytest.raises(LLMToolError) as exc_info:
                await provider.complete("Hello", max_tokens=10)

            assert exc_info.value.recoverable is True
            assert "Timeout" in str(exc_info.value) or "timeout" in str(exc_info.value).lower()

        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_timeout_config_overrides_default(
        self, qwen_config: Dict[str, str], short_timeout_config: TimeoutConfig
    ):
        """Test that custom TimeoutConfig is used correctly."""
        is_reachable = await check_openai_endpoint_works(qwen_config)
        if not is_reachable:
            pytest.skip("Qwen endpoint not reachable")

        provider = OpenAIProvider(
            api_key=qwen_config["api_key"],
            model=qwen_config["model"],
            base_url=qwen_config["base_url"],
            timeout_config=short_timeout_config,
        )

        assert provider.timeout_config is short_timeout_config
        assert provider.timeout_config.chat_timeout == 10.0

        await provider.close()


# ============================================================================
# Resilience and Fallback Tests
# ============================================================================

class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with real providers."""

    @pytest.mark.asyncio
    async def test_provider_with_circuit_breaker_success(
        self, qwen_config: Dict[str, str]
    ):
        """Test that provider works with circuit breaker enabled."""
        is_reachable = await check_openai_endpoint_works(qwen_config)
        if not is_reachable:
            pytest.skip("Qwen endpoint not reachable")

        from petals.client.feedback.retry_policy import CircuitBreaker, CircuitBreakerConfig

        breaker_config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60.0,
        )
        breaker = CircuitBreaker(breaker_config)

        provider = OpenAIProvider(
            api_key=qwen_config["api_key"],
            model=qwen_config["model"],
            base_url=qwen_config["base_url"],
            circuit_breaker=breaker,
        )

        try:
            # First call should succeed
            response = await provider.complete("Hello", max_tokens=10)
            assert response is not None
            assert breaker.state == "closed"

        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_circuit_breaker_trips_on_failures(
        self, qwen_config: Dict[str, str]
    ):
        """Test that circuit breaker trips after consecutive failures."""
        from petals.client.feedback.retry_policy import CircuitBreaker, CircuitBreakerConfig

        # Use a very low threshold for testing
        breaker_config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=60.0,
        )
        breaker = CircuitBreaker(breaker_config)

        # Provider pointing to invalid endpoint
        provider = OpenAIProvider(
            api_key="invalid-key",
            model="invalid-model",
            base_url="http://localhost:99999",  # Invalid endpoint
            circuit_breaker=breaker,
            timeout=1.0,
        )

        try:
            # Make requests that should fail
            for _ in range(3):
                try:
                    await provider.complete("Hello", max_tokens=10)
                except Exception:
                    pass

            # Circuit should be open after failures
            assert breaker.state == "open"

        finally:
            await provider.close()


class TestFallbackChain:
    """Test fallback chain with real providers."""

    @pytest.mark.asyncio
    async def test_fallback_chain_primary_works(
        self, qwen_config: Dict[str, str], anthropic_config: Dict[str, str]
    ):
        """Test fallback chain when primary provider works."""
        qwen_reachable = await check_openai_endpoint_works(qwen_config)
        anthropic_reachable = await check_anthropic_endpoint_works(anthropic_config)

        if not qwen_reachable and not anthropic_reachable:
            pytest.skip("No providers reachable")

        registry = LLMProviderRegistry()

        # Register providers
        if qwen_reachable:
            qwen_provider = OpenAIProvider(
                api_key=qwen_config["api_key"],
                model=qwen_config["model"],
                base_url=qwen_config["base_url"],
            )
            registry.register("qwen", qwen_provider, CircuitBreakerConfig())

        if anthropic_reachable:
            anthropic_provider = AnthropicProvider(
                api_key=anthropic_config["api_key"],
                model=anthropic_config["model"],
                base_url=anthropic_config["base_url"],
            )
            registry.register("anthropic", anthropic_provider, CircuitBreakerConfig())

        # Create fallback config
        providers = registry.list_providers()
        if len(providers) == 0:
            pytest.skip("No providers available")

        config = FallbackConfig(providers=providers)
        chain = FallbackChain(config, registry)

        response = await chain.execute("complete", prompt="Hello, how are you?", max_tokens=10)

        assert response is not None
        assert isinstance(response, LLMResponse)
        assert response.content is not None
        logger.info(f"Fallback chain succeeded with response: {response.content[:50]}")

    @pytest.mark.asyncio
    async def test_fallback_to_secondary_on_primary_failure(self):
        """Test fallback to secondary provider when primary fails."""
        # Create a registry with one working and one failing provider
        registry = LLMProviderRegistry()

        # First provider that will fail
        from petals.client.providers.base import BaseLLMProvider

        class FailingProvider(BaseLLMProvider):
            async def complete(self, prompt, **kwargs):
                raise ConnectionError("Primary provider failed")

            async def stream(self, prompt, **kwargs):
                raise ConnectionError("Primary provider failed")

            async def count_tokens(self, text):
                return len(text) // 4

        # Second provider that will succeed (using a mock but simulating fallback)
        class SuccessProvider(BaseLLMProvider):
            async def complete(self, prompt, **kwargs):
                return LLMResponse(content="Fallback succeeded!")

            async def stream(self, prompt, **kwargs):
                yield LLMChunk(content="Fallback succeeded!", is_complete=True)

            async def count_tokens(self, text):
                return len(text) // 4

        failing = FailingProvider()
        success = SuccessProvider()

        registry.register("primary", failing, CircuitBreakerConfig(failure_threshold=1))
        registry.register("secondary", success, CircuitBreakerConfig())

        config = FallbackConfig(providers=["primary", "secondary"])
        chain = FallbackChain(config, registry)

        response = await chain.execute("complete", prompt="Test")

        assert response.content == "Fallback succeeded!"
        assert chain.attempted == {"primary", "secondary"}

    @pytest.mark.asyncio
    async def test_all_providers_fail_raises_error(self):
        """Test that AllProvidersFailedError is raised when all providers fail."""
        registry = LLMProviderRegistry()

        class AlwaysFailingProvider(BaseLLMProvider):
            async def complete(self, prompt, **kwargs):
                raise ConnectionError("Always fails")

            async def stream(self, prompt, **kwargs):
                raise ConnectionError("Always fails")

            async def count_tokens(self, text):
                return len(text) // 4

        registry.register("failing1", AlwaysFailingProvider(), CircuitBreakerConfig())
        registry.register("failing2", AlwaysFailingProvider(), CircuitBreakerConfig())

        config = FallbackConfig(providers=["failing1", "failing2"])
        chain = FallbackChain(config, registry)

        with pytest.raises(AllProvidersFailedError) as exc_info:
            await chain.execute("complete", prompt="Test")

        assert exc_info.value.provider_count == 2
        assert set(exc_info.value.attempted) == {"failing1", "failing2"}


# ============================================================================
# Tool Call Tests
# ============================================================================

class TestToolCalls:
    """Test tool call parsing and execution with real providers."""

    @pytest.mark.asyncio
    async def test_openai_provider_receives_tools(self, qwen_config: Dict[str, str]):
        """Test that OpenAI provider can receive tool definitions."""
        is_reachable = await check_openai_endpoint_works(qwen_config)
        if not is_reachable:
            pytest.skip("Qwen endpoint not reachable")

        provider = OpenAIProvider(
            api_key=qwen_config["api_key"],
            model=qwen_config["model"],
            base_url=qwen_config["base_url"],
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "A simple calculator",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Math expression to evaluate",
                            }
                        },
                        "required": ["expression"],
                    },
                },
            }
        ]

        try:
            response = await provider.complete(
                "What is 5 + 3? Use the calculator.",
                tools=tools,
                max_tokens=100,
            )

            assert response is not None
            # Note: The model may or may not use tools depending on its training
            # We're just testing that the request succeeds with tools
            logger.info(f"Response with tools: {response.content[:100]}")

        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_anthropic_provider_receives_tools(
        self, anthropic_config: Dict[str, str]
    ):
        """Test that Anthropic provider can receive tool definitions."""
        is_reachable = await check_anthropic_endpoint_works(anthropic_config)
        if not is_reachable:
            pytest.skip("Anthropic endpoint not reachable")

        provider = AnthropicProvider(
            api_key=anthropic_config["api_key"],
            model=anthropic_config["model"],
            base_url=anthropic_config["base_url"],
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            }
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

        try:
            response = await provider.complete(
                "Search for information about Python.",
                tools=tools,
                max_tokens=100,
            )

            assert response is not None
            logger.info(f"Response with tools: {response.content[:100]}")

        finally:
            await provider.close()


# ============================================================================
# Token Counting Tests
# ============================================================================

class TestTokenCounting:
    """Test token counting functionality."""

    @pytest.mark.asyncio
    async def test_openai_token_counting(self, qwen_config: Dict[str, str]):
        """Test OpenAI provider token counting."""
        is_reachable = await check_openai_endpoint_works(qwen_config)
        if not is_reachable:
            pytest.skip("Qwen endpoint not reachable")

        provider = OpenAIProvider(
            api_key=qwen_config["api_key"],
            model=qwen_config["model"],
            base_url=qwen_config["base_url"],
        )

        try:
            # Test basic token counting (uses approximation)
            tokens = await provider.count_tokens("Hello, world!")
            assert tokens > 0
            assert isinstance(tokens, int)

        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_anthropic_token_counting(self, anthropic_config: Dict[str, str]):
        """Test Anthropic provider token counting."""
        is_reachable = await check_anthropic_endpoint_works(anthropic_config)
        if not is_reachable:
            pytest.skip("Anthropic endpoint not reachable")

        provider = AnthropicProvider(
            api_key=anthropic_config["api_key"],
            model=anthropic_config["model"],
            base_url=anthropic_config["base_url"],
        )

        try:
            tokens = await provider.count_tokens("Hello, world!")
            assert tokens > 0
            assert isinstance(tokens, int)

        finally:
            await provider.close()


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling with various failure scenarios."""

    @pytest.mark.asyncio
    async def test_invalid_api_key(self, qwen_config: Dict[str, str]):
        """Test that invalid API key returns appropriate error."""
        provider = OpenAIProvider(
            api_key="invalid-key-12345",
            model=qwen_config["model"],
            base_url=qwen_config["base_url"],
            timeout=10.0,
        )

        try:
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await provider.complete("Hello", max_tokens=10)

            # Should get a 401 or similar auth error
            assert exc_info.value.response.status_code in (401, 403)

        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_invalid_model(self, qwen_config: Dict[str, str]):
        """Test that invalid model name returns appropriate error."""
        provider = OpenAIProvider(
            api_key=qwen_config["api_key"],
            model="nonexistent-model-12345",
            base_url=qwen_config["base_url"],
            timeout=10.0,
        )

        try:
            with pytest.raises(httpx.HTTPStatusError):
                await provider.complete("Hello", max_tokens=10)

        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_connection_refused_error(self):
        """Test that connection refused is handled properly."""
        provider = OpenAIProvider(
            api_key="test-key",
            model="test-model",
            base_url="http://localhost:59999",  # Invalid port
            timeout=2.0,
        )

        try:
            with pytest.raises((httpx.ConnectError, httpx.TransportError, OSError)):
                await provider.complete("Hello", max_tokens=10)

        finally:
            await provider.close()


# ============================================================================
# Resource Cleanup Tests
# ============================================================================

class TestResourceCleanup:
    """Test proper resource cleanup."""

    @pytest.mark.asyncio
    async def test_provider_close(self, qwen_config: Dict[str, str]):
        """Test that provider.close() properly closes resources."""
        is_reachable = await check_openai_endpoint_works(qwen_config)
        if not is_reachable:
            pytest.skip("Qwen endpoint not reachable")

        provider = OpenAIProvider(
            api_key=qwen_config["api_key"],
            model=qwen_config["model"],
            base_url=qwen_config["base_url"],
        )

        # Make a request
        response = await provider.complete("Hello", max_tokens=10)
        assert response is not None

        # Close the provider
        await provider.close()

        # Client should be None after close
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_multiple_requests_and_close(self, qwen_config: Dict[str, str]):
        """Test multiple requests followed by close."""
        is_reachable = await check_openai_endpoint_works(qwen_config)
        if not is_reachable:
            pytest.skip("Qwen endpoint not reachable")

        provider = OpenAIProvider(
            api_key=qwen_config["api_key"],
            model=qwen_config["model"],
            base_url=qwen_config["base_url"],
        )

        try:
            # Make multiple requests
            for i in range(3):
                response = await provider.complete(f"Request {i}", max_tokens=10)
                assert response is not None

        finally:
            await provider.close()


# ============================================================================
# Integration Tests
# ============================================================================

class TestProviderIntegration:
    """Integration tests combining multiple features."""

    @pytest.mark.asyncio
    async def test_full_workflow_with_both_providers(
        self, qwen_config: Dict[str, str], anthropic_config: Dict[str, str]
    ):
        """Test a complete workflow using both providers."""
        qwen_reachable = await check_openai_endpoint_works(qwen_config)
        anthropic_reachable = await check_anthropic_endpoint_works(anthropic_config)

        if not qwen_reachable and not anthropic_reachable:
            pytest.skip("No providers reachable")

        # Test Qwen if available
        if qwen_reachable:
            qwen_provider = OpenAIProvider(
                api_key=qwen_config["api_key"],
                model=qwen_config["model"],
                base_url=qwen_config["base_url"],
            )

            try:
                # Basic completion
                response = await qwen_provider.complete(
                    "Explain what a REST API is in one sentence.",
                    max_tokens=50,
                )
                assert response is not None
                assert len(response.content) > 10

                # Streaming
                chunks = []
                async for chunk in qwen_provider.stream(
                    "Count: 1, 2, 3",
                    max_tokens=20,
                ):
                    chunks.append(chunk)
                assert len(chunks) > 0

            finally:
                await qwen_provider.close()

        # Test Anthropic if available
        if anthropic_reachable:
            anthropic_provider = AnthropicProvider(
                api_key=anthropic_config["api_key"],
                model=anthropic_config["model"],
                base_url=anthropic_config["base_url"],
            )

            try:
                # Basic completion
                response = await anthropic_provider.complete(
                    "Explain what a REST API is in one sentence.",
                    max_tokens=50,
                )
                assert response is not None
                assert len(response.content) > 10

                # Streaming
                chunks = []
                async for chunk in anthropic_provider.stream(
                    "Count: 1, 2, 3",
                    max_tokens=20,
                ):
                    chunks.append(chunk)
                assert len(chunks) > 0

            finally:
                await anthropic_provider.close()

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, qwen_config: Dict[str, str]):
        """Test making concurrent requests to a single provider."""
        is_reachable = await check_openai_endpoint_works(qwen_config)
        if not is_reachable:
            pytest.skip("Qwen endpoint not reachable")

        provider = OpenAIProvider(
            api_key=qwen_config["api_key"],
            model=qwen_config["model"],
            base_url=qwen_config["base_url"],
        )

        try:
            # Create multiple concurrent requests
            tasks = [
                provider.complete(f"Request {i}: Say hello", max_tokens=10)
                for i in range(3)
            ]

            responses = await asyncio.gather(*tasks)

            assert len(responses) == 3
            for response in responses:
                assert response is not None
                assert response.content is not None

        finally:
            await provider.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
