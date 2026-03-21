"""Tests for LLM providers."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from petals.client.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    LLMChunk,
)
from petals.client.providers.openai import OpenAIProvider
from petals.client.providers.anthropic import AnthropicProvider
from petals.client.providers import create_provider


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_llm_response_creation(self):
        """Test basic LLMResponse creation."""
        response = LLMResponse(content="Hello, world!")
        assert response.content == "Hello, world!"
        assert response.tool_calls is None
        assert response.usage is None

    def test_llm_response_with_tool_calls(self):
        """Test LLMResponse with tool calls."""
        tool_calls = [{"name": "search", "arguments": {"query": "test"}}]
        response = LLMResponse(
            content="I'll search for that.",
            tool_calls=tool_calls,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            model="gpt-4",
            finish_reason="tool_calls",
        )
        assert response.content == "I'll search for that."
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "search"
        assert response.usage["total_tokens"] == 15


class TestLLMChunk:
    """Tests for LLMChunk dataclass."""

    def test_llm_chunk_creation(self):
        """Test basic LLMChunk creation."""
        chunk = LLMChunk(content="Hello")
        assert chunk.content == "Hello"
        assert chunk.tool_names is None
        assert chunk.is_complete is False

    def test_llm_chunk_with_tools(self):
        """Test LLMChunk with detected tool names."""
        chunk = LLMChunk(
            content="",
            tool_names=["search", "calculator"],
            is_complete=True,
        )
        assert "search" in chunk.tool_names
        assert "calculator" in chunk.tool_names
        assert chunk.is_complete is True


class TestBaseLLMProvider:
    """Tests for BaseLLMProvider abstract class."""

    def test_base_provider_instantiation_fails(self):
        """Test that BaseLLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLLMProvider()

    def test_base_provider_supports_tools_default(self):
        """Test default supports_tools returns True."""

        class ConcreteProvider(BaseLLMProvider):
            async def complete(self, prompt, **kwargs):
                pass

            async def stream(self, prompt, **kwargs):
                pass

            async def count_tokens(self, text):
                return 0

        provider = ConcreteProvider()
        assert provider.supports_tools() is True


class TestOpenAIProvider:
    """Tests for OpenAIProvider."""

    @pytest.fixture
    def provider(self):
        """Create OpenAIProvider instance."""
        return OpenAIProvider(
            api_key="test-api-key",
            model="gpt-4",
            base_url="https://api.openai.com/v1",
        )

    def test_provider_initialization(self, provider):
        """Test OpenAIProvider initialization."""
        assert provider.api_key == "test-api-key"
        assert provider.model == "gpt-4"
        assert provider.base_url == "https://api.openai.com/v1"
        assert provider.timeout == 60.0
        assert provider.max_retries == 3

    def test_provider_initialization_with_custom_settings(self):
        """Test OpenAIProvider with custom settings."""
        provider = OpenAIProvider(
            api_key="key",
            model="gpt-3.5-turbo",
            base_url="https://custom.api.com/v1",
            timeout=120.0,
            max_retries=5,
        )
        assert provider.timeout == 120.0
        assert provider.max_retries == 5

    @pytest.mark.asyncio
    async def test_complete_basic(self, provider):
        """Test basic completion request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello, user!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "model": "gpt-4",
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            response = await provider.complete("Hello")
            assert response.content == "Hello, user!"
            assert response.finish_reason == "stop"
            assert response.usage["total_tokens"] == 8

    @pytest.mark.asyncio
    async def test_complete_with_system_prompt(self, provider):
        """Test completion with system prompt."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
            "model": "gpt-4",
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            response = await provider.complete("Hello", system="You are helpful.")
            assert response.content == "Response"

            # Verify system prompt was included in messages
            call_args = mock_client.post.call_args
            request_data = call_args.kwargs.get("json", call_args[1].get("json"))
            assert len(request_data["messages"]) == 2
            assert request_data["messages"][0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_complete_with_tools(self, provider):
        """Test completion with tools."""
        tools = [
            {"type": "function", "function": {"name": "search", "description": "Search"}}
        ]

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Searching...",
                        "tool_calls": [
                            {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": '{"query": "test"}'}}
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            "model": "gpt-4",
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            response = await provider.complete("Search for something", tools=tools)
            assert response.content == "Searching..."
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0]["function"]["name"] == "search"

            # Verify tools were included in request
            call_args = mock_client.post.call_args
            request_data = call_args.kwargs.get("json", call_args[1].get("json"))
            assert "tools" in request_data
            assert request_data["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_stream_basic(self, provider):
        """Test basic streaming."""
        chunks_data = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            "data: [DONE]",
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_lines():
            for chunk in chunks_data:
                yield chunk

        mock_response.aiter_lines = mock_aiter_lines

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.stream = MagicMock(return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()))
            mock_get_client.return_value = mock_client

            chunks = []
            async for chunk in provider.stream("Hello"):
                chunks.append(chunk)

            assert len(chunks) == 3
            assert chunks[0].content == "Hello"
            assert chunks[1].content == " world"
            assert chunks[2].is_complete is True

    @pytest.mark.asyncio
    async def test_stream_with_tool_calls(self, provider):
        """Test streaming with tool call detection."""
        chunks_data = [
            'data: {"choices":[{"delta":{"content":"Let me search"}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"function":{"name":"search"}}]}}]}',
            "data: [DONE]",
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_lines():
            for chunk in chunks_data:
                yield chunk

        mock_response.aiter_lines = mock_aiter_lines

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.stream = MagicMock(return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()))
            mock_get_client.return_value = mock_client

            chunks = []
            async for chunk in provider.stream("Search"):
                chunks.append(chunk)

            # Check tool names were detected
            final_chunk = chunks[-1]
            assert "search" in final_chunk.tool_names


class TestAnthropicProvider:
    """Tests for AnthropicProvider."""

    @pytest.fixture
    def provider(self):
        """Create AnthropicProvider instance."""
        return AnthropicProvider(
            api_key="test-api-key",
            model="claude-3-5-sonnet-20241022",
            base_url="https://api.anthropic.com",
        )

    def test_provider_initialization(self, provider):
        """Test AnthropicProvider initialization."""
        assert provider.api_key == "test-api-key"
        assert provider.model == "claude-3-5-sonnet-20241022"
        assert provider.base_url == "https://api.anthropic.com"

    def test_provider_default_model(self):
        """Test AnthropicProvider default model."""
        provider = AnthropicProvider(api_key="key")
        assert "claude" in provider.model

    @pytest.mark.asyncio
    async def test_complete_basic(self, provider):
        """Test basic completion request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Hello, user!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            response = await provider.complete("Hello")
            assert response.content == "Hello, user!"
            assert response.finish_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_complete_with_system(self, provider):
        """Test completion with system prompt."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Response"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 20, "output_tokens": 5},
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            response = await provider.complete("Hello", system="You are helpful.")
            assert response.content == "Response"


class TestProviderFactory:
    """Tests for create_provider factory function."""

    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        provider = create_provider("openai", api_key="key", model="gpt-4")
        assert isinstance(provider, OpenAIProvider)
        assert provider.model == "gpt-4"

    def test_create_anthropic_provider(self):
        """Test creating Anthropic provider."""
        provider = create_provider("anthropic", api_key="key")
        assert isinstance(provider, AnthropicProvider)
        assert "claude" in provider.model

    def test_create_provider_case_insensitive(self):
        """Test provider creation is case insensitive."""
        provider1 = create_provider("OPENAI", api_key="key")
        provider2 = create_provider("OpenAI", api_key="key")
        assert isinstance(provider1, OpenAIProvider)
        assert isinstance(provider2, OpenAIProvider)

    def test_create_unknown_provider_raises(self):
        """Test unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider type"):
            create_provider("unknown", api_key="key")


class TestProviderIntegration:
    """Integration tests for providers."""

    @pytest.mark.asyncio
    async def test_openai_provider_error_handling(self):
        """Test OpenAI provider handles HTTP errors."""
        provider = OpenAIProvider(api_key="key")

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            error_response = MagicMock()
            error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Error", request=MagicMock(), response=MagicMock(status_code=401)
            )
            mock_client.post = AsyncMock(return_value=error_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                await provider.complete("Hello")
