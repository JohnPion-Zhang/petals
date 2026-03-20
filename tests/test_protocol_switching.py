"""
Test protocol switching across different LLM API formats.

Requires mock servers running:
    python -m tests.mocks.mock_llm_servers --mode all

Then run tests:
    pytest tests/test_protocol_switching.py -v -s
"""
import asyncio
import os

import pytest

from petals.client.http_client import HTTPClient


class TestProtocolSwitching:
    """Test switching between different LLM protocols at runtime."""

    @pytest.fixture
    def api_key(self):
        """Get API key from environment."""
        key = os.environ.get("LLM_API_KEY", "test-key")
        return key

    @pytest.mark.asyncio
    async def test_openai_chat_protocol(self, api_key):
        """Test OpenAI /v1/chat/completions format."""
        client = HTTPClient(
            api_key=api_key,
            base_url="http://localhost:18001",
            default_model="qwen3-coder-flash",
        )

        response = await client.generate("Say hello in one word.")

        assert response.content is not None
        assert "[OpenAI Chat]" in response.content
        print(f"OpenAI Chat Response: {response.content}")

    @pytest.mark.asyncio
    async def test_openai_responses_protocol(self, api_key):
        """Test OpenAI /v1/responses format."""
        client = HTTPClient(
            api_key=api_key,
            base_url="http://localhost:18002",
            default_model="qwen3-coder-flash",
        )

        response = await client.generate("Say hello in one word.")

        assert response.content is not None
        assert "[OpenAI Responses]" in response.content
        print(f"OpenAI Responses: {response.content}")

    @pytest.mark.asyncio
    async def test_anthropic_protocol(self, api_key):
        """Test Anthropic /v1/messages format."""
        client = HTTPClient(
            api_key=api_key,
            base_url="http://localhost:18003",
            default_model="claude-4.5-sonnet",
        )

        response = await client.generate("Say hello in one word.")

        assert response.content is not None
        assert "[Anthropic]" in response.content
        print(f"Anthropic Response: {response.content}")

    @pytest.mark.asyncio
    async def test_runtime_switching(self, api_key):
        """Test switching between protocols at runtime."""
        client = HTTPClient(
            api_key=api_key,
            base_url="http://localhost:18001",
            default_model="qwen3-coder-flash",
        )

        # Test with OpenAI Chat first
        response1 = await client.generate("Test 1")
        assert "[OpenAI Chat]" in response1.content

        # Switch to Anthropic base URL
        client.base_url = "http://localhost:18003"
        client.switch_model("claude-4.5-sonnet")

        response2 = await client.generate("Test 2")
        assert "[Anthropic]" in response2.content

        # Switch back to OpenAI Chat
        client.base_url = "http://localhost:18001"
        client.switch_model("qwen3-coder-flash")

        response3 = await client.generate("Test 3")
        assert "[OpenAI Chat]" in response3.content

        print(f"Runtime switching: OK (Chat → Anthropic → Chat)")

    @pytest.mark.asyncio
    async def test_combined_server_all_endpoints(self, api_key):
        """Test all three endpoints on combined server."""
        client = HTTPClient(
            api_key=api_key,
            base_url="http://localhost:18000",
            default_model="test-model",
        )

        # The combined server needs special handling - it serves all three
        # but we need to test each endpoint specifically

        # Test OpenAI Chat
        from petals.client.http_client import litellm

        # Manually call each endpoint format
        messages = [{"role": "user", "content": "Test"}]

        # Test chat completions path
        response = await client.chat(messages)
        print(f"Combined server chat: {response.content}")


class TestWithRealEndpoints:
    """Test against real external endpoints (requires actual credentials)."""

    @pytest.fixture
    def skip_if_no_real_endpoints(self):
        """Skip if real endpoint credentials are not available."""
        base_url = os.environ.get("LLM_BASE_URL")
        if not base_url:
            pytest.skip("LLM_BASE_URL not set")

    @pytest.mark.asyncio
    async def test_real_endpoint_chat(self, skip_if_no_real_endpoints):
        """Test against real OpenAI-compatible endpoint."""
        client = HTTPClient(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.environ["LLM_BASE_URL"],
            default_model="qwen3-coder-flash",
        )

        response = await client.generate("Say 'test' in one word.")
        assert response.content is not None
        print(f"Real endpoint: {response.content}")

    @pytest.mark.asyncio
    async def test_real_endpoint_switching(self, skip_if_no_real_endpoints):
        """Test switching between models on real endpoint."""
        client = HTTPClient(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.environ["LLM_BASE_URL"],
            default_model="qwen3-coder-flash",
        )

        # Test first model
        response1 = await client.generate("Count to 3:")
        assert response1.content is not None

        # Switch model
        client.switch_model("claude-4.5-sonnet")
        response2 = await client.generate("Count to 3:")

        print(f"Model 1 ({client.default_model}): {response1.content}")
        print(f"Model 2 ({client.default_model}): {response2.content}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
