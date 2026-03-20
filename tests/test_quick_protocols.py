#!/usr/bin/env python3
"""
Quick test to verify all three mock LLM servers work correctly.

Run mock servers first:
    python -m tests.mocks.mock_llm_servers --mode all

Then run this script:
    python tests/test_mock_servers_quick.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from petals.client.http_client import HTTPClient


async def test_openai_chat():
    """Test OpenAI /v1/chat/completions."""
    print("\n=== Testing OpenAI /v1/chat/completions ===")
    client = HTTPClient(
        api_key="test-key",
        base_url="http://localhost:18001",
        default_model="qwen3-coder-flash",
    )

    response = await client.generate("Say 'hello' in one word.")
    print(f"Response: {response.content}")
    assert "[OpenAI Chat]" in response.content, "Expected OpenAI Chat prefix"
    print("✓ OpenAI Chat works!")


async def test_openai_responses():
    """Test OpenAI /v1/responses."""
    print("\n=== Testing OpenAI /v1/responses ===")
    client = HTTPClient(
        api_key="test-key",
        base_url="http://localhost:18002",
        default_model="qwen3-coder-flash",
    )

    response = await client.generate("Say 'hello' in one word.")
    print(f"Response: {response.content}")
    assert "[OpenAI Responses]" in response.content, "Expected OpenAI Responses prefix"
    print("✓ OpenAI Responses works!")


async def test_anthropic():
    """Test Anthropic /v1/messages."""
    print("\n=== Testing Anthropic /v1/messages ===")
    client = HTTPClient(
        api_key="kala-0719",
        base_url="http://localhost:18003",
        default_model="claude-4.5-sonnet",
    )

    response = await client.generate("Say 'hello' in one word.")
    print(f"Response: {response.content}")
    assert "[Anthropic]" in response.content, "Expected Anthropic prefix"
    print("✓ Anthropic works!")


async def test_runtime_switching():
    """Test switching between protocols at runtime."""
    print("\n=== Testing Runtime Protocol Switching ===")

    # Start with OpenAI Chat
    client = HTTPClient(
        api_key="test-key",
        base_url="http://localhost:18001",
        default_model="qwen3-coder-flash",
    )

    r1 = await client.generate("Test 1")
    print(f"1. OpenAI Chat: {r1.content[:50]}...")

    # Switch to Anthropic
    client.base_url = "http://localhost:18003"
    client.switch_model("claude-4.5-sonnet")
    r2 = await client.generate("Test 2")
    print(f"2. Anthropic: {r2.content[:50]}...")

    # Switch to OpenAI Responses
    client.base_url = "http://localhost:18002"
    client.switch_model("qwen3-coder-flash")
    r3 = await client.generate("Test 3")
    print(f"3. OpenAI Responses: {r3.content[:50]}...")

    # Switch back to OpenAI Chat
    client.base_url = "http://localhost:18001"
    client.switch_model("qwen3-coder-flash")
    r4 = await client.generate("Test 4")
    print(f"4. OpenAI Chat: {r4.content[:50]}...")

    assert "[OpenAI Chat]" in r1.content
    assert "[Anthropic]" in r2.content
    assert "[OpenAI Responses]" in r3.content
    assert "[OpenAI Chat]" in r4.content

    print("✓ Runtime switching works!")


async def main():
    print("Testing mock LLM servers...")

    try:
        await test_openai_chat()
        await test_openai_responses()
        await test_anthropic()
        await test_runtime_switching()

        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("Protocol switching is working correctly.")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        print("\nMake sure mock servers are running:")
        print("  python -m tests.mocks.mock_llm_servers --mode all")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
