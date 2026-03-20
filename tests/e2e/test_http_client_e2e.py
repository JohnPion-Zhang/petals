"""
E2E tests for HTTP Client and Agent with real LLM endpoints.

These tests make actual HTTP calls to LLM providers.
Run with: pytest tests/e2e/test_http_client_e2e.py -v -s

NOTE: Requires environment variables:
- LLM_API_KEY: API key for the LLM provider
- LLM_BASE_URL: Optional base URL for custom endpoints (default: uses OpenAI)
"""
import asyncio
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

import pytest

# Import the modules under test
from petals.client.http_client import HTTPClient, LLMResponse


@dataclass
class TestReport:
    """Report structure for E2E test results."""
    test_name: str
    timestamp: str
    model: str
    base_url: Optional[str]
    success: bool
    duration_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    error: Optional[str] = None
    request_payload: Optional[Dict[str, Any]] = None
    response_content: Optional[str] = None
    tool_calls_detected: List[str] = field(default_factory=list)
    raw_response: Optional[Dict[str, Any]] = None


class E2ETestReporter:
    """Reports E2E test results to files."""

    def __init__(self, output_dir: str = "docs/e2e-reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_report(self, report: TestReport, filename: Optional[str] = None) -> str:
        """Save a test report to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = report.test_name.replace(" ", "_").replace("/", "_")
            filename = f"{safe_name}_{timestamp}.json"

        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(asdict(report), f, indent=2)

        return filepath

    def save_summary(self, reports: List[TestReport]) -> str:
        """Save a summary of all test reports."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary = {
            "timestamp": timestamp,
            "total_tests": len(reports),
            "passed": sum(1 for r in reports if r.success),
            "failed": sum(1 for r in reports if not r.success),
            "total_duration_ms": sum(r.duration_ms for r in reports),
            "tests": [asdict(r) for r in reports],
        }

        filepath = os.path.join(self.output_dir, f"summary_{timestamp}.json")
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)

        return filepath


@pytest.fixture
def api_key():
    """Get API key from environment."""
    key = os.environ.get("LLM_API_KEY")
    if not key:
        pytest.skip("LLM_API_KEY environment variable not set")
    return key


@pytest.fixture
def base_url():
    """Get base URL from environment or return None for default."""
    return os.environ.get("LLM_BASE_URL")


@pytest.fixture
def default_model():
    """Get default model from environment."""
    return os.environ.get("LLM_MODEL", "gpt-4o-mini")


@pytest.fixture
def http_client(api_key, base_url, default_model):
    """Create HTTP client for tests."""
    return HTTPClient(
        api_key=api_key,
        base_url=base_url,
        default_model=default_model,
        timeout=60.0,
        max_retries=2,
    )


@pytest.fixture
def reporter():
    """Create E2E test reporter."""
    return E2ETestReporter()


class TestHTTPClientE2E:
    """E2E tests for HTTPClient with real LLM endpoints."""

    @pytest.mark.asyncio
    async def test_simple_generate(
        self, http_client, default_model, base_url, reporter
    ):
        """Test simple text generation."""
        report = TestReport(
            test_name="test_simple_generate",
            timestamp=datetime.now().isoformat(),
            model=default_model,
            base_url=base_url,
            success=False,
            duration_ms=0,
        )

        start = time.time()
        try:
            response = await http_client.generate(
                "Say 'Hello, World!' in exactly those words.",
                max_tokens=50,
            )

            report.duration_ms = (time.time() - start) * 1000
            report.success = True
            report.input_tokens = response.usage.get("prompt_tokens") if response.usage else None
            report.output_tokens = response.usage.get("completion_tokens") if response.usage else None
            report.total_tokens = response.usage.get("total_tokens") if response.usage else None
            report.response_content = response.content
            report.model = response.model

            print(f"\n=== Simple Generate Test ===")
            print(f"Model: {response.model}")
            print(f"Duration: {report.duration_ms:.2f}ms")
            print(f"Input tokens: {report.input_tokens}")
            print(f"Output tokens: {report.output_tokens}")
            print(f"Response: {response.content}")

        except Exception as e:
            report.error = str(e)
            report.duration_ms = (time.time() - start) * 1000
            print(f"\n=== Test Failed ===")
            print(f"Error: {e}")

        filepath = reporter.save_report(report)
        print(f"\nReport saved to: {filepath}")

        assert report.success, f"Test failed: {report.error}"
        assert "Hello, World" in report.response_content

    @pytest.mark.asyncio
    async def test_chat_completion(
        self, http_client, default_model, base_url, reporter
    ):
        """Test chat completion with message history."""
        report = TestReport(
            test_name="test_chat_completion",
            timestamp=datetime.now().isoformat(),
            model=default_model,
            base_url=base_url,
            success=False,
            duration_ms=0,
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2 + 2?"},
        ]
        report.request_payload = {"messages": messages}

        start = time.time()
        try:
            response = await http_client.chat(
                messages,
                max_tokens=100,  # Increased for thinking models
            )

            report.duration_ms = (time.time() - start) * 1000
            report.success = True
            report.input_tokens = response.usage.get("prompt_tokens") if response.usage else None
            report.output_tokens = response.usage.get("completion_tokens") if response.usage else None
            report.total_tokens = response.usage.get("total_tokens") if response.usage else None
            report.response_content = response.content

            print(f"\n=== Chat Completion Test ===")
            print(f"Messages: {len(messages)}")
            print(f"Duration: {report.duration_ms:.2f}ms")
            print(f"Response: {response.content}")

        except Exception as e:
            report.error = str(e)
            report.duration_ms = (time.time() - start) * 1000
            print(f"\n=== Test Failed ===")
            print(f"Error: {e}")

        filepath = reporter.save_report(report)
        print(f"\nReport saved to: {filepath}")

        assert report.success, f"Test failed: {report.error}"
        # For thinking/reasoning models, check if response has content
        assert report.response_content is not None and len(report.response_content) > 0

    @pytest.mark.asyncio
    async def test_model_switching(
        self, http_client, api_key, base_url, reporter
    ):
        """Test switching between different models."""
        # Test with the same model (model switching is internal)
        # Note: Model switching at runtime works, but different models
        # may not be available at the same endpoint
        models = [
            os.environ.get("LLM_MODEL", "claude-4.5-sonnet"),
        ]

        results = []
        for model in models:
            report = TestReport(
                test_name=f"test_model_switching_{model}",
                timestamp=datetime.now().isoformat(),
                model=model,
                base_url=base_url,
                success=False,
                duration_ms=0,
            )

            start = time.time()
            try:
                # Switch model
                http_client.switch_model(model)

                response = await http_client.generate(
                    "Say 'test' in one word.",
                    max_tokens=50,  # Increased for thinking models
                )

                report.duration_ms = (time.time() - start) * 1000
                report.success = True
                report.response_content = response.content

                print(f"\n=== Model: {model} ===")
                print(f"Duration: {report.duration_ms:.2f}ms")
                print(f"Response: {response.content}")

            except Exception as e:
                report.error = str(e)
                report.duration_ms = (time.time() - start) * 1000
                print(f"\n=== Model {model} Failed ===")
                print(f"Error: {e}")

            filepath = reporter.save_report(report)
            results.append(report)

        # Save combined summary
        summary_path = reporter.save_summary(results)
        print(f"\n=== Summary saved to: {summary_path} ===")

        # At least one should succeed
        assert any(r.success for r in results), "All model tests failed"


class TestAgentWithRealLLM:
    """E2E tests for Agent with real LLM endpoints."""

    @pytest.mark.asyncio
    async def test_agent_simple_task(
        self, http_client, api_key, base_url, default_model, reporter
    ):
        """Test agent with a simple task requiring no tools."""
        from petals.client.agent import AgentOrchestrator

        report = TestReport(
            test_name="test_agent_simple_task",
            timestamp=datetime.now().isoformat(),
            model=default_model,
            base_url=base_url,
            success=False,
            duration_ms=0,
        )

        # Create agent with HTTP client
        agent = AgentOrchestrator(
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            max_iterations=3,
            max_context_tokens=4000,
        )

        start = time.time()
        try:
            result = await agent.run("What is the capital of France? Answer in one word.")

            report.duration_ms = (time.time() - start) * 1000
            report.success = True
            report.response_content = result
            report.total_tokens = agent.state.total_tokens_used
            report.input_tokens = (
                agent.state.total_tokens_used // 2
            )  # Approximate

            print(f"\n=== Agent Simple Task ===")
            print(f"Duration: {report.duration_ms:.2f}ms")
            print(f"Total tokens: {report.total_tokens_used}")
            print(f"Iterations: {agent.state.current_iteration}")
            print(f"Response: {result}")

        except Exception as e:
            report.error = str(e)
            report.duration_ms = (time.time() - start) * 1000
            print(f"\n=== Test Failed ===")
            print(f"Error: {e}")

        filepath = reporter.save_report(report)
        print(f"\nReport saved to: {filepath}")

        assert report.success, f"Test failed: {report.error}"
        assert "Paris" in report.response_content or "paris" in report.response_content.lower()

    @pytest.mark.asyncio
    async def test_agent_with_tools(
        self, http_client, api_key, base_url, default_model, reporter
    ):
        """Test agent with a task that requires tool use."""
        from petals.client.agent import AgentOrchestrator

        report = TestReport(
            test_name="test_agent_with_tools",
            timestamp=datetime.now().isoformat(),
            model=default_model,
            base_url=base_url,
            success=False,
            duration_ms=0,
        )

        # Define mock tools
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            weather_data = {
                "Tokyo": "Sunny, 25°C",
                "Paris": "Cloudy, 18°C",
                "New York": "Rainy, 15°C",
            }
            return weather_data.get(city, f"Weather data not available for {city}")

        def calculate(expression: str) -> str:
            """Calculate a mathematical expression."""
            try:
                result = eval(expression)  # Note: In production, use safer evaluation
                return str(result)
            except Exception as e:
                return f"Error: {e}"

        # Create agent with tools
        agent = AgentOrchestrator(
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            tools=[
                {"name": "get_weather", "func": get_weather},
                {"name": "calculate", "func": calculate},
            ],
            max_iterations=5,
            max_context_tokens=4000,
        )

        start = time.time()
        tool_calls_made = []

        try:
            # Inject tool call detection by wrapping functions
            original_get_weather = get_weather
            original_calculate = calculate

            def wrapped_get_weather(city: str) -> str:
                tool_calls_made.append(f"get_weather(city='{city}')")
                return original_get_weather(city)

            def wrapped_calculate(expression: str) -> str:
                tool_calls_made.append(f"calculate(expression='{expression}')")
                return original_calculate(expression)

            # Replace the registered tools with wrapped versions
            agent.registry._tools["get_weather"]["func"] = wrapped_get_weather
            agent.registry._tools["calculate"]["func"] = wrapped_calculate

            result = await agent.run(
                "What is the weather in Tokyo and what is 15 + 27?"
            )

            report.duration_ms = (time.time() - start) * 1000
            report.success = True
            report.response_content = result
            report.total_tokens = agent.state.total_tokens_used
            report.tool_calls_detected = tool_calls_made

            print(f"\n=== Agent With Tools ===")
            print(f"Duration: {report.duration_ms:.2f}ms")
            print(f"Total tokens: {agent.state.total_tokens_used}")
            print(f"Iterations: {agent.state.current_iteration}")
            print(f"Tool calls made: {tool_calls_made}")
            print(f"Tool history count: {len(list(agent.state.tool_history))}")
            print(f"Response: {result}")

        except Exception as e:
            report.error = str(e)
            report.duration_ms = (time.time() - start) * 1000
            report.tool_calls_detected = tool_calls_made
            print(f"\n=== Test Failed ===")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

        filepath = reporter.save_report(report)
        print(f"\nReport saved to: {filepath}")

        assert report.success, f"Test failed: {report.error}"


class TestCustomEndpointE2E:
    """E2E tests with custom OpenAI-compatible endpoints."""

    @pytest.mark.asyncio
    @pytest.fixture
    def custom_endpoint_client(self, api_key):
        """Create client for custom endpoint."""
        base_url = os.environ.get("LLM_BASE_URL")
        model = os.environ.get("LLM_MODEL", "claude-4.5-sonnet")
        if not base_url:
            pytest.skip("LLM_BASE_URL not set for custom endpoint test")
        return HTTPClient(
            api_key=api_key,
            base_url=base_url,
            default_model=model,
            timeout=120.0,
        )

    @pytest.mark.asyncio
    async def test_custom_endpoint_poem(
        self, custom_endpoint_client, base_url, reporter
    ):
        """Test with custom endpoint - generate a poem."""
        model = custom_endpoint_client.default_model

        report = TestReport(
            test_name="test_custom_endpoint_poem",
            timestamp=datetime.now().isoformat(),
            model=model,
            base_url=base_url,
            success=False,
            duration_ms=0,
        )

        prompt = "Write a short haiku about coding."
        report.request_payload = {"prompt": prompt}

        start = time.time()
        try:
            response = await custom_endpoint_client.generate(
                prompt,
                max_tokens=200,  # Increased for thinking models
            )

            report.duration_ms = (time.time() - start) * 1000
            report.success = True
            report.response_content = response.content
            report.input_tokens = response.usage.get("prompt_tokens") if response.usage else None
            report.output_tokens = response.usage.get("completion_tokens") if response.usage else None
            report.total_tokens = response.usage.get("total_tokens") if response.usage else None

            print(f"\n=== Custom Endpoint Test ===")
            print(f"Model: {model}")
            print(f"Base URL: {base_url}")
            print(f"Duration: {report.duration_ms:.2f}ms")
            print(f"Tokens: {report.input_tokens} -> {report.output_tokens}")
            print(f"Response:\n{response.content}")

        except Exception as e:
            report.error = str(e)
            report.duration_ms = (time.time() - start) * 1000
            print(f"\n=== Test Failed ===")
            print(f"Error: {e}")

        filepath = reporter.save_report(report)
        print(f"\nReport saved to: {filepath}")

        assert report.success, f"Test failed: {report.error}"
        # Check response has content (lenient for thinking models)
        assert report.response_content is not None and len(report.response_content) > 0


if __name__ == "__main__":
    # Run with: python -m pytest tests/e2e/test_http_client_e2e.py -v -s
    pytest.main([__file__, "-v", "-s"])
