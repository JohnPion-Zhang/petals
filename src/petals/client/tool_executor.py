"""Tool executor for running tool calls with dependency management."""

import asyncio
from typing import List, Optional, Generator

from petals.data_structures import ToolCall, CallStatus
from petals.client.tool_registry import ToolRegistry


class ToolExecutor:
    """Executes tool calls with support for parallel execution and dependencies."""

    def __init__(self, registry: ToolRegistry, timeout: float = 30.0, state=None):
        """
        Initialize the tool executor with a tool registry.

        Args:
            registry: The ToolRegistry containing available tools.
            timeout: Maximum time in seconds to wait for a tool to complete.
            state: Optional AgentState for tracking tool history.
        """
        self.registry = registry
        self.timeout = timeout
        self.state = state

    async def execute(self, tool_call: ToolCall) -> ToolCall:
        """
        Execute a single tool call.

        Args:
            tool_call: The ToolCall to execute.

        Returns:
            The ToolCall with updated status and result.
        """
        tool_call.status = CallStatus.RUNNING

        try:
            # Execute the tool with timeout
            result = await asyncio.wait_for(
                self.registry.execute(
                    tool_call.name,
                    tool_call.arguments or {}
                ),
                timeout=self.timeout
            )

            # Check for error in result
            if isinstance(result, dict) and "error" in result:
                tool_call.status = CallStatus.FAILED
                tool_call.result = result
            else:
                tool_call.status = CallStatus.DONE
                tool_call.result = result

        except asyncio.TimeoutError:
            tool_call.status = CallStatus.FAILED
            tool_call.result = {"error": "Timeout"}
        except ValueError as e:
            # Unknown tool
            tool_call.status = CallStatus.FAILED
            tool_call.result = {"error": str(e)}
        except Exception as e:
            tool_call.status = CallStatus.FAILED
            tool_call.result = {"error": str(e)}

        return tool_call

    async def parallel_execute(self, tool_calls: List[ToolCall]) -> Generator[ToolCall, None, None]:
        """
        Execute tools in parallel, yielding results as they complete.

        Yields results as they finish (not in submission order), enabling
        streaming behavior and reducing memory for large tool sets.

        Args:
            tool_calls: List of ToolCalls to execute.

        Yields:
            ToolCall with updated status and result as each completes.
        """
        # Execute all tool calls concurrently
        tasks = [self.execute(tc) for tc in tool_calls]

        # Use as_completed to yield results as they finish
        for coro in asyncio.as_completed(tasks):
            result = await coro

            # Update state if available
            if self.state is not None:
                self.state.add_tool(result)

            yield result

    async def execute_with_dependencies(
        self,
        tool_calls: List[ToolCall],
        results_cache: dict = None
    ) -> Generator[ToolCall, None, None]:
        """
        Execute tool calls respecting their dependency order.

        Tools in each wave execute in parallel. Results are yielded per wave.

        Args:
            tool_calls: List of ToolCalls to execute.
            results_cache: Dictionary to store results for dependency resolution.

        Yields:
            ToolCall with updated status as each wave completes.
        """
        if results_cache is None:
            results_cache = {}

        completed = set()
        remaining = list(tool_calls)

        while remaining:
            # Find tools whose dependencies are all satisfied
            ready = []
            waiting = []

            for tc in remaining:
                deps_satisfied = all(dep in completed for dep in tc.dependencies)
                if deps_satisfied:
                    ready.append(tc)
                else:
                    waiting.append(tc)

            if not ready:
                # No tools ready but still waiting - cycle dependency
                for tc in waiting:
                    tc.status = CallStatus.FAILED
                    tc.result = {"error": "Unresolved dependency"}
                    yield tc
                break

            # Execute ready tools in parallel
            async for result in self.parallel_execute(ready):
                if result.status == CallStatus.DONE:
                    completed.add(result.id)
                    results_cache[result.id] = result.result
                yield result

            remaining = waiting

    async def parallel_execute_list(self, tool_calls: List[ToolCall]) -> List[ToolCall]:
        """
        Execute tools in parallel, returning a list (non-generator version).

        This is a convenience method for cases where you need all results
        at once instead of streaming.

        Args:
            tool_calls: List of ToolCalls to execute.

        Returns:
            List of ToolCalls with updated status and results.
        """
        results = []
        async for result in self.parallel_execute(tool_calls):
            results.append(result)
        return results
