"""LLM with tool calling integrated with orchestrator."""

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from .providers.base import BaseLLMProvider, LLMChunk
from .dag import ToolCallNode, ToolCallDAG
from .tool_parser import ToolParser
from .async_support.streaming_types import StreamEvent, StreamEventType

logger = logging.getLogger(__name__)


class ToolCallingLLM:
    """LLM wrapper that handles tool parsing and execution.

    Integrates HTTP-based LLM providers with the orchestrator's DAG execution
    for seamless tool-calling workflows.

    Usage:
        >>> from petals.client.providers import create_provider
        >>> from petals.client.tool_calling_llm import ToolCallingLLM
        >>>
        >>> provider = create_provider("openai", api_key="...")
        >>> llm = ToolCallingLLM(
        ...     provider=provider,
        ...     registry=registry,
        ...     orchestrator=orchestrator,
        ... )
        >>> async for event in llm.run_streaming(prompt, tools=tools):
        ...     await sse.send(event.to_sse())

    Attributes:
        provider: The LLM provider to use.
        registry: ToolRegistry for looking up tool definitions.
        orchestrator: Orchestrator for DAG execution.
        tool_parser: Parser for extracting tool calls from text.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        registry: Any,  # ToolRegistry
        orchestrator: Any,  # Orchestrator
        tool_parser: Optional[ToolParser] = None,
    ):
        """Initialize the tool-calling LLM.

        Args:
            provider: HTTP-based LLM provider.
            registry: ToolRegistry containing available tools.
            orchestrator: Orchestrator for DAG execution.
            tool_parser: Optional ToolParser instance. Creates new one if not provided.
        """
        self.provider = provider
        self.registry = registry
        self.orchestrator = orchestrator
        self.tool_parser = tool_parser or ToolParser()

    async def run_streaming(
        self,
        prompt: str,
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Run LLM with streaming, parsing tool calls as they appear.

        This method streams LLM tokens while detecting tool calls in real-time.
        After the stream completes, it parses tool calls and executes them
        via the orchestrator if needed.

        Args:
            prompt: User prompt.
            system: Optional system prompt.
            tools: Optional list of tool definitions.
            context: Optional context for the execution.

        Yields:
            StreamEvents for LLM tokens and tool execution results.

        Example:
            >>> async for event in llm.run_streaming(
            ...     "Search for AI news",
            ...     tools=tools,
            ... ):
            ...     if event.type == StreamEventType.TEXT:
            ...         print(event.data["content"], end="")
            ...     elif event.type == StreamEventType.TOOL_RESULT:
            ...         print(f"\\nTool result: {event.data['result']}")
        """
        # Stream LLM response while detecting tools
        tool_names_detected: List[str] = []
        full_content = ""
        chunk_index = 0
        is_complete = False

        async for chunk in self.provider.stream(prompt, system, tools):
            # Emit text content
            if chunk.content:
                full_content += chunk.content
                yield StreamEvent(
                    type=StreamEventType.TEXT_CHUNK,
                    data={"text": chunk.content, "is_final": False},
                )
                chunk_index += 1

            # Track detected tool names
            for tool_name in chunk.tool_names or []:
                if tool_name not in tool_names_detected:
                    tool_names_detected.append(tool_name)
                    yield StreamEvent(
                        type=StreamEventType.TOOL_CALL_PENDING,
                        data={"tool_name": tool_name, "node_id": f"llm_detected_{tool_name}"},
                    )

            # Track completion
            if chunk.is_complete:
                is_complete = True

        # If LLM is complete, emit final event before tool execution
        if is_complete:
            yield StreamEvent(
                type=StreamEventType.FINAL,
                data={
                    "final_text": full_content,
                    "total_chunks": chunk_index,
                    "total_tools": len(tool_names_detected),
                },
            )

        # Parse tool calls from the full response
        tool_calls = self.tool_parser.parse(full_content)

        if tool_calls:
            # Build DAG from parsed tool calls
            dag = self._build_dag_from_tool_calls(tool_calls)

            # Execute DAG via orchestrator
            async for event in self.orchestrator.execute_streaming(dag):
                yield event

    async def run(
        self,
        prompt: str,
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run LLM completion with optional tool execution.

        Unlike run_streaming, this returns all results at once.

        Args:
            prompt: User prompt.
            system: Optional system prompt.
            tools: Optional list of tool definitions.
            context: Optional execution context.

        Returns:
            Dictionary with:
            - content: Final text response
            - tool_results: Results from tool execution (if any)
            - usage: Token usage statistics
        """
        # Get completion
        response = await self.provider.complete(prompt, system, tools)

        result: Dict[str, Any] = {
            "content": response.content,
            "tool_results": None,
            "usage": response.usage,
            "finish_reason": response.finish_reason,
        }

        # Parse tool calls
        tool_calls = self.tool_parser.parse(response.content)

        if tool_calls:
            # Build and execute DAG
            dag = self._build_dag_from_tool_calls(tool_calls)

            tool_results: Dict[str, Any] = {}
            async for event in self.orchestrator.execute_streaming(dag):
                if event.type == StreamEventType.TOOL_RESULT:
                    tool_results[event.data["id"]] = event.data["result"]

            result["tool_results"] = tool_results

        return result

    def _build_dag_from_tool_calls(self, tool_calls: List[Any]) -> ToolCallDAG:
        """Build DAG from parsed tool calls.

        Args:
            tool_calls: List of ToolCall objects from parser.

        Returns:
            ToolCallDAG ready for execution.
        """
        dag = ToolCallDAG()

        for tc in tool_calls:
            node = ToolCallNode(
                id=tc.id,
                name=tc.name,
                arguments=tc.arguments or {},
                dependencies=list(tc.dependencies) if tc.dependencies else [],
            )
            dag.add_node(node)

            # Add edges based on dependencies
            for dep_id in node.dependencies:
                if dep_id in dag.nodes:
                    dag.add_edge(dep_id, node.id)

        logger.debug(f"Built DAG with {len(dag.nodes)} nodes")
        return dag

    def get_tool_definitions(
        self, tool_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get tool definitions from registry.

        Args:
            tool_names: Optional list of tool names to filter.
                If None, returns all available tools.

        Returns:
            List of tool definitions in OpenAI-compatible format.
        """
        if not self.registry:
            return []

        if tool_names:
            tools = [
                self.registry.get_tool(name)
                for name in tool_names
                if self.registry.has_tool(name)
            ]
        else:
            tools = list(self.registry.list_tools().values())

        # Convert to OpenAI-compatible format
        result = []
        for tool in tools:
            if tool:
                result.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": getattr(tool, "description", ""),
                            "parameters": getattr(tool, "parameters", {"type": "object"}),
                        },
                    }
                )
        return result

    async def close(self) -> None:
        """Close the provider and cleanup resources."""
        if hasattr(self.provider, "close"):
            await self.provider.close()


def create_tool_calling_llm(
    provider_type: str,
    registry: Any,
    orchestrator: Any,
    **provider_kwargs: Any,
) -> ToolCallingLLM:
    """Create a ToolCallingLLM with the specified provider type.

    Args:
        provider_type: Type of provider ("openai", "anthropic", etc.).
        registry: ToolRegistry for available tools.
        orchestrator: Orchestrator for DAG execution.
        **provider_kwargs: Arguments for the provider constructor.

    Returns:
        Configured ToolCallingLLM instance.

    Example:
        >>> llm = create_tool_calling_llm(
        ...     "openai",
        ...     registry=registry,
        ...     orchestrator=orchestrator,
        ...     api_key="sk-...",
        ...     model="gpt-4",
        ... )
    """
    from .providers import create_provider

    provider = create_provider(provider_type, **provider_kwargs)
    return ToolCallingLLM(
        provider=provider,
        registry=registry,
        orchestrator=orchestrator,
    )


__all__ = [
    "ToolCallingLLM",
    "create_tool_calling_llm",
]
