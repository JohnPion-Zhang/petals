"""
StreamingAggregator - Aggregates streaming results for large context handling.

This module provides streaming aggregation capabilities including incremental
accumulation, buffering, deduplication, and partial result handling.
"""
import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from petals.client.async_support.streaming_types import StreamEvent, StreamEventType

logger = logging.getLogger(__name__)


@dataclass
class AggregationResult:
    """Result of aggregating streaming data.

    Attributes:
        items: List of accumulated items from the stream.
        metadata: Metadata collected during aggregation.
        errors: List of error messages encountered.
        status: Overall status ('success', 'partial', 'error').
        total_chunks: Total number of chunks processed.
        processing_time_ms: Total processing time in milliseconds.
    """

    items: List[Any] = None
    metadata: Dict[str, Any] = None
    errors: List[str] = None
    status: str = "partial"
    total_chunks: int = 0
    processing_time_ms: float = 0.0

    def __post_init__(self) -> None:
        """Initialize default values if not provided."""
        if self.items is None:
            self.items = []
        if self.metadata is None:
            self.metadata = {}
        if self.errors is None:
            self.errors = []

    @property
    def is_success(self) -> bool:
        """Check if aggregation was successful."""
        return self.status == "success"

    @property
    def is_partial(self) -> bool:
        """Check if aggregation was partial (incomplete)."""
        return self.status == "partial"

    @property
    def is_error(self) -> bool:
        """Check if aggregation encountered errors."""
        return self.status == "error"

    def add_item(self, item: Any) -> None:
        """Add an item to the aggregation result.

        Args:
            item: Item to add to the items list.
        """
        self.items.append(item)

    def add_error(self, error: str) -> None:
        """Add an error message.

        Args:
            error: Error message to record.
        """
        self.errors.append(error)
        if self.status == "success":
            self.status = "partial"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all result fields.
        """
        return {
            "items": self.items,
            "metadata": self.metadata,
            "errors": self.errors,
            "status": self.status,
            "total_chunks": self.total_chunks,
            "processing_time_ms": self.processing_time_ms,
        }


class StreamingAggregator:
    """Aggregates streaming results for large context handling.

    Features:
    - Incremental accumulation of items
    - Streaming aggregation with buffering
    - Deduplication support
    - Partial result handling
    - Chunk counting and statistics

    Example:
        >>> from petals.client.async import StreamingAggregator, StreamEvent
        >>> from petals.client.async.streaming_types import StreamEventType
        >>>
        >>> aggregator = StreamingAggregator(buffer_size=100)
        >>>
        >>> # Process a stream
        >>> async def process_stream():
        ...     event = StreamEvent(
        ...         type=StreamEventType.TOOL_RESULT,
        ...         data={"result": {"data": [1, 2, 3]}}
        ...     )
        ...     await aggregator.add_chunk(event)
        ...     return aggregator.get_current_result()
        >>>
        >>> result = asyncio.run(process_stream())
        >>> print(result.items)
        [{'data': [1, 2, 3]}]
    """

    def __init__(
        self,
        buffer_size: int = 100,
        deduplicate: bool = True,
        key_field: Optional[str] = None
    ) -> None:
        """Initialize the streaming aggregator.

        Args:
            buffer_size: Maximum items to buffer before auto-flush.
            deduplicate: Whether to enable deduplication.
            key_field: Field name to use for deduplication key.
        """
        self.buffer_size = buffer_size
        self.deduplicate = deduplicate
        self.key_field = key_field

        self._items: List[Any] = []
        self._seen_keys: set = set()
        self._metadata: Dict[str, Any] = {}
        self._errors: List[str] = []
        self._chunk_count: int = 0
        self._start_time: float = 0.0
        self._lock = asyncio.Lock()

        # Statistics
        self._duplicate_count: int = 0
        self._error_count: int = 0

    async def aggregate_stream(
        self,
        stream: AsyncIterator[StreamEvent]
    ) -> AggregationResult:
        """Aggregate a stream of events into a final result.

        Processes all events from the stream and returns a complete
        aggregation result.

        Args:
            stream: Async iterator of StreamEvents.

        Returns:
            AggregationResult with all accumulated data.

        Example:
            >>> async def events():
            ...     yield StreamEvent.tool_result("search", "n1", {"data": [1]})
            ...     yield StreamEvent.tool_result("calc", "n2", {"data": [2]})
            ...
            >>> result = await aggregator.aggregate_stream(events())
            >>> print(len(result.items))
            2
        """
        self._start_time = time.time()
        self._reset()

        try:
            async for event in stream:
                await self.add_chunk(event)

            # Mark as success if no errors
            if self._error_count == 0:
                self._result.status = "success"

        except Exception as e:
            logger.error(f"Error during stream aggregation: {e}")
            self._errors.append(str(e))
            self._result.status = "error"

        # Calculate final stats
        self._result.total_chunks = self._chunk_count
        self._result.processing_time_ms = (time.time() - self._start_time) * 1000

        return self._result

    @property
    def _result(self) -> AggregationResult:
        """Get current aggregation result."""
        return AggregationResult(
            items=self._items,
            metadata=self._metadata,
            errors=self._errors.copy(),
            status="partial" if self._errors else "success",
            total_chunks=self._chunk_count,
            processing_time_ms=(time.time() - self._start_time) * 1000 if self._start_time else 0.0,
        )

    def _reset(self) -> None:
        """Reset internal state for a new aggregation."""
        self._items = []
        self._seen_keys = set()
        self._metadata = {}
        self._errors = []
        self._chunk_count = 0
        self._duplicate_count = 0
        self._error_count = 0

    async def add_chunk(self, event: StreamEvent) -> None:
        """Add a single chunk to the aggregation.

        Handles buffering, deduplication, and metadata extraction.

        Args:
            event: The StreamEvent to add.

        Example:
            >>> event = StreamEvent(
            ...     type=StreamEventType.TOOL_RESULT,
            ...     data={"tool": "search", "result": [1, 2, 3]}
            ... )
            >>> await aggregator.add_chunk(event)
        """
        async with self._lock:
            self._chunk_count += 1

            try:
                if event.type == StreamEventType.ERROR:
                    await self._handle_error_event(event)
                elif event.type == StreamEventType.TOOL_RESULT:
                    await self._handle_tool_result(event)
                elif event.type == StreamEventType.TEXT_CHUNK:
                    await self._handle_text_chunk(event)
                elif event.type == StreamEventType.FINAL:
                    await self._handle_final_event(event)
                else:
                    # For other events, just extract metadata
                    await self._extract_metadata(event)

            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
                self._errors.append(str(e))
                self._error_count += 1

    async def _handle_error_event(self, event: StreamEvent) -> None:
        """Handle an error event.

        Args:
            event: The error event to process.
        """
        error_msg = event.data.get("message", "Unknown error")
        self._errors.append(error_msg)
        self._error_count += 1

    async def _handle_tool_result(self, event: StreamEvent) -> None:
        """Handle a tool result event.

        Args:
            event: The tool result event to process.
        """
        result = event.data.get("result")
        success = event.data.get("success", True)

        if not success:
            error = event.data.get("error", "Tool execution failed")
            self._errors.append(error)
            self._error_count += 1

        if result is not None:
            # Check for deduplication
            if self.deduplicate:
                key = self._get_dedup_key(result)
                if key is not None and key in self._seen_keys:
                    self._duplicate_count += 1
                    logger.debug(f"Skipping duplicate item with key: {key}")
                    return
                if key is not None:
                    self._seen_keys.add(key)

            # Extract items from result (handle nested structures)
            items = self._extract_items(result)
            self._items.extend(items)

    async def _handle_text_chunk(self, event: StreamEvent) -> None:
        """Handle a text chunk event.

        Accumulates text chunks into a single string.

        Args:
            event: The text chunk event to process.
        """
        text = event.data.get("text", "")
        is_final = event.data.get("is_final", False)

        # Accumulate text metadata
        if "text_chunks" not in self._metadata:
            self._metadata["text_chunks"] = []
            self._metadata["full_text"] = ""

        self._metadata["text_chunks"].append(text)
        self._metadata["full_text"] += text

        if is_final:
            self._metadata["text_complete"] = True

    async def _handle_final_event(self, event: StreamEvent) -> None:
        """Handle a final event.

        Extracts metadata from the final event.

        Args:
            event: The final event to process.
        """
        self._metadata.update({
            "total_tools": event.data.get("total_tools", 0),
            "final_text": event.data.get("final_text"),
        })

        # Extract any additional metadata
        if "metadata" in event.data:
            self._metadata.update(event.data["metadata"])

    async def _extract_metadata(self, event: StreamEvent) -> None:
        """Extract metadata from an event.

        Args:
            event: The event to extract metadata from.
        """
        # Extract tool name and node info
        if "tool_name" in event.data:
            if "tools" not in self._metadata:
                self._metadata["tools"] = []
            self._metadata["tools"].append(event.data["tool_name"])

        if "node_id" in event.data:
            if "nodes" not in self._metadata:
                self._metadata["nodes"] = []
            if event.data["node_id"] not in self._metadata["nodes"]:
                self._metadata["nodes"].append(event.data["node_id"])

    def _get_dedup_key(self, item: Any) -> Optional[str]:
        """Get deduplication key for an item.

        Args:
            item: The item to get key for.

        Returns:
            The deduplication key if found, None otherwise.
        """
        if isinstance(item, dict) and self.key_field and self.key_field in item:
            return str(item[self.key_field])
        if isinstance(item, dict) and "id" in item:
            return str(item["id"])
        if isinstance(item, dict) and "name" in item:
            return str(item["name"])
        return None

    def _extract_items(self, result: Any) -> List[Any]:
        """Extract items from a result.

        Args:
            result: The result to extract items from.

        Returns:
            List of items extracted from the result. When extracting from
            dict keys like "data", returns the wrapped list as a single item.
        """
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            # Try common item fields
            for key in ["items", "data", "results", "entries"]:
                if key in result and isinstance(result[key], list):
                    # Wrap the extracted list as a single item
                    return [result[key]]
            # Return the whole dict as a single item
            return [result]
        return [result]

    async def flush(self) -> None:
        """Flush any buffered data.

        Currently a no-op as items are added immediately,
        but can be overridden for custom buffering behavior.
        """
        pass

    def get_current_result(self) -> AggregationResult:
        """Get current aggregation state without waiting for stream completion.

        Returns:
            AggregationResult with current accumulated data.

        Example:
            >>> # During streaming
            >>> partial = aggregator.get_current_result()
            >>> print(f"Processed {partial.total_chunks} chunks so far")
        """
        return AggregationResult(
            items=self._items.copy(),
            metadata=self._metadata.copy(),
            errors=self._errors.copy(),
            status="partial",
            total_chunks=self._chunk_count,
            processing_time_ms=(time.time() - self._start_time) * 1000 if self._start_time else 0.0,
        )

    @staticmethod
    async def aggregate_multiple(
        streams: List[AsyncIterator[StreamEvent]],
        merge_strategy: str = "append"
    ) -> AggregationResult:
        """Aggregate multiple streams concurrently.

        Args:
            streams: List of async iterators to aggregate.
            merge_strategy: How to merge results ('append', 'merge', 'union').

        Returns:
            Combined AggregationResult from all streams.

        Example:
            >>> results = await StreamingAggregator.aggregate_multiple([stream1, stream2])
            >>> print(f"Total items: {len(results.items)}")
        """
        aggregator = StreamingAggregator()

        async def process_all() -> None:
            tasks = [aggregator.aggregate_stream(stream) for stream in streams]
            await asyncio.gather(*tasks, return_exceptions=True)

        await process_all()

        # Combine results based on merge strategy
        if merge_strategy == "union":
            # Remove duplicates
            seen = set()
            unique_items = []
            for item in aggregator._items:
                key = str(item) if not isinstance(item, dict) else str(sorted(item.items()))
                if key not in seen:
                    seen.add(key)
                    unique_items.append(item)
            aggregator._items = unique_items

        return aggregator.get_current_result()

    @property
    def stats(self) -> Dict[str, Any]:
        """Get aggregation statistics.

        Returns:
            Dictionary with aggregation stats.
        """
        return {
            "total_chunks": self._chunk_count,
            "total_items": len(self._items),
            "duplicate_count": self._duplicate_count,
            "error_count": self._error_count,
            "unique_keys": len(self._seen_keys),
            "processing_time_ms": (time.time() - self._start_time) * 1000 if self._start_time else 0.0,
        }
