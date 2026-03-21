"""Resilience configuration and error types for LLM providers."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TimeoutConfig:
    """Timeout configuration for LLM provider calls.

    Provides fine-grained timeout control for different operations.
    Timeouts prevent hanging requests and provide predictable behavior.

    Attributes:
        connect_timeout: Time to establish connection (seconds).
        read_timeout: Time between data chunks (seconds).
        total_timeout: Maximum total time for any operation (seconds).
        chat_timeout: Timeout for chat/completion operations (seconds).
        embedding_timeout: Timeout for embedding operations (seconds).
        stream_timeout: Timeout for streaming operations (seconds).

    Example:
        >>> config = TimeoutConfig(
        ...     connect_timeout=10.0,
        ...     read_timeout=120.0,
        ...     total_timeout=180.0,
        ...     chat_timeout=120.0,
        ...     embedding_timeout=30.0,
        ... )
        >>> provider = OpenAIProvider(api_key="key", timeout_config=config)
    """

    # Connection timeouts
    connect_timeout: float = 10.0

    # Read/response timeouts
    read_timeout: float = 120.0

    # Overall timeouts
    total_timeout: float = 180.0

    # Per-operation timeouts
    chat_timeout: float = 120.0
    embedding_timeout: float = 30.0
    stream_timeout: float = 120.0

    def get_timeout_for_operation(self, operation: str) -> float:
        """Get timeout for a specific operation.

        Args:
            operation: Operation name (complete, stream, count_tokens, etc.).

        Returns:
            Timeout in seconds for the operation.
        """
        operation_map = {
            "complete": self.chat_timeout,
            "chat": self.chat_timeout,
            "stream": self.stream_timeout,
            "embedding": self.embedding_timeout,
            "count_tokens": self.embedding_timeout,
        }
        return operation_map.get(operation, self.total_timeout)


@dataclass
class LLMToolError(Exception):
    """Error raised when LLM provider operations fail.

    Provides detailed error information including operation type,
    duration, and whether the error is recoverable.

    Attributes:
        message: Human-readable error message.
        recoverable: Whether the operation can be retried.
        operation: Name of the operation that failed.
        duration: How long the operation ran before failing.
        cause: Original exception that caused this error.

    Example:
        >>> raise LLMToolError(
        ...     f"Timeout after {timeout}s during {operation}",
        ...     recoverable=True,
        ...     operation=operation,
        ...     duration=timeout,
        ... )
    """

    message: str
    recoverable: bool = True
    operation: Optional[str] = None
    duration: Optional[float] = None
    cause: Optional[Exception] = None

    def __str__(self) -> str:
        """Return string representation of the error."""
        return self.message

    def __post_init__(self) -> None:
        """Initialize exception with cause if provided."""
        super().__init__(self.message)
        if self.cause is not None:
            self.__cause__ = self.cause
