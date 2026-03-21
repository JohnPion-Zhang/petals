"""
Feedback Loop Module - CodeAct Self-Correction Pattern

This module provides feedback loop functionality for self-correction
in LLM-based tool execution.

Components:
- TracebackCapture: Capture and format Python exceptions
- LLMCorrector: Generate corrected tool arguments via LLM
- RetryPolicy: Configurable retry logic with backoff strategies
- CircuitBreaker: Prevent cascade failures
- ExecutionFeedbackLoop: Main feedback loop implementing CodeAct pattern

Example:
    >>> from petals.client.feedback import ExecutionFeedbackLoop, FeedbackLoopConfig
    >>> from petals.client.tool_registry import ToolRegistry
    >>>
    >>> registry = ToolRegistry()
    >>> # ... register tools ...
    >>>
    >>> config = FeedbackLoopConfig(
    ...     max_retries=3,
    ...     enable_correction=True,
    ...     correction_llm=llm_provider
    ... )
    >>> loop = ExecutionFeedbackLoop(registry, config)
    >>>
    >>> # Execute with feedback
    >>> result = await loop.execute_with_feedback(node)
"""

from petals.client.feedback.traceback import (
    ErrorSeverity,
    CapturedTraceback,
    TracebackCapture,
)

from petals.client.feedback.correction import (
    CorrectionStrategy,
    CorrectionPrompt,
    CorrectionResult,
    LLMCorrector,
)

from petals.client.feedback.retry_policy import (
    RetryError,
    CircuitOpenError,
    BackoffStrategy,
    RetryPolicy,
    CircuitBreakerConfig,
    CircuitBreaker,
    CircuitBreakerManager,
    RetryWithCircuitBreaker,
)

from petals.client.feedback.feedback_loop import (
    FeedbackAction,
    FeedbackEntry,
    FeedbackLoopConfig,
    ExecutionFeedbackLoop,
)

__all__ = [
    # Traceback
    "ErrorSeverity",
    "CapturedTraceback",
    "TracebackCapture",
    # Correction
    "CorrectionStrategy",
    "CorrectionPrompt",
    "CorrectionResult",
    "LLMCorrector",
    # Retry Policy
    "RetryError",
    "CircuitOpenError",
    "BackoffStrategy",
    "RetryPolicy",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "CircuitBreakerManager",
    "RetryWithCircuitBreaker",
    # Feedback Loop
    "FeedbackAction",
    "FeedbackEntry",
    "FeedbackLoopConfig",
    "ExecutionFeedbackLoop",
]
