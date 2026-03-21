"""
Traceback capture and formatting for error feedback in CodeAct pattern.

This module provides utilities for capturing, formatting, and analyzing
Python exceptions to generate meaningful error feedback for LLM correction.

Example:
    >>> from petals.client.feedback.traceback import TracebackCapture, CapturedTraceback
    >>> try:
    ...     raise ValueError("Invalid input: expected positive number")
    ... except ValueError as e:
    ...     tb = TracebackCapture.capture_exception(e)
    ...     print(tb.format_for_llm())
"""
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from enum import Enum


class ErrorSeverity(Enum):
    """Severity level of an error."""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class CapturedTraceback:
    """Captured error information with formatted traceback.

    Attributes:
        error_type: The exception class name.
        error_message: The exception message.
        traceback_str: Raw traceback string.
        frame_summaries: List of dictionaries containing frame information.
        severity: Error severity level.
        context: Additional context for the error.

    Example:
        >>> tb = CapturedTraceback(
        ...     error_type="ValueError",
        ...     error_message="Invalid input",
        ...     traceback_str="...",
        ...     severity=ErrorSeverity.ERROR
        ... )
        >>> print(tb.format_compact())
        ValueError: Invalid input
    """

    error_type: str
    error_message: str
    traceback_str: str
    frame_summaries: List[Dict[str, Any]] = field(default_factory=list)
    severity: ErrorSeverity = ErrorSeverity.ERROR
    context: Dict[str, Any] = field(default_factory=dict)

    def format_for_llm(self) -> str:
        """Format traceback in a way suitable for LLM consumption.

        Returns a structured string that helps the LLM understand
        the error and generate a fix.

        Returns:
            Formatted string with error type, message, traceback,
            and suggested fix hints.
        """
        hints = self.get_fix_hints()
        hints_str = "\n".join(f"- {hint}" for hint in hints) if hints else "No specific hints available."

        return f"""## Error Analysis

**Error Type:** {self.error_type}
**Severity:** {self.severity.value}

**Message:**
{self.error_message}

**Traceback:**
{self.traceback_str}

**Code Location:**
{self.format_compact()}

**Suggested Fix Hints:**
{hints_str}
"""

    def format_compact(self) -> str:
        """Compact format showing just the error location.

        Returns:
            A compact string showing error type, message, and location.
        """
        if self.frame_summaries:
            first_frame = self.frame_summaries[0]
            location = f"{first_frame.get('filename', 'unknown')}:{first_frame.get('lineno', '?')}"
            return f"{self.error_type}: {self.error_message} (at {location})"
        return f"{self.error_type}: {self.error_message}"

    def get_fix_hints(self) -> List[str]:
        """Extract potential fix hints from traceback.

        Analyzes the error type and traceback to suggest possible fixes.

        Returns:
            List of potential fix hints.
        """
        hints: List[str] = []

        # Common error type hints
        error_hints = {
            "ValueError": [
                "Check that input values are of the expected type",
                "Verify that value constraints are satisfied",
                "Ensure string/numeric conversions are valid"
            ],
            "TypeError": [
                "Check argument types match function expectations",
                "Verify that None is not passed where a value is required",
                "Ensure correct method calls on objects"
            ],
            "KeyError": [
                "Verify the key exists in the dictionary",
                "Check for typos in key names",
                "Use .get() method with default values"
            ],
            "IndexError": [
                "Check array bounds before accessing elements",
                "Verify list is not empty before indexing",
                "Use len() to validate index ranges"
            ],
            "ConnectionError": [
                "Check network connectivity",
                "Verify the remote service is available",
                "Consider adding retry logic with backoff"
            ],
            "TimeoutError": [
                "Increase timeout duration",
                "Check if the remote service is overloaded",
                "Consider breaking up large requests"
            ],
            "FileNotFoundError": [
                "Verify the file path is correct",
                "Check that the file exists",
                "Ensure proper directory structure"
            ],
            "PermissionError": [
                "Check file/directory permissions",
                "Verify read/write access",
                "Run with appropriate privileges"
            ]
        }

        # Add hints based on error type
        if self.error_type in error_hints:
            hints.extend(error_hints[self.error_type])

        # Add hints based on error message content
        msg_lower = self.error_message.lower()
        if "none" in msg_lower or "null" in msg_lower:
            hints.append("Handle None values explicitly before processing")
        if "timeout" in msg_lower:
            hints.append("Consider increasing timeout or adding retry logic")
        if "connection" in msg_lower:
            hints.append("Check network connection and remote service availability")
        if "permission" in msg_lower:
            hints.append("Verify file/directory permissions")
        if "not found" in msg_lower:
            hints.append("Verify resource exists before accessing")

        # Remove duplicates while preserving order
        seen = set()
        unique_hints = []
        for hint in hints:
            if hint not in seen:
                seen.add(hint)
                unique_hints.append(hint)

        return unique_hints

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the captured traceback to a dictionary.

        Returns:
            Dictionary representation of the captured traceback.
        """
        return {
            "error_type": self.error_type,
            "error_message": self.error_message,
            "traceback_str": self.traceback_str,
            "frame_summaries": self.frame_summaries,
            "severity": self.severity.value,
            "context": self.context
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CapturedTraceback":
        """Create a CapturedTraceback from a dictionary.

        Args:
            data: Dictionary with traceback data.

        Returns:
            A new CapturedTraceback instance.
        """
        severity = data.get("severity", "error")
        if isinstance(severity, str):
            severity = ErrorSeverity(severity)

        return cls(
            error_type=data["error_type"],
            error_message=data["error_message"],
            traceback_str=data["traceback_str"],
            frame_summaries=data.get("frame_summaries", []),
            severity=severity,
            context=data.get("context", {})
        )


class TracebackCapture:
    """Utility class for capturing and formatting tracebacks.

    Provides static methods for capturing exceptions, results,
    and extracting code snippets for error analysis.
    """

    @staticmethod
    def capture_exception(
        exc: BaseException,
        limit: Optional[int] = None,
        include_context: bool = True
    ) -> CapturedTraceback:
        """Capture exception and format as CapturedTraceback.

        Args:
            exc: The exception to capture.
            limit: Maximum number of stack frames to capture.
            include_context: Whether to include frame summaries.

        Returns:
            A CapturedTraceback containing the formatted error information.
        """
        # Get the traceback string
        tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=limit))

        # Extract frame summaries if requested
        frame_summaries: List[Dict[str, Any]] = []
        if include_context:
            for frame_summary in traceback.extract_tb(exc.__traceback__, limit=limit):
                frame_summaries.append({
                    "filename": frame_summary.filename,
                    "lineno": frame_summary.lineno,
                    "name": frame_summary.name,
                    "line": frame_summary.line
                })

        # Determine severity based on exception type
        severity = ErrorSeverity.ERROR
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            severity = ErrorSeverity.CRITICAL
        elif isinstance(exc, (DeprecationWarning, PendingDeprecationWarning)):
            severity = ErrorSeverity.WARNING

        return CapturedTraceback(
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback_str=tb_str,
            frame_summaries=frame_summaries,
            severity=severity
        )

    @staticmethod
    def capture_result(
        result: Any,
        error: Optional[Exception] = None
    ) -> Dict[str, Any]:
        """Capture execution result with optional error.

        Args:
            result: The result to capture.
            error: Optional exception that occurred.

        Returns:
            Dictionary with result and error information.
        """
        captured: Dict[str, Any] = {
            "result": result,
            "success": error is None,
            "error": None
        }

        if error is not None:
            captured["error"] = TracebackCapture.capture_exception(error)

        return captured

    @staticmethod
    def format_code_snippet(
        file_path: str,
        line_number: int,
        context_lines: int = 3
    ) -> str:
        """Extract and format code snippet around error location.

        Args:
            file_path: Path to the source file.
            line_number: Line number of the error.
            context_lines: Number of lines before/after to include.

        Returns:
            Formatted code snippet with line numbers.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            start = max(0, line_number - context_lines - 1)
            end = min(len(lines), line_number + context_lines)

            snippet_lines = []
            for i in range(start, end):
                line_num = i + 1
                prefix = ">>> " if line_num == line_number else "    "
                snippet_lines.append(f"{prefix}{line_num:4d} | {lines[i].rstrip()}")

            return f"File: {file_path}\n" + "\n".join(snippet_lines)

        except (OSError, IOError) as e:
            return f"Could not read file {file_path}: {e}"

    @staticmethod
    def determine_severity(error_type: str, error_message: str) -> ErrorSeverity:
        """Determine error severity based on error type and message.

        Args:
            error_type: The exception class name.
            error_message: The exception message.

        Returns:
            The appropriate ErrorSeverity level.
        """
        critical_types = {"KeyboardInterrupt", "SystemExit", "GeneratorExit"}
        warning_types = {"DeprecationWarning", "PendingDeprecationWarning", "FutureWarning"}

        if error_type in critical_types:
            return ErrorSeverity.CRITICAL
        if error_type in warning_types:
            return ErrorSeverity.WARNING

        msg_lower = error_message.lower()
        if any(word in msg_lower for word in ["critical", "fatal", "crash"]):
            return ErrorSeverity.CRITICAL
        if any(word in msg_lower for word in ["warning", "deprecated", "soon"]):
            return ErrorSeverity.WARNING

        return ErrorSeverity.ERROR
