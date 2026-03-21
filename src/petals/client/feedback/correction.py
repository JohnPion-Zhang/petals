"""
LLM-based error correction using CodeAct pattern.

This module provides the LLMCorrector class that analyzes execution errors
and generates corrected tool arguments using an LLM.

Example:
    >>> from petals.client.feedback.correction import LLMCorrector, CorrectionResult
    >>> corrector = LLMCorrector(llm_provider=mock_llm, max_retries=3)
    >>> result = await corrector.correct(
    ...     tool_name="search",
    ...     arguments={"query": "invalid query"},
    ...     error=captured_traceback
    ... )
"""
import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from enum import Enum

from .traceback import CapturedTraceback, TracebackCapture

logger = logging.getLogger(__name__)


class CorrectionStrategy(Enum):
    """Strategy for handling errors."""
    RETRY_SAME = "retry_same"
    RETRY_MODIFIED = "retry_modified"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"


@dataclass
class CorrectionPrompt:
    """Prompt template for LLM-based error correction.

    Attributes:
        system_prompt: Base system prompt for the correction LLM.
        error_template: Template for formatting error information.
    """
    system_prompt: str = """
You are a code correction assistant specializing in fixing tool call errors.
Analyze the error and provide corrected arguments for the tool.

Guidelines:
1. Examine the error traceback carefully
2. Identify what went wrong (type mismatch, invalid value, missing parameter, etc.)
3. Provide corrected arguments that fix the issue
4. Keep the same tool name unless the error requires a different approach
5. Be conservative - only change what is necessary to fix the error
"""

    error_template: str = """
## Error Information

**Tool Name:** {tool_name}

**Error Type:** {error_type}
**Error Message:** {error_message}

**Traceback:**
{traceback}

**Previous Arguments:**
{arguments}

## Context
{context}

## Your Task

Analyze the error and provide corrected arguments. Return your response as JSON:
{{
    "corrected_arguments": {{...}},  // Corrected arguments for the tool
    "explanation": "...",            // Brief explanation of what was fixed
    "strategy": "retry_modified"     // Strategy: retry_same, retry_modified, fallback, skip, abort
}}
"""


@dataclass
class CorrectionResult:
    """Result of a correction attempt.

    Attributes:
        success: Whether the correction was successful.
        strategy: The strategy used for handling the error.
        corrected_arguments: The corrected tool arguments (if applicable).
        explanation: Explanation of the correction (if applicable).
        retry_count: Number of retry attempts made.
        original_error: String representation of the original error.
    """
    success: bool
    strategy: CorrectionStrategy
    corrected_arguments: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None
    retry_count: int = 0
    original_error: Optional[str] = None


class LLMCorrector:
    """LLM-based error correction using CodeAct pattern.

    Features:
    - Analyze execution errors
    - Generate corrected tool arguments
    - Support multiple correction strategies
    - Retry with backoff

    Example:
        >>> corrector = LLMCorrector(
        ...     llm_provider=mock_llm,
        ...     max_retries=3,
        ...     base_backoff=1.0,
        ...     max_backoff=30.0
        ... )
        >>>
        >>> # In an async context:
        >>> result = await corrector.correct(
        ...     tool_name="search",
        ...     arguments={"query": ""},
        ...     error=error_traceback
        ... )
    """

    def __init__(
        self,
        llm_provider: Any,
        max_retries: int = 3,
        base_backoff: float = 1.0,
        max_backoff: float = 30.0,
        timeout: float = 30.0
    ):
        """Initialize the LLMCorrector.

        Args:
            llm_provider: LLM provider interface with a generate method.
            max_retries: Maximum number of correction attempts.
            base_backoff: Base delay for exponential backoff in seconds.
            max_backoff: Maximum delay cap in seconds.
            timeout: Timeout for LLM generation in seconds.
        """
        self.llm_provider = llm_provider
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.timeout = timeout
        self._correction_count = 0
        self._prompt_template = CorrectionPrompt()

    async def correct(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        error: CapturedTraceback,
        context: Optional[Dict[str, Any]] = None
    ) -> CorrectionResult:
        """Attempt to correct an error using LLM.

        Args:
            tool_name: Name of the failed tool.
            arguments: Original tool arguments.
            error: Captured error information.
            context: Additional context for correction.

        Returns:
            CorrectionResult with corrected arguments or failure info.
        """
        logger.info(f"Attempting correction for tool '{tool_name}' (attempt {self._correction_count + 1})")

        for attempt in range(self.max_retries):
            try:
                # Generate correction
                correction = await self._generate_correction(
                    tool_name, arguments, error, context
                )

                if correction is None:
                    return CorrectionResult(
                        success=False,
                        strategy=CorrectionStrategy.ABORT,
                        original_error=str(error),
                        retry_count=attempt + 1
                    )

                # Parse the correction response
                result = self._parse_correction_response(correction, arguments)

                if result.success and result.corrected_arguments:
                    self._correction_count += 1
                    logger.info(
                        f"Correction successful for '{tool_name}': "
                        f"{result.explanation}"
                    )
                    return result

                # Apply backoff before retry
                if attempt < self.max_retries - 1 and self.max_backoff > 0:
                    await self._apply_backoff(attempt)

            except asyncio.TimeoutError:
                logger.warning(f"Correction attempt {attempt + 1} timed out")
                if attempt < self.max_retries - 1:
                    await self._apply_backoff(attempt)
                    continue
                return CorrectionResult(
                    success=False,
                    strategy=CorrectionStrategy.ABORT,
                    original_error="Correction timed out",
                    retry_count=attempt + 1
                )

            except Exception as e:
                logger.error(f"Correction attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await self._apply_backoff(attempt)
                    continue
                return CorrectionResult(
                    success=False,
                    strategy=CorrectionStrategy.ABORT,
                    original_error=str(e),
                    retry_count=attempt + 1
                )

        return CorrectionResult(
            success=False,
            strategy=CorrectionStrategy.ABORT,
            original_error="Max retries exceeded",
            retry_count=self.max_retries
        )

    async def _generate_correction(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        error: CapturedTraceback,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Generate corrected arguments via LLM.

        Args:
            tool_name: Name of the failed tool.
            arguments: Original tool arguments.
            error: Captured error information.
            context: Additional context for correction.

        Returns:
            LLM response string or None on failure.
        """
        prompt = self._build_prompt(tool_name, arguments, error, context)

        try:
            # Call the LLM provider
            if hasattr(self.llm_provider, 'generate'):
                if asyncio.iscoroutinefunction(self.llm_provider.generate):
                    response = await asyncio.wait_for(
                        self.llm_provider.generate(prompt),
                        timeout=self.timeout
                    )
                else:
                    response = self.llm_provider.generate(prompt)
            elif hasattr(self.llm_provider, 'chat'):
                # Support chat-based interfaces
                if asyncio.iscoroutinefunction(self.llm_provider.chat):
                    response = await asyncio.wait_for(
                        self.llm_provider.chat([{"role": "user", "content": prompt}]),
                        timeout=self.timeout
                    )
                else:
                    response = self.llm_provider.chat([{"role": "user", "content": prompt}])
            else:
                raise ValueError("LLM provider must have 'generate' or 'chat' method")

            return response

        except Exception as e:
            logger.error(f"Failed to generate correction: {e}")
            raise

    def _build_prompt(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        error: CapturedTraceback,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build correction prompt for LLM.

        Args:
            tool_name: Name of the failed tool.
            arguments: Original tool arguments.
            error: Captured error information.
            context: Additional context for correction.

        Returns:
            Formatted prompt string.
        """
        context_str = ""
        if context:
            context_parts = []
            for key, value in context.items():
                if not key.startswith("_"):  # Skip private fields
                    context_parts.append(f"- {key}: {value}")
            if context_parts:
                context_str = "\n".join(context_parts)

        return self._prompt_template.system_prompt + "\n\n" + self._prompt_template.error_template.format(
            tool_name=tool_name,
            error_type=error.error_type,
            error_message=error.error_message,
            traceback=error.traceback_str,
            arguments=json.dumps(arguments, indent=2, default=str),
            context=context_str or "No additional context provided."
        )

    def _parse_correction_response(
        self,
        response: str,
        original_arguments: Dict[str, Any]
    ) -> CorrectionResult:
        """Parse LLM correction response.

        Args:
            response: The LLM response string.
            original_arguments: Original arguments for fallback.

        Returns:
            Parsed CorrectionResult.
        """
        try:
            # Try to extract JSON from the response
            json_str = self._extract_json(response)
            if json_str:
                data = json.loads(json_str)

                strategy_str = data.get("strategy", "retry_modified")
                try:
                    strategy = CorrectionStrategy(strategy_str)
                except ValueError:
                    strategy = CorrectionStrategy.RETRY_MODIFIED

                return CorrectionResult(
                    success=True,
                    strategy=strategy,
                    corrected_arguments=data.get("corrected_arguments", original_arguments),
                    explanation=data.get("explanation")
                )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse correction response as JSON: {e}")

        # Fallback: return original arguments with retry_modified strategy
        return CorrectionResult(
            success=True,
            strategy=CorrectionStrategy.RETRY_MODIFIED,
            corrected_arguments=original_arguments,
            explanation="Could not parse correction; returning original arguments"
        )

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from LLM response text.

        Looks for JSON blocks (with or without markdown formatting).

        Args:
            text: The response text.

        Returns:
            Extracted JSON string or None.
        """
        import re

        # Try to find JSON block (with or without markdown)
        patterns = [
            r'```(?:json)?\s*(\{.*?\})\s*```',  # Markdown code blocks
            r'```(\{.*?\})```',
            r'(\{.*\})',  # Plain JSON object
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1)

        # Try finding array or object starting with { or [
        for start_char in ['{', '[']:
            idx = text.find(start_char)
            if idx != -1:
                # Find matching closing bracket
                bracket_count = 0
                end_idx = idx
                for i, char in enumerate(text[idx:], idx):
                    if char in '{[':
                        bracket_count += 1
                    elif char in '}]':
                        bracket_count -= 1
                    if bracket_count == 0:
                        end_idx = i + 1
                        break
                return text[idx:end_idx]

        return None

    async def _apply_backoff(self, retry_count: int) -> None:
        """Apply exponential backoff between retries.

        Args:
            retry_count: Current retry attempt number.
        """
        delay = min(self.base_backoff * (2 ** retry_count), self.max_backoff)
        logger.debug(f"Applying backoff: {delay}s before retry")
        await asyncio.sleep(delay)

    def get_correction_count(self) -> int:
        """Get total number of successful corrections made.

        Returns:
            Number of successful corrections.
        """
        return self._correction_count

    def reset(self) -> None:
        """Reset correction counter."""
        self._correction_count = 0
