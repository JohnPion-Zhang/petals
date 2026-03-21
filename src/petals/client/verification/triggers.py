"""
Verification Triggers - Configurable Verification Conditions

Defines when verification should be triggered based on various conditions
such as tool type, result size, error rates, or custom conditions.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Any
import logging

logger = logging.getLogger(__name__)


class TriggerType:
    """Types of verification triggers."""

    ALWAYS = "always"
    ON_FLAG = "on_flag"
    ON_ERROR_RATE = "on_error_rate"
    ON_SIZE = "on_size"
    ON_TYPE = "on_type"
    CUSTOM = "custom"


@dataclass
class VerificationTrigger:
    """Configuration for when to trigger verification.

    Defines a condition that, when met, triggers verification of a tool result.

    Attributes:
        trigger_type: Type of trigger (always, on_flag, on_error_rate, etc.).
        tool_names: Set of tool names this trigger applies to (empty = all tools).
        condition: Custom condition function (for CUSTOM type).
        error_threshold: Error rate threshold for ON_ERROR_RATE type.
        size_threshold: Result size threshold for ON_SIZE type.
        required_fields: Required fields for ON_TYPE type.
        max_verifications: Maximum verifications per run for this trigger.
    """

    trigger_type: str
    tool_names: Set[str] = field(default_factory=set)
    condition: Optional[Callable[[Any, Any], bool]] = None

    # Trigger-specific config
    error_threshold: float = 0.1
    size_threshold: int = 1000
    required_fields: List[str] = field(default_factory=list)
    max_verifications: int = 100

    # Internal state
    _trigger_count: int = field(default=0, repr=False)

    def should_trigger(
        self,
        tool_name: str,
        result: Any,
        error: Optional[Exception] = None,
        node: Optional[Any] = None,
    ) -> bool:
        """Check if verification should be triggered.

        Args:
            tool_name: Name of the tool.
            result: The tool execution result.
            error: Optional exception that occurred.
            node: Optional node for flag-based triggers.

        Returns:
            True if verification should be triggered.
        """
        # Check tool name filter
        if self.tool_names and tool_name not in self.tool_names:
            return False

        # Check max verifications
        if self._trigger_count >= self.max_verifications:
            return False

        # Check trigger type and get result
        triggered = False
        if self.trigger_type == TriggerType.ALWAYS:
            triggered = self._check_always(tool_name, result, error)

        elif self.trigger_type == TriggerType.ON_FLAG:
            triggered = self._check_on_flag(result, node)

        elif self.trigger_type == TriggerType.ON_ERROR_RATE:
            triggered = self._check_on_error_rate(error)

        elif self.trigger_type == TriggerType.ON_SIZE:
            triggered = self._check_on_size(result)

        elif self.trigger_type == TriggerType.ON_TYPE:
            triggered = self._check_on_type(result)

        elif self.trigger_type == TriggerType.CUSTOM:
            triggered = self._check_custom(result, error, node)

        # Increment count if triggered
        if triggered:
            self._increment_count()

        return triggered

    def _check_always(
        self,
        tool_name: str,
        result: Any,
        error: Optional[Exception],
    ) -> bool:
        """Check ALWAYS trigger type."""
        return True

    def _check_on_flag(
        self,
        result: Any,
        node: Optional[Any],
    ) -> bool:
        """Check ON_FLAG trigger type.

        Triggers when node has requires_verification flag.
        """
        if node is None:
            return False

        flag = getattr(node, "requires_verification", False)
        return bool(flag)

    def _check_on_error_rate(
        self,
        error: Optional[Exception],
    ) -> bool:
        """Check ON_ERROR_RATE trigger type.

        Triggers when error rate exceeds threshold.
        """
        # In a full implementation, this would track error rates
        # For now, we trigger if an error occurred
        return error is not None

    def _check_on_size(
        self,
        result: Any,
    ) -> bool:
        """Check ON_SIZE trigger type.

        Triggers when result size exceeds threshold.
        """
        if result is None:
            return False

        # Calculate size
        size = self._calculate_size(result)
        return size > self.size_threshold

    def _check_on_type(
        self,
        result: Any,
    ) -> bool:
        """Check ON_TYPE trigger type.

        Triggers when result has specific required fields.
        """
        if not self.required_fields:
            return False

        if not isinstance(result, dict):
            return False

        # Check if all required fields are present
        return all(field in result for field in self.required_fields)

    def _check_custom(
        self,
        result: Any,
        error: Optional[Exception],
        node: Optional[Any],
    ) -> bool:
        """Check CUSTOM trigger type.

        Triggers based on custom condition function.
        """
        if self.condition is None:
            return False

        try:
            return bool(self.condition(result, error, node))
        except Exception as e:
            logger.warning(f"Custom trigger condition raised exception: {e}")
            return False

    def _calculate_size(self, result: Any) -> int:
        """Calculate approximate size of result."""
        import sys

        try:
            return len(str(result))
        except Exception:
            return 0

    def _increment_count(self) -> None:
        """Increment trigger count."""
        self._trigger_count += 1

    def reset_count(self) -> None:
        """Reset trigger count."""
        self._trigger_count = 0

    @classmethod
    def always(cls, tool_names: Optional[Set[str]] = None) -> "VerificationTrigger":
        """Create an ALWAYS trigger for specified tools.

        Args:
            tool_names: Set of tool names to always verify (None = all tools).

        Returns:
            VerificationTrigger configured for always verification.
        """
        return cls(
            trigger_type=TriggerType.ALWAYS,
            tool_names=tool_names or set(),
        )

    @classmethod
    def on_flag(cls, flag: str = "requires_verification") -> "VerificationTrigger":
        """Create a trigger that fires when node has a specific flag.

        Args:
            flag: Name of the flag attribute to check.

        Returns:
            VerificationTrigger configured for flag-based verification.
        """
        return cls(
            trigger_type=TriggerType.ON_FLAG,
            condition=lambda result, error, node: getattr(node, flag, False) if node else False,
        )

    @classmethod
    def on_error_rate(
        cls,
        threshold: float = 0.1,
        tool_names: Optional[Set[str]] = None,
    ) -> "VerificationTrigger":
        """Create a trigger that fires when error rate exceeds threshold.

        Args:
            threshold: Error rate threshold (0.0 to 1.0).
            tool_names: Optional set of tool names to monitor.

        Returns:
            VerificationTrigger configured for error rate monitoring.
        """
        return cls(
            trigger_type=TriggerType.ON_ERROR_RATE,
            tool_names=tool_names or set(),
            error_threshold=threshold,
        )

    @classmethod
    def on_large_result(
        cls,
        size_threshold: int = 1000,
        tool_names: Optional[Set[str]] = None,
    ) -> "VerificationTrigger":
        """Create a trigger that fires when result size exceeds threshold.

        Args:
            size_threshold: Minimum size in characters to trigger verification.
            tool_names: Optional set of tool names to monitor.

        Returns:
            VerificationTrigger configured for size-based verification.
        """
        return cls(
            trigger_type=TriggerType.ON_SIZE,
            tool_names=tool_names or set(),
            size_threshold=size_threshold,
        )

    @classmethod
    def on_type(
        cls,
        required_fields: List[str],
        tool_names: Optional[Set[str]] = None,
    ) -> "VerificationTrigger":
        """Create a trigger that fires when result contains required fields.

        Args:
            required_fields: List of field names that must be present.
            tool_names: Optional set of tool names to monitor.

        Returns:
            VerificationTrigger configured for type-based verification.
        """
        return cls(
            trigger_type=TriggerType.ON_TYPE,
            tool_names=tool_names or set(),
            required_fields=required_fields,
        )

    @classmethod
    def custom(
        cls,
        condition: Callable[[Any, Any], bool],
        tool_names: Optional[Set[str]] = None,
    ) -> "VerificationTrigger":
        """Create a trigger with a custom condition function.

        Args:
            condition: Function that takes (result, error, node) and returns bool.
            tool_names: Optional set of tool names to apply trigger to.

        Returns:
            VerificationTrigger configured with custom condition.
        """
        return cls(
            trigger_type=TriggerType.CUSTOM,
            tool_names=tool_names or set(),
            condition=condition,
        )


@dataclass
class TriggerConfig:
    """Configuration for verification triggers.

    Manages a collection of triggers and provides methods to check
    if verification should be performed for a given tool/result.

    Attributes:
        triggers: List of VerificationTrigger configurations.
        default_level: Default verification level when triggered.
        max_verifications_per_run: Maximum total verifications per execution.
    """

    triggers: List[VerificationTrigger] = field(default_factory=list)
    default_level: str = "structural"  # "none", "basic", "structural", "deep"
    max_verifications_per_run: int = 100

    # Internal state
    _total_verifications: int = field(default=0, repr=False)
    _triggered_tools: Dict[str, int] = field(default_factory=dict)

    def add_trigger(self, trigger: VerificationTrigger) -> None:
        """Add a verification trigger.

        Args:
            trigger: The VerificationTrigger to add.
        """
        self.triggers.append(trigger)
        logger.debug(f"Added trigger: {trigger.trigger_type}")

    def remove_trigger(self, trigger_type: str) -> bool:
        """Remove a trigger by type.

        Args:
            trigger_type: Type of trigger to remove.

        Returns:
            True if a trigger was removed.
        """
        for i, trigger in enumerate(self.triggers):
            if trigger.trigger_type == trigger_type:
                self.triggers.pop(i)
                return True
        return False

    def should_verify(
        self,
        tool_name: str,
        result: Any,
        error: Optional[Exception] = None,
        node: Optional[Any] = None,
    ) -> bool:
        """Check if any trigger should cause verification.

        Args:
            tool_name: Name of the tool.
            result: The tool execution result.
            error: Optional exception that occurred.
            node: Optional node for flag-based triggers.

        Returns:
            True if verification should be performed.
        """
        # Check total limit
        if self._total_verifications >= self.max_verifications_per_run:
            return False

        # Check each trigger
        for trigger in self.triggers:
            if trigger.should_trigger(tool_name, result, error, node):
                self._total_verifications += 1
                self._triggered_tools[tool_name] = self._triggered_tools.get(tool_name, 0) + 1
                return True

        return False

    def get_triggered_count(self, tool_name: str) -> int:
        """Get number of times a tool has triggered verification.

        Args:
            tool_name: Name of the tool.

        Returns:
            Number of times this tool has triggered verification.
        """
        return self._triggered_tools.get(tool_name, 0)

    def get_total_triggered(self) -> int:
        """Get total number of verifications triggered.

        Returns:
            Total count of triggered verifications.
        """
        return self._total_verifications

    def reset_counts(self) -> None:
        """Reset all trigger counts."""
        self._total_verifications = 0
        self._triggered_tools.clear()
        for trigger in self.triggers:
            trigger.reset_count()
        logger.debug("Trigger counts reset")

    def get_stats(self) -> Dict[str, Any]:
        """Get trigger statistics.

        Returns:
            Dictionary with trigger statistics.
        """
        return {
            "total_verifications": self._total_verifications,
            "triggered_by_tool": dict(self._triggered_tools),
            "trigger_types": [t.trigger_type for t in self.triggers],
            "max_verifications": self.max_verifications_per_run,
            "default_level": self.default_level,
        }

    @classmethod
    def default_config(cls) -> "TriggerConfig":
        """Get default trigger configuration.

        Returns:
            TriggerConfig with sensible defaults:
            - Flag-based verification for nodes with requires_verification=True
            - Always verify critical tools (web_search, file_read, code_execute)
        """
        return cls(
            triggers=[
                VerificationTrigger.on_flag(),
                VerificationTrigger.always({"web_search", "file_read", "code_execute"}),
            ],
            default_level="structural",
        )

    @classmethod
    def strict_config(cls) -> "TriggerConfig":
        """Get strict trigger configuration.

        Returns:
            TriggerConfig with strict verification settings:
            - Verify all tools
            - Trigger on large results
        """
        return cls(
            triggers=[
                VerificationTrigger.always(),
                VerificationTrigger.on_large_result(size_threshold=100),
            ],
            default_level="deep",
            max_verifications_per_run=1000,
        )

    @classmethod
    def minimal_config(cls) -> "TriggerConfig":
        """Get minimal trigger configuration.

        Returns:
            TriggerConfig with minimal verification:
            - Only verify flagged nodes
        """
        return cls(
            triggers=[
                VerificationTrigger.on_flag(),
            ],
            default_level="basic",
        )
