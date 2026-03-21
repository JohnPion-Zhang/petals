"""
Structured Output - Schema validation and structured output enforcement.

This module provides tools for enforcing structured output formats,
including schema validation, type checking, and fallback handling
for invalid output.
"""
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Status of structured output validation."""

    VALID = "valid"
    """Output matches schema and all validations pass."""

    INVALID = "invalid"
    """Output does not match schema or validation failed."""

    PARTIAL = "partial"
    """Output partially matches schema with warnings."""


@dataclass
class ValidationResult:
    """Result of validating structured output.

    Attributes:
        status: Overall validation status.
        errors: List of validation error messages.
        warnings: List of non-critical warning messages.
        data: The validated/transformed data.
    """

    status: ValidationStatus
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    data: Optional[Any] = None

    @property
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return self.status == ValidationStatus.VALID

    @property
    def is_invalid(self) -> bool:
        """Check if validation failed."""
        return self.status == ValidationStatus.INVALID

    @property
    def is_partial(self) -> bool:
        """Check if validation is partial with warnings."""
        return self.status == ValidationStatus.PARTIAL

    def add_error(self, error: str) -> None:
        """Add a validation error.

        Args:
            error: Error message to add.
        """
        self.errors.append(error)
        if self.status == ValidationStatus.VALID:
            self.status = ValidationStatus.INVALID

    def add_warning(self, warning: str) -> None:
        """Add a validation warning.

        Args:
            warning: Warning message to add.
        """
        self.warnings.append(warning)
        if self.status == ValidationStatus.VALID:
            self.status = ValidationStatus.PARTIAL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all validation result fields.
        """
        return {
            "status": self.status.value,
            "errors": self.errors,
            "warnings": self.warnings,
            "data": self.data,
        }


@dataclass
class OutputSchema:
    """Schema definition for structured output.

    Defines required/optional fields, types, and custom validators
    for validating structured output data.

    Attributes:
        required_fields: List of required field names.
        optional_fields: List of optional field names.
        field_types: Dictionary mapping field names to expected types.
        validators: Dictionary mapping field names to validation functions.

    Example:
        >>> schema = OutputSchema(
        ...     required_fields=["url", "title"],
        ...     field_types={"url": str, "count": int},
        ...     validators={"count": lambda x: x > 0}
        ... )
        >>> result = schema.validate({"url": "https://...", "count": 5})
        >>> print(result.is_valid)
        True
    """

    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    field_types: Dict[str, type] = field(default_factory=dict)
    validators: Dict[str, Callable[[Any], bool]] = field(default_factory=dict)

    def validate(self, data: Any) -> ValidationResult:
        """Validate data against this schema.

        Args:
            data: The data to validate.

        Returns:
            ValidationResult with validation status and any errors/warnings.

        Example:
            >>> result = schema.validate({"url": "https://example.com"})
            >>> if result.is_valid:
            ...     print("Data is valid!")
        """
        result = ValidationResult(status=ValidationStatus.VALID, data=data)

        # Handle non-dict data
        if not isinstance(data, dict):
            result.add_error(f"Expected dict, got {type(data).__name__}")
            return result

        # Check required fields
        for field_name in self.required_fields:
            if field_name not in data:
                result.add_error(f"Missing required field: {field_name}")

        # Check for unknown fields (warning only)
        all_fields = set(self.required_fields) | set(self.optional_fields)
        for field_name in data:
            if field_name not in all_fields:
                result.add_warning(f"Unknown field: {field_name}")

        # Validate field types
        for field_name, expected_type in self.field_types.items():
            if field_name in data:
                value = data[field_name]
                if not isinstance(value, expected_type):
                    result.add_error(
                        f"Field '{field_name}' has wrong type: "
                        f"expected {expected_type.__name__}, got {type(value).__name__}"
                    )

        # Run custom validators
        for field_name, validator in self.validators.items():
            if field_name in data:
                value = data[field_name]
                try:
                    if not validator(value):
                        result.add_error(f"Field '{field_name}' failed validation")
                except Exception as e:
                    result.add_error(f"Field '{field_name}' validator raised: {e}")

        return result


class StructuredOutputEnforcer:
    """Enforces structured output format for reliable aggregation.

    Features:
    - Schema validation
    - Type checking
    - Custom validators
    - Fallback handling for invalid output

    Example:
        >>> enforcer = StructuredOutputEnforcer(strict=False)
        >>> enforcer.register_schema("search", OutputSchema(
        ...     required_fields=["results"],
        ...     field_types={"results": list}
        ... ))
        >>>
        >>> result = await enforcer.validate_and_extract(
        ...     "search",
        ...     {"results": [1, 2, 3]}
        ... )
        >>> print(result.is_valid)
        True
    """

    def __init__(
        self,
        default_schema: Optional[OutputSchema] = None,
        strict: bool = False,
        on_invalid: str = "warn"
    ) -> None:
        """Initialize the structured output enforcer.

        Args:
            default_schema: Default schema to use when no specific schema matches.
            strict: If True, treat unknown fields as errors instead of warnings.
            on_invalid: Action on invalid output ('warn', 'raise', 'fallback').
        """
        self.default_schema = default_schema
        self.strict = strict
        self.on_invalid = on_invalid
        self._schemas: Dict[str, OutputSchema] = {}

    def register_schema(self, name: str, schema: OutputSchema) -> None:
        """Register a named schema for tool output.

        Args:
            name: Name to associate with the schema (usually tool name).
            schema: The OutputSchema to register.

        Example:
            >>> schema = OutputSchema(required_fields=["data"])
            >>> enforcer.register_schema("my_tool", schema)
        """
        self._schemas[name] = schema
        logger.debug(f"Registered schema '{name}' with {len(schema.required_fields)} required fields")

    def get_schema(self, name: str) -> Optional[OutputSchema]:
        """Get a registered schema by name.

        Args:
            name: Name of the schema.

        Returns:
            The OutputSchema if found, None otherwise.
        """
        return self._schemas.get(name)

    async def validate_and_extract(
        self,
        tool_name: str,
        raw_output: Any,
        schema: Optional[OutputSchema] = None
    ) -> ValidationResult:
        """Validate and extract structured data from tool output.

        Args:
            tool_name: Name of the tool that produced the output.
            raw_output: The raw output from the tool.
            schema: Optional schema override.

        Returns:
            ValidationResult with validation status and extracted data.

        Example:
            >>> result = await enforcer.validate_and_extract(
            ...     "search",
            ...     {"results": [], "count": 0}
            ... )
            >>> if not result.is_valid:
            ...     print(f"Errors: {result.errors}")
        """
        # Determine which schema to use
        use_schema = schema or self._schemas.get(tool_name) or self.default_schema

        if use_schema is None:
            # No schema - just return the raw output as valid
            return ValidationResult(
                status=ValidationStatus.VALID,
                data=raw_output
            )

        # Validate
        result = use_schema.validate(raw_output)

        # Handle strict mode for unknown fields
        if self.strict and result.warnings:
            for warning in result.warnings:
                if "Unknown field" in warning:
                    result.errors.append(warning)
            result.warnings = [w for w in result.warnings if "Unknown field" not in w]

        # Handle invalid output based on on_invalid setting
        if result.is_invalid:
            if self.on_invalid == "raise":
                raise ValueError(f"Invalid output for '{tool_name}': {result.errors}")
            elif self.on_invalid == "fallback":
                result.data = await self._apply_fallback(tool_name, raw_output)

        return result

    async def enforce(
        self,
        raw_output: Any,
        schema: Optional[OutputSchema] = None
    ) -> Any:
        """Ensure output matches schema, applying fallback if needed.

        Args:
            raw_output: The raw output to enforce.
            schema: Optional schema to use.

        Returns:
            The validated/extracted data.

        Example:
            >>> result = await enforcer.enforce(
            ...     {"items": [1, 2, 3]},
            ...     my_schema
            ... )
        """
        result = await self.validate_and_extract(
            tool_name="__enforce__",
            raw_output=raw_output,
            schema=schema
        )
        return result.data

    async def _apply_fallback(
        self,
        tool_name: str,
        raw_output: Any
    ) -> Any:
        """Apply fallback handling for invalid output.

        Args:
            tool_name: Name of the tool.
            raw_output: The invalid raw output.

        Returns:
            Fallback value (often a normalized version of raw_output).
        """
        logger.warning(f"Applying fallback for '{tool_name}' output: {raw_output}")

        # Try to extract valid data from malformed output
        if isinstance(raw_output, dict):
            # Extract items field if present
            for key in ["data", "results", "items", "output"]:
                if key in raw_output:
                    return {key: raw_output[key], "_fallback": True}

        # Return wrapped raw output as last resort
        return {"raw": raw_output, "_fallback": True}

    @staticmethod
    def default_search_schema() -> OutputSchema:
        """Standard schema for search tool output.

        Returns:
            OutputSchema suitable for search results.

        Example:
            >>> schema = StructuredOutputEnforcer.default_search_schema()
            >>> # Expected fields: results (list), query (str), count (int)
        """
        return OutputSchema(
            required_fields=["results"],
            optional_fields=["query", "count", "next_page_token"],
            field_types={
                "results": list,
                "query": str,
                "count": int,
            },
            validators={
                "count": lambda x: x >= 0,
                "results": lambda x: all(isinstance(r, dict) for r in x) if x else True,
            }
        )

    @staticmethod
    def default_list_schema() -> OutputSchema:
        """Standard schema for list/query tool output.

        Returns:
            OutputSchema suitable for list-style results.

        Example:
            >>> schema = StructuredOutputEnforcer.default_list_schema()
            >>> # Expected fields: items (list), total (int, optional)
        """
        return OutputSchema(
            required_fields=["items"],
            optional_fields=["total", "offset", "limit"],
            field_types={
                "items": list,
                "total": int,
                "offset": int,
                "limit": int,
            },
            validators={
                "total": lambda x: x >= 0,
                "offset": lambda x: x >= 0,
                "limit": lambda x: x > 0,
            }
        )

    @staticmethod
    def default_key_value_schema() -> OutputSchema:
        """Standard schema for key-value/map output.

        Returns:
            OutputSchema suitable for key-value results.
        """
        return OutputSchema(
            required_fields=["data"],
            optional_fields=["count"],
            field_types={
                "data": dict,
                "count": int,
            },
        )

    def register_defaults(self) -> None:
        """Register default schemas for common tool types."""
        self._schemas["search"] = self.default_search_schema()
        self._schemas["list"] = self.default_list_schema()
        self._schemas["query"] = self.default_list_schema()
        self._schemas["map"] = self.default_key_value_schema()
        self._schemas["get"] = self.default_key_value_schema()

        logger.debug("Registered default schemas")
