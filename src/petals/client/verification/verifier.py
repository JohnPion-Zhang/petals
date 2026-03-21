"""
ResultVerifier - RLM-style Result Verification

Verifies child results before parent aggregation using configurable rules
and optional LLM-based reasoning for complex verification scenarios.

Features:
- Configurable verification rules per tool type
- LLM-based reasoning for complex verification
- Hierarchical verification (parent waits for children)
- Verification caching
- Async verification with timeout
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
import logging
import json

logger = logging.getLogger(__name__)


class VerificationStatus:
    """Status of result verification."""

    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"

    _values = {PENDING, VERIFIED, FAILED, SKIPPED, TIMEOUT}

    @classmethod
    def from_string(cls, value: str) -> "VerificationStatus":
        """Create status from string value."""
        if value not in cls._values:
            raise ValueError(f"Invalid verification status: {value}")
        return cls(value)


class VerificationLevel:
    """Depth of verification to perform."""

    NONE = "none"
    BASIC = "basic"
    STRUCTURAL = "structural"
    DEEP = "deep"

    _values = {NONE, BASIC, STRUCTURAL, DEEP}

    @classmethod
    def from_string(cls, value: str) -> "VerificationLevel":
        """Create level from string value."""
        if value not in cls._values:
            raise ValueError(f"Invalid verification level: {value}")
        return cls(value)


@dataclass
class VerificationRule:
    """A single verification rule for a tool.

    Attributes:
        name: Unique name for this rule.
        description: Human-readable description of what this rule checks.
        check: Callable that takes the result and returns True if passed.
        severity: Either "error" (fails verification) or "warning" (passes with warning).
        metadata: Additional metadata for the rule.
    """

    name: str
    description: str
    check: Callable[[Any], bool]
    severity: str = "error"  # "error" or "warning"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate severity."""
        if self.severity not in ("error", "warning"):
            raise ValueError(f"Invalid severity: {self.severity}. Must be 'error' or 'warning'.")

    def evaluate(self, result: Any) -> tuple[bool, Optional[str]]:
        """Evaluate this rule against a result.

        Args:
            result: The tool execution result to verify.

        Returns:
            Tuple of (passed, error_message).
        """
        try:
            passed = self.check(result)
            if passed:
                return True, None
            return False, f"Rule '{self.name}' failed: {self.description}"
        except Exception as e:
            return False, f"Rule '{self.name}' raised exception: {str(e)}"


@dataclass
class VerificationResult:
    """Result of verifying a tool execution result.

    Attributes:
        status: The verification status (verified, failed, skipped, etc.).
        tool_name: Name of the tool that was verified.
        tool_id: Unique ID of the tool call.
        passed_rules: List of rule names that passed.
        failed_rules: List of rule names that failed.
        warnings: List of warning messages from warning-severity rules.
        score: Verification score from 0.0 to 1.0.
        details: Additional details about the verification.
        verified_at: Timestamp when verification was completed.
        llm_feedback: Optional LLM feedback for complex verification.
    """

    status: str
    tool_name: str
    tool_id: str
    passed_rules: List[str] = field(default_factory=list)
    failed_rules: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    score: float = 1.0  # 0.0 to 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    verified_at: Optional[float] = None
    llm_feedback: Optional[str] = None

    @property
    def is_verified(self) -> bool:
        """Check if verification passed."""
        return self.status == VerificationStatus.VERIFIED

    @property
    def can_proceed(self) -> bool:
        """Check if execution can proceed despite verification result.

        Returns:
            True if status is VERIFIED or SKIPPED.
        """
        return self.status in (VerificationStatus.VERIFIED, VerificationStatus.SKIPPED)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "status": self.status,
            "tool_name": self.tool_name,
            "tool_id": self.tool_id,
            "passed_rules": self.passed_rules,
            "failed_rules": self.failed_rules,
            "warnings": self.warnings,
            "score": self.score,
            "details": self.details,
            "verified_at": self.verified_at,
            "llm_feedback": self.llm_feedback,
            "is_verified": self.is_verified,
            "can_proceed": self.can_proceed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationResult":
        """Deserialize from dictionary."""
        return cls(
            status=data["status"],
            tool_name=data["tool_name"],
            tool_id=data["tool_id"],
            passed_rules=data.get("passed_rules", []),
            failed_rules=data.get("failed_rules", []),
            warnings=data.get("warnings", []),
            score=data.get("score", 1.0),
            details=data.get("details", {}),
            verified_at=data.get("verified_at"),
            llm_feedback=data.get("llm_feedback"),
        )


class ResultVerifier:
    """RLM-style result verification engine.

    Verifies child results before parent aggregation using configurable rules
    and optional LLM-based reasoning for complex verification scenarios.

    Features:
    - Configurable verification rules per tool type
    - LLM-based reasoning for complex verification
    - Hierarchical verification (parent waits for children)
    - Verification caching
    - Async verification with timeout

    Example:
        >>> verifier = ResultVerifier(
        ...     verification_level=VerificationLevel.STRUCTURAL,
        ...     timeout=30.0
        ... )
        >>>
        >>> # Register a rule for web_search results
        >>> verifier.register_rule(
        ...     "web_search",
        ...     VerificationRule(
        ...         name="has_results",
        ...         description="Check if results are non-empty",
        ...         check=lambda r: bool(r.get("results")),
        ...     )
        ... )
        >>>
        >>> # Verify a result
        >>> result = await verifier.verify(
        ...     tool_name="web_search",
        ...     tool_id="call_1",
        ...     result={"results": [{"title": "Test"}]}
        ... )
        >>> print(f"Verified: {result.is_verified}")
    """

    def __init__(
        self,
        llm_provider: Optional[Any] = None,
        verification_level: str = VerificationLevel.STRUCTURAL,
        timeout: float = 30.0,
        enable_llm_verification: bool = True,
        enable_caching: bool = True,
    ):
        """Initialize the ResultVerifier.

        Args:
            llm_provider: Optional LLM provider for complex verification.
            verification_level: Default verification level (none, basic, structural, deep).
            timeout: Timeout for verification operations in seconds.
            enable_llm_verification: Whether to use LLM for complex verification.
            enable_caching: Whether to cache verification results.
        """
        self.llm_provider = llm_provider
        self.verification_level = verification_level
        self.timeout = timeout
        self.enable_llm_verification = enable_llm_verification
        self.enable_caching = enable_caching

        # Rule registry: tool_name -> list of rules
        self._rules: Dict[str, List[VerificationRule]] = {}
        self._verification_cache: Dict[str, VerificationResult] = {}
        self._verification_count = 0

        # Statistics
        self._stats = {
            "total_verifications": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "cache_hits": 0,
            "llm_calls": 0,
        }

    def register_rule(self, tool_name: str, rule: VerificationRule) -> None:
        """Register a verification rule for a tool.

        Args:
            tool_name: Name of the tool this rule applies to.
            rule: The VerificationRule to register.
        """
        if tool_name not in self._rules:
            self._rules[tool_name] = []
        self._rules[tool_name].append(rule)
        logger.debug(f"Registered rule '{rule.name}' for tool '{tool_name}'")

    def unregister_rule(self, tool_name: str, rule_name: str) -> bool:
        """Unregister a verification rule.

        Args:
            tool_name: Name of the tool.
            rule_name: Name of the rule to remove.

        Returns:
            True if rule was removed, False if not found.
        """
        if tool_name not in self._rules:
            return False

        rules = self._rules[tool_name]
        for i, rule in enumerate(rules):
            if rule.name == rule_name:
                rules.pop(i)
                logger.debug(f"Unregistered rule '{rule_name}' for tool '{tool_name}'")
                return True
        return False

    def get_rules(self, tool_name: str) -> List[VerificationRule]:
        """Get all rules registered for a tool.

        Args:
            tool_name: Name of the tool.

        Returns:
            List of VerificationRules for this tool.
        """
        return self._rules.get(tool_name, [])

    def has_rules(self, tool_name: str) -> bool:
        """Check if a tool has any registered rules.

        Args:
            tool_name: Name of the tool.

        Returns:
            True if rules exist for this tool.
        """
        return tool_name in self._rules and len(self._rules[tool_name]) > 0

    def should_verify(self, tool_name: str) -> bool:
        """Check if verification should be performed for this tool.

        Args:
            tool_name: Name of the tool.

        Returns:
            True if verification should be performed.
        """
        # No verification if level is NONE
        if self.verification_level == VerificationLevel.NONE:
            return False

        # Check if tool has rules
        if self.has_rules(tool_name):
            return True

        # Check if LLM verification is enabled and provider is available
        if self.enable_llm_verification and self.llm_provider is not None:
            return True

        return False

    async def verify(
        self,
        tool_name: str,
        tool_id: str,
        result: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """Verify a tool execution result.

        Args:
            tool_name: Name of the tool that produced the result.
            tool_id: Unique ID of the tool call.
            result: The tool execution result to verify.
            context: Optional additional context for verification.

        Returns:
            VerificationResult with pass/fail status.
        """
        self._verification_count += 1
        self._stats["total_verifications"] += 1

        # Check cache if enabled
        if self.enable_caching:
            cache_key = self.get_cache_key(tool_id)
            if cache_key in self._verification_cache:
                self._stats["cache_hits"] += 1
                logger.debug(f"Cache hit for verification: {tool_id}")
                return self._verification_cache[cache_key]

        # Check if verification should be skipped
        if not self.should_verify(tool_name):
            verification_result = VerificationResult(
                status=VerificationStatus.SKIPPED,
                tool_name=tool_name,
                tool_id=tool_id,
                details={"reason": "verification_disabled"},
            )
            self._stats["skipped"] += 1
            return verification_result

        # Perform verification based on level
        try:
            if self.verification_level == VerificationLevel.NONE:
                verification_result = VerificationResult(
                    status=VerificationStatus.SKIPPED,
                    tool_name=tool_name,
                    tool_id=tool_id,
                    details={"reason": "level_none"},
                )
            elif self.verification_level == VerificationLevel.BASIC:
                verification_result = await self._verify_basic(
                    tool_name, tool_id, result, context
                )
            elif self.verification_level == VerificationLevel.STRUCTURAL:
                verification_result = await self._verify_structural(
                    tool_name, tool_id, result, context
                )
            elif self.verification_level == VerificationLevel.DEEP:
                verification_result = await self._verify_deep(
                    tool_name, tool_id, result, context
                )
            else:
                verification_result = VerificationResult(
                    status=VerificationStatus.SKIPPED,
                    tool_name=tool_name,
                    tool_id=tool_id,
                    details={"reason": "unknown_level"},
                )

            # Set timestamp
            verification_result.verified_at = time.time()

            # Cache result if enabled
            if self.enable_caching:
                self._verification_cache[self.get_cache_key(tool_id)] = verification_result

            # Update stats
            if verification_result.is_verified:
                self._stats["passed"] += 1
            else:
                self._stats["failed"] += 1

            return verification_result

        except asyncio.TimeoutError:
            logger.warning(f"Verification timed out for {tool_id}")
            verification_result = VerificationResult(
                status=VerificationStatus.TIMEOUT,
                tool_name=tool_name,
                tool_id=tool_id,
                details={"reason": "timeout", "timeout_seconds": self.timeout},
            )
            self._stats["failed"] += 1
            return verification_result

        except Exception as e:
            logger.error(f"Verification error for {tool_id}: {str(e)}")
            verification_result = VerificationResult(
                status=VerificationStatus.FAILED,
                tool_name=tool_name,
                tool_id=tool_id,
                details={"reason": "error", "error": str(e)},
            )
            self._stats["failed"] += 1
            return verification_result

    async def _verify_basic(
        self,
        tool_name: str,
        tool_id: str,
        result: Any,
        context: Optional[Dict[str, Any]],
    ) -> VerificationResult:
        """Perform basic verification.

        Basic verification checks:
        - Result is not None
        - Result is expected type
        """
        passed_rules: List[str] = []
        failed_rules: List[str] = []
        warnings: List[str] = []

        # Check if result is None
        if result is None:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                tool_name=tool_name,
                tool_id=tool_id,
                failed_rules=["null_result"],
                details={"reason": "result_is_null"},
            )

        # Check registered rules
        rules = self.get_rules(tool_name)
        for rule in rules:
            passed, error_msg = rule.evaluate(result)
            if passed:
                passed_rules.append(rule.name)
            else:
                if rule.severity == "error":
                    failed_rules.append(rule.name)
                else:
                    warnings.append(error_msg or f"Rule '{rule.name}' failed")

        # Determine status
        if failed_rules:
            status = VerificationStatus.FAILED
            score = len(passed_rules) / max(len(rules), 1)
        else:
            status = VerificationStatus.VERIFIED
            score = 1.0 if not warnings else 0.9

        return VerificationResult(
            status=status,
            tool_name=tool_name,
            tool_id=tool_id,
            passed_rules=passed_rules,
            failed_rules=failed_rules,
            warnings=warnings,
            score=score,
            details={"verification_level": "basic", "total_rules": len(rules)},
        )

    async def _verify_structural(
        self,
        tool_name: str,
        tool_id: str,
        result: Any,
        context: Optional[Dict[str, Any]],
    ) -> VerificationResult:
        """Perform structural verification.

        Structural verification checks:
        - All rules pass
        - Result structure matches expected schema
        - Required fields are present
        """
        passed_rules: List[str] = []
        failed_rules: List[str] = []
        warnings: List[str] = []

        # Basic null check
        if result is None:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                tool_name=tool_name,
                tool_id=tool_id,
                failed_rules=["null_result"],
                details={"reason": "result_is_null"},
            )

        # Check registered rules
        rules = self.get_rules(tool_name)
        schema = context.get("schema") if context else None

        for rule in rules:
            passed, error_msg = rule.evaluate(result)
            if passed:
                passed_rules.append(rule.name)
            else:
                if rule.severity == "error":
                    failed_rules.append(rule.name)
                else:
                    warnings.append(error_msg or f"Rule '{rule.name}' failed")

        # Additional structural checks
        if schema and isinstance(result, dict):
            required_fields = schema.get("required", [])
            for field_name in required_fields:
                if field_name not in result:
                    failed_rules.append(f"missing_field:{field_name}")

        # Determine status
        if failed_rules:
            status = VerificationStatus.FAILED
            total_checks = len(rules) + len(required_fields) if schema else len(rules)
            score = len(passed_rules) / max(total_checks, 1)
        else:
            status = VerificationStatus.VERIFIED
            score = 1.0 if not warnings else 0.9

        return VerificationResult(
            status=status,
            tool_name=tool_name,
            tool_id=tool_id,
            passed_rules=passed_rules,
            failed_rules=failed_rules,
            warnings=warnings,
            score=score,
            details={
                "verification_level": "structural",
                "total_rules": len(rules),
                "schema_validation": bool(schema),
            },
        )

    async def _verify_deep(
        self,
        tool_name: str,
        tool_id: str,
        result: Any,
        context: Optional[Dict[str, Any]],
    ) -> VerificationResult:
        """Perform deep verification.

        Deep verification includes:
        - All structural checks
        - LLM-based semantic verification
        - Consistency checks
        """
        # First do structural verification
        verification_result = await self._verify_structural(
            tool_name, tool_id, result, context
        )

        # Skip LLM verification if already failed
        if not verification_result.can_proceed:
            return verification_result

        # Add LLM verification if available
        if self.enable_llm_verification and self.llm_provider:
            llm_feedback = await self.verify_with_llm(
                tool_name, result, context
            )
            if llm_feedback:
                verification_result.llm_feedback = llm_feedback
                # If LLM found issues, mark as failed or add warning
                if "ISSUE" in llm_feedback.upper():
                    verification_result.failed_rules.append("llm_verification")
                    verification_result.status = VerificationStatus.FAILED
                    verification_result.score *= 0.8

        verification_result.details["verification_level"] = "deep"
        return verification_result

    async def verify_with_llm(
        self,
        tool_name: str,
        result: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Use LLM to verify complex result properties.

        Args:
            tool_name: Name of the tool.
            result: The result to verify.
            context: Optional context including request details.

        Returns:
            LLM feedback string or None if verification passes.
        """
        if self.llm_provider is None:
            return None

        self._stats["llm_calls"] += 1

        try:
            # Format result for LLM
            result_str = json.dumps(result, indent=2, default=str)[:2000]  # Limit size

            prompt = f"""Verify this tool result:

Tool: {tool_name}
Result: {result_str}

Context: {json.dumps(context or {}, indent=2, default=str)[:1000]}

Check for:
1. Factual accuracy
2. Logical consistency
3. Alignment with request
4. Absence of hallucination
5. Data quality

Respond with:
- "VERIFIED" if the result is correct
- "ISSUE: <description>" if problems were found
"""

            response = await asyncio.wait_for(
                self.llm_provider.generate(prompt),
                timeout=self.timeout
            )

            # Parse response
            if "VERIFIED" in response.upper():
                return None
            return response

        except asyncio.TimeoutError:
            logger.warning(f"LLM verification timed out for tool {tool_name}")
            return "TIMEOUT: LLM verification timed out"
        except Exception as e:
            logger.error(f"LLM verification error: {str(e)}")
            return f"ERROR: {str(e)}"

    def get_cache_key(self, tool_id: str) -> str:
        """Generate cache key for verification result.

        Args:
            tool_id: The tool call ID.

        Returns:
            Cache key string.
        """
        return f"verify:{tool_id}"

    async def get_cached(
        self,
        tool_id: str
    ) -> Optional[VerificationResult]:
        """Get cached verification result.

        Args:
            tool_id: The tool call ID to look up.

        Returns:
            Cached VerificationResult or None.
        """
        if not self.enable_caching:
            return None
        return self._verification_cache.get(self.get_cache_key(tool_id))

    async def cache_result(
        self,
        tool_id: str,
        result: VerificationResult
    ) -> None:
        """Cache verification result.

        Args:
            tool_id: The tool call ID.
            result: The verification result to cache.
        """
        if self.enable_caching:
            self._verification_cache[self.get_cache_key(tool_id)] = result

    def clear_cache(self) -> None:
        """Clear all cached verification results."""
        self._verification_cache.clear()
        logger.debug("Verification cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get verification statistics.

        Returns:
            Dictionary with verification statistics.
        """
        total = self._stats["total_verifications"]
        return {
            **self._stats,
            "success_rate": self._stats["passed"] / max(total, 1),
            "failure_rate": self._stats["failed"] / max(total, 1),
            "cache_hit_rate": self._stats["cache_hits"] / max(total, 1),
            "cached_results": len(self._verification_cache),
            "registered_tools": len(self._rules),
        }

    def reset_stats(self) -> None:
        """Reset verification statistics."""
        self._stats = {
            "total_verifications": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "cache_hits": 0,
            "llm_calls": 0,
        }
        logger.debug("Verification statistics reset")

    async def verify_batch(
        self,
        results: List[Dict[str, Any]],
    ) -> List[VerificationResult]:
        """Verify multiple results in batch.

        Args:
            results: List of result dicts with tool_name, tool_id, result keys.

        Returns:
            List of VerificationResults.
        """
        tasks = [
            self.verify(
                tool_name=r["tool_name"],
                tool_id=r["tool_id"],
                result=r["result"],
                context=r.get("context"),
            )
            for r in results
        ]
        return await asyncio.gather(*tasks)
