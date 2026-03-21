"""
Verification Module - RLM-style Result Verification

This module provides RLM (Recursive Language Model) style verification
for tool execution results before parent aggregation.

Components:
- ResultVerifier: Core verification engine with configurable rules
- VerificationTrigger: Configurable conditions for when to verify
- VerificationPromptTemplate: LLM prompt templates for complex verification
- VerificationAwareExecutor: DAG executor with integrated verification

Example:
    >>> from petals.client.verification import ResultVerifier, VerificationLevel
    >>> verifier = ResultVerifier(verification_level=VerificationLevel.STRUCTURAL)
    >>> result = await verifier.verify("search", "call_1", {"results": [...]})
    >>> if result.is_verified:
    ...     print("Result passed verification")
"""

from petals.client.verification.verifier import (
    ResultVerifier,
    VerificationResult,
    VerificationRule,
    VerificationStatus,
    VerificationLevel,
)
from petals.client.verification.prompts import (
    VerificationPromptTemplate,
    VERIFICATION_SYSTEM_PROMPT,
    BASIC_VERIFICATION_PROMPT,
    STRUCTURAL_VERIFICATION_PROMPT,
    DEEP_VERIFICATION_PROMPT,
    CHILD_AGGREGATION_PROMPT,
)
from petals.client.verification.triggers import (
    VerificationTrigger,
    TriggerConfig,
    TriggerType,
)
from petals.client.verification.verification_aware_executor import (
    VerificationAwareExecutor,
)

__all__ = [
    # Core verifier
    "ResultVerifier",
    "VerificationResult",
    "VerificationRule",
    "VerificationStatus",
    "VerificationLevel",
    # Prompts
    "VerificationPromptTemplate",
    "VERIFICATION_SYSTEM_PROMPT",
    "BASIC_VERIFICATION_PROMPT",
    "STRUCTURAL_VERIFICATION_PROMPT",
    "DEEP_VERIFICATION_PROMPT",
    "CHILD_AGGREGATION_PROMPT",
    # Triggers
    "VerificationTrigger",
    "TriggerConfig",
    "TriggerType",
    # Executor
    "VerificationAwareExecutor",
]
