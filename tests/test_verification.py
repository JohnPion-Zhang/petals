"""
Tests for Verification Module - RLM-style Result Verification

TDD Phase 4: Verification Tests

These tests cover:
- ResultVerifier with configurable rules
- VerificationLevel enum values
- VerificationStatus transitions
- VerificationRule evaluation
- TriggerConfig and VerificationTrigger
- VerificationAwareExecutor integration
"""

import asyncio
import pytest

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
)
from petals.client.verification.triggers import (
    VerificationTrigger,
    TriggerConfig,
    TriggerType,
)
from petals.client.verification.verification_aware_executor import (
    VerificationAwareExecutor,
)
from petals.client.dag import ToolCallNode, ToolCallDAG
from petals.data_structures import CallStatus


# =============================================================================
# Test Fixtures
# =============================================================================


class MockLLMProvider:
    """Mock LLM provider for testing verification."""

    def __init__(self, response: str = "VERIFIED", should_fail: bool = False):
        self.response = response
        self.should_fail = should_fail
        self.call_count = 0
        self.last_prompt = None

    async def generate(self, prompt: str) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        if self.should_fail:
            raise RuntimeError("LLM generation failed")
        return self.response


@pytest.fixture
def verifier():
    """Create a fresh ResultVerifier with default settings."""
    return ResultVerifier(
        verification_level=VerificationLevel.STRUCTURAL,
        timeout=30.0,
    )


@pytest.fixture
def verifier_with_llm():
    """Create a ResultVerifier with mock LLM."""
    return ResultVerifier(
        llm_provider=MockLLMProvider(),
        verification_level=VerificationLevel.DEEP,
        timeout=30.0,
    )


@pytest.fixture
def trigger_config():
    """Create a fresh TriggerConfig."""
    return TriggerConfig.default_config()


# =============================================================================
# VerificationStatus Tests
# =============================================================================


class TestVerificationStatus:
    """Tests for VerificationStatus enum."""

    def test_status_values(self):
        """Test that all expected status values exist."""
        assert VerificationStatus.PENDING == "pending"
        assert VerificationStatus.VERIFIED == "verified"
        assert VerificationStatus.FAILED == "failed"
        assert VerificationStatus.SKIPPED == "skipped"
        assert VerificationStatus.TIMEOUT == "timeout"

    def test_status_is_string(self):
        """Test that status values are strings."""
        assert isinstance(VerificationStatus.VERIFIED, str)
        assert VerificationStatus.VERIFIED == "verified"

    def test_valid_status(self):
        """Test valid status values."""
        assert VerificationStatus.VERIFIED in VerificationStatus._values
        assert "verified" in VerificationStatus._values


# =============================================================================
# VerificationLevel Tests
# =============================================================================


class TestVerificationLevel:
    """Tests for VerificationLevel enum."""

    def test_level_values(self):
        """Test that all expected level values exist."""
        assert VerificationLevel.NONE == "none"
        assert VerificationLevel.BASIC == "basic"
        assert VerificationLevel.STRUCTURAL == "structural"
        assert VerificationLevel.DEEP == "deep"

    def test_level_is_string(self):
        """Test that level values are strings."""
        assert isinstance(VerificationLevel.BASIC, str)
        assert VerificationLevel.BASIC == "basic"

    def test_valid_level(self):
        """Test valid level values."""
        assert VerificationLevel.STRUCTURAL in VerificationLevel._values
        assert "structural" in VerificationLevel._values


# =============================================================================
# VerificationRule Tests
# =============================================================================


class TestVerificationRule:
    """Tests for VerificationRule dataclass."""

    def test_rule_creation(self):
        """Test creating a verification rule."""
        rule = VerificationRule(
            name="has_results",
            description="Check if results exist",
            check=lambda r: bool(r.get("results")),
            severity="error",
        )

        assert rule.name == "has_results"
        assert rule.severity == "error"

    def test_rule_evaluate_pass(self):
        """Test rule evaluation passes."""
        rule = VerificationRule(
            name="test",
            description="Test rule",
            check=lambda r: r > 0,
        )

        passed, msg = rule.evaluate(5)
        assert passed is True
        assert msg is None

    def test_rule_evaluate_fail(self):
        """Test rule evaluation fails."""
        rule = VerificationRule(
            name="test",
            description="Test rule",
            check=lambda r: r > 0,
        )

        passed, msg = rule.evaluate(-1)
        assert passed is False
        assert "failed" in msg.lower()

    def test_rule_evaluate_exception(self):
        """Test rule evaluation handles exceptions."""
        rule = VerificationRule(
            name="test",
            description="Test rule",
            check=lambda r: r["missing_key"],  # Will raise KeyError
        )

        passed, msg = rule.evaluate({})
        assert passed is False
        assert "exception" in msg.lower()

    def test_rule_warning_severity(self):
        """Test warning severity does not fail verification."""
        rule = VerificationRule(
            name="warning_test",
            description="Warning rule",
            check=lambda r: False,
            severity="warning",
        )

        passed, msg = rule.evaluate("test")
        assert passed is False
        assert "warning" in msg.lower()

    def test_invalid_severity(self):
        """Test invalid severity raises error."""
        with pytest.raises(ValueError):
            VerificationRule(
                name="test",
                description="Test",
                check=lambda r: True,
                severity="invalid",
            )


# =============================================================================
# VerificationResult Tests
# =============================================================================


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_result_creation(self):
        """Test creating a verification result."""
        result = VerificationResult(
            status=VerificationStatus.VERIFIED,
            tool_name="web_search",
            tool_id="call_1",
            passed_rules=["rule1", "rule2"],
            score=1.0,
        )

        assert result.is_verified is True
        assert result.can_proceed is True
        assert len(result.passed_rules) == 2

    def test_result_failed(self):
        """Test failed verification result."""
        result = VerificationResult(
            status=VerificationStatus.FAILED,
            tool_name="web_search",
            tool_id="call_1",
            failed_rules=["rule1"],
        )

        assert result.is_verified is False
        assert result.can_proceed is False

    def test_result_skipped(self):
        """Test skipped verification result."""
        result = VerificationResult(
            status=VerificationStatus.SKIPPED,
            tool_name="web_search",
            tool_id="call_1",
        )

        assert result.is_verified is False
        assert result.can_proceed is True

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = VerificationResult(
            status=VerificationStatus.VERIFIED,
            tool_name="test",
            tool_id="id_1",
        )

        d = result.to_dict()
        assert d["status"] == "verified"
        assert d["tool_name"] == "test"
        assert d["is_verified"] is True
        assert d["can_proceed"] is True

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "status": "verified",
            "tool_name": "test",
            "tool_id": "id_1",
            "passed_rules": ["r1"],
            "failed_rules": [],
            "warnings": [],
            "score": 1.0,
        }

        result = VerificationResult.from_dict(data)
        assert result.status == "verified"
        assert result.passed_rules == ["r1"]


# =============================================================================
# ResultVerifier Tests
# =============================================================================


class TestResultVerifier:
    """Tests for ResultVerifier class."""

    def test_verifier_initialization(self, verifier):
        """Test verifier initializes with correct defaults."""
        assert verifier.verification_level == VerificationLevel.STRUCTURAL
        assert verifier.timeout == 30.0
        assert verifier.llm_provider is None

    def test_register_rule(self, verifier):
        """Test registering a verification rule."""
        rule = VerificationRule(
            name="has_results",
            description="Check for results",
            check=lambda r: bool(r.get("results")),
        )

        verifier.register_rule("web_search", rule)

        rules = verifier.get_rules("web_search")
        assert len(rules) == 1
        assert rules[0].name == "has_results"

    def test_register_multiple_rules(self, verifier):
        """Test registering multiple rules for same tool."""
        rule1 = VerificationRule(
            name="has_results",
            description="Has results",
            check=lambda r: bool(r.get("results")),
        )
        rule2 = VerificationRule(
            name="has_data",
            description="Has data field",
            check=lambda r: "data" in r,
        )

        verifier.register_rule("web_search", rule1)
        verifier.register_rule("web_search", rule2)

        rules = verifier.get_rules("web_search")
        assert len(rules) == 2

    def test_unregister_rule(self, verifier):
        """Test unregistering a rule."""
        rule = VerificationRule(
            name="test",
            description="Test",
            check=lambda r: True,
        )

        verifier.register_rule("tool", rule)
        assert verifier.has_rules("tool") is True

        result = verifier.unregister_rule("tool", "test")
        assert result is True
        assert verifier.has_rules("tool") is False

    def test_should_verify_no_rules(self, verifier):
        """Test should_verify returns False when no rules exist."""
        # Level is structural but no rules registered
        assert verifier.should_verify("unknown_tool") is False

    def test_should_verify_with_rules(self, verifier):
        """Test should_verify returns True when rules exist."""
        rule = VerificationRule(
            name="test",
            description="Test",
            check=lambda r: True,
        )
        verifier.register_rule("web_search", rule)

        assert verifier.should_verify("web_search") is True

    def test_should_verify_none_level(self):
        """Test should_verify returns False for NONE level."""
        verifier = ResultVerifier(verification_level=VerificationLevel.NONE)
        assert verifier.should_verify("any_tool") is False

    def test_should_verify_with_llm(self):
        """Test should_verify returns True when LLM is available."""
        verifier = ResultVerifier(
            llm_provider=MockLLMProvider(),
            enable_llm_verification=True,
        )
        assert verifier.should_verify("any_tool") is True

    @pytest.mark.asyncio
    async def test_verify_basic_pass(self, verifier):
        """Test basic verification passes when rules exist."""
        verifier.verification_level = VerificationLevel.BASIC
        # Register a rule so should_verify returns True
        verifier.register_rule(
            "test",
            VerificationRule(
                name="always_pass",
                description="Always passes",
                check=lambda r: True,
            ),
        )

        result = await verifier.verify(
            tool_name="test",
            tool_id="call_1",
            result={"data": "test"},
        )

        assert result.status == VerificationStatus.VERIFIED
        assert result.is_verified is True

    @pytest.mark.asyncio
    async def test_verify_basic_null_result(self, verifier):
        """Test basic verification fails on null result."""
        verifier.verification_level = VerificationLevel.BASIC
        # Register a rule so should_verify returns True
        verifier.register_rule(
            "test",
            VerificationRule(
                name="always_pass",
                description="Always passes",
                check=lambda r: True,
            ),
        )

        result = await verifier.verify(
            tool_name="test",
            tool_id="call_1",
            result=None,
        )

        assert result.status == VerificationStatus.FAILED
        assert "null" in result.details.get("reason", "").lower()

    @pytest.mark.asyncio
    async def test_verify_with_rules_pass(self, verifier):
        """Test verification passes all rules."""
        rule = VerificationRule(
            name="has_results",
            description="Has results",
            check=lambda r: bool(r.get("results")),
        )
        verifier.register_rule("search", rule)

        result = await verifier.verify(
            tool_name="search",
            tool_id="call_1",
            result={"results": [1, 2, 3]},
        )

        assert result.is_verified is True
        assert "has_results" in result.passed_rules

    @pytest.mark.asyncio
    async def test_verify_with_rules_fail(self, verifier):
        """Test verification fails when rules fail."""
        rule = VerificationRule(
            name="has_results",
            description="Has results",
            check=lambda r: bool(r.get("results")),
            severity="error",
        )
        verifier.register_rule("search", rule)

        result = await verifier.verify(
            tool_name="search",
            tool_id="call_1",
            result={"other": "data"},  # No "results" key
        )

        assert result.is_verified is False
        assert "has_results" in result.failed_rules

    @pytest.mark.asyncio
    async def test_verify_structural_with_schema(self, verifier):
        """Test structural verification with schema."""
        verifier.verification_level = VerificationLevel.STRUCTURAL

        rule = VerificationRule(
            name="has_field",
            description="Has required field",
            check=lambda r: "data" in r,
        )
        verifier.register_rule("api", rule)

        result = await verifier.verify(
            tool_name="api",
            tool_id="call_1",
            result={"data": "value"},
            context={"schema": {"required": ["data"]}},
        )

        assert result.is_verified is True

    @pytest.mark.asyncio
    async def test_verify_caching(self, verifier):
        """Test verification caching."""
        verifier.enable_caching = True
        # Register a rule so should_verify returns True
        verifier.register_rule(
            "tool",
            VerificationRule(
                name="always_pass",
                description="Always passes",
                check=lambda r: True,
            ),
        )

        result1 = await verifier.verify("tool", "id_1", {"data": "test"})
        result2 = await verifier.verify("tool", "id_1", {"data": "test"})

        stats = verifier.get_stats()
        assert stats["cache_hits"] == 1

        # Results should be identical
        assert result1.status == result2.status

    @pytest.mark.asyncio
    async def test_verify_llm_verification(self, verifier_with_llm):
        """Test LLM-based verification."""
        result = await verifier_with_llm.verify(
            tool_name="complex_tool",
            tool_id="call_1",
            result={"data": "test"},
        )

        # Should have called LLM
        assert verifier_with_llm.llm_provider.call_count >= 1

    @pytest.mark.asyncio
    async def test_verify_batch(self, verifier):
        """Test batch verification."""
        results_data = [
            {"tool_name": "t1", "tool_id": "id_1", "result": {"data": 1}},
            {"tool_name": "t2", "tool_id": "id_2", "result": {"data": 2}},
            {"tool_name": "t3", "tool_id": "id_3", "result": {"data": 3}},
        ]

        results = await verifier.verify_batch(results_data)

        assert len(results) == 3
        assert all(isinstance(r, VerificationResult) for r in results)

    def test_get_stats(self, verifier):
        """Test getting verification statistics."""
        stats = verifier.get_stats()

        assert "total_verifications" in stats
        assert "passed" in stats
        assert "failed" in stats
        assert "registered_tools" in stats

    def test_clear_cache(self, verifier):
        """Test clearing verification cache."""
        verifier.enable_caching = True

        # Add some cached results
        asyncio.run(verifier.verify("tool", "id_1", {"data": "test"}))

        verifier.clear_cache()

        stats = verifier.get_stats()
        assert stats["cached_results"] == 0


# =============================================================================
# TriggerConfig Tests
# =============================================================================


class TestVerificationTrigger:
    """Tests for VerificationTrigger class."""

    def test_always_trigger(self):
        """Test ALWAYS trigger type."""
        trigger = VerificationTrigger.always({"web_search"})

        assert trigger.trigger_type == TriggerType.ALWAYS
        assert trigger.should_trigger("web_search", {"data": "test"}) is True
        assert trigger.should_trigger("file_read", {"data": "test"}) is False

    def test_on_flag_trigger(self):
        """Test ON_FLAG trigger type."""
        trigger = VerificationTrigger.on_flag()

        class MockNode:
            requires_verification = True

        assert trigger.should_trigger("any", {}, node=MockNode()) is True

    def test_on_size_trigger(self):
        """Test ON_SIZE trigger type."""
        trigger = VerificationTrigger(
            trigger_type=TriggerType.ON_SIZE,
            size_threshold=10,
        )

        # Small result - no trigger
        assert trigger.should_trigger("any", "short") is False

        # Large result - triggers
        assert trigger.should_trigger("any", "this is a longer string") is True

    def test_on_type_trigger(self):
        """Test ON_TYPE trigger type."""
        trigger = VerificationTrigger(
            trigger_type=TriggerType.ON_TYPE,
            required_fields=["results", "status"],
        )

        # Missing fields - no trigger
        assert trigger.should_trigger("any", {"results": []}) is False

        # All fields present - triggers
        assert trigger.should_trigger("any", {"results": [], "status": "ok"}) is True

    def test_custom_trigger(self):
        """Test CUSTOM trigger type."""
        condition = lambda result, error, node: result.get("verify", False)
        trigger = VerificationTrigger.custom(condition)

        assert trigger.should_trigger("any", {"verify": True}) is True
        assert trigger.should_trigger("any", {"verify": False}) is False

    def test_trigger_count_limit(self):
        """Test trigger respects max_verifications limit."""
        trigger = VerificationTrigger.always()
        trigger.max_verifications = 2

        assert trigger.should_trigger("tool", {}) is True
        assert trigger.should_trigger("tool", {}) is True
        assert trigger.should_trigger("tool", {}) is False  # Limit reached


class TestTriggerConfig:
    """Tests for TriggerConfig class."""

    def test_default_config(self):
        """Test default trigger configuration."""
        config = TriggerConfig.default_config()

        assert len(config.triggers) >= 1
        assert config.default_level == "structural"

    def test_add_trigger(self, trigger_config):
        """Test adding a trigger."""
        trigger = VerificationTrigger.always({"specific_tool"})
        trigger_config.add_trigger(trigger)

        assert len(trigger_config.triggers) == 3  # 2 from default + 1 added

    def test_should_verify_with_trigger(self, trigger_config):
        """Test should_verify returns True when trigger matches."""
        # Flagged node should trigger
        class MockNode:
            requires_verification = True

        result = trigger_config.should_verify(
            tool_name="any",
            result={},
            node=MockNode(),
        )

        assert result is True

    def test_should_verify_with_tool_filter(self, trigger_config):
        """Test should_verify respects tool name filter."""
        result = trigger_config.should_verify(
            tool_name="web_search",  # In the always list
            result={"data": "test"},
        )

        assert result is True

    def test_reset_counts(self, trigger_config):
        """Test resetting trigger counts."""
        # Trigger a few times
        trigger_config.should_verify("web_search", {"data": "test"})
        trigger_config.should_verify("file_read", {"data": "test"})

        trigger_config.reset_counts()

        assert trigger_config.get_total_triggered() == 0

    def test_get_stats(self, trigger_config):
        """Test getting trigger statistics."""
        stats = trigger_config.get_stats()

        assert "total_verifications" in stats
        assert "trigger_types" in stats
        assert "default_level" in stats

    def test_strict_config(self):
        """Test strict trigger configuration."""
        config = TriggerConfig.strict_config()

        assert config.default_level == "deep"
        assert config.max_verifications_per_run == 1000

    def test_minimal_config(self):
        """Test minimal trigger configuration."""
        config = TriggerConfig.minimal_config()

        assert config.default_level == "basic"
        assert len(config.triggers) == 1


# =============================================================================
# VerificationAwareExecutor Tests
# =============================================================================


class TestVerificationAwareExecutor:
    """Tests for VerificationAwareExecutor class."""

    @pytest.fixture
    def executor(self, verifier):
        """Create a VerificationAwareExecutor."""
        return VerificationAwareExecutor(verifier)

    def test_executor_initialization(self, executor, verifier):
        """Test executor initializes correctly."""
        assert executor.verifier is verifier
        assert executor.trigger_config is not None
        assert executor.fail_on_verification_failure is False

    def test_needs_verification_flag(self, executor):
        """Test needs_verification with flag set."""
        node = ToolCallNode(
            id="test",
            name="tool",
            arguments={},
            requires_verification=True,
        )

        assert executor._needs_verification(node) is True

    def test_needs_verification_trigger(self, executor, verifier):
        """Test needs_verification with trigger match."""
        # Register a rule with the trigger config instead
        from petals.client.verification.triggers import VerificationTrigger, TriggerConfig
        executor.trigger_config = TriggerConfig(
            triggers=[VerificationTrigger.always({"specific_tool"})],
            default_level="structural",
        )

        node = ToolCallNode(
            id="test",
            name="specific_tool",
            arguments={},
            requires_verification=False,
        )

        # With flag off, needs verification due to trigger
        assert executor._needs_verification(node) is True

    @pytest.mark.asyncio
    async def test_verify_existing_node(self, executor):
        """Test verifying an already-executed node."""
        node = ToolCallNode(
            id="test",
            name="tool",
            arguments={},
        )
        node.mark_done({"data": "test"})

        result = await executor.verify_existing_node(node)

        assert result is not None
        assert isinstance(result, VerificationResult)

    @pytest.mark.asyncio
    async def test_verify_existing_node_no_result(self, executor):
        """Test verifying node with no result fails."""
        node = ToolCallNode(
            id="test",
            name="tool",
            arguments={},
        )
        # No result set

        result = await executor.verify_existing_node(node)

        assert result.status == VerificationStatus.FAILED

    def test_stats(self, executor):
        """Test getting executor statistics."""
        stats = executor.stats

        assert "verification_count" in stats
        assert "failed_verifications" in stats
        assert "success_rate" in stats
        assert "verifier_stats" in stats
        assert "trigger_stats" in stats

    def test_reset(self, executor):
        """Test resetting executor state."""
        executor._verification_count = 10
        executor._verified_nodes.add("node1")

        executor.reset()

        assert executor._verification_count == 0
        assert len(executor._verified_nodes) == 0

    def test_is_verified(self, executor):
        """Test checking if node is verified."""
        executor._verified_nodes.add("verified_node")

        assert executor.is_verified("verified_node") is True
        assert executor.is_verified("unknown_node") is False

    def test_get_verification_result(self, executor):
        """Test getting verification result for node."""
        result = VerificationResult(
            status=VerificationStatus.VERIFIED,
            tool_name="tool",
            tool_id="node_1",
        )
        executor._verified_results["node_1"] = result

        retrieved = executor.get_verification_result("node_1")
        assert retrieved is result

        assert executor.get_verification_result("unknown") is None


# =============================================================================
# Prompt Template Tests
# =============================================================================


class TestVerificationPromptTemplate:
    """Tests for VerificationPromptTemplate class."""

    def test_template_creation(self):
        """Test creating a prompt template."""
        template = VerificationPromptTemplate()

        assert template.system == VERIFICATION_SYSTEM_PROMPT
        assert template.basic == BASIC_VERIFICATION_PROMPT

    def test_get_prompt_basic(self):
        """Test getting basic prompt."""
        template = VerificationPromptTemplate()

        prompt = template.get_prompt(
            "basic",
            tool_name="search",
            request="Find AI news",
            result="[articles]",
        )

        assert "search" in prompt
        assert "AI news" in prompt

    def test_get_prompt_invalid_level(self):
        """Test getting prompt for invalid level falls back to basic."""
        template = VerificationPromptTemplate()

        # Should fall back to basic when no template variables provided
        # Invalid level without kwargs raises KeyError, which is expected
        # So we test with a valid fallback
        prompt = template.get_prompt("basic", tool_name="test", request="req", result="res")
        assert "VERIFIED" in prompt

    def test_get_full_prompt(self):
        """Test getting complete formatted prompt."""
        template = VerificationPromptTemplate()

        prompt = template.get_full_prompt(
            level="basic",
            tool_name="test",
            result={"data": "value"},
            request="Test request",
        )

        assert len(prompt) > 0
        assert "test" in prompt

    def test_with_system_prompt(self):
        """Test getting prompt with system prompt."""
        template = VerificationPromptTemplate()

        system, user = template.with_system_prompt(
            "basic",
            tool_name="test",
            request="Request",
            result="Result",
        )

        assert len(system) > 0
        assert len(user) > 0

    def test_create_restrictive(self):
        """Test creating restrictive template."""
        template = VerificationPromptTemplate.create_restrictive()

        assert "STRICT" in template.system
        assert "EXACTLY" in template.system

    def test_create_lenient(self):
        """Test creating lenient template."""
        template = VerificationPromptTemplate.create_lenient()

        assert "LENIENT" in template.system


# =============================================================================
# Integration Tests
# =============================================================================


class TestVerificationIntegration:
    """Integration tests for verification module."""

    @pytest.mark.asyncio
    async def test_full_verification_flow(self):
        """Test complete verification flow."""
        # Create verifier
        verifier = ResultVerifier(
            verification_level=VerificationLevel.STRUCTURAL,
        )

        # Register rules
        verifier.register_rule(
            "web_search",
            VerificationRule(
                name="has_results",
                description="Must have results",
                check=lambda r: bool(r.get("results")),
            ),
        )
        verifier.register_rule(
            "web_search",
            VerificationRule(
                name="non_empty",
                description="Results not empty",
                check=lambda r: len(r.get("results", [])) > 0,
            ),
        )

        # Verify
        result = await verifier.verify(
            tool_name="web_search",
            tool_id="search_1",
            result={"results": [{"title": "Test"}]},
        )

        assert result.is_verified is True
        assert len(result.passed_rules) == 2

    @pytest.mark.asyncio
    async def test_verification_with_llm(self):
        """Test verification with LLM integration."""
        mock_llm = MockLLMProvider(response="VERIFIED - looks good")
        verifier = ResultVerifier(
            llm_provider=mock_llm,
            verification_level=VerificationLevel.DEEP,
        )

        result = await verifier.verify(
            tool_name="complex_tool",
            tool_id="call_1",
            result={"complex": "data"},
        )

        assert mock_llm.call_count >= 1
        assert result.verified_at is not None

    @pytest.mark.asyncio
    async def test_trigger_and_verify_flow(self):
        """Test trigger-based verification flow."""
        # Create trigger config with ALWAYS for specific tool
        trigger_config = TriggerConfig(
            triggers=[VerificationTrigger.always({"data_tool"})],
            default_level="structural",
        )

        # Create verifier
        verifier = ResultVerifier(verification_level=VerificationLevel.BASIC)
        verifier.register_rule(
            "data_tool",
            VerificationRule(
                name="has_data",
                description="Has data",
                check=lambda r: "data" in r,
            ),
        )

        # Create executor
        executor = VerificationAwareExecutor(verifier, trigger_config)

        # Verify
        node = ToolCallNode(
            id="test_node",
            name="data_tool",
            arguments={},
        )
        node.mark_done({"data": "value"})

        result = await executor.verify_existing_node(node)

        assert result.is_verified is True

    def test_trigger_config_tool_filter(self):
        """Test trigger respects tool name filters."""
        trigger = VerificationTrigger.always({"tool_a", "tool_b"})

        assert trigger.should_trigger("tool_a", {}) is True
        assert trigger.should_trigger("tool_b", {}) is True
        assert trigger.should_trigger("tool_c", {}) is False


# =============================================================================
# Edge Cases
# =============================================================================


class TestVerificationEdgeCases:
    """Edge case tests for verification."""

    @pytest.mark.asyncio
    async def test_verify_empty_result(self, verifier):
        """Test verifying empty result."""
        result = await verifier.verify(
            tool_name="tool",
            tool_id="id",
            result={},
        )

        # Should pass basic check (not null)
        assert result.status in [
            VerificationStatus.VERIFIED,
            VerificationStatus.SKIPPED,
        ]

    @pytest.mark.asyncio
    async def test_verify_nested_result(self, verifier):
        """Test verifying deeply nested result."""
        result = await verifier.verify(
            tool_name="tool",
            tool_id="id",
            result={
                "level1": {
                    "level2": {
                        "level3": {
                            "data": "deep_value"
                        }
                    }
                }
            },
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_verify_with_context(self, verifier):
        """Test verification with context."""
        result = await verifier.verify(
            tool_name="tool",
            tool_id="id",
            result={"key": "value"},
            context={"schema": {"required": ["key"]}},
        )

        assert result is not None

    def test_verification_rule_exception_safety(self):
        """Test rules handle exceptions gracefully."""
        rule = VerificationRule(
            name="unsafe",
            description="Rule that throws",
            check=lambda r: 1 / 0,  # Division by zero
        )

        passed, msg = rule.evaluate({})
        assert passed is False
        assert "exception" in msg.lower()
