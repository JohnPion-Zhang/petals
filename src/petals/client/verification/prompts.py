"""
Verification Prompt Templates

Prompt templates for RLM-style result verification using LLM reasoning.
"""

from dataclasses import dataclass, field
from typing import Dict, Any

# System prompt for verification
VERIFICATION_SYSTEM_PROMPT = """You are a result verification assistant following the RLM (Recursive Language Model) pattern.
Your role is to verify tool execution results before they are aggregated by parent nodes.

Verification principles:
1. Check semantic correctness - does the result make sense?
2. Check completeness - is all expected data present?
3. Check consistency - does it match what was requested?
4. Check plausibility - could this be hallucinated data?
5. Check data quality - are values reasonable and well-formed?

Provide honest assessment. It's better to flag uncertain results than to pass invalid data.

Response format:
- "VERIFIED" if the result is correct and can proceed
- "ISSUE: <description>" if problems were found that need attention
"""

# Basic verification prompt
BASIC_VERIFICATION_PROMPT = """Tool: {tool_name}
Request: {request}
Result: {result}

Is this result correct and complete? Provide a brief assessment.

Respond with:
- "VERIFIED" if the result is correct
- "ISSUE: <description>" if problems were found"""

# Structural verification prompt
STRUCTURAL_VERIFICATION_PROMPT = """Verify the structure and content of this result:

Tool: {tool_name}
Schema: {schema}
Result: {result}

Check:
1. Are all required fields present?
2. Are the data types correct?
3. Are values within expected ranges?
4. Is the data semantically correct?

Respond with:
- "VERIFIED" if structure is valid
- "ISSUE: <description>" if structure problems were found"""

# Deep verification prompt
DEEP_VERIFICATION_PROMPT = """Perform deep verification of this tool result.

Tool: {tool_name}
Request: {request}
Context: {context}
Result: {result}

Verify:
1. Factual accuracy
2. Logical consistency
3. Alignment with request
4. Absence of hallucination
5. Data quality

For each check, note if there are any concerns.
Then provide a final assessment:
- "VERIFIED" if all checks pass
- "ISSUE: <description>" if any concerns were found"""

# Child aggregation verification prompt
CHILD_AGGREGATION_PROMPT = """Before aggregating child results, verify each one:

Child Results:
{child_results}

Parent Tool: {parent_tool}
Parent Request: {parent_request}

Verify each child is correct before aggregation.
Return JSON with verification status for each child:
{{
    "child_id": {{
        "status": "verified" | "issue" | "error",
        "concerns": ["list of concerns if any"]
    }}
}}

If all children are verified, aggregate can proceed.
If any child has issues, note which ones need attention."""


@dataclass
class VerificationPromptTemplate:
    """Template for generating verification prompts.

    Provides customizable prompts for different verification levels
    and scenarios in the RLM verification flow.

    Attributes:
        system: System prompt for the verifier.
        basic: Basic verification prompt template.
        structural: Structural verification prompt template.
        deep: Deep verification prompt template.
        aggregation: Child aggregation verification prompt.
    """

    system: str = VERIFICATION_SYSTEM_PROMPT
    basic: str = BASIC_VERIFICATION_PROMPT
    structural: str = STRUCTURAL_VERIFICATION_PROMPT
    deep: str = DEEP_VERIFICATION_PROMPT
    aggregation: str = CHILD_AGGREGATION_PROMPT

    def get_prompt(self, level: str, **kwargs) -> str:
        """Get prompt for verification level.

        Args:
            level: Verification level (basic, structural, deep, aggregation).
            **kwargs: Template variables for the prompt.

        Returns:
            Formatted prompt string.
        """
        templates = {
            "basic": self.basic,
            "structural": self.structural,
            "deep": self.deep,
            "aggregation": self.aggregation,
        }

        template = templates.get(level, self.basic)

        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")

    def get_full_prompt(
        self,
        level: str,
        tool_name: str,
        result: Any,
        request: str = None,
        context: Dict[str, Any] = None,
        schema: Dict[str, Any] = None,
    ) -> str:
        """Get a complete prompt for verification.

        Args:
            level: Verification level.
            tool_name: Name of the tool.
            result: The result to verify.
            request: Optional request description.
            context: Optional context dictionary.
            schema: Optional schema for structural verification.

        Returns:
            Complete formatted prompt.
        """
        import json

        # Format result
        result_str = json.dumps(result, indent=2, default=str)

        # Format schema
        schema_str = json.dumps(schema, indent=2, default=str) if schema else "No schema provided"

        # Format context
        context_str = json.dumps(context, indent=2, default=str) if context else "No context provided"

        if level == "basic":
            return self.get_prompt(
                "basic",
                tool_name=tool_name,
                request=request or "No request specified",
                result=result_str[:2000],  # Limit size
            )

        elif level == "structural":
            return self.get_prompt(
                "structural",
                tool_name=tool_name,
                schema=schema_str,
                result=result_str[:2000],
            )

        elif level == "deep":
            return self.get_prompt(
                "deep",
                tool_name=tool_name,
                request=request or "No request specified",
                context=context_str,
                result=result_str[:2000],
            )

        else:
            return self.basic.format(
                tool_name=tool_name,
                request=request or "No request",
                result=result_str[:2000],
            )

    def get_aggregation_prompt(
        self,
        child_results: Dict[str, Dict[str, Any]],
        parent_tool: str,
        parent_request: str = None,
    ) -> str:
        """Get prompt for verifying child results before aggregation.

        Args:
            child_results: Dict mapping child IDs to their results.
            parent_tool: Name of the parent tool.
            parent_request: Optional parent request description.

        Returns:
            Formatted aggregation verification prompt.
        """
        import json

        # Format child results for readability
        formatted_results = []
        for child_id, child_data in child_results.items():
            formatted_results.append({
                "id": child_id,
                "tool": child_data.get("tool_name", "unknown"),
                "result_summary": str(child_data.get("result", "no result"))[:200],
            })

        return self.get_prompt(
            "aggregation",
            child_results=json.dumps(formatted_results, indent=2),
            parent_tool=parent_tool,
            parent_request=parent_request or "No request specified",
        )

    def with_system_prompt(self, level: str, **kwargs) -> tuple[str, str]:
        """Get prompt with system prompt prepended.

        Args:
            level: Verification level.
            **kwargs: Template variables.

        Returns:
            Tuple of (system_prompt, user_prompt).
        """
        user_prompt = self.get_prompt(level, **kwargs)
        return self.system, user_prompt

    @classmethod
    def create_restrictive(cls) -> "VerificationPromptTemplate":
        """Create a stricter verification template.

        Returns:
            VerificationPromptTemplate with stricter verification criteria.
        """
        return cls(
            system="""You are a STRICT verification assistant.
Only approve results that are EXACTLY correct.
Reject anything that is uncertain, incomplete, or potentially hallucinated.""",
            basic=BASIC_VERIFICATION_PROMPT,
            structural=STRUCTURAL_VERIFICATION_PROMPT,
            deep=DEEP_VERIFICATION_PROMPT,
            aggregation=CHILD_AGGREGATION_PROMPT,
        )

    @classmethod
    def create_lenient(cls) -> "VerificationPromptTemplate":
        """Create a more lenient verification template.

        Returns:
            VerificationPromptTemplate with lenient verification criteria.
        """
        return cls(
            system="""You are a LENIENT verification assistant.
Approve results unless there are CLEAR and SERIOUS problems.
Minor issues can be noted but should not block processing.""",
            basic=BASIC_VERIFICATION_PROMPT,
            structural=STRUCTURAL_VERIFICATION_PROMPT,
            deep=DEEP_VERIFICATION_PROMPT,
            aggregation=CHILD_AGGREGATION_PROMPT,
        )
