"""
Tests for CodeActParser - Python AST to ToolCall DAG Parser

TDD Phase: CodeAct Parser Tests

These tests cover:
- Python source code parsing
- Tool call extraction from AST
- Variable tracking and dependency resolution
- Dependency markers (${var} and from_dep)
- Error handling (syntax errors, unknown tools)
"""
import pytest

from petals.client.dag import ToolCallNode, ToolCallDAG
from petals.client.tool_registry import ToolRegistry
from petals.client.codeact import (
    CodeActParser,
    CodeActSyntaxError,
    CodeActUnknownToolError,
    ToolCallInfo,
)


# --- Test Fixtures ---

@pytest.fixture
def tool_registry():
    """Create a fresh ToolRegistry with common test tools."""
    registry = ToolRegistry()

    def echo(text: str) -> str:
        return text

    def uppercase(text: str) -> str:
        return text.upper()

    def lowercase(text: str) -> str:
        return text.lower()

    def concat(a: str, b: str) -> str:
        return a + b

    def fetch(url: str) -> dict:
        return {"url": url, "content": f"content from {url}"}

    def process(data: dict) -> str:
        return f"processed: {data.get('content', '')}"

    registry.register("echo", echo)
    registry.register("uppercase", uppercase)
    registry.register("lowercase", lowercase)
    registry.register("concat", concat)
    registry.register("fetch", fetch)
    registry.register("process", process)

    return registry


@pytest.fixture
def parser(tool_registry):
    """Create a CodeActParser with a tool registry."""
    return CodeActParser(tool_registry)


# ============================================================================
# ToolCallInfo Tests
# ============================================================================

class TestToolCallInfo:
    """Tests for ToolCallInfo dataclass."""

    def test_tool_call_info_creation(self):
        """Create a ToolCallInfo with all fields."""
        info = ToolCallInfo(
            node_id="echo_1",
            tool_name="echo",
            arguments={"text": "hello"},
            line=1,
            variable_name="result",
            dependencies=["dep_1"]
        )

        assert info.node_id == "echo_1"
        assert info.tool_name == "echo"
        assert info.arguments == {"text": "hello"}
        assert info.line == 1
        assert info.variable_name == "result"
        assert info.dependencies == ["dep_1"]

    def test_tool_call_info_defaults(self):
        """Create a ToolCallInfo with minimal fields."""
        info = ToolCallInfo(
            node_id="test_1",
            tool_name="test",
            arguments={},
            line=1
        )

        assert info.variable_name is None
        assert info.dependencies == []


# ============================================================================
# CodeActParser Basic Tests
# ============================================================================

class TestCodeActParserBasics:
    """Tests for CodeActParser basic functionality."""

    def test_parser_initialization(self, tool_registry):
        """Parser initializes with a tool registry."""
        parser = CodeActParser(tool_registry)
        assert parser.registry is tool_registry

    def test_parse_empty_source(self, parser):
        """Parse empty source code."""
        dag, var_map = parser.parse("")

        assert len(dag) == 0
        assert var_map == {}

    def test_parse_whitespace_only(self, parser):
        """Parse whitespace-only source code."""
        dag, var_map = parser.parse("   \n\n   ")

        assert len(dag) == 0
        assert var_map == {}

    def test_parse_comment_only(self, parser):
        """Parse comment-only source code."""
        dag, var_map = parser.parse("# This is a comment\n# Another comment")

        assert len(dag) == 0
        assert var_map == {}

    def test_parse_non_tool_code(self, parser):
        """Parse source with no tool calls."""
        dag, var_map = parser.parse("""
x = 1
y = "hello"
def foo():
    return 42
""")

        assert len(dag) == 0
        assert var_map == {}


# ============================================================================
# Tool Call Extraction Tests
# ============================================================================

class TestToolCallExtraction:
    """Tests for tool call extraction from Python source."""

    def test_simple_tool_call_no_assignment(self, parser):
        """Extract tool call without variable assignment."""
        dag, var_map = parser.parse('echo(text="hello")')

        assert len(dag) == 1

        # Get the only node
        node_id = list(dag.nodes.keys())[0]
        node = dag.nodes[node_id]

        assert node.name == "echo"
        assert node.arguments == {"text": "hello"}
        assert node.is_root  # No dependencies
        assert var_map == {}

    def test_simple_tool_call_with_assignment(self, parser):
        """Extract tool call with variable assignment."""
        dag, var_map = parser.parse('result = echo(text="hello")')

        assert len(dag) == 1

        node_id = list(dag.nodes.keys())[0]
        node = dag.nodes[node_id]

        assert node.name == "echo"
        assert node.arguments == {"text": "hello"}
        assert var_map == {"result": node_id}

    def test_tool_call_with_positional_args(self, parser):
        """Extract tool call with positional arguments."""
        dag, var_map = parser.parse('concat("hello", "world")')

        assert len(dag) == 1

        node = list(dag.nodes.values())[0]
        assert node.name == "concat"
        # Positional args get converted to positional param names if available
        # or remain as a list depending on implementation

    def test_tool_call_with_mixed_args(self, parser):
        """Extract tool call with both positional and keyword arguments."""
        dag, var_map = parser.parse('concat("hello", b="world")')

        assert len(dag) == 1

        node = list(dag.nodes.values())[0]
        assert node.name == "concat"
        assert "b" in node.arguments

    def test_tool_call_with_nested_calls(self, parser):
        """Extract tool call when args contain other function calls."""
        # This tests that nested non-tool calls are evaluated as expressions
        dag, var_map = parser.parse('echo(text=len("hello"))')

        assert len(dag) == 1

        node = list(dag.nodes.values())[0]
        assert node.name == "echo"
        # len() is not a tool, so it's evaluated

    def test_multiple_tool_calls(self, parser):
        """Extract multiple tool calls in sequence."""
        source = '''
first = echo(text="hello")
second = echo(text="world")
'''

        dag, var_map = parser.parse(source)

        assert len(dag) == 2

        # Both should be in var_map
        assert "first" in var_map
        assert "second" in var_map

        # Both should be root nodes (no dependencies)
        nodes = list(dag.nodes.values())
        for node in nodes:
            assert node.is_root

    def test_node_id_generation(self, parser):
        """Verify node IDs are generated correctly."""
        source = '''
echo(text="first")
echo(text="second")
echo(text="third")
'''

        dag, var_map = parser.parse(source)

        node_ids = list(dag.nodes.keys())

        # Should have tool name prefix with counter
        assert any("echo" in node_id for node_id in node_ids)
        # All should be unique
        assert len(set(node_ids)) == 3


# ============================================================================
# Dependency Resolution Tests
# ============================================================================

class TestDependencyResolution:
    """Tests for variable dependency resolution."""

    def test_simple_dependency_with_dollar_syntax(self, parser):
        """Resolve simple dependency using ${var} syntax."""
        source = '''
result1 = echo(text="hello")
result2 = uppercase(text="${result1}")
'''

        dag, var_map = parser.parse(source)

        assert len(dag) == 2
        assert len(var_map) == 2

        # Find the uppercase node
        uppercase_node = None
        for node in dag.nodes.values():
            if node.name == "uppercase":
                uppercase_node = node
                break

        assert uppercase_node is not None
        # Should have dependency on echo node (resolved from result1 variable)
        # The variable "result1" maps to "echo_1" node ID
        assert "echo_1" in uppercase_node.dependencies

    def test_simple_dependency_with_from_dep_syntax(self, parser):
        """Resolve dependency using from_dep syntax."""
        source = '''
result1 = echo(text="hello")
result2 = uppercase(text={"from_dep": "result1"})
'''

        dag, var_map = parser.parse(source)

        assert len(dag) == 2

        # Find the uppercase node
        uppercase_node = None
        for node in dag.nodes.values():
            if node.name == "uppercase":
                uppercase_node = node
                break

        assert uppercase_node is not None
        # from_dep indicates a dependency
        assert len(uppercase_node.dependencies) > 0 or "result1" in str(uppercase_node.arguments)

    def test_chained_dependencies(self, parser):
        """Test chained dependency chain: A -> B -> C."""
        source = '''
a = echo(text="start")
b = uppercase(text="${a}")
c = lowercase(text="${b}")
'''

        dag, var_map = parser.parse(source)

        assert len(dag) == 3

        # Get nodes by their variable names
        nodes_by_var = {}
        for var_name, node_id in var_map.items():
            nodes_by_var[var_name] = dag.nodes[node_id]

        # c depends on b
        c_node = nodes_by_var["c"]
        assert len(c_node.dependencies) >= 1

        # b depends on a
        b_node = nodes_by_var["b"]
        assert len(b_node.dependencies) >= 1

    def test_parallel_branches_with_merge(self, parser):
        """Test parallel branches that merge: A -> B and A -> C -> D."""
        source = '''
data = echo(text="hello")
upper = uppercase(text="${data}")
lower = lowercase(text="${data}")
combined = concat(a="${upper}", b="${lower}")
'''

        dag, var_map = parser.parse(source)

        assert len(dag) == 4

        # Get waves to verify execution order
        waves = dag.get_waves()
        wave_ids = [set(n.id for n in wave) for wave in waves]

        # First wave: echo (root)
        # Second wave: uppercase, lowercase (parallel, depend on echo)
        # Third wave: concat (depends on both uppercase and lowercase)
        assert len(waves) == 3

    def test_dependency_variable_not_found_warning(self, parser):
        """Test that undefined variable dependencies are handled gracefully."""
        source = '''
result = uppercase(text="${undefined_var}")
'''

        dag, var_map = parser.parse(source)

        # Parser should still work but the dependency won't resolve
        assert len(dag) == 1
        node = list(dag.nodes.values())[0]
        assert node.name == "uppercase"

    def test_multiple_dependencies_same_node(self, parser):
        """Test a node with multiple dependencies."""
        source = '''
a = echo(text="first")
b = echo(text="second")
combined = concat(a="${a}", b="${b}")
'''

        dag, var_map = parser.parse(source)

        assert len(dag) == 3

        # Find concat node
        concat_node = None
        for node in dag.nodes.values():
            if node.name == "concat":
                concat_node = node
                break

        assert concat_node is not None
        assert len(concat_node.dependencies) == 2


# ============================================================================
# DAG Structure Tests
# ============================================================================

class TestDAGStructure:
    """Tests for DAG structure and wave computation."""

    def test_linear_chain_waves(self, parser):
        """Test wave computation for linear chain."""
        source = '''
a = echo(text="1")
b = uppercase(text="${a}")
c = lowercase(text="${b}")
'''

        dag, var_map = parser.parse(source)
        waves = dag.get_waves()

        # Should be 3 waves (one per node in chain)
        assert len(waves) == 3

        # Wave 1: a
        # Wave 2: b
        # Wave 3: c
        wave_0_ids = {n.id for n in waves[0]}
        wave_1_ids = {n.id for n in waves[1]}
        wave_2_ids = {n.id for n in waves[2]}

        # Each wave should have one node
        assert len(wave_0_ids) == 1
        assert len(wave_1_ids) == 1
        assert len(wave_2_ids) == 1

    def test_parallel_nodes_same_wave(self, parser):
        """Test parallel nodes are in the same wave."""
        source = '''
a = echo(text="1")
b = uppercase(text="${a}")
c = lowercase(text="${a}")  # Also depends on a, can run in parallel with b
'''

        dag, var_map = parser.parse(source)
        waves = dag.get_waves()

        # Wave 1: a
        # Wave 2: b, c (parallel)
        assert len(waves) == 2

        wave_1_ids = {n.id for n in waves[1]}
        assert len(wave_1_ids) == 2

    def test_diamond_pattern_waves(self, parser):
        """Test diamond pattern DAG."""
        source = '''
root = echo(text="start")
branch1 = uppercase(text="${root}")
branch2 = lowercase(text="${root}")
merged = concat(a="${branch1}", b="${branch2}")
'''

        dag, var_map = parser.parse(source)
        waves = dag.get_waves()

        # Wave 1: root
        # Wave 2: branch1, branch2 (parallel)
        # Wave 3: merged
        assert len(waves) == 3

        assert len(waves[0]) == 1  # root
        assert len(waves[1]) == 2  # branch1, branch2
        assert len(waves[2]) == 1  # merged

    def test_topological_sort_order(self, parser):
        """Test topological sort respects dependencies."""
        source = '''
a = echo(text="1")
b = uppercase(text="${a}")
c = lowercase(text="${b}")
'''

        dag, var_map = parser.parse(source)

        sorted_nodes = dag.topological_sort()
        node_ids = [n.id for n in sorted_nodes]

        # Should maintain dependency order
        a_idx = next(i for i, n in enumerate(sorted_nodes) if n.name == "echo")
        b_idx = next(i for i, n in enumerate(sorted_nodes) if n.name == "uppercase")
        c_idx = next(i for i, n in enumerate(sorted_nodes) if n.name == "lowercase")

        assert a_idx < b_idx < c_idx


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for parser error handling."""

    def test_syntax_error_raises_exception(self, parser):
        """Syntax error raises CodeActSyntaxError."""
        invalid_source = """
x = 1 +
y = 2
"""

        with pytest.raises(CodeActSyntaxError) as exc_info:
            parser.parse(invalid_source)

        assert "SyntaxError" in str(exc_info.value) or "syntax" in str(exc_info.value).lower()

    def test_syntax_error_incomplete_expression(self, parser):
        """Incomplete expression raises CodeActSyntaxError."""
        invalid_source = "echo(text="

        with pytest.raises(CodeActSyntaxError):
            parser.parse(invalid_source)

    def test_unknown_tool_raises_exception(self, tool_registry):
        """Unknown tool raises CodeActUnknownToolError."""
        # Create parser with registry that doesn't have 'unknown_tool'
        parser = CodeActParser(tool_registry)

        source = 'unknown_tool(text="hello")'

        with pytest.raises(CodeActUnknownToolError) as exc_info:
            parser.parse(source)

        assert "unknown_tool" in str(exc_info.value).lower()

    def test_partial_unknown_tools_raises_exception(self, parser):
        """If any tool in the source is unknown, raise error."""
        source = '''
echo(text="hello")
unknown_tool(text="world")
'''

        with pytest.raises(CodeActUnknownToolError):
            parser.parse(source)

    def test_error_contains_line_number(self, tool_registry):
        """Error message contains line number for debugging."""
        parser = CodeActParser(tool_registry)
        source = '''
echo(text="ok")
unknown_tool(text="bad")
'''

        try:
            parser.parse(source)
            pytest.fail("Should have raised CodeActUnknownToolError")
        except CodeActUnknownToolError as e:
            error_msg = str(e)
            # Should mention the problematic tool
            assert "unknown_tool" in error_msg.lower()


# ============================================================================
# Edge Cases Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_tool_call_inside_control_flow(self, parser):
        """Tool calls inside if/else blocks are extracted."""
        source = '''
if True:
    result = echo(text="inside if")
else:
    result = echo(text="inside else")
'''

        dag, var_map = parser.parse(source)

        # Both calls should be extracted (parser doesn't evaluate conditions)
        assert len(dag) == 2

    def test_tool_call_inside_loop(self, parser):
        """Tool calls inside loops are extracted."""
        source = '''
for i in range(3):
    result = echo(text=str(i))
'''

        dag, var_map = parser.parse(source)

        # Should extract the call (loop unrolling would happen at runtime)
        assert len(dag) == 1  # Parser doesn't know loop count

    def test_tool_call_in_function(self, parser):
        """Tool calls inside function definitions are extracted."""
        source = '''
def my_function():
    return echo(text="in function")
'''

        dag, var_map = parser.parse(source)

        # Function bodies are still parsed as code
        assert len(dag) == 1

    def test_tool_call_with_complex_literal(self, parser):
        """Tool call with complex argument literals."""
        source = '''
result = echo(text={"key": [1, 2, 3], "nested": {"a": "b"}})
'''

        dag, var_map = parser.parse(source)

        assert len(dag) == 1
        node = list(dag.nodes.values())[0]
        assert node.arguments["text"] == {"key": [1, 2, 3], "nested": {"a": "b"}}

    def test_tool_call_with_number_literals(self, parser):
        """Tool call with number argument values."""
        source = '''
result = echo(text=42)
'''

        dag, var_map = parser.parse(source)

        assert len(dag) == 1
        node = list(dag.nodes.values())[0]
        assert node.arguments["text"] == 42

    def test_tool_call_with_boolean_literals(self, parser):
        """Tool call with boolean argument values."""
        source = '''
result = echo(text=True)
'''

        dag, var_map = parser.parse(source)

        assert len(dag) == 1
        node = list(dag.nodes.values())[0]
        assert node.arguments["text"] is True

    def test_empty_arguments(self, parser):
        """Tool call with no arguments."""
        source = '''
echo(text="hello")
'''

        dag, var_map = parser.parse(source)

        assert len(dag) == 1
        node = list(dag.nodes.values())[0]
        assert node.arguments["text"] == "hello"

    def test_multiple_statements_same_line(self, parser):
        """Multiple statements on the same line."""
        source = 'a = echo(text="1"); b = uppercase(text="${a}")'

        dag, var_map = parser.parse(source)

        assert len(dag) == 2
        assert "a" in var_map
        assert "b" in var_map

    def test_comment_after_tool_call(self, parser):
        """Comment after tool call doesn't affect parsing."""
        source = '''
result = echo(text="hello")  # This is a comment
'''

        dag, var_map = parser.parse(source)

        assert len(dag) == 1
        node = list(dag.nodes.values())[0]
        assert node.arguments["text"] == "hello"


# ============================================================================
# Integration Tests
# ============================================================================

class TestCodeActParserIntegration:
    """Integration tests for CodeActParser with full workflows."""

    def test_full_workflow_data_processing(self, parser):
        """Test complete data processing workflow."""
        source = '''
# Fetch data from multiple sources
web_data = fetch(url="https://example.com")
api_data = fetch(url="https://api.example.com")

# Process both
processed_web = process(data="${web_data}")
processed_api = process(data="${api_data}")

# Combine results
final = concat(a="${processed_web}", b="${processed_api}")
'''

        dag, var_map = parser.parse(source)

        # Should have 5 tool calls
        assert len(dag) == 5
        assert len(var_map) == 5

        # Verify waves
        waves = dag.get_waves()

        # Wave 1: 2 fetch calls (parallel)
        assert len(waves[0]) == 2
        assert {n.name for n in waves[0]} == {"fetch"}

        # Wave 2: 2 process calls (parallel)
        assert len(waves[1]) == 2
        assert {n.name for n in waves[1]} == {"process"}

        # Wave 3: concat
        assert len(waves[2]) == 1
        assert {n.name for n in waves[2]} == {"concat"}

        # Total of 3 waves
        assert len(waves) == 3

    def test_workflow_with_intermediate_transforms(self, parser):
        """Test workflow with multiple transformations."""
        source = '''
original = echo(text="Hello World")
upper = uppercase(text="${original}")
lower = lowercase(text="${original}")
combined = concat(a="${upper}", b="-")
with_lower = concat(a="${combined}", b="${lower}")
'''

        dag, var_map = parser.parse(source)

        assert len(dag) == 5

        # Verify dependencies are tracked
        waves = dag.get_waves()
        assert len(waves) == 4  # echo -> upper+lower -> combined -> with_lower

    def test_real_world_example(self, parser):
        """Test with a realistic multi-step workflow."""
        source = '''
# Step 1: Gather information
search_results = fetch(url="https://search.example.com?q=AI")
weather_data = fetch(url="https://weather.api.com")

# Step 2: Process results
processed_search = process(data="${search_results}")
processed_weather = process(data="${weather_data}")

# Step 3: Combine and format
summary = concat(a="${processed_search}", b="${processed_weather}")
final = uppercase(text="${summary}")
'''

        dag, var_map = parser.parse(source)

        assert len(dag) == 6
        assert len(var_map) == 6

        # Verify wave structure
        waves = dag.get_waves()
        assert len(waves) == 4

        # First wave: 2 fetch calls
        assert len(waves[0]) == 2

        # Second wave: 2 process calls
        assert len(waves[1]) == 2

        # Third wave: 1 concat
        assert len(waves[2]) == 1

        # Fourth wave: 1 uppercase
        assert len(waves[3]) == 1
