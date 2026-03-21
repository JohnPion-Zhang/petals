# Round 2 Epic: CodeAct AST Pattern Support

> **Status**: Queued
> **Priority**: High
> **Round**: 2 (after simple DAG orchestration)

## Overview

CodeAct DAG represents Python AST (Abstract Syntax Tree) execution graphs, which can express complex control flow patterns beyond simple serial/parallel execution.

## Motivation

Current DAG implementation handles:
- ✅ Linear chains (A → B → C)
- ✅ Parallel waves (A, B, C execute together)
- ✅ Diamond patterns (A→B, A→C, B→D, C→D)
- ✅ Simple dependency resolution

CodeAct AST extends this to Python language patterns:
- **For-loops** with break/continue
- **Conditionals** (if/elif/else)
- **Exception handling** (try/except/finally)
- **Pattern matching** (match/case)
- **Early returns** from nested functions
- **Complex control flow** with goto-like semantics

## CodeAct Pattern Examples

### For-Loop Pattern
```python
# Traditional: A1, A2, A3, A4, A5 sequentially
# CodeAct: Parallel inner iterations, sequential outer
for item in items:
    process(item)  # Can parallelize within iteration
```

### Conditional Pattern
```python
if condition_a:
    do_alpha()
elif condition_b:
    do_beta()
else:
    do_gamma()
```
- Branches are mutually exclusive
- Only one path executes
- DAG: Diamond with exclusive path selection

### Try/Except/Finally Pattern
```python
try:
    risky_operation()
except SpecificError:
    handle_specific()
except OtherError:
    handle_other()
finally:
    cleanup()
```
- All branches potentially execute
- finally always runs
- Multiple exception handlers (OR-like)

### Match/Case Pattern (Python 3.10+)
```python
match value:
    case 1:
        action_one()
    case 2:
        action_two()
    case _:
        default_action()
```

### Break/Continue in Loops
```python
for i in range(10):
    if should_skip(i):
        continue  # Skip this iteration
    if should_stop(i):
        break  # Exit loop
    process(i)
```

## Design Considerations

### 1. Execution Semantics
| Pattern | Execution Model |
|---------|-----------------|
| For-loop | Sequential iterations OR parallel batches |
| If/elif/else | Single path (exclusive OR) |
| Try/except | All handlers (OR), finally always |
| Match/case | Single matching case |
| Break | Early exit from loop |
| Continue | Skip to next iteration |

### 2. DAG Representation
```python
# BranchNode for conditionals
class BranchNode:
    condition: str
    true_path: List[ToolCallNode]
    false_path: List[ToolCallNode]
    # Only one path executes

# LoopNode for iteration
class LoopNode:
    items: List[Any]
    body: List[ToolCallNode]
    max_iterations: int
    break_deps: List[str]
    continue_deps: List[str]
```

### 3. State Management
- Loop variables must persist between iterations
- Break/continue flags need coordination
- Exception state must propagate correctly

### 4. Serialization
- AST must be serializable for persistence
- Round-trip: Python code → AST → DAG → Execute → Result

## Implementation Phases

### Phase 1: Core AST Nodes
- [ ] `BranchNode` - if/elif/else support
- [ ] `LoopNode` - for/while support
- [ ] `TryNode` - try/except/finally support
- [ ] `MatchNode` - match/case support
- [ ] `BreakNode` / `ContinueNode` - loop control

### Phase 2: Parser Integration
- [ ] Python AST parser (ast module)
- [ ] CodeAct DSL parser
- [ ] AST → DAG converter

### Phase 3: Execution Engine
- [ ] Conditional execution logic
- [ ] Loop execution with state
- [ ] Exception handling flow
- [ ] Break/continue coordination

### Phase 4: Tests & Documentation
- [ ] Comprehensive test suite
- [ ] Pattern documentation
- [ ] Examples gallery

## Technical Challenges

1. **State Persistence**: Loop variables and break/continue flags
2. **Cycle Boundaries**: Loops create dynamic cycles
3. **Back-edges**: Traditional DAG assumes no back-edges; loops need them
4. **Exclusive OR**: Branching paths vs parallel execution

## Comparison: Traditional DAG vs CodeAct AST

| Aspect | Traditional DAG | CodeAct AST |
|--------|-----------------|-------------|
| Edges | Forward only | May have back-edges |
| Execution | Fixed order | Data-dependent paths |
| Loops | Not supported | First-class citizen |
| Conditionals | Diamond pattern | Single path execution |
| Exceptions | Propagate up | Handled inline |
| State | Stateless | Mutable state |

## References

- [CodeAct Paper](https://arxiv.org/abs/...) - Agent system with code actions
- [Python AST Module](https://docs.python.org/3/library/ast.html) - AST parsing
- [Petals Agent Layer](src/petals/client/CLAUDE.md) - Current implementation

## Future Considerations

- Parallel loop iterations (batch mode)
- Async iteration support
- Generator patterns
- Context managers (with/finally)

---

**Ready for implementation when Round 1 is validated in production.**
