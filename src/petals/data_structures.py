import dataclasses
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Generator
import uuid

import pydantic.v1 as pydantic
from hivemind.p2p import PeerID
from hivemind.moe.expert_uid import ExpertUID


class CallStatus(Enum):
    """Status of a tool call execution."""
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclasses.dataclass
class ToolCall:
    """Represents a single tool call with its arguments and execution state."""
    name: str
    arguments: Optional[Dict[str, Any]] = None
    id: str = dataclasses.field(default_factory=lambda: f"tool_{uuid.uuid4().hex[:12]}")
    status: CallStatus = CallStatus.PENDING
    result: Any = None
    dependencies: List[str] = dataclasses.field(default_factory=list)
    error: Optional[str] = None


@dataclasses.dataclass
class Message:
    """Represents a message in the conversation history."""
    role: str  # "user", "assistant", "tool"
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


@dataclasses.dataclass
class ContextWindow:
    """Represents the current context window with conversation history."""
    max_tokens: int = 4096
    system_prompt: str = ""
    tool_descriptions: str = ""
    conversation_history: List[Message] = dataclasses.field(default_factory=list)
    active_context: str = ""

    def add_message(self, message: Message):
        """Add a message to conversation history."""
        self.conversation_history.append(message)


@dataclasses.dataclass
class AgentState:
    """Tracks the state of an agent execution.

    Attributes:
        current_iteration: Current iteration count in the agent loop.
        max_iterations: Maximum iterations before stopping.
        _tool_history: Internal list storing tool history.
        total_tokens_used: Total tokens consumed by LLM calls.
        stopped_early: Flag indicating if agent stopped before completion.
    """
    current_iteration: int = 0
    max_iterations: int = 10
    _tool_history: List[ToolCall] = dataclasses.field(default_factory=list)
    total_tokens_used: int = 0
    stopped_early: bool = False

    @property
    def tool_history(self) -> Generator[ToolCall, None, None]:
        """Generator yielding tool history as tools complete.

        Note: Generators are single-use. Iterate once, or use `as_list()`
        for multiple iterations.

        Usage:
            for tool in agent.state.tool_history:
                print(f"Tool: {tool.name}, Status: {tool.status}")
        """
        yield from self._tool_history

    def add_tool(self, tool_call: ToolCall) -> None:
        """Add tool to history (for streaming/generator pattern)."""
        self._tool_history.append(tool_call)

    def clear(self) -> None:
        """Clear tool history and reset iteration state."""
        self._tool_history.clear()
        self.current_iteration = 0
        self.stopped_early = False

    def as_list(self) -> List[ToolCall]:
        """Materialize full history for debugging.

        Returns:
            Complete list of all tool calls.
        """
        return list(self._tool_history)

    def update_token_usage(self, usage: Dict[str, int]) -> None:
        """Update total tokens used from LLM response.

        Args:
            usage: Dict with 'prompt_tokens', 'completion_tokens', 'total_tokens'.
        """
        if usage and "total_tokens" in usage:
            self.total_tokens_used += usage["total_tokens"]

ModuleUID = str
UID_DELIMITER = "."  # delimits parts of one module uid, e.g. "bloom.transformer.h.4.self_attention"
CHAIN_DELIMITER = " "  # delimits multiple uids in a sequence, e.g. "bloom.layer3 bloom.layer4"


def parse_uid(uid: ModuleUID) -> Tuple[str, int]:
    assert CHAIN_DELIMITER not in uid, "parse_uid() does not support chained UIDs"
    dht_prefix, index = uid.split(UID_DELIMITER)
    return dht_prefix, int(index)


@pydantic.dataclasses.dataclass
class ModelInfo:
    num_blocks: pydantic.conint(ge=1, strict=True)
    repository: Optional[str] = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, source: dict):
        return cls(**source)


class ServerState(Enum):
    OFFLINE = 0
    JOINING = 1
    ONLINE = 2


RPS = pydantic.confloat(ge=0, allow_inf_nan=False, strict=True)


@pydantic.dataclasses.dataclass
class ServerInfo:
    state: ServerState
    throughput: RPS

    start_block: Optional[pydantic.conint(ge=0, strict=True)] = None
    end_block: Optional[pydantic.conint(ge=0, strict=True)] = None

    public_name: Optional[str] = None
    version: Optional[str] = None

    network_rps: Optional[RPS] = None
    forward_rps: Optional[RPS] = None
    inference_rps: Optional[RPS] = None

    adapters: Sequence[str] = ()
    torch_dtype: Optional[str] = None
    quant_type: Optional[str] = None
    using_relay: Optional[bool] = None
    cache_tokens_left: Optional[pydantic.conint(ge=0, strict=True)] = None
    next_pings: Optional[Dict[str, pydantic.confloat(ge=0, strict=True)]] = None

    def to_tuple(self) -> Tuple[int, float, dict]:
        extra_info = dataclasses.asdict(self)
        del extra_info["state"], extra_info["throughput"]
        return (self.state.value, self.throughput, extra_info)

    @classmethod
    def from_tuple(cls, source: tuple):
        state, throughput = source[:2]
        extra_info = source[2] if len(source) > 2 else {}
        # pydantic will validate existing fields and ignore extra ones
        return cls(state=ServerState(state), throughput=throughput, **extra_info)


@dataclasses.dataclass
class RemoteModuleInfo:
    """A remote module that is served by one or more servers"""

    uid: ModuleUID
    servers: Dict[PeerID, ServerInfo]


@dataclasses.dataclass
class RemoteSpanInfo:
    """A chain of remote blocks served by one specific remote peer"""

    peer_id: PeerID
    start: int
    end: int
    server_info: ServerInfo

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def state(self) -> ServerState:
        return self.server_info.state

    @property
    def throughput(self) -> float:
        return self.server_info.throughput


RPCInfo = Dict[str, Any]

Handle = int


@dataclasses.dataclass(frozen=True)
class InferenceMetadata:
    uid: ExpertUID
    prefix_length: int
    cache_handles: Tuple[Handle, ...]
    active_adapter: Optional[str]
