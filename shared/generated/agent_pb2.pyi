from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

DESCRIPTOR: _descriptor.FileDescriptor

class TaskState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TASK_STATE_UNSPECIFIED: _ClassVar[TaskState]
    TASK_STATE_SUBMITTED: _ClassVar[TaskState]
    TASK_STATE_WORKING: _ClassVar[TaskState]
    TASK_STATE_COMPLETED: _ClassVar[TaskState]
    TASK_STATE_FAILED: _ClassVar[TaskState]
TASK_STATE_UNSPECIFIED: TaskState
TASK_STATE_SUBMITTED: TaskState
TASK_STATE_WORKING: TaskState
TASK_STATE_COMPLETED: TaskState
TASK_STATE_FAILED: TaskState

class Skill(_message.Message):
    __slots__ = ("id", "name", "description")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    description: str
    def __init__(self, id: str | None = ..., name: str | None = ..., description: str | None = ...) -> None: ...

class AgentCard(_message.Message):
    __slots__ = ("name", "version", "description", "skills", "input_modes", "output_modes")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    SKILLS_FIELD_NUMBER: _ClassVar[int]
    INPUT_MODES_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_MODES_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    description: str
    skills: _containers.RepeatedCompositeFieldContainer[Skill]
    input_modes: _containers.RepeatedScalarFieldContainer[str]
    output_modes: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, name: str | None = ..., version: str | None = ..., description: str | None = ..., skills: _Iterable[Skill | _Mapping] | None = ..., input_modes: _Iterable[str] | None = ..., output_modes: _Iterable[str] | None = ...) -> None: ...

class GetAgentCardRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class TaskEvent(_message.Message):
    __slots__ = ("task_id", "state", "message", "timestamp_ms")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    state: TaskState
    message: str
    timestamp_ms: int
    def __init__(self, task_id: str | None = ..., state: TaskState | str | None = ..., message: str | None = ..., timestamp_ms: int | None = ...) -> None: ...

class UsageMetadata(_message.Message):
    __slots__ = ("input_tokens", "output_tokens", "model")
    INPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    input_tokens: int
    output_tokens: int
    model: str
    def __init__(self, input_tokens: int | None = ..., output_tokens: int | None = ..., model: str | None = ...) -> None: ...

class SendTaskRequest(_message.Message):
    __slots__ = ("task_id", "trace_id", "query", "context")
    class ContextEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: str | None = ..., value: str | None = ...) -> None: ...
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    trace_id: str
    query: str
    context: _containers.ScalarMap[str, str]
    def __init__(self, task_id: str | None = ..., trace_id: str | None = ..., query: str | None = ..., context: _Mapping[str, str] | None = ...) -> None: ...

class SendTaskResponse(_message.Message):
    __slots__ = ("task_id", "state", "result_text", "sufficient", "metadata", "usage", "error")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: str | None = ..., value: str | None = ...) -> None: ...
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    RESULT_TEXT_FIELD_NUMBER: _ClassVar[int]
    SUFFICIENT_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    USAGE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    state: TaskState
    result_text: str
    sufficient: bool
    metadata: _containers.ScalarMap[str, str]
    usage: _containers.RepeatedCompositeFieldContainer[UsageMetadata]
    error: str
    def __init__(self, task_id: str | None = ..., state: TaskState | str | None = ..., result_text: str | None = ..., sufficient: bool | None = ..., metadata: _Mapping[str, str] | None = ..., usage: _Iterable[UsageMetadata | _Mapping] | None = ..., error: str | None = ...) -> None: ...
