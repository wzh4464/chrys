# Copyright (c) 2026 Chrys. All rights reserved.

"""Acceptance tests for the chrys-owned FunctionTool surface.

Behavior matrix for ``chrys.kernel.tools``: the ``invoke`` mirror
(result-parsing five states, validation, ctx injection, owned ``__call__``
limit semantics), the ``tool`` factory passthrough, and ``normalize_tools``
surface.
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import inspect
import json
import logging
import typing
from collections import deque
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from threading import Event as ThreadEvent
from typing import Annotated, Any, Literal, NewType, NotRequired, TypedDict

import annotated_types
import pytest
import typing_extensions
from pydantic import (
    AfterValidator,
    AnyUrl,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    PositiveInt,
    RootModel,
    SecretStr,
    StringConstraints,
    computed_field,
    field_serializer,
    model_serializer,
    model_validator,
)
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic.fields import FieldInfo
from pydantic_core import core_schema

from chrys.kernel import SKIP_PARSING, FunctionTool, ToolException, normalize_tools, tool
from chrys.kernel._null_overlay import OverlayTier, classify
from chrys.kernel.middleware import FunctionInvocationContext
from chrys.kernel.tools import SyncToolCancelledAfterCompletion
from chrys.kernel.types import ChatResponse, Content, Message
from chrys.service.mcp.owned import MCPStdioTool

# ---------------------------------------------------------------------------
# Helpers — the two tool forms (risk 3: schema-supplied vs func-introspection)
# ---------------------------------------------------------------------------

_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "mode": {"type": "string", "enum": ["a", "b"]},
    },
    "required": ["text"],
    "additionalProperties": False,
}


class _NullableValue(BaseModel):
    value: str | None


_FIELD_INFO_SERIALIZER_CARRIER = FieldInfo.from_annotation(
    Annotated[list[_NullableValue], PlainSerializer(lambda values: [{"owned": True} for _ in values])]
)


class _FieldInfoCarrierArguments(BaseModel):
    items: Annotated[list[_NullableValue], _FIELD_INFO_SERIALIZER_CARRIER]


class _FieldConstraintArguments(BaseModel):
    count: int = Field(gt=0)
    positive: PositiveInt


class _CoreSchemaChild(BaseModel):
    value: str | None = None

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        schema = handler(source)
        schema["serialization"] = core_schema.plain_serializer_function_ser_schema(lambda _: {"owned": True})
        return schema


class _CoreSchemaArguments(BaseModel):
    child: _CoreSchemaChild


@dataclasses.dataclass
class _CoreSchemaDataclass:
    value: str | None = None

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        schema = handler(source)
        schema["serialization"] = core_schema.plain_serializer_function_ser_schema(lambda _: {"owned": True})
        return schema


class _CoreSchemaDataclassArguments(BaseModel):
    child: _CoreSchemaDataclass


class _CoreSchemaKey(enum.StrEnum):
    VALUE = "value"

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        schema = handler(source)
        schema["serialization"] = core_schema.plain_serializer_function_ser_schema(lambda _: "SERIALIZED")
        return schema


class _CoreSchemaKeyArguments(BaseModel):
    payload: dict[_CoreSchemaKey, _NullableValue]


class _RootDumpOverrideArguments(BaseModel):
    value: str | None = None

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        return {"owned": True}


class _NestedDumpOverride(BaseModel):
    value: str | None = None

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        return {"owned": True}


class _NestedDumpOverrideArguments(BaseModel):
    child: _NestedDumpOverride


class _CollisionOwned(BaseModel):
    value: str | None = None


class _CollisionConditional(BaseModel):
    value: str | None = None


class _ExcludeIfCollisionArguments(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True, validate_by_name=True)

    owned: _CollisionOwned = Field(serialization_alias="slot")
    conditional: _CollisionConditional = Field(serialization_alias="slot", exclude_if=lambda _: False)


class _ExcludeIfPlainArguments(BaseModel):
    safe: str | None
    conditional: _CollisionConditional = Field(exclude_if=lambda _: False)


@dataclasses.dataclass
class _ComputedAliasBaseDataclass:
    owned: _NullableValue

    @computed_field
    @property
    def view(self) -> dict[str, Any]:
        return {"base": True}


@dataclasses.dataclass
class _ComputedAliasDerivedDataclass(_ComputedAliasBaseDataclass):
    @computed_field(alias="owned")
    @property
    def view(self) -> dict[str, Any]:
        return {}


class _ComputedAliasDataclassArguments(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)

    child: _ComputedAliasDerivedDataclass


class _SlotBase(BaseModel):
    value: str | None = None


class _SlotSub(_SlotBase):
    extra: str | None = None


def _promote_slot(value: _SlotBase) -> _SlotSub:
    return _SlotSub(value=value.value, extra=None)


class _UnionLeft(BaseModel):
    kind: Literal["left"]
    value: str | None = None


class _UnionRight(BaseModel):
    kind: Literal["right"]
    value: str | None = None


class _SlotClassArguments(BaseModel):
    child: Annotated[_SlotBase, AfterValidator(_promote_slot)]
    other: _SlotSub | None = None
    choice: _UnionLeft | _UnionRight


class _ChildListRoot(RootModel[list[_NullableValue]]):
    pass


class _RootModelArguments(BaseModel):
    payload: _ChildListRoot


class _SerializationDefaultsRequiredArguments(BaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    value: str | None = None


class _DivergentDefaults(BaseModel):
    count: int = 7


class _NestedArguments(BaseModel):
    child: _NullableValue


class _ListArguments(BaseModel):
    items: list[_NullableValue]


class _TupleArguments(BaseModel):
    items: tuple[_NullableValue, ...]


class _PlainMembers(TypedDict):
    value: str | None
    optional: NotRequired[str | None]


class _TypedDictArguments(BaseModel):
    payload: _PlainMembers


@dataclasses.dataclass
class _StaticDataclass:
    required: str | None
    omitted: str | None = None
    non_none_default: str | None = "seed"
    factory: str | None = dataclasses.field(default_factory=lambda: None)


class _DataclassArguments(BaseModel):
    child: _StaticDataclass


@pydantic_dataclass
class _PydanticDataclass:
    value: str | None


class _PydanticDataclassArguments(BaseModel):
    child: _PydanticDataclass


class _AliasedNullable(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)

    value: str | None = Field(alias="wire")


class _ExtraArguments(BaseModel):
    model_config = ConfigDict(extra="allow")

    anchor: int = 1


class _ExcludedArguments(BaseModel):
    visible: str | None
    hidden: str | None = Field(exclude=True)


class _ComputedArguments(BaseModel):
    count: int

    @computed_field
    @property
    def ghost(self) -> None:
        return None


class _SplitAliasArguments(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)

    value: str | None = Field(validation_alias="inputKey", serialization_alias="callableKey")


class _EmptyAliasArguments(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)

    value: str | None = Field(alias="")


class _UnschemableComputedArguments(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)

    value: str | None = Field(validation_alias="inKey", serialization_alias="outKey")

    @computed_field
    @property
    def factory(self) -> Callable[[], int]:
        return lambda: 1


_StdNewTypeChild = NewType("_StdNewTypeChild", _NullableValue)
_ExtNewTypeChild = typing_extensions.NewType("_ExtNewTypeChild", _NullableValue)


def _parse_int(value: Any) -> int:
    return int(value)


class _FullGateArguments(BaseModel):
    child: _NullableValue
    members: _PlainMembers
    items: list[_NullableValue]
    mapping: dict[str, _NullableValue]
    std_new: _StdNewTypeChild
    ext_new: _ExtNewTypeChild
    validated: Annotated[int, BeforeValidator(_parse_int)]
    choice: _NullableValue | None
    sequence: Sequence[_NullableValue]
    abstract_mapping: Mapping[str, _NullableValue]
    mutable_mapping: MutableMapping[str, _NullableValue]
    members_set: set[str | None]
    frozen_members: frozenset[str | None]
    mode: Literal["plain"]


class _Mode(enum.Enum):
    PLAIN = "plain"


class _EnumGateArguments(BaseModel):
    mode: _Mode


class _ValidationMetadataArguments(BaseModel):
    count: PositiveInt | None = None
    name: Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)] | None = None
    score: Annotated[int, annotated_types.Gt(0)] | None = None
    interval: Annotated[int, annotated_types.Interval(gt=0, lt=10)] | None = None
    length: Annotated[str, annotated_types.Len(1, 5)] | None = None


class _SafeMetadataWithSerializerArguments(BaseModel):
    value: (
        Annotated[
            str,
            annotated_types.MinLen(1),
            PlainSerializer(lambda value: value.upper()),
        ]
        | None
    ) = None


class _SerializerGroupedMetadata(annotated_types.GroupedMetadata):
    def __iter__(self) -> Iterator[Any]:
        yield PlainSerializer(lambda _: {"grouped_owned": True})


class _GroupedSerializerArguments(BaseModel):
    safe: str | None
    child: Annotated[_NullableValue, _SerializerGroupedMetadata()]


_CoreSchemaNewType = NewType("_CoreSchemaNewType", _NullableValue)


def _new_type_core_schema(source: Any, handler: Any) -> Any:
    schema = dict(handler(_NullableValue))
    schema["serialization"] = core_schema.plain_serializer_function_ser_schema(lambda _: {"new_type_owned": True})
    return schema


_CoreSchemaNewType.__get_pydantic_core_schema__ = _new_type_core_schema  # type: ignore[attr-defined]


class _NewTypeSerializerArguments(BaseModel):
    child: _CoreSchemaNewType


class _CoreSchemaTypedDict(typing_extensions.TypedDict):
    child: _NullableValue


def _typed_dict_core_schema(cls: Any, source: Any, handler: Any) -> Any:
    schema = handler(source)
    schema["serialization"] = core_schema.plain_serializer_function_ser_schema(lambda _: {"typed_dict_owned": True})
    return schema


_CoreSchemaTypedDict.__get_pydantic_core_schema__ = classmethod(_typed_dict_core_schema)  # type: ignore[attr-defined]


class _TypedDictSerializerArguments(BaseModel):
    payload: _CoreSchemaTypedDict


class _CoreSchemaPropertyMeta(type(BaseModel)):
    @property
    def __get_pydantic_core_schema__(cls) -> Any:
        def hook(source: Any, handler: Any) -> Any:
            return core_schema.no_info_after_validator_function(
                lambda value: value,
                handler(source),
                serialization=core_schema.plain_serializer_function_ser_schema(lambda _: {"metaclass_owned": True}),
            )

        return hook


class _MetaclassSerializerArguments(BaseModel, metaclass=_CoreSchemaPropertyMeta):
    value: str | None


class _NestedPlainSerializerArguments(BaseModel):
    safe: str | None
    child: Annotated[_NullableValue, PlainSerializer(lambda _: {"plain_owned": True})]


_ModelSerializationNewType = NewType("_ModelSerializationNewType", _NullableValue)


def _model_serialization_core_schema(source: Any, handler: Any) -> Any:
    schema = dict(handler(_NullableValue))
    schema["serialization"] = core_schema.model_ser_schema(
        _NullableValue,
        core_schema.model_fields_schema({}),
    )
    return schema


_ModelSerializationNewType.__get_pydantic_core_schema__ = _model_serialization_core_schema  # type: ignore[attr-defined]


class _ModelSerializationArguments(BaseModel):
    child: _ModelSerializationNewType


class _CustomSlotSerializer:
    def __get_pydantic_core_schema__(self, source: Any, handler: Any) -> Any:
        return core_schema.no_info_after_validator_function(
            lambda value: value,
            handler(source),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda _: "SERIALIZER_OWNED",
                when_used="always",
            ),
        )


class _CustomSlotSerializerArguments(BaseModel):
    safe: str | None = None
    value: Annotated[str | None, _CustomSlotSerializer()] = None


class _PydanticInternalSerializationArguments(BaseModel):
    url: AnyUrl
    secret: SecretStr


_RootSerializationNewType = NewType("_RootSerializationNewType", dict[str, Any] | None)


def _root_serialization_core_schema(source: Any, handler: Any) -> Any:
    schema = dict(handler(dict[str, Any] | None))
    schema["serialization"] = core_schema.plain_serializer_function_ser_schema(lambda _: {"owned": True})
    return schema


_RootSerializationNewType.__get_pydantic_core_schema__ = _root_serialization_core_schema  # type: ignore[attr-defined]


class _SerializedRootArguments(RootModel[_RootSerializationNewType]):
    @model_validator(mode="before")
    @classmethod
    def _empty_to_null(cls, value: Any) -> Any:
        return None if value == {} else value


class _AnyUrlWithSafeRootFieldArguments(BaseModel):
    safe: str | None
    url: AnyUrl


class _SerializationNamedFieldArguments(BaseModel):
    serialization: str
    child: _NullableValue


class _ModelFieldsSerializerMeta(type(BaseModel)):
    @property
    def __get_pydantic_core_schema__(cls) -> Any:
        def hook(source: Any, handler: Any) -> Any:
            schema = handler(source)
            schema["schema"]["serialization"] = core_schema.plain_serializer_function_ser_schema(
                lambda _: {"owned": True}
            )
            return schema

        return hook


class _ModelFieldsSerializerArguments(BaseModel, metaclass=_ModelFieldsSerializerMeta):
    value: str | None


class _RecursiveSerializedChild(BaseModel):
    next: _RecursiveSerializedChild | None = None

    @model_serializer(mode="plain")
    def _serialize(self) -> dict[str, bool]:
        return {"owned": True}


class _RecursiveSerializedArguments(BaseModel):
    child: _RecursiveSerializedChild | None


class _RecursivePlainChild(BaseModel):
    next: _RecursivePlainChild | None = None


class _RecursivePlainDegradedArguments(BaseModel):
    child: _RecursivePlainChild | None
    url: AnyUrl


class _DefaultedDataclassPayload(TypedDict):
    value: str | None


@pydantic_dataclass
class _DefaultedNestedDataclass:
    payload: _DefaultedDataclassPayload = dataclasses.field(default_factory=lambda: {"value": None})


class _DefaultedNestedDataclassArguments(BaseModel):
    child: _DefaultedNestedDataclass


@pydantic_dataclass
class _RequiredNestedDataclass:
    payload: _DefaultedDataclassPayload


class _RequiredNestedDataclassArguments(BaseModel):
    child: _RequiredNestedDataclass


@pydantic_dataclass
class _NonNoneDefaultDataclass:
    value: str | None = "seed"


class _NonNoneDefaultDataclassArguments(BaseModel):
    child: _NonNoneDefaultDataclass


class _AliasedFieldWithInternalNameExtraArguments(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True, validate_by_name=False, extra="allow")

    value: str | None = Field(alias="wire")


class _ExcludedFieldWithAliasedExtraArguments(BaseModel):
    model_config = ConfigDict(extra="allow", serialize_by_alias=True)

    hidden: str | None = Field(
        default="seed",
        validation_alias="inputHidden",
        serialization_alias="x",
        exclude=True,
    )


class _EmittingFieldWithCollidingExtraArguments(BaseModel):
    model_config = ConfigDict(extra="allow", serialize_by_alias=True)

    visible: str | None = Field(
        default=None,
        validation_alias="inputVisible",
        serialization_alias="x",
    )


class _ConsumingInvokeArguments(BaseModel):
    items: list[dict[str, int]]

    @model_validator(mode="before")
    @classmethod
    def _consume_nested(cls, data: Any) -> dict[str, Any]:
        item = data["items"].pop(0)
        return {"items": [{"x": item.pop("x")}]}


class _CycleNormalizingArguments(BaseModel):
    saw_cycle: bool

    @model_validator(mode="before")
    @classmethod
    def _normalize_cycle(cls, data: Any) -> dict[str, bool]:
        payload = data["payload"]
        return {"saw_cycle": payload["self"] is payload}


class _UnhashableAnnotation:
    __hash__ = None

    def __get_pydantic_core_schema__(self, source: Any, handler: Any) -> Any:
        return core_schema.str_schema()


class _UnhashableAnnotationArguments(BaseModel):
    value: _UnhashableAnnotation()


class _InheritedAliasBase(typing_extensions.TypedDict):
    __pydantic_config__ = ConfigDict(alias_generator=str.upper)

    first: str | None


class _InheritedAliasPayload(_InheritedAliasBase):
    second: str | None


class _InheritedAliasArguments(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)

    payload: _InheritedAliasPayload


class _PlainInheritedBase(typing_extensions.TypedDict):
    first: str | None


class _PlainInheritedPayload(_PlainInheritedBase):
    second: str | None


class _PlainInheritedArguments(BaseModel):
    payload: _PlainInheritedPayload


def _base_alias(name: str) -> str:
    return f"base_{name}"


def _own_alias(name: str) -> str:
    return f"own_{name}"


class _PrecedenceAliasBase(typing_extensions.TypedDict):
    __pydantic_config__ = ConfigDict(alias_generator=_base_alias)

    first: str | None


class _PrecedenceAliasPayload(_PrecedenceAliasBase):
    __pydantic_config__ = ConfigDict(alias_generator=_own_alias)

    second: str | None


class _PrecedenceAliasArguments(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)

    payload: _PrecedenceAliasPayload


class _CoDeclaredTupleArguments(BaseModel):
    pair: tuple[Annotated[_SlotBase, AfterValidator(_promote_slot)], _SlotSub]


class _CoDeclaredUnionArguments(BaseModel):
    choice: _SlotBase | _SlotSub


class _UnknownContainerArguments(BaseModel):
    items: deque[int]


_NESTED_SERIALIZER_CALLS = 0


class _SerializedChild(BaseModel):
    value: str | None

    @model_serializer(mode="plain")
    def _serialize(self) -> dict[str, Any]:
        global _NESTED_SERIALIZER_CALLS
        _NESTED_SERIALIZER_CALLS += 1
        return {"owned": True}


class _NestedSerializerArguments(BaseModel):
    safe: str | None
    child: _SerializedChild


class _RootSerializerArguments(BaseModel):
    value: str | None

    @model_serializer(mode="plain")
    def _serialize(self) -> dict[str, Any]:
        return {"owned": True}


_FIELD_SERIALIZER_CALLS = 0


class _FieldSerializerArguments(BaseModel):
    safe: str | None
    text: str

    @field_serializer("text")
    def _serialize_text(self, value: str) -> str:
        global _FIELD_SERIALIZER_CALLS
        _FIELD_SERIALIZER_CALLS += 1
        return value.upper()


_KEY_SERIALIZER_CALLS = 0


def _upper_key(value: str) -> str:
    global _KEY_SERIALIZER_CALLS
    _KEY_SERIALIZER_CALLS += 1
    return value.upper()


_SerializedKey = Annotated[str, PlainSerializer(_upper_key)]


class _OptionalKeyChild(BaseModel):
    value: str | None = None


class _KeySerializerArguments(BaseModel):
    payload: dict[_SerializedKey, _OptionalKeyChild]


class _SupertypeAttributeChild(_NullableValue):
    __supertype__: typing.ClassVar[type] = _NullableValue

    @model_serializer(mode="plain")
    def _serialize(self) -> dict[str, Any]:
        return {"owned": True}


class _SupertypeAttributeArguments(BaseModel):
    child: _SupertypeAttributeChild


class _AliasedTypedDict(TypedDict):
    value: Annotated[str | None, Field(serialization_alias="wire")]


class _AliasedTypedDictArguments(BaseModel):
    payload: _AliasedTypedDict


class _DefaultedTypedDict(TypedDict, total=False):
    value: Annotated[str | None, Field(default=None)]


class _DefaultedTypedDictArguments(BaseModel):
    payload: _DefaultedTypedDict


@dataclasses.dataclass
class _AliasedDataclass:
    value: Annotated[str | None, Field(serialization_alias="wire")]


class _AliasedDataclassArguments(BaseModel):
    payload: _AliasedDataclass


@dataclasses.dataclass
class _MetadataDefaultDataclass:
    required: str | None
    annotated_default: Annotated[str | None, Field(default=None)]
    slot_default: str | None = Field(default=None)


class _MetadataDefaultDataclassArguments(BaseModel):
    child: _MetadataDefaultDataclass


_PDC_ALIAS_GENERATOR_CALLS = 0


def _pdc_alias_generator(name: str) -> str:
    global _PDC_ALIAS_GENERATOR_CALLS
    _PDC_ALIAS_GENERATOR_CALLS += 1
    return name.upper()


@pydantic_dataclass(
    config=ConfigDict(alias_generator=_pdc_alias_generator, serialize_by_alias=True, validate_by_name=True)
)
class _GeneratedPydanticDataclass:
    value: str | None


class _AnyPydanticDataclassArguments(BaseModel):
    payload: Any


class _DeclaredPydanticDataclassArguments(BaseModel):
    payload: _GeneratedPydanticDataclass


def _introspected_tool(**kwargs: Any) -> FunctionTool:
    @tool(name="echo", **kwargs)
    async def echo(text: str) -> str:
        """Echo the text."""
        return f"echo:{text}"

    return echo


def _schema_supplied_tool(func: Any = None, **kwargs: Any) -> FunctionTool:
    async def echo(**call_kwargs: Any) -> str:
        return f"echo:{call_kwargs.get('text')}"

    return FunctionTool(
        name="echo",
        description="Echo the text.",
        func=func or echo,
        input_model=_SCHEMA,
        **kwargs,
    )


def _foreign_content(text: str) -> object:
    class ForeignContent:
        type = "text"
        raw_representation = None

        def to_dict(self, exclude_none: bool = False) -> dict[str, Any]:
            del exclude_none
            return {"type": "text", "text": text, "additional_properties": {}}

    return ForeignContent()


# ---------------------------------------------------------------------------
# A. Surface pins — owned behavior
# ---------------------------------------------------------------------------


class TestDriftPins:
    def test_invoke_dependency_attributes_exist_on_owned_class(self) -> None:
        """The invoke path reads these through ``self`` — they must keep existing."""
        t = tool(lambda text: text, name="t", description="d")
        assert hasattr(t, "_schema_supplied")
        assert hasattr(t, "_context_parameter_name")
        assert hasattr(t, "_dump_arguments")
        assert hasattr(t, "_invoke_function")
        assert hasattr(t, "_null_overlay_policy")
        assert hasattr(t, "result_parser")
        assert hasattr(t, "declaration_only")
        assert hasattr(t, "input_model")

    def test_parse_result_is_chrys_owned_and_returns_chrys_content(self) -> None:
        assert FunctionTool.parse_result.__module__ == "chrys.kernel.tools"
        parsed = FunctionTool.parse_result("owned")
        assert type(parsed[0]) is Content

    def test_parse_result_converts_foreign_content_items(self) -> None:
        parsed = FunctionTool.parse_result([_foreign_content("mcp residual")])
        assert [type(item) for item in parsed] == [Content]
        assert parsed[0].text == "mcp residual"

    def test_parse_result_falls_back_for_invalid_content_like_objects(self) -> None:
        class ContentLike:
            type = "not_a_content_type"

            def to_dict(self, exclude_none: bool = False) -> dict[str, Any]:
                return {"type": self.type, "payload": {"kept": True}, "exclude_none": exclude_none}

        parsed = FunctionTool.parse_result(ContentLike())

        assert [type(item) for item in parsed] == [Content]
        assert json.loads(parsed[0].text or "{}") == {
            "type": "not_a_content_type",
            "payload": {"kept": True},
            "exclude_none": False,
        }

    def test_call_and_invoke_function_are_chrys_owned(self) -> None:
        assert FunctionTool.__call__.__module__ == "chrys.kernel.tools"
        assert FunctionTool._invoke_function.__module__ == "chrys.kernel.tools"
        assert FunctionTool.invoke.__module__ == "chrys.kernel.tools"

    def test_invoke_signature_stays_stable(self) -> None:
        params = inspect.signature(FunctionTool.invoke).parameters
        assert list(params) == ["self", "arguments", "context", "tool_call_id", "skip_parsing", "kwargs"]

    def test_tool_factory_signature_stays_stable(self) -> None:
        params = inspect.signature(tool).parameters
        assert list(params) == [
            "func",
            "name",
            "description",
            "schema",
            "kind",
            "max_invocations",
            "max_invocation_exceptions",
            "additional_properties",
            "result_parser",
        ]

    def test_skip_parsing_sentinel_repr_stays_stable(self) -> None:
        assert repr(SKIP_PARSING) == "SKIP_PARSING"

    def test_serialization_type_identifier_unchanged(self) -> None:
        """Same class name → same snake_case type id → byte-identical dumps."""
        chrys_t = _introspected_tool()
        dumped = chrys_t.to_dict()
        assert dumped["type"] == "function_tool"

    def test_subclass_round_trips_through_inherited_type_identifier(self) -> None:
        # Serialized payloads always carry the instance ``type`` field
        # ("function_tool"), so a subclass's from_dict must validate against
        # that same inherited identifier — the snake_case class-name
        # fallback would reject every payload the subclass itself emitted.
        class _CustomTool(FunctionTool):
            pass

        payload = _CustomTool(name="mytool", description="d").to_dict()
        assert payload["type"] == "function_tool"
        restored = _CustomTool.from_dict(payload)
        assert isinstance(restored, _CustomTool)
        assert restored.name == "mytool"
        assert restored.description == "d"


# ---------------------------------------------------------------------------
# B. invoke — result parsing five states + validation + ctx injection
# ---------------------------------------------------------------------------


class TestInvokeResultParsing:
    @pytest.mark.asyncio
    async def test_default_parse_to_content_list(self) -> None:
        result = await _introspected_tool().invoke(arguments={"text": "hi"})
        assert isinstance(result, list)
        assert [c.type for c in result] == ["text"]
        assert result[0].text == "echo:hi"

    @pytest.mark.asyncio
    async def test_skip_parsing_per_call_returns_raw(self) -> None:
        raw = await _introspected_tool().invoke(arguments={"text": "hi"}, skip_parsing=True)
        assert raw == "echo:hi"

    @pytest.mark.asyncio
    async def test_skip_parsing_via_configured_sentinel(self) -> None:
        t = _introspected_tool(result_parser=SKIP_PARSING)
        raw = await t.invoke(arguments={"text": "hi"})
        assert raw == "echo:hi"

    @pytest.mark.asyncio
    async def test_skip_parsing_sentinel_does_not_coerce_foreign_content(self) -> None:
        marker = _foreign_content("raw foreign content")

        async def impl() -> Any:
            return marker

        t = tool(impl, name="raw_content", description="d", result_parser=SKIP_PARSING)
        raw = await t.invoke(arguments={})

        assert raw is marker

    @pytest.mark.asyncio
    async def test_custom_parser_str_product_is_content_wrapped(self) -> None:
        t = _introspected_tool(result_parser=lambda r: f"parsed:{r}")
        result = await t.invoke(arguments={"text": "hi"})
        assert [c.type for c in result] == ["text"]
        assert result[0].text == "parsed:echo:hi"

    @pytest.mark.asyncio
    async def test_custom_parser_list_product_passes_through(self) -> None:
        marker = [Content.from_text("a"), Content.from_text("b")]
        t = _introspected_tool(result_parser=lambda r: marker)
        result = await t.invoke(arguments={"text": "hi"})
        assert result is marker

    @pytest.mark.asyncio
    async def test_custom_parser_foreign_content_product_is_converted(self) -> None:
        marker = _foreign_content("from custom parser")
        t = _introspected_tool(result_parser=lambda r: marker)

        result = await t.invoke(arguments={"text": "hi"})

        assert [type(item) for item in result] == [Content]
        assert result[0].text == "from custom parser"

    @pytest.mark.asyncio
    async def test_custom_parser_foreign_content_list_is_converted(self) -> None:
        marker = [_foreign_content("a"), _foreign_content("b")]
        t = _introspected_tool(result_parser=lambda r: marker)

        result = await t.invoke(arguments={"text": "hi"})

        assert result is not marker
        assert [type(item) for item in result] == [Content, Content]
        assert [item.text for item in result] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_parser_exception_falls_back_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        def boom(result: Any) -> str:
            raise RuntimeError("parser broke")

        t = _introspected_tool(result_parser=boom)
        with caplog.at_level(logging.WARNING, logger="chrys.kernel.tools"):
            result = await t.invoke(arguments={"text": "hi"})
        assert [c.type for c in result] == ["text"]
        assert result[0].text == "echo:hi"
        assert any("result parser failed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_empty_parsed_list_returned_as_is(self) -> None:
        t = _introspected_tool(result_parser=lambda r: [])
        assert await t.invoke(arguments={"text": "hi"}) == []

    @pytest.mark.asyncio
    async def test_default_parser_wire_shape_stays_stable(self) -> None:
        async def impl(text: str) -> str:
            return f"echo:{text}"

        chrys_t = tool(impl, name="echo", description="d")
        chrys_out = await chrys_t.invoke(arguments={"text": "x"})
        assert [c.to_dict() for c in chrys_out] == [{"type": "text", "text": "echo:x", "additional_properties": {}}]


class TestInvokeValidation:
    @pytest.mark.asyncio
    async def test_declaration_only_refused(self) -> None:
        t = FunctionTool(name="decl", description="d", func=None)
        with pytest.raises(ToolException, match="declaration only"):
            await t.invoke(arguments={})

    @pytest.mark.asyncio
    async def test_derived_model_validates_mapping_arguments(self) -> None:
        with pytest.raises(TypeError, match="Invalid arguments for 'echo'"):
            await _introspected_tool().invoke(arguments={"text": 123})

    @pytest.mark.asyncio
    async def test_derived_model_rejects_foreign_base_model(self) -> None:
        class Other(BaseModel):
            text: str = "x"

        with pytest.raises(TypeError, match="Expected"):
            await _introspected_tool().invoke(arguments=Other())

    def test_static_gate_assigns_full_top_level_and_off_tiers(self) -> None:
        assert classify(_FullGateArguments).tier is OverlayTier.FULL
        assert classify(_NestedSerializerArguments).tier is OverlayTier.TOP_LEVEL
        assert classify(_FieldSerializerArguments).tier is OverlayTier.TOP_LEVEL
        assert classify(_KeySerializerArguments).tier is OverlayTier.TOP_LEVEL
        assert classify(_AliasedTypedDictArguments).tier is OverlayTier.TOP_LEVEL
        assert classify(_DefaultedTypedDictArguments).tier is OverlayTier.TOP_LEVEL
        assert classify(_AliasedDataclassArguments).tier is OverlayTier.TOP_LEVEL
        assert classify(_UnknownContainerArguments).tier is OverlayTier.TOP_LEVEL
        assert classify(_RootSerializerArguments).tier is OverlayTier.OFF
        assert classify(_AnyPydanticDataclassArguments).tier is OverlayTier.FULL
        assert classify(_EnumGateArguments).tier is OverlayTier.FULL

    def test_new_type_detection_does_not_unwrap_ordinary_classes(self) -> None:
        full_policy = classify(_FullGateArguments)
        assert _NullableValue in full_policy.approved_classes
        assert classify(_SupertypeAttributeArguments).tier is OverlayTier.TOP_LEVEL

    @pytest.mark.asyncio
    async def test_validation_metadata_is_full_and_restores_nulls(self) -> None:
        constraints = StringConstraints(min_length=1)
        assert isinstance(constraints, annotated_types.GroupedMetadata)

        received: list[dict[str, Any]] = []

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        metadata_tool = FunctionTool(
            name="validation_metadata",
            description="Restore fields with validation-only metadata.",
            func=capture,
            input_model=_ValidationMetadataArguments,
        )

        assert metadata_tool._null_overlay_policy.tier is OverlayTier.FULL
        await metadata_tool.invoke(arguments={"count": None, "name": None, "score": None})
        assert received == [{"count": None, "name": None, "score": None}]

    def test_serializer_metadata_still_degrades_with_safe_metadata(self) -> None:
        policy = classify(_SafeMetadataWithSerializerArguments)

        assert policy.tier is OverlayTier.TOP_LEVEL
        assert policy.reason == "Annotated serializer metadata"
        assert "value" not in policy.top_level_restorable

    @pytest.mark.asyncio
    async def test_field_info_metadata_cannot_hide_a_serializer(self) -> None:
        received: list[Any] = []

        async def capture(items: Any) -> str:
            received.append(items)
            return "ok"

        carrier_tool = FunctionTool(
            name="field_info_carrier",
            description="FieldInfo metadata is classified recursively.",
            func=capture,
            input_model=_FieldInfoCarrierArguments,
        )

        assert carrier_tool._null_overlay_policy.tier is OverlayTier.TOP_LEVEL
        assert classify(_FieldConstraintArguments).tier is OverlayTier.FULL
        assert classify(_ValidationMetadataArguments).tier is OverlayTier.FULL
        await carrier_tool.invoke(arguments={"items": [{"value": None}]})
        assert received == [[{"owned": True}]]

    @pytest.mark.asyncio
    async def test_custom_core_schema_hooks_degrade_reachable_classes(self) -> None:
        received: list[dict[str, Any]] = []

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        child_tool = FunctionTool(
            name="core_schema_child",
            description="Custom core-schema serialization is outside the full gate.",
            func=capture,
            input_model=_CoreSchemaArguments,
        )
        key_tool = FunctionTool(
            name="core_schema_key",
            description="Custom key serialization is outside the full gate.",
            func=capture,
            input_model=_CoreSchemaKeyArguments,
        )
        dataclass_tool = FunctionTool(
            name="core_schema_dataclass",
            description="Custom dataclass core-schema serialization is outside the full gate.",
            func=capture,
            input_model=_CoreSchemaDataclassArguments,
        )

        assert child_tool._null_overlay_policy.tier is OverlayTier.TOP_LEVEL
        assert key_tool._null_overlay_policy.tier is OverlayTier.TOP_LEVEL
        assert dataclass_tool._null_overlay_policy.tier is OverlayTier.TOP_LEVEL
        assert classify(_NestedArguments).tier is OverlayTier.FULL
        await child_tool.invoke(arguments={"child": {"value": None}})
        await key_tool.invoke(arguments={"payload": {"value": {"value": None}}})
        await dataclass_tool.invoke(arguments={"child": {"value": None}})
        assert received == [
            {"child": {"owned": True}},
            {"payload": {"SERIALIZED": {}}},
            {"child": {"owned": True}},
        ]

    @pytest.mark.asyncio
    async def test_root_model_dump_override_is_off_but_nested_override_is_plain(self) -> None:
        received: list[dict[str, Any]] = []

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        root_tool = FunctionTool(
            name="root_dump_override",
            description="Root model_dump overrides own the root output.",
            func=capture,
            input_model=_RootDumpOverrideArguments,
        )
        nested_tool = FunctionTool(
            name="nested_dump_override",
            description="Compiled nested serialization ignores model_dump overrides.",
            func=capture,
            input_model=_NestedDumpOverrideArguments,
        )

        assert root_tool._null_overlay_policy.tier is OverlayTier.OFF
        assert nested_tool._null_overlay_policy.tier is OverlayTier.FULL
        await root_tool.invoke(arguments={"value": None})
        await nested_tool.invoke(arguments={"child": {"value": None}})
        assert received == [
            {"owned": True},
            {"child": {"value": None}},
        ]

    @pytest.mark.asyncio
    async def test_exclude_if_fields_keep_claims_and_annotation_checks(self) -> None:
        received: list[dict[str, Any]] = []

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        collision_tool = FunctionTool(
            name="exclude_if_collision",
            description="Conditionally emitting fields retain dump-key claims.",
            func=capture,
            input_model=_ExcludeIfCollisionArguments,
        )
        plain_tool = FunctionTool(
            name="exclude_if_plain",
            description="A non-colliding exclude_if field keeps the graph plain.",
            func=capture,
            input_model=_ExcludeIfPlainArguments,
        )

        assert collision_tool._null_overlay_policy.tier is OverlayTier.TOP_LEVEL
        assert plain_tool._null_overlay_policy.tier is OverlayTier.FULL
        await collision_tool.invoke(arguments={"owned": {"value": None}, "conditional": {"value": None}})
        await plain_tool.invoke(arguments={"safe": None, "conditional": {"value": None}})
        assert received == [
            {"slot": {}},
            {"safe": None, "conditional": {}},
        ]

    @pytest.mark.asyncio
    async def test_derived_dataclass_computed_metadata_wins(self) -> None:
        received: list[Any] = []

        async def capture(child: Any) -> str:
            received.append(child)
            return "ok"

        dataclass_tool = FunctionTool(
            name="derived_computed_alias",
            description="Derived computed-field metadata controls the dataclass gate.",
            func=capture,
            input_model=_ComputedAliasDataclassArguments,
        )

        assert dataclass_tool._null_overlay_policy.tier is OverlayTier.TOP_LEVEL
        await dataclass_tool.invoke(arguments={"child": {"owned": {"value": None}}})
        assert received == [{"owned": {}}]

    @pytest.mark.asyncio
    async def test_declared_slot_classes_guard_runtime_substitutions(self) -> None:
        received: list[dict[str, Any]] = []

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        slot_tool = FunctionTool(
            name="slot_classes",
            description="Runtime descent stays within each declared slot closure.",
            func=capture,
            input_model=_SlotClassArguments,
        )

        assert slot_tool._null_overlay_policy.tier is OverlayTier.FULL
        await slot_tool.invoke(
            arguments={
                "child": {"value": "child"},
                "other": {"value": "other", "extra": None},
                "choice": {"kind": "right", "value": None},
            }
        )
        assert received == [
            {
                "child": {"value": "child"},
                "other": {"value": "other", "extra": None},
                "choice": {"kind": "right", "value": None},
            }
        ]

    @pytest.mark.asyncio
    async def test_root_model_unwraps_before_dump_shape_guard(self) -> None:
        received: list[Any] = []

        async def capture(payload: Any) -> str:
            received.append(payload)
            return "ok"

        root_tool = FunctionTool(
            name="root_model_restore",
            description="RootModel values restore against their direct dump node.",
            func=capture,
            input_model=_RootModelArguments,
        )

        assert root_tool._null_overlay_policy.tier is OverlayTier.FULL
        await root_tool.invoke(arguments={"payload": [{"value": None}]})
        assert received == [[{"value": None}]]

    @pytest.mark.asyncio
    async def test_serialization_defaults_required_does_not_require_omitted_defaults(self) -> None:
        received: list[dict[str, Any]] = []

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        defaults_tool = FunctionTool(
            name="serialization_defaults_required",
            description="The callable mirror requires validation-required fields only.",
            func=capture,
            input_model=_SerializationDefaultsRequiredArguments,
        )

        await defaults_tool.invoke(arguments={})
        await defaults_tool.invoke(arguments={"value": None})
        assert received == [{}, {"value": None}]

    @pytest.mark.parametrize(
        ("input_model", "arguments", "expected", "tier"),
        [
            (
                _GroupedSerializerArguments,
                {"safe": None, "child": {"value": None}},
                {"safe": None, "child": {"grouped_owned": True}},
                OverlayTier.TOP_LEVEL,
            ),
            (
                _NewTypeSerializerArguments,
                {"child": {"value": None}},
                {"child": {"new_type_owned": True}},
                OverlayTier.TOP_LEVEL,
            ),
            (
                _TypedDictSerializerArguments,
                {"payload": {"child": {"value": None}}},
                {"payload": {"typed_dict_owned": True}},
                OverlayTier.TOP_LEVEL,
            ),
            (
                _MetaclassSerializerArguments,
                {"value": None},
                {"metaclass_owned": True},
                OverlayTier.OFF,
            ),
        ],
        ids=["grouped-metadata", "new-type", "typed-dict", "metaclass-property"],
    )
    @pytest.mark.asyncio
    async def test_compiled_schema_scan_covers_every_serializer_ingress(
        self,
        input_model: type[BaseModel],
        arguments: dict[str, Any],
        expected: dict[str, Any],
        tier: OverlayTier,
    ) -> None:
        received: list[dict[str, Any]] = []

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        schema_tool = FunctionTool(
            name="compiled_schema_gate",
            description="Compiled function serialization controls the final gate.",
            func=capture,
            input_model=input_model,
        )

        assert schema_tool._null_overlay_policy.tier is tier
        await schema_tool.invoke(arguments=arguments)
        assert received == [expected]

    @pytest.mark.asyncio
    async def test_compiled_scan_preserves_plain_and_precise_early_rules(self) -> None:
        received: list[dict[str, Any]] = []

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        root_policy = classify(_RootSerializerArguments)
        nested_tool = FunctionTool(
            name="nested_plain_serializer",
            description="A nested serializer degrades without owning the root mapping.",
            func=capture,
            input_model=_NestedPlainSerializerArguments,
        )

        assert classify(_FullGateArguments).tier is OverlayTier.FULL
        assert classify(_ValidationMetadataArguments).tier is OverlayTier.FULL
        assert root_policy.tier is OverlayTier.OFF
        assert root_policy.reason == "model serializer on _RootSerializerArguments"
        assert nested_tool._null_overlay_policy.tier is OverlayTier.TOP_LEVEL
        await nested_tool.invoke(arguments={"safe": None, "child": {"value": None}})
        assert received == [{"safe": None, "child": {"plain_owned": True}}]

    @pytest.mark.asyncio
    async def test_typed_dict_config_follows_orig_bases_precedence(self) -> None:
        received: list[Any] = []

        async def capture(payload: Any) -> str:
            received.append(payload)
            return "ok"

        inherited_tool = FunctionTool(
            name="inherited_typed_dict_config",
            description="Inherited TypedDict alias generators reject the full gate.",
            func=capture,
            input_model=_InheritedAliasArguments,
        )
        precedence_value = _PrecedenceAliasArguments.model_validate(
            {"payload": {"own_first": None, "own_second": "kept"}}
        )

        assert inherited_tool._null_overlay_policy.tier is OverlayTier.TOP_LEVEL
        assert classify(_PlainInheritedArguments).tier is OverlayTier.FULL
        assert classify(_PrecedenceAliasArguments).tier is OverlayTier.TOP_LEVEL
        assert precedence_value.model_dump() == {"payload": {"own_first": None, "own_second": "kept"}}
        await inherited_tool.invoke(arguments={"payload": {"FIRST": None, "SECOND": "kept"}})
        assert received == [{"SECOND": "kept"}]

    @pytest.mark.parametrize("input_model", [_CoDeclaredTupleArguments, _CoDeclaredUnionArguments])
    def test_subclass_pairs_in_one_slot_degrade(self, input_model: type[BaseModel]) -> None:
        policy = classify(input_model)

        assert policy.tier is OverlayTier.TOP_LEVEL
        assert policy.reason == "co-declared subclass pair _SlotBase/_SlotSub in one slot"

    @pytest.mark.asyncio
    async def test_tuple_subclass_pair_output_is_not_descended(self) -> None:
        received: list[Any] = []

        async def capture(pair: Any) -> str:
            received.append(pair)
            return "ok"

        pair_tool = FunctionTool(
            name="subclass_pair",
            description="A co-declared subclass pair leaves nested output untouched.",
            func=capture,
            input_model=_CoDeclaredTupleArguments,
        )
        await pair_tool.invoke(
            arguments={
                "pair": [
                    {"value": "first"},
                    {"value": "second", "extra": "kept"},
                ]
            }
        )

        assert received == [
            (
                {"value": "first"},
                {"value": "second", "extra": "kept"},
            )
        ]

    @pytest.mark.asyncio
    async def test_non_function_compiled_serialization_degrades_nested_restore(self) -> None:
        received: list[Any] = []

        async def capture(child: Any) -> str:
            received.append(child)
            return "ok"

        serialized_tool = FunctionTool(
            name="model_serialization_node",
            description="Any compiled serialization node owns its output.",
            func=capture,
            input_model=_ModelSerializationArguments,
        )

        assert serialized_tool._null_overlay_policy.tier is OverlayTier.TOP_LEVEL
        assert classify(_FullGateArguments).tier is OverlayTier.FULL
        assert classify(_ValidationMetadataArguments).tier is OverlayTier.FULL
        assert classify(_PydanticInternalSerializationArguments).tier is OverlayTier.TOP_LEVEL
        await serialized_tool.invoke(arguments={"child": {"value": None}})
        assert received == [{}]

    @pytest.mark.asyncio
    async def test_degraded_root_model_without_compiled_fields_restores_no_slots(self) -> None:
        received: list[dict[str, Any]] = []

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        root_tool = FunctionTool(
            name="serialized_root",
            description="An unverifiable RootModel slot remains untouched.",
            func=capture,
            input_model=_SerializedRootArguments,
        )

        policy = root_tool._null_overlay_policy
        assert policy.tier is OverlayTier.TOP_LEVEL
        assert policy.top_level_restorable == frozenset()
        assert classify(_AnyUrlWithSafeRootFieldArguments).top_level_restorable == frozenset({"safe", "url"})
        await root_tool.invoke(arguments={})
        assert received == [{"owned": True}]

    @pytest.mark.asyncio
    async def test_serialization_named_field_is_not_a_compiled_schema_node(self) -> None:
        received: list[dict[str, Any]] = []

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        plain_tool = FunctionTool(
            name="serialization_named_field",
            description="Field names cannot impersonate compiled schema entries.",
            func=capture,
            input_model=_SerializationNamedFieldArguments,
        )

        assert plain_tool._null_overlay_policy.tier is OverlayTier.FULL
        assert classify(_ModelSerializationArguments).tier is OverlayTier.TOP_LEVEL
        assert classify(_PydanticInternalSerializationArguments).tier is OverlayTier.TOP_LEVEL
        await plain_tool.invoke(arguments={"serialization": "plain", "child": {"value": None}})
        assert received == [{"serialization": "plain", "child": {"value": None}}]

    @pytest.mark.asyncio
    async def test_root_model_fields_serialization_disables_restoration(self) -> None:
        received: list[dict[str, Any]] = []

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        root_tool = FunctionTool(
            name="model_fields_serializer",
            description="Serialization on root model fields owns the output mapping.",
            func=capture,
            input_model=_ModelFieldsSerializerArguments,
        )

        assert root_tool._null_overlay_policy.tier is OverlayTier.OFF
        await root_tool.invoke(arguments={"value": None})
        assert received == [{"owned": True}]

    @pytest.mark.asyncio
    async def test_definition_references_preserve_compiled_field_ownership(self) -> None:
        received: list[dict[str, Any]] = []

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        serialized_tool = FunctionTool(
            name="recursive_serialized_child",
            description="Definition references retain nested serializer ownership.",
            func=capture,
            input_model=_RecursiveSerializedArguments,
        )

        serialized_policy = serialized_tool._null_overlay_policy
        plain_policy = classify(_RecursivePlainDegradedArguments)
        assert serialized_policy.tier is OverlayTier.TOP_LEVEL
        assert "child" not in serialized_policy.top_level_restorable
        assert plain_policy.tier is OverlayTier.TOP_LEVEL
        assert plain_policy.top_level_restorable == frozenset({"child", "url"})
        await serialized_tool.invoke(arguments={"child": None})
        assert received == [{}]

    @pytest.mark.asyncio
    async def test_defaulted_dataclass_fields_are_not_descended_without_provenance(self) -> None:
        received: list[Any] = []

        async def capture(child: Any) -> str:
            received.append(child)
            return "ok"

        defaulted_tool = FunctionTool(
            name="defaulted_dataclass_descent",
            description="Defaulted dataclass fields have no nested provenance.",
            func=capture,
            input_model=_DefaultedNestedDataclassArguments,
        )

        await defaulted_tool.invoke(arguments={"child": {}})
        await defaulted_tool.invoke(arguments={"child": {"payload": {"value": None}}})
        assert received == [
            {"payload": {}},
            {"payload": {}},
        ]

    @pytest.mark.asyncio
    async def test_required_dataclass_fields_descend_and_non_none_defaults_restore_null(self) -> None:
        received: list[Any] = []

        async def capture(child: Any) -> str:
            received.append(child)
            return "ok"

        required_tool = FunctionTool(
            name="required_dataclass_descent",
            description="Required dataclass fields retain nested provenance.",
            func=capture,
            input_model=_RequiredNestedDataclassArguments,
        )
        direct_null_tool = FunctionTool(
            name="non_none_dataclass_default",
            description="A non-None dataclass default proves an explicit null.",
            func=capture,
            input_model=_NonNoneDefaultDataclassArguments,
        )

        await required_tool.invoke(arguments={"child": {"payload": {"value": None}}})
        await direct_null_tool.invoke(arguments={"child": {"value": None}})
        assert received == [
            {"payload": {"value": None}},
            {"value": None},
        ]

    @pytest.mark.asyncio
    async def test_compiled_field_serialization_excludes_only_its_top_level_slot(self) -> None:
        received: list[dict[str, Any]] = []

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        serialized_tool = FunctionTool(
            name="custom_slot_serialization",
            description="Compiled field serialization owns its root slot.",
            func=capture,
            input_model=_CustomSlotSerializerArguments,
        )

        policy = serialized_tool._null_overlay_policy
        assert policy.tier is OverlayTier.TOP_LEVEL
        assert policy.top_level_restorable == frozenset({"safe"})
        await serialized_tool.invoke(arguments={"safe": None, "value": None})
        assert received == [{"safe": None}]

    @pytest.mark.asyncio
    async def test_declared_field_and_same_named_extra_restore_independently(self) -> None:
        received: list[dict[str, Any]] = []

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        aliased_tool = FunctionTool(
            name="field_extra_internal_name",
            description="Declared fields and extras have independent provenance.",
            func=capture,
            input_model=_AliasedFieldWithInternalNameExtraArguments,
        )

        await aliased_tool.invoke(arguments={"wire": None, "value": "extra"})
        assert received == [{"wire": None, "value": "extra"}]

    @pytest.mark.asyncio
    async def test_excluded_fields_do_not_reserve_extra_dump_keys(self) -> None:
        received: list[dict[str, Any]] = []

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        excluded_tool = FunctionTool(
            name="excluded_field_extra_key",
            description="Excluded fields cannot collide with emitted extras.",
            func=capture,
            input_model=_ExcludedFieldWithAliasedExtraArguments,
        )
        emitting_tool = FunctionTool(
            name="emitting_field_extra_key",
            description="Emitting fields still claim their dump keys.",
            func=capture,
            input_model=_EmittingFieldWithCollidingExtraArguments,
        )

        await excluded_tool.invoke(arguments={"x": None})
        await emitting_tool.invoke(arguments={"inputVisible": None, "x": None})
        assert received == [{"x": None}, {}]

    @pytest.mark.asyncio
    async def test_mapping_validation_cannot_mutate_the_caller_payload(self) -> None:
        received: list[list[dict[str, int]]] = []
        original = {"items": [{"x": 1}]}

        async def capture(items: list[dict[str, int]]) -> str:
            received.append(items)
            return "ok"

        consuming_tool = FunctionTool(
            name="isolated_mapping_validation",
            description="Before validators receive an isolated mapping.",
            func=capture,
            input_model=_ConsumingInvokeArguments,
        )

        await consuming_tool.invoke(arguments=original)
        assert received == [[{"x": 1}]]
        assert original == {"items": [{"x": 1}]}

    @pytest.mark.asyncio
    async def test_cyclic_mapping_validation_remains_supported(self) -> None:
        received: list[bool] = []
        cycle: dict[str, Any] = {}
        cycle["self"] = cycle

        async def capture(saw_cycle: bool) -> str:
            received.append(saw_cycle)
            return "ok"

        cyclic_tool = FunctionTool(
            name="cyclic_mapping_validation",
            description="Deep-copy validation preserves cyclic container topology.",
            func=capture,
            input_model=_CycleNormalizingArguments,
        )

        await cyclic_tool.invoke(arguments={"payload": cycle})
        assert received == [True]
        assert cycle["self"] is cycle

    @pytest.mark.asyncio
    async def test_unhashable_annotation_degrades_without_bricking_invocation(self) -> None:
        received: list[str] = []

        async def capture(value: str) -> str:
            received.append(value)
            return "ok"

        annotation_tool = FunctionTool(
            name="unhashable_annotation",
            description="Unsupported annotation objects degrade classification.",
            func=capture,
            input_model=_UnhashableAnnotationArguments,
        )

        assert annotation_tool._null_overlay_policy.tier is OverlayTier.TOP_LEVEL
        await annotation_tool.invoke(arguments={"value": "kept"})
        assert received == ["kept"]

    @pytest.mark.asyncio
    async def test_classifier_exceptions_use_a_cached_degraded_policy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        received: list[dict[str, Any]] = []

        class _Arguments(BaseModel):
            value: str | None

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        def fail_classification(model: type[BaseModel]) -> Any:
            raise RuntimeError("classification failed")

        monkeypatch.setattr("chrys.kernel.tools.classify", fail_classification)
        degraded_tool = FunctionTool(
            name="classifier_failure",
            description="Classifier failures conservatively retain top-level restore.",
            func=capture,
            input_model=_Arguments,
        )

        first = degraded_tool._null_overlay_policy
        second = degraded_tool._null_overlay_policy
        assert first is second
        assert first.tier is OverlayTier.TOP_LEVEL
        await degraded_tool.invoke(arguments={"value": None})
        assert received == [{"value": None}]

    def test_tool_caches_one_policy_and_logs_one_degraded_classification(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        degraded_tool = FunctionTool(
            name="degraded_policy",
            description="Cache a static policy.",
            func=lambda **kwargs: kwargs,
            input_model=_NestedSerializerArguments,
        )
        with caplog.at_level(logging.DEBUG, logger="chrys.kernel.tools"):
            first = degraded_tool._null_overlay_policy
            second = degraded_tool._null_overlay_policy

        assert first is second
        records = [record for record in caplog.records if "degraded_policy" in record.getMessage()]
        assert len(records) == 1

    @pytest.mark.asyncio
    async def test_explicit_and_nested_nulls_reach_callable_under_full_policy(self) -> None:
        received: list[dict[str, Any]] = []

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        explicit_tool = FunctionTool(
            name="full_restore",
            description="Restore statically approved nulls.",
            func=capture,
            input_model=_FullGateArguments,
        )
        await explicit_tool.invoke(
            arguments={
                "child": {"value": None},
                "members": {"value": None},
                "items": [{"value": None}],
                "mapping": {"x": {"value": None}},
                "std_new": {"value": None},
                "ext_new": {"value": None},
                "validated": "7",
                "choice": None,
                "sequence": [{"value": None}],
                "abstract_mapping": {"a": {"value": None}},
                "mutable_mapping": {"b": {"value": None}},
                "members_set": [None, "x"],
                "frozen_members": [None, "x"],
                "mode": "plain",
            }
        )

        assert received == [
            {
                "child": {"value": None},
                "members": {"value": None},
                "items": [{"value": None}],
                "mapping": {"x": {"value": None}},
                "std_new": {"value": None},
                "ext_new": {"value": None},
                "validated": 7,
                "choice": None,
                "sequence": [{"value": None}],
                "abstract_mapping": {"a": {"value": None}},
                "mutable_mapping": {"b": {"value": None}},
                "members_set": {None, "x"},
                "frozen_members": frozenset({None, "x"}),
                "mode": "plain",
            }
        ]

    @pytest.mark.asyncio
    async def test_tuple_restore_preserves_python_mode_tuple(self) -> None:
        received: list[Any] = []

        async def capture(items: Any) -> str:
            received.append(items)
            return "ok"

        tuple_tool = FunctionTool(
            name="tuple_restore",
            description="Tuple dumps retain tuple structure.",
            func=capture,
            input_model=_TupleArguments,
        )
        await tuple_tool.invoke(arguments={"items": [{"value": None}, {"value": "x"}]})

        assert received == [({"value": None}, {"value": "x"})]

    @pytest.mark.asyncio
    async def test_typed_dict_omitted_member_stays_absent(self) -> None:
        received: list[Any] = []

        async def capture(payload: Any) -> str:
            received.append(payload)
            return "ok"

        td_tool = FunctionTool(
            name="typed_dict_restore",
            description="Restore present members only.",
            func=capture,
            input_model=_TypedDictArguments,
        )
        await td_tool.invoke(arguments={"payload": {"value": None}})

        assert received == [{"value": None}]

    @pytest.mark.asyncio
    async def test_dataclass_static_null_provenance_and_pydantic_dataclass_restore(self) -> None:
        received: list[Any] = []

        async def capture(child: Any) -> str:
            received.append(child)
            return "ok"

        std_tool = FunctionTool(
            name="stdlib_dataclass_restore",
            description="Use static dataclass provenance.",
            func=capture,
            input_model=_DataclassArguments,
        )
        pdc_tool = FunctionTool(
            name="pydantic_dataclass_restore",
            description="Restore a required pydantic-dataclass null.",
            func=capture,
            input_model=_PydanticDataclassArguments,
        )

        await std_tool.invoke(arguments={"child": {"required": None, "non_none_default": None}})
        await pdc_tool.invoke(arguments={"child": {"value": None}})

        assert received == [
            {"required": None, "non_none_default": None},
            {"value": None},
        ]

    @pytest.mark.asyncio
    async def test_dataclass_field_metadata_defaults_do_not_invent_nulls(self) -> None:
        received: list[Any] = []

        async def capture(child: Any) -> str:
            received.append(child)
            return "ok"

        metadata_tool = FunctionTool(
            name="metadata_defaults",
            description="FieldInfo defaults are not caller provenance.",
            func=capture,
            input_model=_MetadataDefaultDataclassArguments,
        )
        await metadata_tool.invoke(arguments={"child": {"required": None}})

        assert received == [{"required": None}]

    @pytest.mark.asyncio
    async def test_alias_extras_exclusion_and_computed_fields(self) -> None:
        received: list[dict[str, Any]] = []

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        alias_tool = FunctionTool(
            name="alias_restore",
            description="Use explicit model aliases.",
            func=capture,
            input_model=_AliasedNullable,
        )
        extra_tool = FunctionTool(
            name="extra_restore",
            description="Restore explicit extra nulls by raw key.",
            func=capture,
            input_model=_ExtraArguments,
        )
        excluded_tool = FunctionTool(
            name="excluded_restore",
            description="Never restore excluded fields.",
            func=capture,
            input_model=_ExcludedArguments,
        )
        computed_tool = FunctionTool(
            name="computed_output",
            description="Computed fields are not callable arguments.",
            func=capture,
            input_model=_ComputedArguments,
        )

        await alias_tool.invoke(arguments={"wire": None})
        await extra_tool.invoke(arguments={"runtime": None})
        await excluded_tool.invoke(arguments={"visible": None, "hidden": None})
        await computed_tool.invoke(arguments={"count": 2})

        assert received == [
            {"wire": None},
            {"anchor": 1, "runtime": None},
            {"visible": None},
            {"count": 2},
        ]

    @pytest.mark.asyncio
    async def test_top_level_tier_restores_only_safe_root_fields(self) -> None:
        global _NESTED_SERIALIZER_CALLS, _FIELD_SERIALIZER_CALLS
        _NESTED_SERIALIZER_CALLS = 0
        _FIELD_SERIALIZER_CALLS = 0
        received: list[dict[str, Any]] = []

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        nested_tool = FunctionTool(
            name="nested_serializer_gate",
            description="Nested serializer output remains authoritative.",
            func=capture,
            input_model=_NestedSerializerArguments,
        )
        field_tool = FunctionTool(
            name="field_serializer_gate",
            description="Field serializer output remains authoritative.",
            func=capture,
            input_model=_FieldSerializerArguments,
        )
        assert nested_tool._null_overlay_policy.tier is OverlayTier.TOP_LEVEL
        assert field_tool._null_overlay_policy.tier is OverlayTier.TOP_LEVEL
        assert _NESTED_SERIALIZER_CALLS == 0
        assert _FIELD_SERIALIZER_CALLS == 0

        await nested_tool.invoke(arguments={"safe": None, "child": {"value": None}})
        await field_tool.invoke(arguments={"safe": None, "text": "hello"})

        assert received == [
            {"safe": None, "child": {"owned": True}},
            {"safe": None, "text": "HELLO"},
        ]
        assert _NESTED_SERIALIZER_CALLS == 1
        assert _FIELD_SERIALIZER_CALLS == 1

    @pytest.mark.asyncio
    async def test_off_tier_preserves_root_serializer_output(self) -> None:
        received: list[dict[str, Any]] = []

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        root_tool = FunctionTool(
            name="root_serializer_gate",
            description="Root serializer output is not modified.",
            func=capture,
            input_model=_RootSerializerArguments,
        )
        await root_tool.invoke(arguments={"value": None})

        assert received == [{"owned": True}]

    @pytest.mark.asyncio
    async def test_serialized_mapping_keys_are_uniformly_left_untouched(self) -> None:
        global _KEY_SERIALIZER_CALLS
        _KEY_SERIALIZER_CALLS = 0
        received: list[Any] = []

        async def capture(payload: Any) -> str:
            received.append(payload)
            return "ok"

        key_tool = FunctionTool(
            name="serialized_keys",
            description="Serialized mapping keys reject nested restoration.",
            func=capture,
            input_model=_KeySerializerArguments,
        )
        assert key_tool._null_overlay_policy.tier is OverlayTier.TOP_LEVEL
        assert _KEY_SERIALIZER_CALLS == 0
        await key_tool.invoke(arguments={"payload": {"X": {"value": None}, "x": {}}})
        await key_tool.invoke(arguments={"payload": {"x": {"value": None}}})

        assert received == [{"X": {}}, {"X": {}}]
        assert _KEY_SERIALIZER_CALLS == 3

    @pytest.mark.asyncio
    async def test_supertype_class_attribute_does_not_bypass_serializer_gate(self) -> None:
        received: list[Any] = []

        async def capture(child: Any) -> str:
            received.append(child)
            return "ok"

        supertype_tool = FunctionTool(
            name="supertype_attribute",
            description="Ordinary classes are not NewType objects.",
            func=capture,
            input_model=_SupertypeAttributeArguments,
        )
        await supertype_tool.invoke(arguments={"child": {"value": None}})

        assert received == [{"owned": True}]

    @pytest.mark.asyncio
    async def test_runtime_dataclass_under_any_is_an_unapproved_leaf(self) -> None:
        baseline = _PDC_ALIAS_GENERATOR_CALLS
        received: list[Any] = []

        async def capture(payload: Any) -> str:
            received.append(payload)
            return "ok"

        any_tool = FunctionTool(
            name="any_pydantic_dataclass",
            description="Any-position runtime classes require static approval.",
            func=capture,
            input_model=_AnyPydanticDataclassArguments,
        )
        declared_tool = FunctionTool(
            name="declared_generated_dataclass",
            description="Generated dataclass aliases reject nested restoration.",
            func=capture,
            input_model=_DeclaredPydanticDataclassArguments,
        )
        payload = _GeneratedPydanticDataclass(value=None)

        await any_tool.invoke(arguments={"payload": payload})
        await declared_tool.invoke(arguments={"payload": payload})

        assert received == [{}, {}]
        assert baseline == _PDC_ALIAS_GENERATOR_CALLS

    @pytest.mark.parametrize("value", [None, "x"])
    @pytest.mark.asyncio
    async def test_split_alias_invokes_in_serialization_keyspace(self, value: Any) -> None:
        received: list[dict[str, Any]] = []

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        alias_tool = FunctionTool(
            name="split_alias",
            description="Validation and callable aliases differ.",
            func=capture,
            input_model=_SplitAliasArguments,
        )
        await alias_tool.invoke(arguments={"inputKey": value})

        assert received == [{"callableKey": value}]

    @pytest.mark.asyncio
    async def test_unschemable_serialization_mirror_degrades_without_crashing(self) -> None:
        received: list[dict[str, Any]] = []

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        unschemable_tool = FunctionTool(
            name="unschemable",
            description="Python-mode output can exceed JSON Schema.",
            func=capture,
            input_model=_UnschemableComputedArguments,
        )
        await unschemable_tool.invoke(arguments={"inKey": None})

        assert received[0]["outKey"] is None
        assert callable(received[0]["factory"])

    @pytest.mark.asyncio
    async def test_empty_alias_is_a_real_dump_key(self) -> None:
        received: list[dict[str, Any]] = []

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        alias_tool = FunctionTool(
            name="empty_alias",
            description="Empty aliases remain keys.",
            func=capture,
            input_model=_EmptyAliasArguments,
        )
        await alias_tool.invoke(arguments={"": None})

        assert received == [{"": None}]

    @pytest.mark.asyncio
    async def test_omitted_model_default_keeps_callable_default(self) -> None:
        received: list[int] = []

        async def capture(count: int = 3) -> str:
            received.append(count)
            return "ok"

        default_tool = FunctionTool(
            name="model_default",
            description="Omitted fields stay omitted.",
            func=capture,
            input_model=_DivergentDefaults,
        )
        await default_tool.invoke(arguments={})

        assert received == [7]

    @pytest.mark.asyncio
    async def test_schema_supplied_array_stays_strict_and_raw_null_passes_through(self) -> None:
        received: list[dict[str, Any]] = []

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        schema_tool = FunctionTool(
            name="schema_tool",
            description="Schema-supplied tools bypass typed restoration.",
            func=capture,
            input_model={
                "type": "object",
                "properties": {"items": {"type": "array"}, "value": {"type": ["string", "null"]}},
                "required": ["items", "value"],
                "additionalProperties": False,
            },
        )

        await schema_tool.invoke(arguments={"items": [1, 2], "value": None})
        with pytest.raises(TypeError, match="Invalid type for 'items'"):
            await schema_tool.invoke(arguments={"items": (1, 2), "value": None})

        assert received == [{"items": [1, 2], "value": None}]

    @pytest.mark.asyncio
    async def test_non_mapping_arguments_rejected(self) -> None:
        with pytest.raises(TypeError, match="mapping-like"):
            await _introspected_tool().invoke(arguments=[("text", "hi")])  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_schema_supplied_validation_errors_and_happy_path(self) -> None:
        schema_tool = _schema_supplied_tool()
        with pytest.raises(TypeError, match="Missing required argument"):
            await schema_tool.invoke(arguments={})
        with pytest.raises(TypeError, match="is not in"):
            await schema_tool.invoke(arguments={"text": "hi", "mode": "z"})
        with pytest.raises(TypeError, match="Invalid type for 'text'"):
            await schema_tool.invoke(arguments={"text": 5})
        with pytest.raises(TypeError, match="Unexpected argument"):
            await schema_tool.invoke(arguments={"text": "hi", "extra": 1})
        result = await schema_tool.invoke(arguments={"text": "hi", "mode": "a"})
        assert result[0].text == "echo:hi"

    @pytest.mark.asyncio
    async def test_direct_kwargs_context_and_unexpected_kwargs(self) -> None:
        direct = await _introspected_tool().invoke(text="hi")
        assert direct[0].text == "echo:hi"

        with pytest.raises(TypeError, match="Unexpected keyword argument"):
            await _introspected_tool().invoke(text="hi", runtime_thing=1)

        t = _introspected_tool()
        ctx = FunctionInvocationContext(function=t, arguments={"text": "hi"})
        contextual = await t.invoke(context=ctx)
        assert contextual[0].text == "echo:hi"
        assert ctx.arguments == {"text": "hi"}


class TestInvokeContextInjection:
    """Uses the schema-supplied + unannotated-``ctx`` form — the production
    (MCP) shape. Step 3 owns context detection, so annotating the chrys context
    class is recognized directly."""

    def _ctx_tool(self, seen: list[Any]) -> FunctionTool:
        async def impl(ctx=None, **call_kwargs: Any) -> str:
            seen.append(ctx)
            return str(call_kwargs.get("text"))

        return FunctionTool(name="ctx_tool", description="d", func=impl, input_model=_SCHEMA)

    @pytest.mark.asyncio
    async def test_ctx_parameter_gets_internally_built_chrys_context(self) -> None:
        seen: list[Any] = []
        t = self._ctx_tool(seen)
        assert t._context_parameter_name == "ctx"
        await t.invoke(arguments={"text": "hi"})
        assert len(seen) == 1
        assert isinstance(seen[0], FunctionInvocationContext)
        assert seen[0].arguments == {"text": "hi"}

    @pytest.mark.asyncio
    async def test_passed_context_is_synced_and_injected(self) -> None:
        seen: list[Any] = []
        t = self._ctx_tool(seen)
        ctx = FunctionInvocationContext(function=t, arguments={}, kwargs={"runtime": 1})
        await t.invoke(arguments={"text": "hi"}, context=ctx)
        assert seen[0] is ctx
        assert ctx.arguments == {"text": "hi"}
        assert ctx.kwargs == {"runtime": 1}

    def test_chrys_context_annotation_is_detected(self) -> None:
        @tool(name="annotated")
        async def annotated(text: str, ctx: FunctionInvocationContext) -> str:
            return text

        assert annotated._context_parameter_name == "ctx"
        assert "ctx" not in annotated.parameters()["properties"]


class TestOwnedCallSemantics:
    @pytest.mark.asyncio
    async def test_max_invocations_enforced_via_owned_call(self) -> None:
        t = _introspected_tool(max_invocations=1)
        await t.invoke(arguments={"text": "a"})
        assert t.invocation_count == 1
        with pytest.raises(ToolException, match="maximum invocation limit"):
            await t.invoke(arguments={"text": "b"})

    @pytest.mark.asyncio
    async def test_max_invocation_exceptions_enforced(self) -> None:
        # Sync tool on purpose: ``__call__`` only counts exceptions raised
        # synchronously inside it; an async tool's exception surfaces later,
        # at the ``_invoke_function`` await.
        @tool(name="boom", max_invocation_exceptions=1)
        def boom(text: str) -> str:
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError):
            await boom.invoke(arguments={"text": "a"})
        assert boom.invocation_exception_count == 1
        with pytest.raises(ToolException, match="maximum exception limit"):
            await boom.invoke(arguments={"text": "b"})

    def test_descriptor_binding_preserves_subclass_and_sidecar(self) -> None:
        class Tools:
            def execute(self, command: str) -> str:
                return command

        Tools.execute = FunctionTool(name="execute", description="d", func=Tools.execute)
        Tools.execute.chrys_kind = "shell"
        bound = Tools().execute
        assert bound is not Tools.__dict__["execute"]
        assert type(bound) is FunctionTool
        assert bound.chrys_kind == "shell"
        assert bound._instance is not None


# ---------------------------------------------------------------------------
# C. tool factory — full kwargs passthrough
# ---------------------------------------------------------------------------


class TestToolFactory:
    def test_bare_and_parenthesized_decorator_forms(self) -> None:
        @tool
        def bare(text: str) -> str:
            """Bare."""
            return text

        @tool()
        def parens(text: str) -> str:
            """Parens."""
            return text

        assert type(bare) is FunctionTool
        assert type(parens) is FunctionTool
        assert bare.name == "bare"
        assert bare.description == "Bare."

    def test_full_kwargs_passthrough(self) -> None:
        def parser(result: Any) -> str:
            return str(result)

        built = tool(
            lambda text: text,
            name="custom",
            description="desc",
            schema=_SCHEMA,
            kind="shell",
            max_invocations=3,
            max_invocation_exceptions=2,
            additional_properties={"a": 1},
            result_parser=parser,
        )
        assert type(built) is FunctionTool
        assert built.name == "custom"
        assert built.description == "desc"
        assert built.kind == "shell"
        assert built.max_invocations == 3
        assert built.max_invocation_exceptions == 2
        assert built.additional_properties == {"a": 1}
        assert built.result_parser is parser
        assert built._schema_supplied is True
        assert built.parameters()["properties"] == _SCHEMA["properties"]

    def test_name_and_description_fall_back_to_function(self) -> None:
        @tool
        def named(text: str) -> str:
            """Doc here."""
            return text

        assert named.name == "named"
        assert named.description == "Doc here."


# ---------------------------------------------------------------------------
# D. normalize_tools — full surface
# ---------------------------------------------------------------------------


class TestNormalizeToolsThinLayer:
    def test_callable_wrapped_to_chrys_class(self) -> None:
        def impl(text: str) -> str:
            return text

        out = normalize_tools([impl])
        assert len(out) == 1
        assert type(out[0]) is FunctionTool

    def test_single_noncallable_leaf_is_preserved(self) -> None:
        spec = {"type": "web_search"}
        assert normalize_tools(spec) == [spec]

    def test_dict_and_mcp_tool_pass_through_untouched(self) -> None:
        spec = {"type": "web_search"}
        mcp = MCPStdioTool(name="m", command="true")
        out = normalize_tools([spec, mcp])
        assert out[0] is spec
        assert out[1] is mcp
        assert type(mcp) is MCPStdioTool

    def test_tools_collection_object_flattened(self) -> None:
        # NOT a dict subclass: dicts are preserved before the ``.tools``
        # collection check.
        class Box:
            @property
            def tools(self) -> list[Any]:
                return [lambda text: text]

        out = normalize_tools([Box()])
        assert len(out) == 1
        assert type(out[0]) is FunctionTool

    def test_toolbox_iterable_flattened_with_nested_callable_upcast(self) -> None:
        class Toolbox:
            def __iter__(self) -> Any:
                return iter([lambda text: text, {"type": "web_search"}])

        out = normalize_tools([Toolbox()])
        assert len(out) == 2
        assert type(out[0]) is FunctionTool
        assert out[1] == {"type": "web_search"}

    def test_none_and_empty_return_empty_list(self) -> None:
        assert normalize_tools(None) == []
        assert normalize_tools([]) == []

    def test_chrys_instances_pass_through_unchanged(self) -> None:
        t = _introspected_tool()
        out = normalize_tools([t])
        assert out[0] is t
        assert type(t) is FunctionTool

    def test_non_framework_function_tool_like_callable_is_wrapped(self) -> None:
        class ForeignTool:
            name = "foreign"
            func = None

            def __call__(self, text: str) -> str:
                return f"foreign:{text}"

            def parameters(self) -> dict[str, Any]:
                return {}

            def to_json_schema_spec(self) -> dict[str, Any]:
                return {}

        foreign = ForeignTool()
        out = normalize_tools([foreign])
        assert len(out) == 1
        assert type(out[0]) is FunctionTool
        assert out[0].func is foreign
        assert out[0](text="x") == "foreign:x"


# ---------------------------------------------------------------------------
# E. Production-path guarantee — everything reaching the loop through the
# agent is the chrys subclass
# ---------------------------------------------------------------------------


class TestAgentPathUpcastGuarantee:
    @pytest.mark.asyncio
    async def test_agent_normalization_delivers_chrys_subclass_to_the_wire(self) -> None:
        from chrys.kernel import Agent

        captured: dict[str, Any] = {}

        class _Wire:
            def get_response(self, messages: Any, *, stream: bool = False, options: Any = None, **kwargs: Any) -> Any:
                captured["tools"] = list((options or {}).get("tools") or [])

                async def _resolve() -> ChatResponse:
                    return ChatResponse(messages=[Message(role="assistant", contents=["done"])])

                return _resolve()

        ctor_tool = tool(lambda text: text, name="ctor_tool", description="d")

        def runtime_impl(text: str) -> str:
            return text

        agent = Agent(client=_Wire(), tools=[ctor_tool])

        await agent.run("hi", tools=[runtime_impl])
        delivered = captured["tools"]
        assert {t.name for t in delivered} == {"ctor_tool", "runtime_impl"}
        assert all(type(t) is FunctionTool for t in delivered)


class TestSyncWorkerCancellationDrain:
    """Cancellation racing a threaded sync tool that already returned."""

    @staticmethod
    async def _poll_until_done(pending: asyncio.Future) -> None:
        # Poll instead of awaiting: the manual ``coro.send`` drive already
        # consumed the future's ``__await__`` yield, so a second await would
        # trip its resumed-while-pending guard.
        for _ in range(1000):
            if pending.done():
                return
            await asyncio.sleep(0.005)
        raise AssertionError("worker future never resolved")

    @pytest.mark.asyncio
    async def test_cancel_after_worker_completion_carries_parsed_result(self) -> None:
        """Cancellation in the worker-done/await-suspended window drains the value."""
        release = ThreadEvent()

        @tool(name="threaded")
        def threaded() -> str:
            release.wait(timeout=5)
            return "done"

        coro = threaded.invoke(arguments={}, tool_call_id="c1")
        pending = coro.send(None)
        assert isinstance(pending, asyncio.Future)
        assert not pending.done()
        release.set()
        # The worker records its outcome strictly before the executor future
        # resolves, so a resolved future pins the exact race window: value
        # produced, ``await`` in the invoke coroutine not yet resumed.
        await self._poll_until_done(pending)
        with pytest.raises(SyncToolCancelledAfterCompletion) as exc_info:
            coro.throw(asyncio.CancelledError())
        completed = exc_info.value.completed_result
        assert [content.text for content in completed] == ["done"]

    @pytest.mark.asyncio
    async def test_cancelled_future_with_completed_worker_still_drains(self) -> None:
        """Race shape (b): the future loses to cancellation but the worker still finished.

        Pins the drain signal to the completion record rather than future
        state — a future-state implementation would see only the cancelled
        future here and drop the completed value.
        """
        release = ThreadEvent()
        body_done = ThreadEvent()

        @tool(name="threaded")
        def threaded() -> str:
            release.wait(timeout=30)
            body_done.set()
            return "done"

        coro = threaded.invoke(arguments={}, tool_call_id="c1")
        pending = coro.send(None)
        assert not pending.done()
        pending.cancel()
        release.set()
        assert await asyncio.to_thread(body_done.wait, 2)
        # The worker holds the GIL from ``body_done.set()`` through the
        # completion record, so the record precedes this coroutine's
        # resumption; the sleep is belt-and-braces on top of that ordering.
        await asyncio.sleep(0.05)
        with pytest.raises(SyncToolCancelledAfterCompletion) as exc_info:
            coro.throw(asyncio.CancelledError())
        completed = exc_info.value.completed_result
        assert [content.text for content in completed] == ["done"]

    @pytest.mark.asyncio
    async def test_cancel_while_worker_still_running_stays_plain_cancellation(self) -> None:
        started = ThreadEvent()
        release = ThreadEvent()

        @tool(name="blocked")
        def blocked() -> str:
            started.set()
            release.wait(timeout=5)
            return "late"

        coro = blocked.invoke(arguments={}, tool_call_id="c1")
        pending = coro.send(None)
        assert await asyncio.to_thread(started.wait, 2)
        assert not pending.done()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            coro.throw(asyncio.CancelledError())
        assert not isinstance(exc_info.value, SyncToolCancelledAfterCompletion)
        release.set()
        await self._poll_until_done(pending)
        assert pending.result() == "late"
