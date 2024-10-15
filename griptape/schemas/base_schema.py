from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from typing import Any, Literal, TypeVar, Union, _SpecialForm, get_args, get_origin

import attrs
from marshmallow import INCLUDE, Schema, fields

from griptape.schemas.bytes_field import Bytes
from griptape.schemas.union_field import UnionField


class BaseSchema(Schema):
    class Meta:
        unknown = INCLUDE

    DATACLASS_TYPE_MAPPING = {**Schema.TYPE_MAPPING, dict: fields.Dict, bytes: Bytes, Any: fields.Raw}

    @classmethod
    def from_attrs_cls(cls, attrs_cls: type) -> type:
        """Generate a Schema from an attrs class.

        Args:
            attrs_cls: An attrs class.
        """
        from marshmallow import post_load

        from griptape.mixins.serializable_mixin import SerializableMixin

        class SubSchema(cls):
            @post_load
            def make_obj(self, data: Any, **kwargs) -> Any:
                return attrs_cls(**data)

        if issubclass(attrs_cls, SerializableMixin):
            cls._resolve_types(attrs_cls)
            return SubSchema.from_dict(
                {
                    a.alias or a.name: cls._get_field_for_type(a.type)
                    for a in attrs.fields(attrs_cls)
                    if a.metadata.get("serializable")
                },
                name=f"{attrs_cls.__name__}Schema",
            )
        else:
            raise ValueError(f"Class must implement SerializableMixin: {attrs_cls}")

    @classmethod
    def _get_field_for_type(cls, field_type: type) -> fields.Field | fields.Nested:
        """Generate a marshmallow Field instance from a Python type.

        Args:
            field_type: A field type.
        """
        from enum import Enum

        from griptape.schemas.polymorphic_schema import PolymorphicSchema

        field_class, args, optional = cls._get_field_type_info(field_type)

        # Handle TypeVars
        if isinstance(field_class, TypeVar):
            field_class = field_class.__bound__

        if field_class is None:
            return fields.Constant(None, allow_none=True)

        if cls.is_union_(field_type):
            return cls._handle_union(field_type, optional=optional)

        elif attrs.has(field_class):
            schema = PolymorphicSchema if ABC in field_class.__bases__ else cls.from_attrs_cls
            return fields.Nested(schema(field_class), allow_none=optional)

        elif isinstance(field_class, type) and issubclass(field_class, Enum):
            return fields.String(allow_none=optional)

        elif cls.is_list_sequence(field_class):
            if args:
                return cls._handle_list(args[0], optional=optional)
            else:
                raise ValueError(f"Missing type for list field: {field_type}")
        field_class = cls.DATACLASS_TYPE_MAPPING.get(field_class)
        if field_class is None:
            raise ValueError(f"Unsupported field type: {field_type}")
        return field_class(allow_none=optional)

    @classmethod
    def _handle_list(cls, list_type: type, *, optional: bool) -> fields.Field:
        """Handle List Fields, including Union Types.

        Args:
            list_type: The List type to handle.
            optional: Whether the List can be none.

        Returns:
            A marshmallow List field.
        """
        if cls.is_union_(list_type):
            union_field = cls._handle_union(list_type, optional=optional)
            return fields.List(cls_or_instance=union_field, allow_none=optional)
        list_field = cls._get_field_for_type(list_type)
        if isinstance(list_field, fields.Constant) and list_field.constant is None:
            raise ValueError(f"List elements cannot be None: {list_type}")
        return fields.List(cls_or_instance=list_field, allow_none=optional)

    @classmethod
    def _handle_union(cls, union_type: type, *, optional: bool) -> fields.Field:
        """Handle Union Fields, including Unions with List Types.

        Args:
            union_type: The Union Type to handle.
            optional: Whether the Union can be None.

        Returns:
            A marshmallow Union field.
        """
        candidate_fields = [cls._get_field_for_type(arg) for arg in get_args(union_type) if arg is not type(None)]

        if any(arg is type(None) for arg in get_args(union_type)):
            optional = True
        if not candidate_fields:
            raise ValueError(f"Unsupported UnionType field: {union_type}")

        return UnionField(fields=candidate_fields, allow_none=optional)

    @classmethod
    def _get_field_type_info(cls, field_type: type) -> tuple[type, tuple[type, ...], bool]:
        """Get information about a field type.

        Args:
            field_type: A field type.
        """
        origin = get_origin(field_type) or field_type
        args = get_args(field_type)
        optional = False

        if origin is Union:
            origin = args[0]
            if len(args) > 1 and args[1] is type(None):
                optional = True

            origin, args, _ = cls._get_field_type_info(origin)
        elif origin is Literal:
            origin = type(args[0])
            args = ()

        return origin, args, optional

    @classmethod
    def _resolve_types(cls, attrs_cls: type) -> None:
        """Resolve types in an attrs class.

        Args:
            attrs_cls: An attrs class.
        """
        from collections.abc import Sequence
        from typing import Any

        from griptape.artifacts import BaseArtifact
        from griptape.common import (
            BaseDeltaMessageContent,
            BaseMessageContent,
            Message,
            PromptStack,
            Reference,
            ToolAction,
        )
        from griptape.drivers import (
            BaseAudioTranscriptionDriver,
            BaseConversationMemoryDriver,
            BaseEmbeddingDriver,
            BaseImageGenerationDriver,
            BaseImageQueryDriver,
            BasePromptDriver,
            BaseRulesetDriver,
            BaseTextToSpeechDriver,
            BaseVectorStoreDriver,
        )
        from griptape.events import EventListener
        from griptape.memory import TaskMemory
        from griptape.memory.structure import BaseConversationMemory, Run
        from griptape.memory.task.storage import BaseArtifactStorage
        from griptape.structures import Structure
        from griptape.tasks import BaseTask
        from griptape.tokenizers import BaseTokenizer
        from griptape.tools import BaseTool
        from griptape.utils import import_optional_dependency, is_dependency_installed

        attrs.resolve_types(
            attrs_cls,
            localns={
                "Any": Any,
                "BasePromptDriver": BasePromptDriver,
                "BaseImageQueryDriver": BaseImageQueryDriver,
                "BaseEmbeddingDriver": BaseEmbeddingDriver,
                "BaseVectorStoreDriver": BaseVectorStoreDriver,
                "BaseTextToSpeechDriver": BaseTextToSpeechDriver,
                "BaseAudioTranscriptionDriver": BaseAudioTranscriptionDriver,
                "BaseConversationMemoryDriver": BaseConversationMemoryDriver,
                "BaseRulesetDriver": BaseRulesetDriver,
                "BaseImageGenerationDriver": BaseImageGenerationDriver,
                "BaseArtifact": BaseArtifact,
                "PromptStack": PromptStack,
                "EventListener": EventListener,
                "BaseMessageContent": BaseMessageContent,
                "BaseDeltaMessageContent": BaseDeltaMessageContent,
                "BaseTool": BaseTool,
                "BaseTask": BaseTask,
                "State": BaseTask.State,
                "TaskMemory": TaskMemory,
                "Usage": Message.Usage,
                "Structure": Structure,
                "BaseTokenizer": BaseTokenizer,
                "ToolAction": ToolAction,
                "Reference": Reference,
                "Run": Run,
                "Sequence": Sequence,
                "BaseConversationMemory": BaseConversationMemory,
                "BaseArtifactStorage": BaseArtifactStorage,
                # Third party modules
                "Client": import_optional_dependency("cohere").Client if is_dependency_installed("cohere") else Any,
                "GenerativeModel": import_optional_dependency("google.generativeai").GenerativeModel
                if is_dependency_installed("google.generativeai")
                else Any,
                "boto3": import_optional_dependency("boto3") if is_dependency_installed("boto3") else Any,
                "Anthropic": import_optional_dependency("anthropic").Anthropic
                if is_dependency_installed("anthropic")
                else Any,
                "BedrockClient": import_optional_dependency("mypy_boto3_bedrock").BedrockClient
                if is_dependency_installed("mypy_boto3_bedrock")
                else Any,
                "voyageai": import_optional_dependency("voyageai") if is_dependency_installed("voyageai") else Any,
            },
        )

    @classmethod
    def is_list_sequence(cls, field_type: type | _SpecialForm) -> bool:
        if isinstance(field_type, type):
            if issubclass(field_type, str) or issubclass(field_type, bytes) or issubclass(field_type, tuple):
                return False
            else:
                return issubclass(field_type, Sequence)
        else:
            return False

    @classmethod
    def is_union_(cls, field_type: type) -> bool:
        return field_type is Union or get_origin(field_type) is Union
