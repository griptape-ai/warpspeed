from __future__ import annotations

from typing import Optional

from attr import define, field

from griptape.artifacts import BaseArtifact, TextArtifact
from griptape.mixins import SerializableMixin


@define
class ActionChunkArtifact(TextArtifact, SerializableMixin):
    """An Artifact that represents a chunk of an Action.
    Can be used when streaming with Prompt Drivers that support native function calling.

    Attributes:
        tag: The tag of the action.
        name: The name of the action.
        path: The path of the action.
        partial_input: The partial input of the action.
        index: The index of the action.

    """

    tag: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": True})
    name: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": True})
    path: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": True})
    partial_input: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": True})
    index: Optional[int] = field(default=None, kw_only=True, metadata={"serializable": True})

    def __add__(self, other: BaseArtifact) -> ActionChunkArtifact:
        if isinstance(other, ActionChunkArtifact):
            return ActionChunkArtifact(
                value=self.value + other.value,
                tag=(self.tag or "") + (other.tag or ""),
                name=(self.name or "") + (other.name or ""),
                path=(self.path or "") + (other.path or ""),
                partial_input=(self.partial_input or "") + (other.partial_input or ""),
                index=self.index,
            )
        else:
            return ActionChunkArtifact(
                value=self.value + other.value,
                tag=self.tag,
                name=self.name,
                path=self.path,
                partial_input=self.partial_input,
                index=self.index,
            )
