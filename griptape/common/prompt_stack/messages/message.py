from __future__ import annotations

from typing import Any, TypeVar

from attrs import define, field

from griptape.artifacts import BaseArtifact, ListArtifact, TextArtifact
from griptape.common import BaseMessageContent, TextMessageContent

from .base_message import BaseMessage

T = TypeVar("T", bound=BaseMessageContent)


@define
class Message(BaseMessage):
    def __init__(self, content: str | list[BaseMessageContent], **kwargs: Any) -> None:
        if isinstance(content, str):
            content = [TextMessageContent(TextArtifact(value=content))]
        self.__attrs_init__(content, **kwargs)  # pyright: ignore[reportAttributeAccessIssue]

    content: list[BaseMessageContent] = field(metadata={"serializable": True})

    @property
    def value(self) -> Any:
        return self.to_artifact().value

    def __str__(self) -> str:
        return self.to_text()

    def has_all_content_type(self, content_type: type[T]) -> bool:
        return all(isinstance(content, content_type) for content in self.content)

    def has_any_content_type(self, content_type: type[T]) -> bool:
        return any(isinstance(content, content_type) for content in self.content)

    def get_content_type(self, content_type: type[T]) -> list[T]:
        return [content for content in self.content if isinstance(content, content_type)]

    def exclude_content_type(self, content_type: type[T] | tuple[type[T]]) -> list[BaseMessageContent]:
        return [content for content in self.content if not isinstance(content, content_type)]

    def is_text(self) -> bool:
        return all(isinstance(content, TextMessageContent) for content in self.content)

    def to_text(self) -> str:
        return "".join(
            [content.artifact.to_text() for content in self.content if isinstance(content, TextMessageContent)],
        )

    def to_artifact(self) -> BaseArtifact:
        if len(self.content) == 1:
            return self.content[0].artifact
        else:
            return ListArtifact([content.artifact for content in self.content])
