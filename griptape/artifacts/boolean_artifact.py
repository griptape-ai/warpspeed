from __future__ import annotations
from typing import Any, Union
from attrs import define, field
from griptape.artifacts import BaseArtifact


@define
class BooleanArtifact(BaseArtifact):
    value: bool = field(converter=bool, metadata={"serializable": True})
    meta: dict[str, Any] = field(factory=dict, kw_only=True, metadata={"serializable": True})

    @classmethod
    def parse_bool(cls, value: Union[str, bool, BaseArtifact]) -> BooleanArtifact:
        """
        Convert a string literal or bool to a BooleanArtifact. The string must be either "true" or "false" with any casing.
        """
        if value is not None:
            if isinstance(value, BaseArtifact):
                value = str(value)
            if isinstance(value, str):
                if value.lower() == "true":
                    return BooleanArtifact(True)
                elif value.lower() == "false":
                    return BooleanArtifact(False)
            elif isinstance(value, bool):
                return BooleanArtifact(value)
        raise ValueError(f"Cannot convert '{value}' to BooleanArtifact")

    def __add__(self, other: BaseArtifact) -> BooleanArtifact:
        raise ValueError("Cannot add BooleanArtifact with other artifacts")

    def __eq__(self, value: object) -> bool:
        return self.value is value
