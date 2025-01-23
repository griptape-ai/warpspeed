from __future__ import annotations

from attrs import define, field

from griptape.common import BaseDeltaMessageContent


@define
class AudioTranscriptDeltaMessageContent(BaseDeltaMessageContent):
    text: str = field(metadata={"serializable": True})
