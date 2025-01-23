from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define, field

from griptape.artifacts import AudioArtifact
from griptape.common import (
    AudioDeltaMessageContent,
    AudioTranscriptDeltaMessageContent,
    BaseDeltaMessageContent,
    BaseMessageContent,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


@define
class AudioMessageContent(BaseMessageContent):
    artifact: AudioArtifact = field(metadata={"serializable": True})

    @classmethod
    def from_deltas(cls, deltas: Sequence[BaseDeltaMessageContent]) -> AudioMessageContent:
        audio_deltas = [delta for delta in deltas if isinstance(delta, AudioDeltaMessageContent)]
        audio_transcript_deltas = [delta for delta in deltas if isinstance(delta, AudioTranscriptDeltaMessageContent)]
        audio_id = audio_deltas[0].id

        audio_transcript = "".join(delta.text for delta in audio_transcript_deltas)

        artifact = AudioArtifact(
            value=b"".join(delta.data for delta in audio_deltas),
            format="wav",
            meta={
                "audio_id": audio_id,
                "transcript": audio_transcript,
            },
        )

        return cls(artifact=artifact)
