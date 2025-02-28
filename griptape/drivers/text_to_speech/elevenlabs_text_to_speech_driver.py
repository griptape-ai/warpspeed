from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from attrs import define, field

from griptape.artifacts.audio_artifact import AudioArtifact
from griptape.drivers.text_to_speech import BaseTextToSpeechDriver
from griptape.utils import import_optional_dependency
from griptape.utils.decorators import lazy_property

if TYPE_CHECKING:
    from elevenlabs.client import ElevenLabs


@define
class ElevenLabsTextToSpeechDriver(BaseTextToSpeechDriver):
    api_key: str = field(kw_only=True, metadata={"serializable": True})
    voice: str = field(kw_only=True, metadata={"serializable": True})
    output_format: str = field(default="mp3_44100_128", kw_only=True, metadata={"serializable": True})
    _client: Optional[ElevenLabs] = field(default=None, kw_only=True, alias="client", metadata={"serializable": False})

    @lazy_property()
    def client(self) -> ElevenLabs:
        return import_optional_dependency("elevenlabs.client").ElevenLabs(api_key=self.api_key)

    def try_text_to_audio(self, prompts: list[str]) -> AudioArtifact:
        audio = self.client.generate(
            text=". ".join(prompts),
            voice=self.voice,
            model=self.model,
            output_format=self.output_format,
        )

        content = b""
        for chunk in audio:
            content += chunk

        # All ElevenLabs audio format strings have the following structure:
        # {format}_{sample_rate}_{bitrate}
        artifact_format = self.output_format.split("_")[0]

        return AudioArtifact(value=content, format=artifact_format)
