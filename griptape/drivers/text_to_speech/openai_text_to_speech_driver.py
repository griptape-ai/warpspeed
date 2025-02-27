from __future__ import annotations

from typing import Literal, Optional

import openai
from attrs import define, field

from griptape.artifacts.audio_artifact import AudioArtifact
from griptape.drivers.text_to_speech import BaseTextToSpeechDriver
from griptape.utils.decorators import lazy_property


@define
class OpenAiTextToSpeechDriver(BaseTextToSpeechDriver):
    model: str = field(default="tts-1", kw_only=True, metadata={"serializable": True})
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = field(
        default="alloy",
        kw_only=True,
        metadata={"serializable": True},
    )
    format: Literal["mp3", "opus", "aac", "flac"] = field(default="mp3", kw_only=True, metadata={"serializable": True})
    api_type: Optional[str] = field(default=openai.api_type, kw_only=True)
    api_version: Optional[str] = field(default=openai.api_version, kw_only=True, metadata={"serializable": True})
    base_url: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": True})
    api_key: Optional[str] = field(default=None, kw_only=True)
    organization: Optional[str] = field(default=openai.organization, kw_only=True, metadata={"serializable": True})
    _client: Optional[openai.OpenAI] = field(
        default=None, kw_only=True, alias="client", metadata={"serializable": False}
    )

    @lazy_property()
    def client(self) -> openai.OpenAI:
        return openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            organization=self.organization,
        )

    def try_text_to_audio(self, prompts: list[str]) -> AudioArtifact:
        response = self.client.audio.speech.create(
            input=". ".join(prompts),
            voice=self.voice,
            model=self.model,
            response_format=self.format,
        )

        return AudioArtifact(value=response.content, format=self.format)
