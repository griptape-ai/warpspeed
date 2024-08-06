from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Optional

from attrs import define, field

from griptape.config import BaseConfig
from griptape.utils import dict_merge

if TYPE_CHECKING:
    from griptape.drivers import (
        BaseAudioTranscriptionDriver,
        BaseConversationMemoryDriver,
        BaseEmbeddingDriver,
        BaseImageGenerationDriver,
        BaseImageQueryDriver,
        BasePromptDriver,
        BaseTextToSpeechDriver,
        BaseVectorStoreDriver,
    )


@define
class BaseStructureConfig(BaseConfig, ABC, EventPublisherMixin):
    prompt_driver: BasePromptDriver = field(kw_only=True, metadata={"serializable": True})
    image_generation_driver: BaseImageGenerationDriver = field(kw_only=True, metadata={"serializable": True})
    image_query_driver: BaseImageQueryDriver = field(kw_only=True, metadata={"serializable": True})
    embedding_driver: BaseEmbeddingDriver = field(kw_only=True, metadata={"serializable": True})
    vector_store_driver: BaseVectorStoreDriver = field(kw_only=True, metadata={"serializable": True})
    conversation_memory_driver: Optional[BaseConversationMemoryDriver] = field(
        default=None,
        kw_only=True,
        metadata={"serializable": True},
    )
    text_to_speech_driver: BaseTextToSpeechDriver = field(kw_only=True, metadata={"serializable": True})
    audio_transcription_driver: BaseAudioTranscriptionDriver = field(kw_only=True, metadata={"serializable": True})

    def merge_config(self, config: dict) -> BaseStructureConfig:
        base_config = self.to_dict()
        merged_config = dict_merge(base_config, config)

        return BaseStructureConfig.from_dict(merged_config)
