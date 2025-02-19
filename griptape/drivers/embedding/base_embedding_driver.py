from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np
from attrs import define, field

from griptape.artifacts import ImageArtifact, TextArtifact
from griptape.artifacts.list_artifact import ListArtifact
from griptape.chunkers import BaseChunker, TextChunker
from griptape.mixins.exponential_backoff_mixin import ExponentialBackoffMixin
from griptape.mixins.serializable_mixin import SerializableMixin

if TYPE_CHECKING:
    from griptape.tokenizers import BaseTokenizer


@define
class BaseEmbeddingDriver(SerializableMixin, ExponentialBackoffMixin, ABC):
    """Base Embedding Driver.

    Attributes:
        model: The name of the model to use.
        tokenizer: An instance of `BaseTokenizer` to use when calculating tokens.
    """

    model: str = field(kw_only=True, metadata={"serializable": True})
    tokenizer: Optional[BaseTokenizer] = field(default=None, kw_only=True)
    chunker: Optional[BaseChunker] = field(init=False)

    def __attrs_post_init__(self) -> None:
        self.chunker = TextChunker(tokenizer=self.tokenizer) if self.tokenizer else None

    def embed_artifact(
        self, artifact: TextArtifact | ImageArtifact | ListArtifact[TextArtifact | ImageArtifact]
    ) -> list[float]:
        if isinstance(artifact, TextArtifact):
            return self.embed_text_artifact(artifact)
        elif isinstance(artifact, ImageArtifact):
            return self.embed_image_artifact(artifact)
        else:
            embeddings = [self.embed_artifact(artifact) for artifact in artifact]

            return np.average(embeddings, axis=0).tolist()

    def embed_image_artifact(self, artifact: ImageArtifact) -> list[float]:
        return self.try_embed_chunk(artifact.value)

    def embed_text_artifact(self, artifact: TextArtifact) -> list[float]:
        return self.embed_string(artifact.to_text())

    def embed_string(self, string: str) -> list[float]:
        for attempt in self.retrying():
            with attempt:
                if self.tokenizer is not None and self.tokenizer.count_tokens(string) > self.tokenizer.max_input_tokens:
                    return self._embed_long_string(string)
                else:
                    return self.try_embed_chunk(string)

        else:
            raise RuntimeError("Failed to embed string.")

    @abstractmethod
    def try_embed_chunk(self, chunk: str | bytes) -> list[float]: ...

    def _embed_long_string(self, string: str) -> list[float]:
        """Embeds a string that is too long to embed in one go.

        Adapted from: https://github.com/openai/openai-cookbook/blob/683e5f5a71bc7a1b0e5b7a35e087f53cc55fceea/examples/Embedding_long_inputs.ipynb
        """
        chunks = self.chunker.chunk(string)  # pyright: ignore[reportOptionalMemberAccess] In practice this is never None

        embedding_chunks = []
        length_chunks = []
        for chunk in chunks:
            embedding_chunks.append(self.try_embed_chunk(chunk.value))
            length_chunks.append(len(chunk))

        # generate weighted averages
        embedding_chunks = np.average(embedding_chunks, axis=0, weights=length_chunks)

        # normalize length to 1
        embedding_chunks = embedding_chunks / np.linalg.norm(embedding_chunks)

        return embedding_chunks.tolist()
