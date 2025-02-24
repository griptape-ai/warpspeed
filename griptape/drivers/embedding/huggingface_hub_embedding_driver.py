from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define, field

from griptape.drivers.embedding import BaseEmbeddingDriver
from griptape.utils import import_optional_dependency
from griptape.utils.decorators import lazy_property

if TYPE_CHECKING:
    from huggingface_hub import InferenceClient


@define
class HuggingFaceHubEmbeddingDriver(BaseEmbeddingDriver):
    """Hugging Face Hub Embedding Driver.

    Attributes:
        api_token: Hugging Face Hub API token.
        model: Hugging Face Hub model name.
        client: Custom `InferenceApi`.
    """

    api_token: str = field(kw_only=True, metadata={"serializable": True})
    _client: InferenceClient = field(default=None, kw_only=True, alias="client", metadata={"serializable": False})

    @lazy_property()
    def client(self) -> InferenceClient:
        return import_optional_dependency("huggingface_hub").InferenceClient(
            model=self.model,
            token=self.api_token,
        )

    def try_embed_chunk(self, chunk: str | bytes) -> list[float]:
        if isinstance(chunk, bytes):
            raise ValueError(f"{self.__class__.__name__} does not support embedding bytes.")
        response = self.client.feature_extraction(chunk)

        return response.flatten().tolist()
