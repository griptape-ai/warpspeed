from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from attrs import define, field

from griptape.drivers.embedding import BaseEmbeddingDriver
from griptape.utils import import_optional_dependency
from griptape.utils.decorators import lazy_property

if TYPE_CHECKING:
    from ollama import Client


@define
class OllamaEmbeddingDriver(BaseEmbeddingDriver):
    """Ollama Embedding Driver.

    Attributes:
        model: Ollama embedding model name.
        host: Optional Ollama host.
        client: Ollama `Client`.
    """

    model: str = field(kw_only=True, metadata={"serializable": True})
    host: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": True})
    _client: Client = field(default=None, kw_only=True, alias="client", metadata={"serializable": False})

    @lazy_property()
    def client(self) -> Client:
        return import_optional_dependency("ollama").Client(host=self.host)

    def try_embed_chunk(self, chunk: str | bytes) -> list[float]:
        if isinstance(chunk, bytes):
            raise ValueError(f"{self.__class__.__name__} does not support embedding bytes.")
        return list(self.client.embeddings(model=self.model, prompt=chunk)["embedding"])
