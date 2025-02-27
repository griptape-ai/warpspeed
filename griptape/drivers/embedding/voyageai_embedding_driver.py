from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from attrs import Factory, define, field

from griptape.drivers.embedding import BaseEmbeddingDriver
from griptape.tokenizers import VoyageAiTokenizer
from griptape.utils import import_optional_dependency
from griptape.utils.decorators import lazy_property

if TYPE_CHECKING:
    from voyageai.client import Client


@define
class VoyageAiEmbeddingDriver(BaseEmbeddingDriver):
    """VoyageAI Embedding Driver.

    Attributes:
        model: VoyageAI embedding model name. Defaults to `voyage-large-2`.
        api_key: API key to pass directly. Defaults to `VOYAGE_API_KEY` environment variable.
        tokenizer: Optionally provide custom `VoyageAiTokenizer`.
        client: Optionally provide custom VoyageAI `Client`.
        input_type: VoyageAI input type. Defaults to `document`.
    """

    DEFAULT_MODEL = "voyage-large-2"

    model: str = field(default=DEFAULT_MODEL, kw_only=True, metadata={"serializable": True})
    api_key: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": False})
    tokenizer: VoyageAiTokenizer = field(
        default=Factory(lambda self: VoyageAiTokenizer(model=self.model, api_key=self.api_key), takes_self=True),
        kw_only=True,
    )
    input_type: str = field(default="document", kw_only=True, metadata={"serializable": True})
    _client: Optional[Client] = field(default=None, kw_only=True, alias="client", metadata={"serializable": False})

    @lazy_property()
    def client(self) -> Any:
        return import_optional_dependency("voyageai").Client(api_key=self.api_key)

    def try_embed_chunk(self, chunk: str) -> list[float]:
        return self.client.embed([chunk], model=self.model, input_type=self.input_type).embeddings[0]
