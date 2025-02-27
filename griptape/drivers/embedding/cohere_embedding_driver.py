from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from attrs import Factory, define, field

from griptape.drivers.embedding import BaseEmbeddingDriver
from griptape.tokenizers import CohereTokenizer
from griptape.utils import import_optional_dependency
from griptape.utils.decorators import lazy_property

if TYPE_CHECKING:
    from cohere import Client


@define
class CohereEmbeddingDriver(BaseEmbeddingDriver):
    """Cohere Embedding Driver.

    Attributes:
        api_key: Cohere API key.
        model: 	Cohere model name.
        client: Custom `cohere.Client`.
        tokenizer: Custom `CohereTokenizer`.
        input_type: Cohere embedding input type.
    """

    DEFAULT_MODEL = "models/embedding-001"

    api_key: str = field(kw_only=True, metadata={"serializable": False})
    input_type: str = field(kw_only=True, metadata={"serializable": True})
    _client: Optional[Client] = field(default=None, kw_only=True, alias="client", metadata={"serializable": False})
    tokenizer: CohereTokenizer = field(
        default=Factory(lambda self: CohereTokenizer(model=self.model, client=self.client), takes_self=True),
        kw_only=True,
    )

    @lazy_property()
    def client(self) -> Client:
        return import_optional_dependency("cohere").Client(self.api_key)

    def try_embed_chunk(self, chunk: str) -> list[float]:
        result = self.client.embed(texts=[chunk], model=self.model, input_type=self.input_type)

        if isinstance(result.embeddings, list):
            return result.embeddings[0]
        else:
            raise ValueError("Non-float embeddings are not supported.")
