from __future__ import annotations

from typing import Optional

from attrs import define, field

from griptape.drivers.embedding import BaseEmbeddingDriver
from griptape.utils import import_optional_dependency


@define
class GoogleEmbeddingDriver(BaseEmbeddingDriver):
    """Google Embedding Driver.

    Attributes:
        api_key: Google API key.
        model: Google model name.
        task_type: Embedding model task type (https://ai.google.dev/tutorials/python_quickstart#use_embeddings). Defaults to `retrieval_document`.
        title: Optional title for the content. Only works with `retrieval_document` task type.
    """

    DEFAULT_MODEL = "models/embedding-001"

    model: str = field(default=DEFAULT_MODEL, kw_only=True, metadata={"serializable": True})
    api_key: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": False})
    task_type: str = field(default="retrieval_document", kw_only=True, metadata={"serializable": True})
    title: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": True})

    def try_embed_chunk(self, chunk: str | bytes) -> list[float]:
        if isinstance(chunk, bytes):
            raise ValueError(f"{self.__class__.__name__} does not support embedding bytes.")
        genai = import_optional_dependency("google.generativeai")
        genai.configure(api_key=self.api_key)

        result = genai.embed_content(model=self.model, content=chunk, task_type=self.task_type, title=self.title)

        return result["embedding"]

    def _params(self, chunk: str) -> dict:
        return {"input": chunk, "model": self.model}
