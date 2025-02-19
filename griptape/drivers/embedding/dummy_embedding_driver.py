from __future__ import annotations

from attrs import define, field

from griptape.drivers.embedding import BaseEmbeddingDriver
from griptape.exceptions import DummyError


@define
class DummyEmbeddingDriver(BaseEmbeddingDriver):
    model: None = field(init=False, default=None, kw_only=True)

    def try_embed_chunk(self, chunk: str | bytes) -> list[float]:
        raise DummyError(__class__.__name__, "try_embed_chunk")
