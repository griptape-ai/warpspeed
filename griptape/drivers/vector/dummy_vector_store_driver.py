from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from attrs import Factory, define, field

from griptape.drivers.embedding.dummy import DummyEmbeddingDriver
from griptape.drivers.vector import BaseVectorStoreDriver
from griptape.exceptions import DummyError

if TYPE_CHECKING:
    from griptape.artifacts import BaseArtifact
    from griptape.drivers.embedding import BaseEmbeddingDriver


@define()
class DummyVectorStoreDriver(BaseVectorStoreDriver):
    embedding_driver: BaseEmbeddingDriver = field(
        kw_only=True,
        default=Factory(lambda: DummyEmbeddingDriver()),
        metadata={"serializable": True},
    )

    def delete_vector(self, vector_id: str) -> None:
        raise DummyError(__class__.__name__, "delete_vector")

    def upsert_vector(
        self,
        vector: list[float],
        vector_id: Optional[str] = None,
        namespace: Optional[str] = None,
        meta: Optional[dict] = None,
        **kwargs,
    ) -> str:
        raise DummyError(__class__.__name__, "upsert_vector")

    def load_entry(self, vector_id: str, *, namespace: Optional[str] = None) -> Optional[BaseVectorStoreDriver.Entry]:
        raise DummyError(__class__.__name__, "load_entry")

    def load_entries(self, *, namespace: Optional[str] = None) -> list[BaseVectorStoreDriver.Entry]:
        raise DummyError(__class__.__name__, "load_entries")

    def query_vector(
        self,
        vector: list[float],
        *,
        count: Optional[int] = None,
        namespace: Optional[str] = None,
        include_vectors: bool = False,
        **kwargs,
    ) -> list[BaseVectorStoreDriver.Entry]:
        raise DummyError(__class__.__name__, "query_vector")

    def query(
        self,
        query: str | BaseArtifact,
        *,
        count: Optional[int] = None,
        namespace: Optional[str] = None,
        include_vectors: bool = False,
        **kwargs,
    ) -> list[BaseVectorStoreDriver.Entry]:
        raise DummyError(__class__.__name__, "query")
