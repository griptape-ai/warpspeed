from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, NoReturn, Optional
from urllib.parse import urljoin

import requests
from attrs import Factory, define, field

from griptape.artifacts import BaseArtifact
from griptape.drivers.embedding.dummy import DummyEmbeddingDriver
from griptape.drivers.vector import BaseVectorStoreDriver

if TYPE_CHECKING:
    from griptape.artifacts import ListArtifact, TextArtifact
    from griptape.drivers.embedding import BaseEmbeddingDriver


@define
class GriptapeCloudVectorStoreDriver(BaseVectorStoreDriver):
    """A vector store driver for Griptape Cloud Knowledge Bases.

    Attributes:
        api_key: API Key for Griptape Cloud.
        knowledge_base_id: Knowledge Base ID for Griptape Cloud.
        base_url: Base URL for Griptape Cloud.
        headers: Headers for Griptape Cloud.
    """

    base_url: str = field(
        default=Factory(lambda: os.getenv("GT_CLOUD_BASE_URL", "https://cloud.griptape.ai")),
    )
    api_key: str = field(default=Factory(lambda: os.environ["GT_CLOUD_API_KEY"]))
    knowledge_base_id: str = field(kw_only=True, metadata={"serializable": True})
    headers: dict = field(
        default=Factory(lambda self: {"Authorization": f"Bearer {self.api_key}"}, takes_self=True),
        kw_only=True,
    )
    embedding_driver: BaseEmbeddingDriver = field(
        default=Factory(lambda: DummyEmbeddingDriver()),
        metadata={"serializable": True},
        kw_only=True,
        init=False,
    )

    def upsert_vector(
        self,
        vector: list[float],
        vector_id: Optional[str] = None,
        namespace: Optional[str] = None,
        meta: Optional[dict] = None,
        **kwargs,
    ) -> str:
        raise NotImplementedError(f"{self.__class__.__name__} does not support vector upsert.")

    def upsert_text_artifact(
        self,
        artifact: TextArtifact,
        namespace: Optional[str] = None,
        meta: Optional[dict] = None,
        vector_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        raise NotImplementedError(f"{self.__class__.__name__} does not support text artifact upsert.")

    def upsert_text(
        self,
        string: str,
        vector_id: Optional[str] = None,
        namespace: Optional[str] = None,
        meta: Optional[dict] = None,
        **kwargs,
    ) -> str:
        raise NotImplementedError(f"{self.__class__.__name__} does not support text upsert.")

    def load_entry(self, vector_id: str, *, namespace: Optional[str] = None) -> BaseVectorStoreDriver.Entry:
        raise NotImplementedError(f"{self.__class__.__name__} does not support entry loading.")

    def load_entries(self, *, namespace: Optional[str] = None) -> list[BaseVectorStoreDriver.Entry]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support entry loading.")

    def load_artifacts(self, *, namespace: Optional[str] = None) -> ListArtifact:
        raise NotImplementedError(f"{self.__class__.__name__} does not support Artifact loading.")

    def query(
        self,
        query: str | BaseArtifact,
        *,
        count: Optional[int] = None,
        namespace: Optional[str] = None,
        include_vectors: Optional[bool] = None,
        distance_metric: Optional[str] = None,
        # GriptapeCloudVectorStoreDriver-specific params:
        filter: Optional[dict] = None,  # noqa: A002
        **kwargs,
    ) -> list[BaseVectorStoreDriver.Entry]:
        """Performs a query on the Knowledge Base.

        Performs a query on the Knowledge Base and returns Artifacts with close vector proximity to the query, optionally filtering to only those that match the provided filter(s).
        """
        if isinstance(query, BaseArtifact):
            raise ValueError(f"{self.__class__.__name__} does not support querying with Artifacts.")
        url = urljoin(self.base_url.strip("/"), f"/api/knowledge-bases/{self.knowledge_base_id}/query")

        query_args = {
            "count": count,
            "distance_metric": distance_metric,
            "filter": filter,
            "include_vectors": include_vectors,
        }
        query_args = {k: v for k, v in query_args.items() if v is not None}

        request: dict[str, Any] = {
            "query": query,
            "query_args": query_args,
        }

        response = requests.post(url, json=request, headers=self.headers).json()
        entries = response.get("entries", [])
        return [BaseVectorStoreDriver.Entry.from_dict(entry) for entry in entries]

    def delete_vector(self, vector_id: str) -> NoReturn:
        raise NotImplementedError(f"{self.__class__.__name__} does not support deletion.")
