from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Optional

from attrs import define, field

from griptape.drivers.vector import BaseVectorStoreDriver
from griptape.utils import import_optional_dependency
from griptape.utils.decorators import lazy_property

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


DEFAULT_DISTANCE = "Cosine"
CONTENT_PAYLOAD_KEY = "data"


@define
class QdrantVectorStoreDriver(BaseVectorStoreDriver):
    """Vector Store Driver for Qdrant.

    Attributes:
        location: An optional location for the Qdrant client. If set to ':memory:', an in-memory client is used.
        url: An optional Qdrant API URL.
        host: An optional Qdrant host.
        path: Persistence path for QdrantLocal. Default: None
        port: The port number for the Qdrant client. Defaults: 6333.
        grpc_port: The gRPC port number for the Qdrant client. Defaults: 6334.
        prefer_grpc: A boolean indicating whether to prefer gRPC over HTTP. Defaults: False.
        force_disable_check_same_thread: For QdrantLocal, force disable check_same_thread. Default: False Only use this if you can guarantee that you can resolve the thread safety outside QdrantClient.
        timeout: Timeout for REST and gRPC API requests. Default: 5 seconds for REST and unlimited for gRPC
        api_key: API key for authentication in Qdrant Cloud. Defaults: False
        https: If true - use HTTPS(SSL) protocol. Default: True
        prefix: Add prefix to the REST URL path. Example: service/v1 will result in Example: service/v1 will result in http://localhost:6333/service/v1/{qdrant-endpoint} for REST API. Defaults: None
        distance: The distance metric to be used for the vectors. Defaults: 'COSINE'.
        collection_name: The name of the Qdrant collection.
        vector_name: An optional name for the vectors.
        content_payload_key: The key for the content payload in the metadata. Defaults: 'data'.
    """

    location: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": True})
    url: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": True})
    host: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": True})
    path: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": True})
    port: int = field(default=6333, kw_only=True, metadata={"serializable": True})
    grpc_port: int = field(default=6334, kw_only=True, metadata={"serializable": True})
    prefer_grpc: bool = field(default=False, kw_only=True, metadata={"serializable": True})
    api_key: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": True})
    https: bool = field(default=True, kw_only=True, metadata={"serializable": True})
    prefix: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": True})
    force_disable_check_same_thread: Optional[bool] = field(
        default=False,
        kw_only=True,
        metadata={"serializable": True},
    )
    timeout: Optional[int] = field(default=5, kw_only=True, metadata={"serializable": True})
    distance: str = field(default=DEFAULT_DISTANCE, kw_only=True, metadata={"serializable": True})
    collection_name: str = field(kw_only=True, metadata={"serializable": True})
    vector_name: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": True})
    content_payload_key: str = field(default=CONTENT_PAYLOAD_KEY, kw_only=True, metadata={"serializable": True})
    _client: Optional[QdrantClient] = field(
        default=None, kw_only=True, alias="client", metadata={"serializable": False}
    )

    @lazy_property()
    def client(self) -> QdrantClient:
        return import_optional_dependency("qdrant_client").QdrantClient(
            location=self.location,
            url=self.url,
            host=self.host,
            path=self.path,
            port=self.port,
            prefer_grpc=self.prefer_grpc,
            grpc_port=self.grpc_port,
            api_key=self.api_key,
            https=self.https,
            prefix=self.prefix,
            force_disable_check_same_thread=self.force_disable_check_same_thread,
            timeout=self.timeout,
        )

    def delete_vector(self, vector_id: str) -> None:
        """Delete a vector from the Qdrant collection based on its ID.

        Parameters:
            vector_id (str | id): ID of the vector to delete.
        """
        deletion_response = self.client.delete(
            collection_name=self.collection_name,
            points_selector=import_optional_dependency("qdrant_client.http.models").PointIdsList(points=[vector_id]),
        )
        if deletion_response.status == import_optional_dependency("qdrant_client.http.models").UpdateStatus.COMPLETED:
            logging.info("ID %s is successfully deleted", vector_id)

    def query_vector(
        self,
        vector: list[float],
        *,
        count: Optional[int] = None,
        namespace: Optional[str] = None,
        include_vectors: bool = False,
        **kwargs,
    ) -> list[BaseVectorStoreDriver.Entry]:
        """Query the Qdrant collection based on a query vector.

        Parameters:
            vector (list[float]): Query vector.
            count (Optional[int]): Optional number of results to return.
            namespace (Optional[str]): Optional namespace of the vectors.
            include_vectors (bool): Whether to include vectors in the results.

        Returns:
            list[BaseVectorStoreDriver.Entry]: List of Entry objects.
        """
        # Create a search request
        request = {"collection_name": self.collection_name, "query_vector": vector, "limit": count}
        request = {k: v for k, v in request.items() if v is not None}
        results = self.client.search(**request)

        # Convert results to QueryResult objects
        return [
            BaseVectorStoreDriver.Entry(
                id=result.id,
                vector=result.vector if include_vectors else [],
                score=result.score,
                meta={k: v for k, v in result.payload.items() if k not in ["_score", "_tensor_facets"]},
            )
            for result in results
        ]

    def upsert_vector(
        self,
        vector: list[float],
        *,
        vector_id: Optional[str] = None,
        namespace: Optional[str] = None,
        meta: Optional[dict] = None,
        content: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Upsert vectors into the Qdrant collection.

        Parameters:
            vector (list[float]): The vector to be upserted.
            vector_id (Optional[str]): Optional vector ID.
            namespace (Optional[str]): Optional namespace for the vector.
            meta (Optional[dict]): Optional dictionary containing metadata.
            content (Optional[str]): The text content to be included in the payload.

        Returns:
            str: The ID of the upserted vector.
        """
        if vector_id is None:
            vector_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(vector)))

        if meta is None:
            meta = {}

        if content:
            meta[self.content_payload_key] = content

        points = import_optional_dependency("qdrant_client.http.models").Batch(
            ids=[vector_id],
            vectors=[vector],
            payloads=[meta] if meta else None,
        )

        self.client.upsert(collection_name=self.collection_name, points=points)
        return vector_id

    def load_entry(self, vector_id: str, *, namespace: Optional[str] = None) -> Optional[BaseVectorStoreDriver.Entry]:
        """Load a vector entry from the Qdrant collection based on its ID.

        Parameters:
            vector_id (str): ID of the vector to load.
            namespace (str, optional): Optional namespace of the vector.

        Returns:
            Optional[BaseVectorStoreDriver.Entry]: Vector entry if found, else None.
        """
        results = self.client.retrieve(collection_name=self.collection_name, ids=[vector_id])
        if results:
            entry = results[0]
            return BaseVectorStoreDriver.Entry(
                id=entry.id,
                vector=entry.vector,
                meta={k: v for k, v in entry.payload.items() if k not in ["_score", "_tensor_facets"]},
            )
        else:
            return None

    def load_entries(self, *, namespace: Optional[str] = None, **kwargs) -> list[BaseVectorStoreDriver.Entry]:
        """Load vector entries from the Qdrant collection.

        Parameters:
            namespace: Optional namespace of the vectors.

        Returns:
            List of points.
        """
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=kwargs.get("ids", []),
            with_payload=kwargs.get("with_payload", True),
            with_vectors=kwargs.get("with_vectors", True),
        )

        return [
            BaseVectorStoreDriver.Entry(
                id=entry.id,
                vector=entry.vector if kwargs.get("with_vectors", True) else [],
                meta={k: v for k, v in entry.payload.items() if k not in ["_score", "_tensor_facets"]},
            )
            for entry in results
        ]
