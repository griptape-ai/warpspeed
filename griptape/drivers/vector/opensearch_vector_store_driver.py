from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NoReturn, Optional

from attrs import define, field

from griptape import utils
from griptape.drivers.vector import BaseVectorStoreDriver
from griptape.utils import import_optional_dependency
from griptape.utils.decorators import lazy_property

if TYPE_CHECKING:
    from opensearchpy import OpenSearch


@define
class OpenSearchVectorStoreDriver(BaseVectorStoreDriver):
    """A Vector Store Driver for OpenSearch.

    Attributes:
        host: The host of the OpenSearch cluster.
        port: The port of the OpenSearch cluster.
        http_auth: The HTTP authentication credentials to use.
        use_ssl: Whether to use SSL.
        verify_certs: Whether to verify SSL certificates.
        index_name: The name of the index to use.
    """

    host: str = field(kw_only=True, metadata={"serializable": True})
    port: int = field(default=443, kw_only=True, metadata={"serializable": True})
    http_auth: str | tuple[str, Optional[str]] = field(default=None, kw_only=True, metadata={"serializable": True})
    use_ssl: bool = field(default=True, kw_only=True, metadata={"serializable": True})
    verify_certs: bool = field(default=True, kw_only=True, metadata={"serializable": True})
    index_name: str = field(kw_only=True, metadata={"serializable": True})
    _client: Optional[OpenSearch] = field(default=None, kw_only=True, alias="client", metadata={"serializable": False})

    @lazy_property()
    def client(self) -> OpenSearch:
        opensearchpy = import_optional_dependency("opensearchpy")

        return opensearchpy.OpenSearch(
            hosts=[{"host": self.host, "port": self.port}],
            http_auth=self.http_auth,
            use_ssl=self.use_ssl,
            verify_certs=self.verify_certs,
            connection_class=opensearchpy.RequestsHttpConnection,
        )

    def upsert_vector(
        self,
        vector: list[float],
        *,
        vector_id: Optional[str] = None,
        namespace: Optional[str] = None,
        meta: Optional[dict] = None,
        **kwargs,
    ) -> str:
        """Inserts or updates a vector in OpenSearch.

        If a vector with the given vector ID already exists, it is updated; otherwise, a new vector is inserted.
        Metadata associated with the vector can also be provided.
        """
        vector_id = vector_id or utils.str_to_hash(str(vector))
        doc = {"vector": vector, "namespace": namespace, "metadata": meta}
        doc.update(kwargs)
        response = self.client.index(index=self.index_name, id=vector_id, body=doc)

        return response["_id"]

    def load_entry(self, vector_id: str, *, namespace: Optional[str] = None) -> Optional[BaseVectorStoreDriver.Entry]:
        """Retrieves a specific vector entry from OpenSearch based on its identifier and optional namespace.

        Returns:
            If the entry is found, it returns an instance of BaseVectorStoreDriver.Entry; otherwise, None is returned.
        """
        try:
            query = {"bool": {"must": [{"term": {"_id": vector_id}}]}}

            if namespace:
                query["bool"]["must"].append({"term": {"namespace": namespace}})

            response = self.client.search(index=self.index_name, body={"query": query, "size": 1})

            if response["hits"]["total"]["value"] > 0:
                vector_data = response["hits"]["hits"][0]["_source"]
                return BaseVectorStoreDriver.Entry(
                    id=vector_id,
                    meta=vector_data.get("metadata"),
                    vector=vector_data.get("vector"),
                    namespace=vector_data.get("namespace"),
                )
            else:
                return None
        except Exception as e:
            logging.exception("Error while loading entry: %s", e)
            return None

    def load_entries(self, *, namespace: Optional[str] = None) -> list[BaseVectorStoreDriver.Entry]:
        """Retrieves all vector entries from OpenSearch that match the optional namespace.

        Returns:
            A list of BaseVectorStoreDriver.Entry objects.
        """
        query_body = {"size": 10000, "query": {"match_all": {}}}

        if namespace:
            query_body["query"] = {"match": {"namespace": namespace}}

        response = self.client.search(index=self.index_name, body=query_body)

        return [
            BaseVectorStoreDriver.Entry(
                id=hit["_id"],
                vector=hit["_source"].get("vector"),
                meta=hit["_source"].get("metadata"),
                namespace=hit["_source"].get("namespace"),
            )
            for hit in response["hits"]["hits"]
        ]

    def query_vector(
        self,
        vector: list[float],
        *,
        count: Optional[int] = None,
        namespace: Optional[str] = None,
        include_vectors: bool = False,
        include_metadata: bool = True,
        field_name: str = "vector",
        **kwargs,
    ) -> list[BaseVectorStoreDriver.Entry]:
        """Performs a nearest neighbor search on OpenSearch to find vectors similar to the provided vector list.

        Results can be limited using the count parameter and optionally filtered by a namespace.

        Returns:
            A list of BaseVectorStoreDriver.Entry objects, each encapsulating the retrieved vector, its similarity score, metadata, and namespace.
        """
        count = count or BaseVectorStoreDriver.DEFAULT_QUERY_COUNT
        # Base k-NN query
        query_body = {"size": count, "query": {"knn": {field_name: {"vector": vector, "k": count}}}}

        if namespace:
            query_body["query"] = {
                "bool": {
                    "must": [
                        {"match": {"namespace": namespace}},
                        {"knn": {field_name: {"vector": vector, "k": count}}},
                    ],
                },
            }

        response = self.client.search(index=self.index_name, body=query_body)

        return [
            BaseVectorStoreDriver.Entry(
                id=hit["_id"],
                namespace=hit["_source"].get("namespace") if namespace else None,
                score=hit["_score"],
                vector=hit["_source"].get("vector") if include_vectors else None,
                meta=hit["_source"].get("metadata") if include_metadata else None,
            )
            for hit in response["hits"]["hits"]
        ]

    def query(
        self,
        query: str,
        *,
        count: Optional[int] = None,
        namespace: Optional[str] = None,
        include_vectors: bool = False,
        include_metadata: bool = True,
        field_name: str = "vector",
        **kwargs,
    ) -> list[BaseVectorStoreDriver.Entry]:
        """Performs a nearest neighbor search on OpenSearch to find vectors similar to the provided query string.

        Results can be limited using the count parameter and optionally filtered by a namespace.

        Returns:
            A list of BaseVectorStoreDriver.Entry objects, each encapsulating the retrieved vector, its similarity score, metadata, and namespace.
        """
        vector = self.embedding_driver.embed_string(query)
        return self.query_vector(
            vector,
            count=count,
            namespace=namespace,
            include_vectors=include_vectors,
            include_metadata=include_metadata,
            field_name=field_name,
            **kwargs,
        )

    def delete_vector(self, vector_id: str) -> NoReturn:
        raise NotImplementedError(f"{self.__class__.__name__} does not support deletion.")
