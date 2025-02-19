from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn, Optional

from attrs import define, field

from griptape import utils
from griptape.drivers.vector import BaseVectorStoreDriver
from griptape.utils import import_optional_dependency
from griptape.utils.decorators import lazy_property

if TYPE_CHECKING:
    import marqo

    from griptape.artifacts import BaseArtifact, TextArtifact


@define
class MarqoVectorStoreDriver(BaseVectorStoreDriver):
    """A Vector Store Driver for Marqo.

    Attributes:
        api_key: The API key for the Marqo API.
        url: The URL to the Marqo API.
        client: An optional Marqo client. Defaults to a new client with the given URL and API key.
        index: The name of the index to use.
    """

    api_key: str = field(kw_only=True, metadata={"serializable": True})
    url: str = field(kw_only=True, metadata={"serializable": True})
    index: str = field(kw_only=True, metadata={"serializable": True})
    _client: marqo.Client = field(default=None, kw_only=True, alias="client", metadata={"serializable": False})

    @lazy_property()
    def client(self) -> marqo.Client:
        return import_optional_dependency("marqo").Client(self.url, api_key=self.api_key)

    def upsert_text(
        self,
        string: str,
        *,
        vector_id: Optional[str] = None,
        namespace: Optional[str] = None,
        meta: Optional[dict] = None,
        **kwargs: Any,
    ) -> str:
        """Upsert a text document into the Marqo index.

        Args:
            string: The string to be indexed.
            vector_id: The ID for the vector. If None, Marqo will generate an ID.
            namespace: An optional namespace for the document.
            meta: An optional dictionary of metadata for the document.
            kwargs: Additional keyword arguments to pass to the Marqo client.

        Returns:
            str: The ID of the document that was added.
        """
        doc = {"_id": vector_id, "Description": string}  # Description will be treated as tensor field

        # Non-tensor fields
        if meta:
            doc["meta"] = str(meta)
        if namespace:
            doc["namespace"] = namespace

        response = self.client.index(self.index).add_documents([doc], tensor_fields=["Description"])
        if isinstance(response, dict) and "items" in response and response["items"]:
            return response["items"][0]["_id"]
        else:
            raise ValueError(f"Failed to upsert text: {response}")

    def upsert_text_artifact(
        self,
        artifact: TextArtifact,
        *,
        namespace: Optional[str] = None,
        meta: Optional[dict] = None,
        vector_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Upsert a text artifact into the Marqo index.

        Args:
            artifact: The text artifact to be indexed.
            namespace: An optional namespace for the artifact.
            meta: An optional dictionary of metadata for the artifact.
            vector_id: An optional explicit vector_id.
            kwargs: Additional keyword arguments to pass to the Marqo client.

        Returns:
            str: The ID of the artifact that was added.
        """
        artifact_json = artifact.to_json()
        vector_id = utils.str_to_hash(artifact.value) if vector_id is None else vector_id

        doc = {
            "_id": vector_id,
            "Description": artifact.value,  # Description will be treated as tensor field
            "artifact": str(artifact_json),
            "namespace": namespace,
        }

        response = self.client.index(self.index).add_documents([doc], tensor_fields=["Description", "artifact"])
        if isinstance(response, dict) and "items" in response and response["items"]:
            return response["items"][0]["_id"]
        else:
            raise ValueError(f"Failed to upsert text: {response}")

    def load_entry(self, vector_id: str, *, namespace: Optional[str] = None) -> Optional[BaseVectorStoreDriver.Entry]:
        """Load a document entry from the Marqo index.

        Args:
            vector_id: The ID of the vector to load.
            namespace: The namespace of the vector to load.

        Returns:
            The loaded Entry if found, otherwise None.
        """
        result = self.client.index(self.index).get_document(document_id=vector_id, expose_facets=True)

        if result and "_tensor_facets" in result and len(result["_tensor_facets"]) > 0:
            return BaseVectorStoreDriver.Entry(
                id=result["_id"],
                meta={k: v for k, v in result.items() if k != "_id"},
                vector=result["_tensor_facets"][0]["_embedding"],
            )
        else:
            return None

    def load_entries(self, *, namespace: Optional[str] = None) -> list[BaseVectorStoreDriver.Entry]:
        """Load all document entries from the Marqo index.

        Args:
            namespace: The namespace to filter entries by.

        Returns:
            The list of loaded Entries.
        """
        filter_string = f"namespace:{namespace}" if namespace else None

        if filter_string is not None:
            results = self.client.index(self.index).search("", limit=10000, filter_string=filter_string)
        else:
            results = self.client.index(self.index).search("", limit=10000)

        # get all _id's from search results
        ids = [r["_id"] for r in results["hits"]]

        # get documents corresponding to the ids
        documents = self.client.index(self.index).get_documents(document_ids=ids, expose_facets=True)

        # for each document, if it's found, create an Entry object
        entries = []
        for doc in documents["results"]:
            if doc["_found"]:
                entries.append(
                    BaseVectorStoreDriver.Entry(
                        id=doc["_id"],
                        vector=doc["_tensor_facets"][0]["_embedding"],
                        meta={k: v for k, v in doc.items() if k not in ["_id", "_tensor_facets", "_found"]},
                        namespace=doc.get("namespace"),
                    ),
                )

        return entries

    def query_vector(
        self,
        vector: list[float],
        *,
        count: Optional[int] = None,
        namespace: Optional[str] = None,
        include_vectors: bool = False,
        include_metadata: bool = True,
        **kwargs: Any,
    ) -> list[BaseVectorStoreDriver.Entry]:
        """Query the Marqo index for documents.

        Args:
            vector: The vector to query by.
            count: The maximum number of results to return.
            namespace: The namespace to filter results by.
            include_vectors: Whether to include vector data in the results.
            include_metadata: Whether to include metadata in the results.
            kwargs: Additional keyword arguments to pass to the Marqo client.

        Returns:
            The list of query results.
        """
        params = {
            "limit": count or BaseVectorStoreDriver.DEFAULT_QUERY_COUNT,
            "attributes_to_retrieve": None if include_metadata else ["_id"],
            "filter_string": f"namespace:{namespace}" if namespace else None,
        } | kwargs

        results = self.client.index(self.index).search(**params, context={"tensor": [vector], "weight": 1})

        return self.__process_results(results, include_vectors=include_vectors)

    def query(
        self,
        query: str | BaseArtifact,
        *,
        count: Optional[int] = None,
        namespace: Optional[str] = None,
        include_vectors: bool = False,
        include_metadata: bool = True,
        **kwargs: Any,
    ) -> list[BaseVectorStoreDriver.Entry]:
        """Query the Marqo index for documents.

        Args:
            query: The query string.
            count: The maximum number of results to return.
            namespace: The namespace to filter results by.
            include_vectors: Whether to include vector data in the results.
            include_metadata: Whether to include metadata in the results.
            kwargs: Additional keyword arguments to pass to the Marqo client.

        Returns:
            The list of query results.
        """
        params = {
            "limit": count or BaseVectorStoreDriver.DEFAULT_QUERY_COUNT,
            "attributes_to_retrieve": None if include_metadata else ["_id"],
            "filter_string": f"namespace:{namespace}" if namespace else None,
        } | kwargs

        results = self.client.index(self.index).search(query, **params)
        return self.__process_results(results, include_vectors=include_vectors)

    def delete_index(self, name: str) -> dict[str, Any]:
        """Delete an index in the Marqo client.

        Args:
            name: The name of the index to delete.
        """
        return self.client.delete_index(name)

    def get_indexes(self) -> list[str]:
        """Get a list of all indexes in the Marqo client.

        Returns:
            The list of all indexes.
        """
        return [index["index"] for index in self.client.get_indexes()["results"]]

    def upsert_vector(
        self,
        vector: list[float],
        *,
        vector_id: Optional[str] = None,
        namespace: Optional[str] = None,
        meta: Optional[dict] = None,
        **kwargs: Any,
    ) -> str:
        """Upsert a vector into the Marqo index.

        Args:
            vector: The vector to be indexed.
            vector_id: The ID for the vector. If None, Marqo will generate an ID.
            namespace: An optional namespace for the vector.
            meta: An optional dictionary of metadata for the vector.
            kwargs: Additional keyword arguments to pass to the Marqo client.

        Raises:
            Exception: This function is not yet implemented.

        Returns:
            The ID of the vector that was added.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support upserting a vector.")

    def delete_vector(self, vector_id: str) -> NoReturn:
        raise NotImplementedError(f"{self.__class__.__name__} does not support deletion.")

    def __process_results(self, results: dict, *, include_vectors: bool) -> list[BaseVectorStoreDriver.Entry]:
        if include_vectors:
            results["hits"] = [
                {**r, **self.client.index(self.index).get_document(r["_id"], expose_facets=True)}
                for r in results["hits"]
            ]

        return [
            BaseVectorStoreDriver.Entry(
                id=r["_id"],
                vector=r["_tensor_facets"][0]["_embedding"] if include_vectors else [],
                score=r["_score"],
                meta={k: v for k, v in r.items() if k not in ["_score", "_tensor_facets"]},
            )
            for r in results["hits"]
        ]
