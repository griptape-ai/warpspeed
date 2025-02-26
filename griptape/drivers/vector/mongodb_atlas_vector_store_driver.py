from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from attrs import define, field

from griptape.drivers.vector import BaseVectorStoreDriver
from griptape.utils import import_optional_dependency
from griptape.utils.decorators import lazy_property

if TYPE_CHECKING:
    from pymongo import MongoClient
    from pymongo.collection import Collection


@define
class MongoDbAtlasVectorStoreDriver(BaseVectorStoreDriver):
    """A Vector Store Driver for MongoDb Atlas.

    Attributes:
        connection_string: The connection string for the MongoDb Atlas cluster.
        database_name: The name of the database to use.
        collection_name: The name of the collection to use.
        index_name: The name of the index to use.
        vector_path: The path to the vector field in the collection.
        client: An optional MongoDb client to use. Defaults to a new client using the connection string.
    """

    MAX_NUM_CANDIDATES = 10000

    connection_string: str = field(kw_only=True, metadata={"serializable": True})
    database_name: str = field(kw_only=True, metadata={"serializable": True})
    collection_name: str = field(kw_only=True, metadata={"serializable": True})
    index_name: str = field(kw_only=True, metadata={"serializable": True})
    vector_path: str = field(kw_only=True, metadata={"serializable": True})
    num_candidates_multiplier: int = field(
        default=10,
        kw_only=True,
        metadata={"serializable": True},
    )  # https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/#fields
    _client: MongoClient = field(default=None, kw_only=True, alias="client", metadata={"serializable": False})

    @lazy_property()
    def client(self) -> MongoClient:
        return import_optional_dependency("pymongo").MongoClient(self.connection_string)

    def get_collection(self) -> Collection:
        """Returns the MongoDB Collection instance for the specified database and collection name."""
        return self.client[self.database_name][self.collection_name]

    def upsert_vector(
        self,
        vector: list[float],
        *,
        vector_id: Optional[str] = None,
        namespace: Optional[str] = None,
        meta: Optional[dict] = None,
        **kwargs,
    ) -> str:
        """Inserts or updates a vector in the collection.

        If a vector with the given vector ID already exists, it is updated; otherwise, a new vector is inserted.
        """
        collection = self.get_collection()

        if vector_id is None:
            result = collection.insert_one({self.vector_path: vector, "namespace": namespace, "meta": meta})
            vector_id = str(result.inserted_id)
        else:
            collection.replace_one(
                {"_id": vector_id},
                {self.vector_path: vector, "namespace": namespace, "meta": meta},
                upsert=True,
            )
        return vector_id

    def load_entry(self, vector_id: str, *, namespace: Optional[str] = None) -> Optional[BaseVectorStoreDriver.Entry]:
        """Loads a document entry from the MongoDB collection based on the vector ID.

        Returns:
            The loaded Entry if found; otherwise, None is returned.
        """
        collection = self.get_collection()
        if namespace:
            doc = collection.find_one({"_id": vector_id, "namespace": namespace})
        else:
            doc = collection.find_one({"_id": vector_id})

        if doc is None:
            return doc
        else:
            return BaseVectorStoreDriver.Entry(
                id=str(doc["_id"]),
                vector=doc[self.vector_path],
                namespace=doc["namespace"],
                meta=doc["meta"],
            )

    def load_entries(self, *, namespace: Optional[str] = None) -> list[BaseVectorStoreDriver.Entry]:
        """Loads all document entries from the MongoDB collection.

        Entries can optionally be filtered by namespace.
        """
        collection = self.get_collection()
        cursor = collection.find() if namespace is None else collection.find({"namespace": namespace})

        return [
            BaseVectorStoreDriver.Entry(
                id=str(doc["_id"]),
                vector=doc[self.vector_path],
                namespace=doc["namespace"],
                meta=doc["meta"],
            )
            for doc in cursor
        ]

    def query_vector(
        self,
        vector: list[float],
        *,
        count: Optional[int] = None,
        namespace: Optional[str] = None,
        include_vectors: bool = False,
        offset: Optional[int] = None,
        **kwargs,
    ) -> list[BaseVectorStoreDriver.Entry]:
        """Queries the MongoDB collection for documents that match the provided vector list.

        Results can be customized based on parameters like count, namespace, inclusion of vectors, offset, and index.
        """
        collection = self.get_collection()

        count = count or BaseVectorStoreDriver.DEFAULT_QUERY_COUNT
        offset = offset or 0

        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.index_name,
                    "path": self.vector_path,
                    "queryVector": vector,
                    "numCandidates": min(count * self.num_candidates_multiplier, self.MAX_NUM_CANDIDATES),
                    "limit": count,
                },
            },
            {
                "$project": {
                    "_id": 1,
                    self.vector_path: 1,
                    "namespace": 1,
                    "meta": 1,
                    "score": {"$meta": "vectorSearchScore"},
                },
            },
        ]

        if namespace:
            pipeline[0]["$vectorSearch"]["filter"] = {"namespace": namespace}

        return [
            BaseVectorStoreDriver.Entry(
                id=str(doc["_id"]),
                vector=doc[self.vector_path] if include_vectors else [],
                score=doc["score"],
                meta=doc["meta"],
                namespace=namespace,
            )
            for doc in collection.aggregate(pipeline)
        ]

    def delete_vector(self, vector_id: str) -> None:
        """Deletes the vector from the collection."""
        collection = self.get_collection()
        collection.delete_one({"_id": vector_id})
