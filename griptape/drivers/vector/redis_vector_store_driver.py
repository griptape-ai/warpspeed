from __future__ import annotations

import json
from typing import TYPE_CHECKING, NoReturn, Optional

import numpy as np
from attrs import define, field

from griptape.drivers.vector import BaseVectorStoreDriver
from griptape.utils import import_optional_dependency, str_to_hash
from griptape.utils.decorators import lazy_property

if TYPE_CHECKING:
    from redis import Redis


@define
class RedisVectorStoreDriver(BaseVectorStoreDriver):
    """A Vector Store Driver for Redis.

    This driver interfaces with a Redis instance and utilizes the Redis hashes and RediSearch module to store, retrieve, and query vectors in a structured manner.
    Proper setup of the Redis instance and RediSearch is necessary for the driver to function correctly.

    Attributes:
        host: The host of the Redis instance.
        port: The port of the Redis instance.
        db: The database of the Redis instance.
        password: The password of the Redis instance.
        index: The name of the index to use.
    """

    host: str = field(kw_only=True, metadata={"serializable": True})
    port: int = field(kw_only=True, metadata={"serializable": True})
    db: int = field(kw_only=True, default=0, metadata={"serializable": True})
    password: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": False})
    index: str = field(kw_only=True, metadata={"serializable": True})
    _client: Optional[Redis] = field(default=None, kw_only=True, alias="client", metadata={"serializable": False})

    @lazy_property()
    def client(self) -> Redis:
        return import_optional_dependency("redis").Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            decode_responses=False,
        )

    def upsert_vector(
        self,
        vector: list[float],
        vector_id: Optional[str] = None,
        namespace: Optional[str] = None,
        meta: Optional[dict] = None,
        **kwargs,
    ) -> str:
        """Inserts or updates a vector in Redis.

        If a vector with the given vector ID already exists, it is updated; otherwise, a new vector is inserted.
        Metadata associated with the vector can also be provided.
        """
        vector_id = vector_id or str_to_hash(str(vector))
        key = self._generate_key(vector_id, namespace)
        bytes_vector = json.dumps(vector).encode("utf-8")

        mapping = {}
        mapping["vector"] = np.array(vector, dtype=np.float32).tobytes()
        mapping["vec_string"] = bytes_vector

        if namespace:
            mapping["namespace"] = namespace

        if meta:
            mapping["metadata"] = json.dumps(meta)

        self.client.hset(key, mapping=mapping)

        return vector_id

    def load_entry(self, vector_id: str, *, namespace: Optional[str] = None) -> Optional[BaseVectorStoreDriver.Entry]:
        """Retrieves a specific vector entry from Redis based on its identifier and optional namespace.

        Returns:
            If the entry is found, it returns an instance of BaseVectorStoreDriver.Entry; otherwise, None is returned.
        """
        key = self._generate_key(vector_id, namespace)
        result = self.client.hgetall(key)
        vector = np.frombuffer(result[b"vector"], dtype=np.float32).tolist()
        meta = json.loads(result[b"metadata"]) if b"metadata" in result else None

        return BaseVectorStoreDriver.Entry(id=vector_id, meta=meta, vector=vector, namespace=namespace)

    def load_entries(self, *, namespace: Optional[str] = None) -> list[BaseVectorStoreDriver.Entry]:
        """Retrieves all vector entries from Redis that match the optional namespace.

        Returns:
            A list of `BaseVectorStoreDriver.Entry` objects.
        """
        pattern = f"{namespace}:*" if namespace else "*"
        keys = self.client.keys(pattern)

        entries = []
        for key in keys:
            entry = self.load_entry(key.decode("utf-8"), namespace=namespace)
            if entry:
                entries.append(entry)

        return entries

    def query_vector(
        self,
        vector: list[float],
        *,
        count: Optional[int] = None,
        namespace: Optional[str] = None,
        include_vectors: bool = False,
        **kwargs,
    ) -> list[BaseVectorStoreDriver.Entry]:
        """Performs a nearest neighbor search on Redis to find vectors similar to the provided input vector.

        Results can be limited using the count parameter and optionally filtered by a namespace.

        Returns:
            A list of BaseVectorStoreDriver.Entry objects, each encapsulating the retrieved vector, its similarity score, metadata, and namespace.
        """
        search_query = import_optional_dependency("redis.commands.search.query")

        filter_expression = f"(@namespace:{{{namespace}}})" if namespace else "*"
        query_expression = (
            search_query.Query(f"{filter_expression}=>[KNN {count or 10} @vector $vector as score]")
            .sort_by("score")
            .return_fields("id", "score", "metadata", "vec_string")
            .paging(0, count or 10)
            .dialect(2)
        )

        query_params = {"vector": np.array(vector, dtype=np.float32).tobytes()}

        results = self.client.ft(self.index).search(query_expression, query_params).docs  # pyright: ignore[reportArgumentType]

        query_results = []
        for document in results:
            metadata = json.loads(document.metadata) if hasattr(document, "metadata") else None
            namespace = document.id.split(":")[0] if ":" in document.id else None
            vector_id = document.id.split(":")[1] if ":" in document.id else document.id
            vector_float_list = json.loads(document.vec_string) if include_vectors else None
            query_results.append(
                BaseVectorStoreDriver.Entry(
                    id=vector_id,
                    vector=vector_float_list,
                    score=float(document.score),
                    meta=metadata,
                    namespace=namespace,
                ),
            )
        return query_results

    def _generate_key(self, vector_id: str, namespace: Optional[str] = None) -> str:
        """Generates a Redis key using the provided vector ID and optionally a namespace."""
        return f"{namespace}:{vector_id}" if namespace else vector_id

    def _get_doc_prefix(self, namespace: Optional[str] = None) -> str:
        """Get the document prefix based on the provided namespace."""
        return f"{namespace}:" if namespace else ""

    def delete_vector(self, vector_id: str) -> NoReturn:
        raise NotImplementedError(f"{self.__class__.__name__} does not support deletion.")
