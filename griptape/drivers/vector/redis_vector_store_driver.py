from typing import Optional, List
from griptape import utils
from griptape.drivers import BaseVectorStoreDriver
import redis
from attr import define, field
import json
from redis.commands.search.query import Query
import numpy as np


@define
class RedisVectorStoreDriver(BaseVectorStoreDriver):
    host: str = field(kw_only=True)
    port: int = field(kw_only=True)
    db: int = field(kw_only=True, default=0)
    password: Optional[str] = field(default=None, kw_only=True)
    client: redis.StrictRedis = field(init=False)
    index: str = field(kw_only=True)

    def __attrs_post_init__(self) -> None:
        self.client = redis.StrictRedis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password
        )

    def upsert_vector(
            self,
            vector: list[float],
            vector_id: Optional[str] = None,
            namespace: Optional[str] = None,
            meta: Optional[dict] = None,
            **kwargs
    ) -> str:
        vector_id = vector_id if vector_id else utils.str_to_hash(str(vector))
        key = f'{namespace}:{vector_id}' if namespace else vector_id
        bytes_vector = np.array(vector, dtype=np.float32).tobytes()
        mapping_obj = {
            "vector": bytes_vector
        }
        if meta:
            mapping_obj["metadata"] = json.dumps(meta)
        self.client.hset(key, mapping=mapping_obj)
        return vector_id

    def load_entry(self, vector_id: str, namespace: Optional[str] = None) -> Optional[BaseVectorStoreDriver.Entry]:
        key = f'{namespace}:{vector_id}' if namespace else vector_id
        result = self.client.hgetall(key)
        #print("Load result:", result)
        #reversed_vector = np.frombuffer(result[b"vector"], dtype=np.float32).tolist()
        reversed_vector = np.frombuffer(result[b"vector"], dtype=np.float32)
        reversed_vector_list = reversed_vector.tolist()

        if result:
            value = {
                "vector": reversed_vector_list,
                "metadata": json.loads(result[b"metadata"]) if b"metadata" in result else None
            }
            return BaseVectorStoreDriver.Entry(
                id=vector_id,
                meta=value["metadata"],
                vector=value["vector"],
                namespace=namespace
            )
        else:
            return None

    # decode("utf-8")
    def load_entries(self, namespace: Optional[str] = None) -> list[BaseVectorStoreDriver.Entry]:
        pattern = f'{namespace}:*' if namespace else '*'
        keys = self.client.keys(pattern)
        #print("Keys: ", keys)
        entries = []
        for key in keys:
            entries.append(self.load_entry(key.decode("utf-8"), namespace=namespace))
        #print("Entries: ", entries)
        return entries

    def query(self, query: str, count: Optional[int] = None, namespace: Optional[str] = None, **kwargs) -> \
            List[BaseVectorStoreDriver.QueryResult]:

        # Define the query expression
        query_expression = (
            Query(f"*=>[KNN {count or 10} @vector $vec as score]")
            .sort_by("score")
            .return_fields("vector", "score")
            .paging(0, count or 10)
            .dialect(2)
        )

        query_params = {
            "vector": query
        }

        # Execute the search
        results = self.client.ft(self.index).search(query_expression, query_params).docs

        query_results = []
        for document in results:
            vector_float = np.frombuffer(document['vector'], dtype=np.float32)
            vector_float_list = vector_float.tolist()
            print("Document:", document)

            query_results.append(
                BaseVectorStoreDriver.QueryResult(
                    vector=vector_float_list,
                    score=float(document['score']),
                    meta=None,  # Adjust if metadata is available
                    namespace=None  # Adjust if namespace is needed
                )
            )

        return query_results
