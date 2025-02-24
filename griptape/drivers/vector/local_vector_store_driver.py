from __future__ import annotations

import json
import operator
import os
import threading
from typing import Callable, NoReturn, Optional, TextIO

from attrs import Factory, define, field
from numpy import dot
from numpy.linalg import norm

from griptape import utils
from griptape.drivers.vector import BaseVectorStoreDriver


@define(kw_only=True)
class LocalVectorStoreDriver(BaseVectorStoreDriver):
    entries: dict[str, BaseVectorStoreDriver.Entry] = field(factory=dict)
    persist_file: Optional[str] = field(default=None)
    calculate_relatedness: Callable = field(default=lambda x, y: dot(x, y) / (norm(x) * norm(y)))
    thread_lock: threading.Lock = field(default=Factory(lambda: threading.Lock()))

    def __attrs_post_init__(self) -> None:
        if self.persist_file is not None:
            directory = os.path.dirname(self.persist_file)

            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            if not os.path.isfile(self.persist_file):
                with open(self.persist_file, "w") as file:
                    self.__save_entries_to_file(file)

            with open(self.persist_file, "r+") as file:
                if os.path.getsize(self.persist_file) > 0:
                    self.entries = self.load_entries_from_file(file)
                else:
                    self.__save_entries_to_file(file)

    def load_entries_from_file(self, json_file: TextIO) -> dict[str, BaseVectorStoreDriver.Entry]:
        with self.thread_lock:
            data = json.load(json_file)

            return {k: BaseVectorStoreDriver.Entry.from_dict(v) for k, v in data.items()}

    def upsert_vector(
        self,
        vector: list[float],
        *,
        vector_id: Optional[str] = None,
        namespace: Optional[str] = None,
        meta: Optional[dict] = None,
        **kwargs,
    ) -> str:
        vector_id = vector_id or utils.str_to_hash(str(vector))

        with self.thread_lock:
            self.entries[self.__namespaced_vector_id(vector_id, namespace=namespace)] = self.Entry(
                id=vector_id,
                vector=vector,
                meta=meta,
                namespace=namespace,
            )

        if self.persist_file is not None:
            # TODO: optimize later since it reserializes all entries from memory and stores them in the JSON file
            #  every time a new vector is inserted
            with open(self.persist_file, "w") as file:
                self.__save_entries_to_file(file)

        return vector_id

    def load_entry(self, vector_id: str, *, namespace: Optional[str] = None) -> Optional[BaseVectorStoreDriver.Entry]:
        return self.entries.get(self.__namespaced_vector_id(vector_id, namespace=namespace), None)

    def load_entries(self, *, namespace: Optional[str] = None) -> list[BaseVectorStoreDriver.Entry]:
        return [entry for key, entry in self.entries.items() if namespace is None or entry.namespace == namespace]

    def query_vector(
        self,
        vector: list[float],
        *,
        count: Optional[int] = None,
        namespace: Optional[str] = None,
        include_vectors: bool = False,
        **kwargs,
    ) -> list[BaseVectorStoreDriver.Entry]:
        if namespace:
            entries = {k: v for (k, v) in self.entries.items() if k.startswith(f"{namespace}-")}
        else:
            entries = self.entries

        entries_and_relatednesses = [
            (entry, self.calculate_relatedness(vector, entry.vector)) for entry in list(entries.values())
        ]

        entries_and_relatednesses.sort(key=operator.itemgetter(1), reverse=True)

        result = [
            BaseVectorStoreDriver.Entry(
                id=er[0].id, vector=er[0].vector, score=er[1], meta=er[0].meta, namespace=er[0].namespace
            )
            for er in entries_and_relatednesses
        ][:count]

        if include_vectors:
            return result
        else:
            return [
                BaseVectorStoreDriver.Entry(id=r.id, vector=[], score=r.score, meta=r.meta, namespace=r.namespace)
                for r in result
            ]

    def delete_vector(self, vector_id: str) -> NoReturn:
        raise NotImplementedError(f"{self.__class__.__name__} does not support deletion.")

    def __save_entries_to_file(self, json_file: TextIO) -> None:
        with self.thread_lock:
            serialized_data = {k: v.to_dict() for k, v in self.entries.items()}

            json.dump(serialized_data, json_file)

    def __namespaced_vector_id(self, vector_id: str, *, namespace: Optional[str]) -> str:
        return vector_id if namespace is None else f"{namespace}-{vector_id}"
