from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, overload

from attrs import define, field

from griptape import utils
from griptape.artifacts import BaseArtifact, ImageArtifact, ListArtifact, TextArtifact
from griptape.mixins.futures_executor_mixin import FuturesExecutorMixin
from griptape.mixins.serializable_mixin import SerializableMixin
from griptape.utils import with_contextvars

if TYPE_CHECKING:
    from griptape.drivers.embedding import BaseEmbeddingDriver


@define
class BaseVectorStoreDriver(SerializableMixin, FuturesExecutorMixin, ABC):
    DEFAULT_QUERY_COUNT = 5

    @define
    class Entry(SerializableMixin):
        id: str = field(metadata={"serializable": True})
        vector: Optional[list[float]] = field(default=None, metadata={"serializable": True})
        score: Optional[float] = field(default=None, metadata={"serializable": True})
        meta: Optional[dict] = field(default=None, metadata={"serializable": True})
        namespace: Optional[str] = field(default=None, metadata={"serializable": True})

        def to_artifact(self) -> BaseArtifact:
            return BaseArtifact.from_json(self.meta["artifact"])  # pyright: ignore[reportOptionalSubscript]

    embedding_driver: BaseEmbeddingDriver = field(kw_only=True, metadata={"serializable": True})

    def upsert_text_artifacts(
        self,
        artifacts: list[TextArtifact] | dict[str, list[TextArtifact] | list[ImageArtifact]],
        *,
        meta: Optional[dict] = None,
        **kwargs,
    ) -> list[str] | dict[str, list[str]]:
        return self.upsert_collection(artifacts, meta=meta, **kwargs)

    def upsert_text_artifact(
        self,
        artifact: TextArtifact,
        *,
        namespace: Optional[str] = None,
        meta: Optional[dict] = None,
        vector_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        return self.upsert(artifact, namespace=namespace, meta=meta, vector_id=vector_id, **kwargs)

    def upsert_text(
        self,
        string: str,
        *,
        namespace: Optional[str] = None,
        meta: Optional[dict] = None,
        vector_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        return self.upsert(string, namespace=namespace, meta=meta, vector_id=vector_id, **kwargs)

    @overload
    def upsert_collection(
        self,
        artifacts: list[TextArtifact],
        *,
        meta: Optional[dict] = None,
        **kwargs,
    ) -> list[str]: ...

    @overload
    def upsert_collection(
        self,
        artifacts: dict[str, list[TextArtifact] | list[ImageArtifact]],
        *,
        meta: Optional[dict] = None,
        **kwargs,
    ) -> dict[str, list[str]]: ...

    def upsert_collection(
        self,
        artifacts: list[TextArtifact] | dict[str, list[TextArtifact] | list[ImageArtifact]],
        *,
        meta: Optional[dict] = None,
        **kwargs,
    ):
        with self.create_futures_executor() as futures_executor:
            if isinstance(artifacts, list):
                return utils.execute_futures_list(
                    [
                        futures_executor.submit(with_contextvars(self.upsert), a, namespace=None, meta=meta, **kwargs)
                        for a in artifacts
                    ],
                )
            else:
                futures_dict = {}

                for namespace, artifact_list in artifacts.items():
                    for a in artifact_list:
                        if not futures_dict.get(namespace):
                            futures_dict[namespace] = []

                        futures_dict[namespace].append(
                            futures_executor.submit(
                                with_contextvars(self.upsert), a, namespace=namespace, meta=meta, **kwargs
                            )
                        )

                return utils.execute_futures_list_dict(futures_dict)

    def upsert(
        self,
        value: str | TextArtifact | ImageArtifact,
        *,
        namespace: Optional[str] = None,
        meta: Optional[dict] = None,
        vector_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        artifact = TextArtifact(value) if isinstance(value, str) else value

        meta = {} if meta is None else meta

        if vector_id is None:
            value = artifact.to_text() if artifact.reference is None else artifact.to_text() + str(artifact.reference)
            vector_id = self._get_default_vector_id(value)

        if self.does_entry_exist(vector_id, namespace=namespace):
            return vector_id
        else:
            meta["artifact"] = artifact.to_json()

            vector = self.embedding_driver.embed(artifact)

            return self.upsert_vector(vector, vector_id=vector_id, namespace=namespace, meta=meta, **kwargs)

    def does_entry_exist(self, vector_id: str, *, namespace: Optional[str] = None) -> bool:
        try:
            return self.load_entry(vector_id, namespace=namespace) is not None
        except Exception:
            return False

    def load_artifacts(self, *, namespace: Optional[str] = None) -> ListArtifact:
        result = self.load_entries(namespace=namespace)
        artifacts = [r.to_artifact() for r in result]

        return ListArtifact([a for a in artifacts if isinstance(a, TextArtifact)])

    @abstractmethod
    def delete_vector(self, vector_id: str) -> None: ...

    @abstractmethod
    def upsert_vector(
        self,
        vector: list[float],
        *,
        vector_id: Optional[str] = None,
        namespace: Optional[str] = None,
        meta: Optional[dict] = None,
        **kwargs,
    ) -> str: ...

    @abstractmethod
    def load_entry(self, vector_id: str, *, namespace: Optional[str] = None) -> Optional[Entry]: ...

    @abstractmethod
    def load_entries(self, *, namespace: Optional[str] = None) -> list[Entry]: ...

    def query_vector(
        self,
        vector: list[float],
        *,
        count: Optional[int] = None,
        namespace: Optional[str] = None,
        include_vectors: bool = False,
        **kwargs,
    ) -> list[Entry]:
        # TODO: Mark as abstract method for griptape 2.0
        raise NotImplementedError(f"{self.__class__.__name__} does not support vector query.")

    def query(
        self,
        query: str | TextArtifact | ImageArtifact,
        *,
        count: Optional[int] = None,
        namespace: Optional[str] = None,
        include_vectors: bool = False,
        **kwargs,
    ) -> list[Entry]:
        vector = self.embedding_driver.embed(query)
        return self.query_vector(vector, count=count, namespace=namespace, include_vectors=include_vectors, **kwargs)

    def _get_default_vector_id(self, value: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_OID, value))
