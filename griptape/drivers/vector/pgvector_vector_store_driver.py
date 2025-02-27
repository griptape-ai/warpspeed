from __future__ import annotations

import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn, Optional

from attrs import Attribute, Factory, define, field

from griptape.drivers.vector import BaseVectorStoreDriver
from griptape.utils import import_optional_dependency
from griptape.utils.decorators import lazy_property

if TYPE_CHECKING:
    import sqlalchemy


@define
class PgVectorVectorStoreDriver(BaseVectorStoreDriver):
    """A vector store driver to Postgres using the PGVector extension.

    Attributes:
        connection_string: An optional string describing the target Postgres database instance.
        create_engine_params: Additional configuration params passed when creating the database connection.
        engine: An optional sqlalchemy Postgres engine to use.
        table_name: Optionally specify the name of the table to used to store vectors.
    """

    connection_string: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": True})
    create_engine_params: dict = field(factory=dict, kw_only=True, metadata={"serializable": True})
    table_name: str = field(kw_only=True, metadata={"serializable": True})
    _model: Any = field(default=Factory(lambda self: self.default_vector_model(), takes_self=True))
    _engine: Optional[sqlalchemy.Engine] = field(
        default=None, kw_only=True, alias="engine", metadata={"serializable": False}
    )

    @connection_string.validator  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
    def validate_connection_string(self, _: Attribute, connection_string: Optional[str]) -> None:
        # If an engine is provided, the connection string is not used.
        if self._engine is not None:
            return

        # If an engine is not provided, a connection string is required.
        if connection_string is None:
            raise ValueError("An engine or connection string is required")

        if not connection_string.startswith("postgresql://"):
            raise ValueError("The connection string must describe a Postgres database connection")

    @lazy_property()
    def engine(self) -> sqlalchemy.Engine:
        return import_optional_dependency("sqlalchemy").create_engine(
            self.connection_string, **self.create_engine_params
        )

    def setup(
        self,
        *,
        create_schema: bool = True,
        install_uuid_extension: bool = True,
        install_vector_extension: bool = True,
    ) -> None:
        """Provides a mechanism to initialize the database schema and extensions."""
        sqlalchemy_sql = import_optional_dependency("sqlalchemy.sql")

        if install_uuid_extension:
            with self.engine.begin() as conn:
                conn.execute(sqlalchemy_sql.text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'))

        if install_vector_extension:
            with self.engine.begin() as conn:
                conn.execute(sqlalchemy_sql.text('CREATE EXTENSION IF NOT EXISTS "vector";'))

        if create_schema:
            self._model.metadata.create_all(self.engine)

    def upsert_vector(
        self,
        vector: list[float],
        *,
        vector_id: Optional[str] = None,
        namespace: Optional[str] = None,
        meta: Optional[dict] = None,
        **kwargs,
    ) -> str:
        """Inserts or updates a vector in the collection."""
        sqlalchemy_orm = import_optional_dependency("sqlalchemy.orm")

        with sqlalchemy_orm.Session(self.engine) as session:
            obj = self._model(id=vector_id, vector=vector, namespace=namespace, meta=meta, **kwargs)

            obj = session.merge(obj)
            session.commit()

            return str(getattr(obj, "id"))

    def load_entry(self, vector_id: str, *, namespace: Optional[str] = None) -> BaseVectorStoreDriver.Entry:
        """Retrieves a specific vector entry from the collection based on its identifier and optional namespace."""
        sqlalchemy_orm = import_optional_dependency("sqlalchemy.orm")

        with sqlalchemy_orm.Session(self.engine) as session:
            result = session.get(self._model, vector_id)

            return BaseVectorStoreDriver.Entry(
                id=getattr(result, "id"),
                vector=getattr(result, "vector"),
                namespace=getattr(result, "namespace"),
                meta=getattr(result, "meta"),
            )

    def load_entries(self, *, namespace: Optional[str] = None) -> list[BaseVectorStoreDriver.Entry]:
        """Retrieves all vector entries from the collection, optionally filtering to only those that match the provided namespace."""
        sqlalchemy_orm = import_optional_dependency("sqlalchemy.orm")

        with sqlalchemy_orm.Session(self.engine) as session:
            query = session.query(self._model)
            if namespace:
                query = query.filter_by(namespace=namespace)

            results = query.all()

            return [
                BaseVectorStoreDriver.Entry(
                    id=str(result.id),
                    vector=result.vector,
                    namespace=result.namespace,
                    meta=result.meta,
                )
                for result in results
            ]

    def query_vector(
        self,
        vector: list[float],
        *,
        count: Optional[int] = BaseVectorStoreDriver.DEFAULT_QUERY_COUNT,
        namespace: Optional[str] = None,
        include_vectors: bool = False,
        distance_metric: str = "cosine_distance",
        **kwargs,
    ) -> list[BaseVectorStoreDriver.Entry]:
        """Performs a search on the collection to find vectors similar to the provided input vector, optionally filtering to only those that match the provided namespace."""
        sqlalchemy_orm = import_optional_dependency("sqlalchemy.orm")

        distance_metrics = {
            "cosine_distance": self._model.vector.cosine_distance,
            "l2_distance": self._model.vector.l2_distance,
            "inner_product": self._model.vector.max_inner_product,
        }

        if distance_metric not in distance_metrics:
            raise ValueError("Invalid distance metric provided")

        op = distance_metrics[distance_metric]

        with sqlalchemy_orm.Session(self.engine) as session:
            # The query should return both the vector and the distance metric score.
            query_result = session.query(self._model, op(vector).label("score")).order_by(op(vector))  # pyright: ignore[reportOptionalCall]

            filter_kwargs: Optional[OrderedDict] = None

            if namespace is not None:
                filter_kwargs = OrderedDict(namespace=namespace)

            if "filter" in kwargs and isinstance(kwargs["filter"], dict):
                filter_kwargs = filter_kwargs or OrderedDict()
                filter_kwargs.update(kwargs["filter"])

            if filter_kwargs is not None:
                query_result = query_result.filter_by(**filter_kwargs)

            results = query_result.limit(count).all()

            return [
                BaseVectorStoreDriver.Entry(
                    id=str(result[0].id),
                    vector=result[0].vector if include_vectors else None,
                    score=result[1],
                    meta=result[0].meta,
                    namespace=result[0].namespace,
                )
                for result in results
            ]

    def query(
        self,
        query: str,
        *,
        count: Optional[int] = BaseVectorStoreDriver.DEFAULT_QUERY_COUNT,
        namespace: Optional[str] = None,
        include_vectors: bool = False,
        distance_metric: str = "cosine_distance",
        **kwargs,
    ) -> list[BaseVectorStoreDriver.Entry]:
        """Performs a search on the collection to find vectors similar to the provided input vector, optionally filtering to only those that match the provided namespace."""
        vector = self.embedding_driver.embed_string(query)
        return self.query_vector(
            vector,
            count=count,
            namespace=namespace,
            include_vectors=include_vectors,
            distance_metric=distance_metric,
            **kwargs,
        )

    def default_vector_model(self) -> Any:
        pgvector_sqlalchemy = import_optional_dependency("pgvector.sqlalchemy")
        sqlalchemy = import_optional_dependency("sqlalchemy")
        sqlalchemy_dialects_postgresql = import_optional_dependency("sqlalchemy.dialects.postgresql")
        sqlalchemy_orm = import_optional_dependency("sqlalchemy.orm")

        @dataclass
        class VectorModel(sqlalchemy_orm.declarative_base()):
            __tablename__ = self.table_name

            id = sqlalchemy.Column(
                sqlalchemy_dialects_postgresql.UUID(as_uuid=True),
                primary_key=True,
                default=uuid.uuid4,
                unique=True,
                nullable=False,
            )
            vector = sqlalchemy.Column(pgvector_sqlalchemy.Vector())
            namespace = sqlalchemy.Column(sqlalchemy.String)
            meta = sqlalchemy.Column(sqlalchemy.JSON)

        return VectorModel

    def delete_vector(self, vector_id: str) -> NoReturn:
        raise NotImplementedError(f"{self.__class__.__name__} does not support deletion.")
