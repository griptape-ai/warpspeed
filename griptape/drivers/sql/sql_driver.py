from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional

from attrs import define, field

from griptape.drivers.sql import BaseSqlDriver
from griptape.utils import import_optional_dependency
from griptape.utils.decorators import lazy_property

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


@define
class SqlDriver(BaseSqlDriver):
    engine_url: str = field(kw_only=True)
    create_engine_params: dict = field(factory=dict, kw_only=True)
    _engine: Optional[Engine] = field(default=None, kw_only=True, alias="engine", metadata={"serializable": False})

    @lazy_property()
    def engine(self) -> Engine:
        return import_optional_dependency("sqlalchemy").create_engine(self.engine_url, **self.create_engine_params)

    def execute_query(self, query: str) -> Optional[list[BaseSqlDriver.RowResult]]:
        rows = self.execute_query_raw(query)

        if rows:
            return [BaseSqlDriver.RowResult(row) for row in rows]
        else:
            return None

    def execute_query_raw(self, query: str) -> Optional[list[dict[str, Optional[Any]]]]:
        sqlalchemy = import_optional_dependency("sqlalchemy")

        with self.engine.connect() as con:
            results = con.execute(sqlalchemy.text(query))

            if results is not None:
                if results.returns_rows:
                    return [dict(result._mapping) for result in results]
                else:
                    con.commit()
                    return None
            else:
                raise ValueError("No result found")

    def get_table_schema(self, table_name: str, schema: Optional[str] = None) -> Optional[str]:
        sqlalchemy_exc = import_optional_dependency("sqlalchemy.exc")

        try:
            return str(SqlDriver._get_table_schema(self.engine, table_name, schema))
        except sqlalchemy_exc.NoSuchTableError:
            return None

    @staticmethod
    @lru_cache
    def _get_table_schema(
        engine: Engine, table_name: str, schema: Optional[str] = None
    ) -> Optional[list[tuple[str, str]]]:
        sqlalchemy = import_optional_dependency("sqlalchemy")

        return [(col["name"], col["type"]) for col in sqlalchemy.inspect(engine).get_columns(table_name, schema=schema)]
