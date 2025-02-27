from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from attrs import Attribute, define, field

from griptape.drivers.sql import BaseSqlDriver
from griptape.utils import import_optional_dependency
from griptape.utils.decorators import lazy_property

if TYPE_CHECKING:
    from snowflake.connector import SnowflakeConnection
    from sqlalchemy.engine import Engine


@define
class SnowflakeSqlDriver(BaseSqlDriver):
    get_connection: Callable[[], SnowflakeConnection] = field(kw_only=True)
    _engine: Optional[Engine] = field(default=None, kw_only=True, alias="engine", metadata={"serializable": False})

    @get_connection.validator  # pyright: ignore[reportFunctionMemberAccess]
    def validate_get_connection(self, _: Attribute, get_connection: Callable[[], SnowflakeConnection]) -> None:
        snowflake_connection = get_connection()
        snowflake = import_optional_dependency("snowflake")

        if not isinstance(snowflake_connection, snowflake.connector.SnowflakeConnection):
            raise ValueError("The get_connection function must return a SnowflakeConnection")
        if not snowflake_connection.schema or not snowflake_connection.database:
            raise ValueError("Provide a schema and database for the Snowflake connection")

    @lazy_property()
    def engine(self) -> Engine:
        return import_optional_dependency("sqlalchemy").create_engine(
            "snowflake://not@used/db",
            creator=self.get_connection,
        )

    def execute_query(self, query: str) -> Optional[list[BaseSqlDriver.RowResult]]:
        rows = self.execute_query_raw(query)

        if rows:
            return [BaseSqlDriver.RowResult(row) for row in rows]
        else:
            return None

    def execute_query_raw(self, query: str) -> Optional[list[dict[str, Any]]]:
        sqlalchemy = import_optional_dependency("sqlalchemy")

        with self.engine.connect() as con:
            results = con.execute(sqlalchemy.text(query))

            if results is not None:
                if results.returns_rows:
                    return [dict(result._mapping) for result in results]
                else:
                    return None
            else:
                raise ValueError("No results found")

    def get_table_schema(self, table_name: str, schema: Optional[str] = None) -> Optional[str]:
        sqlalchemy = import_optional_dependency("sqlalchemy")

        try:
            metadata_obj = sqlalchemy.MetaData()
            metadata_obj.reflect(bind=self.engine)
            table = sqlalchemy.Table(table_name, metadata_obj, schema=schema, autoload=True, autoload_with=self.engine)
            return str([(c.name, c.type) for c in table.columns])
        except sqlalchemy.exc.NoSuchTableError:
            return None
