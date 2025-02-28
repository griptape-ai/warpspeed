from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Optional

from attrs import Attribute, define, field

from griptape.drivers.sql import BaseSqlDriver
from griptape.utils.decorators import lazy_property

if TYPE_CHECKING:
    import boto3
    from mypy_boto3_redshift_data import RedshiftDataAPIServiceClient


@define
class AmazonRedshiftSqlDriver(BaseSqlDriver):
    database: str = field(kw_only=True)
    session: boto3.Session = field(kw_only=True)
    cluster_identifier: Optional[str] = field(default=None, kw_only=True)
    workgroup_name: Optional[str] = field(default=None, kw_only=True)
    db_user: Optional[str] = field(default=None, kw_only=True)
    database_credentials_secret_arn: Optional[str] = field(default=None, kw_only=True)
    wait_for_query_completion_sec: float = field(default=0.3, kw_only=True)
    _client: Optional[RedshiftDataAPIServiceClient] = field(
        default=None, kw_only=True, alias="client", metadata={"serializable": False}
    )

    @lazy_property()
    def client(self) -> RedshiftDataAPIServiceClient:
        return self.session.client("redshift-data")

    @workgroup_name.validator  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
    def validate_params(self, _: Attribute, workgroup_name: Optional[str]) -> None:
        if not self.cluster_identifier and not self.workgroup_name:
            raise ValueError("Provide a value for one of `cluster_identifier` or `workgroup_name`")
        if self.cluster_identifier and self.workgroup_name:
            raise ValueError("Provide a value for either `cluster_identifier` or `workgroup_name`, but not both")

    @classmethod
    def _process_rows_from_records(cls, records: list) -> list[list]:
        return [[c[list(c.keys())[0]] for c in r] for r in records]

    @classmethod
    def _process_cells_from_rows_and_columns(cls, columns: list, rows: list[list]) -> list[dict[str, Any]]:
        return [{column: r[idx] for idx, column in enumerate(columns)} for r in rows]

    @classmethod
    def _process_columns_from_column_metadata(cls, meta: dict) -> list:
        return [k["name"] for k in meta]

    @classmethod
    def _post_process(cls, meta: dict, records: list) -> list[dict[str, Any]]:
        columns = cls._process_columns_from_column_metadata(meta)
        rows = cls._process_rows_from_records(records)
        return cls._process_cells_from_rows_and_columns(columns, rows)

    def execute_query(self, query: str) -> Optional[list[BaseSqlDriver.RowResult]]:
        rows = self.execute_query_raw(query)
        if rows:
            return [BaseSqlDriver.RowResult(row) for row in rows]
        else:
            return None

    def execute_query_raw(self, query: str) -> Optional[list[dict[str, Optional[Any]]]]:
        function_kwargs = {"Sql": query, "Database": self.database}
        if self.workgroup_name:
            function_kwargs["WorkgroupName"] = self.workgroup_name
        if self.cluster_identifier:
            function_kwargs["ClusterIdentifier"] = self.cluster_identifier
        if self.db_user:
            function_kwargs["DbUser"] = self.db_user
        if self.database_credentials_secret_arn:
            function_kwargs["SecretArn"] = self.database_credentials_secret_arn

        response = self.client.execute_statement(**function_kwargs)
        response_id = response["Id"]

        statement = self.client.describe_statement(Id=response_id)

        while statement["Status"] in ["SUBMITTED", "PICKED", "STARTED"]:
            time.sleep(self.wait_for_query_completion_sec)
            statement = self.client.describe_statement(Id=response_id)

        if statement["Status"] == "FINISHED":
            statement_result = self.client.get_statement_result(Id=response_id)
            results = statement_result.get("Records", [])

            while "NextToken" in statement_result:
                statement_result = self.client.get_statement_result(
                    Id=response_id,
                    NextToken=statement_result["NextToken"],
                )
                results = results + response.get("Records", [])

            return self._post_process(statement_result["ColumnMetadata"], results)

        elif statement["Status"] in ["FAILED", "ABORTED"]:
            return None
        return None

    def get_table_schema(self, table_name: str, schema: Optional[str] = None) -> Optional[str]:
        function_kwargs = {"Database": self.database, "Table": table_name}
        if schema:
            function_kwargs["Schema"] = schema
        if self.workgroup_name:
            function_kwargs["WorkgroupName"] = self.workgroup_name
        if self.cluster_identifier:
            function_kwargs["ClusterIdentifier"] = self.cluster_identifier
        if self.db_user:
            function_kwargs["DbUser"] = self.db_user
        if self.database_credentials_secret_arn:
            function_kwargs["SecretArn"] = self.database_credentials_secret_arn
        response = self.client.describe_table(**function_kwargs)
        return str([col["name"] for col in response["ColumnList"]])
