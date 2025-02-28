from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urljoin

import requests
from attrs import Attribute, Factory, define, field

from griptape.drivers.ruleset import BaseRulesetDriver
from griptape.utils import dict_merge

if TYPE_CHECKING:
    from griptape.rules import BaseRule


@define(kw_only=True)
class GriptapeCloudRulesetDriver(BaseRulesetDriver):
    """A driver for storing conversation memory in the Griptape Cloud.

    Attributes:
        ruleset_id: The ID of the Thread to store the conversation memory in. If not provided, the driver will attempt to
            retrieve the ID from the environment variable `GT_CLOUD_THREAD_ID`. If that is not set, a new Thread will be
            created.
        base_url: The base URL of the Griptape Cloud API. Defaults to the value of the environment variable
            `GT_CLOUD_BASE_URL` or `https://cloud.griptape.ai`.
        api_key: The API key to use for authenticating with the Griptape Cloud API. If not provided, the driver will
            attempt to retrieve the API key from the environment variable `GT_CLOUD_API_KEY`.

    Raises:
        ValueError: If `api_key` is not provided.
    """

    ruleset_id: Optional[str] = field(
        default=None,
        metadata={"serializable": True},
    )
    base_url: str = field(
        default=Factory(lambda: os.getenv("GT_CLOUD_BASE_URL", "https://cloud.griptape.ai")),
    )
    api_key: Optional[str] = field(default=Factory(lambda: os.getenv("GT_CLOUD_API_KEY")))
    headers: dict = field(
        default=Factory(lambda self: {"Authorization": f"Bearer {self.api_key}"}, takes_self=True),
        init=False,
    )

    @api_key.validator  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
    def validate_api_key(self, _: Attribute, value: Optional[str]) -> str:
        if value is None:
            raise ValueError(f"{self.__class__.__name__} requires an API key")
        return value

    def load(self, ruleset_name: str) -> tuple[list[BaseRule], dict[str, Any]]:
        """Load the ruleset from Griptape Cloud, using the ruleset name as an alias if ruleset_id is not provided."""
        ruleset = None

        if self.ruleset_id is not None:
            res = self._call_api("get", f"/rulesets/{self.ruleset_id}", raise_for_status=False)
            if res.status_code == 200:
                ruleset = res.json()

        # use name as 'alias' to get ruleset
        if ruleset is None:
            res = self._call_api("get", f"/rulesets?alias={ruleset_name}").json()
            if res.get("rulesets"):
                ruleset = res["rulesets"][0]

        # no ruleset by name or ruleset_id
        if ruleset is None:
            if self.raise_not_found:
                raise ValueError(f"No ruleset found with alias: {ruleset_name} or ruleset_id: {self.ruleset_id}")
            return [], {}

        rules = self._call_api("get", f"/rules?ruleset_id={ruleset['ruleset_id']}").json().get("rules", [])

        for rule in rules:
            rule["metadata"] = dict_merge(rule.get("metadata", {}), {"griptape_cloud_rule_id": rule["rule_id"]})

        return [self._get_rule(rule["rule"], rule["metadata"]) for rule in rules], ruleset.get("metadata", {})

    def _get_url(self, path: str) -> str:
        path = path.lstrip("/")
        return urljoin(self.base_url, f"/api/{path}")

    def _call_api(self, method: str, path: str, *, raise_for_status: bool = True) -> requests.Response:
        res = requests.request(method, self._get_url(path), headers=self.headers)
        if raise_for_status:
            res.raise_for_status()
        return res
