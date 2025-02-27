from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

from attrs import define, field

from griptape.artifacts import ListArtifact, TextArtifact
from griptape.drivers.web_search import BaseWebSearchDriver
from griptape.utils import import_optional_dependency
from griptape.utils.decorators import lazy_property

if TYPE_CHECKING:
    from duckduckgo_search import DDGS


@define
class DuckDuckGoWebSearchDriver(BaseWebSearchDriver):
    language: str = field(default="en", kw_only=True)
    country: str = field(default="us", kw_only=True)
    _client: Optional[DDGS] = field(default=None, kw_only=True, alias="client", metadata={"serializable": False})

    @lazy_property()
    def client(self) -> DDGS:
        return import_optional_dependency("duckduckgo_search").DDGS()

    def search(self, query: str, **kwargs) -> ListArtifact:
        try:
            results = self.client.text(
                query, region=f"{self.language}-{self.country}", max_results=self.results_count, **kwargs
            )
            return ListArtifact(
                [
                    TextArtifact(
                        json.dumps({"title": result["title"], "url": result["href"], "description": result["body"]}),
                    )
                    for result in results
                ],
            )
        except Exception as e:
            raise Exception(f"Error searching '{query}' with DuckDuckGo: {e}") from e
