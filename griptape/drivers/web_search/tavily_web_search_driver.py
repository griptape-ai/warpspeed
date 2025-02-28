from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from attrs import define, field

from griptape.artifacts import JsonArtifact, ListArtifact
from griptape.drivers.web_search import BaseWebSearchDriver
from griptape.utils import import_optional_dependency
from griptape.utils.decorators import lazy_property

if TYPE_CHECKING:
    from tavily import TavilyClient


@define
class TavilyWebSearchDriver(BaseWebSearchDriver):
    api_key: str = field(kw_only=True)
    params: dict[str, Any] = field(factory=dict, kw_only=True, metadata={"serializable": True})
    _client: Optional[TavilyClient] = field(
        default=None, kw_only=True, alias="client", metadata={"serializable": False}
    )

    @lazy_property()
    def client(self) -> TavilyClient:
        return import_optional_dependency("tavily").TavilyClient(self.api_key)

    def search(self, query: str, **kwargs) -> ListArtifact:
        response = self.client.search(query, max_results=self.results_count, **self.params, **kwargs)
        results = response.get("results", [])
        return ListArtifact([(JsonArtifact(result)) for result in results])
