from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional, TypeVar

from attrs import define, field
from typing_extensions import ParamSpec

from griptape.artifacts import (
    BaseArtifact,
    BlobArtifact,
    BooleanArtifact,
    GenericArtifact,
    JsonArtifact,
    ListArtifact,
    TextArtifact,
)
from griptape.events import EventBus, EventListener, FinishStructureRunEvent
from griptape.utils.decorators import lazy_property

P = ParamSpec("P")
T = TypeVar("T")

if TYPE_CHECKING:
    from types import TracebackType

    from griptape.observability.observability import Observability


@define()
class GriptapeCloudStructure:
    """Utility for working with Griptape Cloud Structures.

    Attributes:
        _event_listener: Event Listener to use. Defaults to an EventListener with a GriptapeCloudEventListenerDriver.
        _observability: Observability to use. Defaults to an Observability with a GriptapeCloudObservabilityDriver.
        observe: Whether to enable observability. Enabling requires the `drivers-observability-griptape-cloud` extra.
    """

    _event_listener: Optional[EventListener] = field(default=None, kw_only=True, alias="event_listener")
    _observability: Optional[Observability] = field(default=None, kw_only=True, alias="observability")
    observe: bool = field(default=False, kw_only=True)
    _output: Optional[BaseArtifact] = field(default=None, init=False)

    @lazy_property()
    def event_listener(self) -> EventListener:
        from griptape.drivers.event_listener.griptape_cloud import GriptapeCloudEventListenerDriver

        return EventListener(event_listener_driver=GriptapeCloudEventListenerDriver())

    @lazy_property()
    def observability(self) -> Observability:
        from griptape.drivers.observability.griptape_cloud import GriptapeCloudObservabilityDriver
        from griptape.observability.observability import Observability

        return Observability(observability_driver=GriptapeCloudObservabilityDriver())

    @property
    def output(self) -> Optional[BaseArtifact]:
        return self._output

    @output.setter
    def output(self, value: BaseArtifact | Any) -> None:
        if isinstance(value, BaseArtifact):
            self._output = value
        else:
            if isinstance(value, list):
                self._output = ListArtifact([self._to_artifact(item) for item in value])
            else:
                self._output = self._to_artifact(value)

    @property
    def structure_run_id(self) -> str:
        return os.environ["GT_CLOUD_STRUCTURE_RUN_ID"]

    @property
    def in_managed_environment(self) -> bool:
        return "GT_CLOUD_STRUCTURE_RUN_ID" in os.environ

    def __enter__(self) -> GriptapeCloudStructure:
        from griptape.observability.observability import Observability

        if self.in_managed_environment:
            EventBus.add_event_listener(self.event_listener)

            if self.observe:
                Observability.set_global_driver(self.observability.observability_driver)
                self.observability.observability_driver.__enter__()

        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        exc_traceback: Optional[TracebackType],
    ) -> None:
        from griptape.observability.observability import Observability

        if self.in_managed_environment:
            if self.output is not None:
                EventBus.publish_event(FinishStructureRunEvent(output_task_output=self.output), flush=True)
            EventBus.remove_event_listener(self.event_listener)

            if self.observe:
                Observability.set_global_driver(None)
                self.observability.observability_driver.__exit__(exc_type, exc_value, exc_traceback)

    def _to_artifact(self, value: Any) -> BaseArtifact:
        if isinstance(value, str):
            return TextArtifact(value)
        elif isinstance(value, bool):
            return BooleanArtifact(value)
        elif isinstance(value, dict):
            return JsonArtifact(value)
        elif isinstance(value, bytes):
            return BlobArtifact(value)
        else:
            return GenericArtifact(value)
