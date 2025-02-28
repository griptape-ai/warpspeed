from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

from attrs import Factory, define, field

from griptape.drivers.event_listener.base_event_listener_driver import BaseEventListenerDriver
from griptape.utils import import_optional_dependency
from griptape.utils.decorators import lazy_property

if TYPE_CHECKING:
    import boto3
    from mypy_boto3_iot_data import IoTDataPlaneClient


@define
class AwsIotCoreEventListenerDriver(BaseEventListenerDriver):
    iot_endpoint: str = field(kw_only=True)
    topic: str = field(kw_only=True)
    session: boto3.Session = field(default=Factory(lambda: import_optional_dependency("boto3").Session()), kw_only=True)
    _client: Optional[IoTDataPlaneClient] = field(
        default=None, kw_only=True, alias="client", metadata={"serializable": False}
    )

    @lazy_property()
    def client(self) -> IoTDataPlaneClient:
        return self.session.client("iot-data")

    def try_publish_event_payload(self, event_payload: dict) -> None:
        self.client.publish(topic=self.topic, payload=json.dumps(event_payload))

    def try_publish_event_payload_batch(self, event_payload_batch: list[dict]) -> None:
        self.client.publish(topic=self.topic, payload=json.dumps(event_payload_batch))
