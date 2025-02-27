from __future__ import annotations

from typing import Optional

import requests
from attrs import define, field

from griptape.drivers.event_listener.base_event_listener_driver import BaseEventListenerDriver


@define
class WebhookEventListenerDriver(BaseEventListenerDriver):
    webhook_url: str = field(kw_only=True)
    headers: Optional[dict] = field(default=None, kw_only=True)

    def try_publish_event_payload(self, event_payload: dict) -> None:
        response = requests.post(url=self.webhook_url, json=event_payload, headers=self.headers)
        response.raise_for_status()

    def try_publish_event_payload_batch(self, event_payload_batch: list[dict]) -> None:
        response = requests.post(url=self.webhook_url, json=event_payload_batch, headers=self.headers)
        response.raise_for_status()
