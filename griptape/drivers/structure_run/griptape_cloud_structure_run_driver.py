from __future__ import annotations

import logging
import time
from urllib.parse import urljoin

import requests
from attrs import Factory, define, field

from griptape.artifacts import BaseArtifact, ErrorArtifact, InfoArtifact
from griptape.configs.defaults_config import Defaults
from griptape.drivers.structure_run.base_structure_run_driver import BaseStructureRunDriver
from griptape.events import BaseEvent, EventBus

logger = logging.getLogger(Defaults.logging_config.logger_name)


@define
class GriptapeCloudStructureRunDriver(BaseStructureRunDriver):
    base_url: str = field(default="https://cloud.griptape.ai", kw_only=True)
    api_key: str = field(kw_only=True)
    headers: dict = field(
        default=Factory(lambda self: {"Authorization": f"Bearer {self.api_key}"}, takes_self=True),
        kw_only=True,
    )
    structure_id: str = field(kw_only=True)
    structure_run_wait_time_interval: int = field(default=2, kw_only=True)
    structure_run_max_wait_time_attempts: int = field(default=20, kw_only=True)
    async_run: bool = field(default=False, kw_only=True)

    def try_run(self, *args: BaseArtifact) -> BaseArtifact | InfoArtifact:
        structure_run_id = self._create_run(*args)

        if self.async_run:
            return InfoArtifact("Run started successfully")
        else:
            return self._get_run_result(structure_run_id)

    def _create_run(self, *args: BaseArtifact) -> str:
        url = urljoin(self.base_url.strip("/"), f"/api/structures/{self.structure_id}/runs")

        env_vars = [{"name": key, "value": value, "source": "manual"} for key, value in self.env.items()]

        response = requests.post(
            url,
            json={"args": [arg.value for arg in args], "env_vars": env_vars},
            headers=self.headers,
        )
        response.raise_for_status()
        response_json = response.json()

        return response_json["structure_run_id"]

    def _get_run_result(self, structure_run_id: str) -> BaseArtifact | InfoArtifact:
        events, next_offset = self._get_run_events(structure_run_id)
        attempts = 0
        output = None

        while output is None and attempts < self.structure_run_max_wait_time_attempts:
            for event in events:
                event_type = event["type"]
                event_payload = event.get("payload", {})
                if event["origin"] == "USER":
                    try:
                        EventBus.publish_event(BaseEvent.from_dict(event_payload))
                    except ValueError as e:
                        logger.warning("Failed to deserialize event: %s", e)
                    if event["type"] == "FinishStructureRunEvent":
                        output = BaseArtifact.from_dict(event_payload["output_task_output"])
                elif event["origin"] == "SYSTEM":
                    if event_type == "StructureRunError":
                        output = ErrorArtifact(event_payload["status_detail"]["error"])

            if output is None and not events:
                time.sleep(self.structure_run_wait_time_interval)
                attempts += 1
            events, next_offset = self._get_run_events(structure_run_id, offset=next_offset)

        if output is None:
            raise TimeoutError("The structure run did not finish in time.")

        return output

    def _get_run_events(self, structure_run_id: str, offset: int = 0) -> tuple[list[dict], int]:
        url = urljoin(self.base_url.strip("/"), f"/api/structure-runs/{structure_run_id}/events")
        response = requests.get(url, headers=self.headers, params={"offset": offset})
        response.raise_for_status()

        response_json = response.json()

        events = response_json.get("events", [])
        next_offset = response_json.get("next_offset", 0)

        return events, next_offset
