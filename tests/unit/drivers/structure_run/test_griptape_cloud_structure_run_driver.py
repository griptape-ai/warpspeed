from unittest.mock import Mock

import pytest

from griptape.artifacts import InfoArtifact, TextArtifact
from griptape.artifacts.error_artifact import ErrorArtifact
from griptape.drivers.structure_run.griptape_cloud import GriptapeCloudStructureRunDriver
from griptape.events import EventBus
from griptape.events.event_listener import EventListener


class TestGriptapeCloudStructureRunDriver:
    @pytest.fixture(autouse=True)
    def mock_requests_get(self, mocker):
        mock_response = mocker.Mock()
        mock_response.json.return_value = {
            "events": [
                {
                    "origin": "USER",
                    "type": "FooBarEvent",
                    "payload": {
                        "type": "FooBarEvent",
                    },
                },
                {
                    "origin": "USER",
                    "type": "FinishStructureRunEvent",
                    "payload": {
                        "type": "FinishStructureRunEvent",
                        "output_task_input": {
                            "type": "TextArtifact",
                            "value": "foo bar",
                        },
                        "output_task_output": {
                            "type": "TextArtifact",
                            "value": "foo bar",
                        },
                    },
                },
                {"origin": "FOO", "type": "BAR"},
                {
                    "origin": "SYSTEM",
                    "type": "StructureRunError",
                    "payload": {
                        "status_detail": {
                            "error": "foo bar",
                        },
                    },
                },
            ],
            "next_offset": 0,
        }

        return mocker.patch(
            "requests.get",
            return_value=mock_response,
        )

    @pytest.fixture(autouse=True)
    def mock_requests_post(self, mocker):
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"structure_run_id": "1"}

        return mocker.patch(
            "requests.post",
            return_value=mock_response,
        )

    @pytest.fixture()
    def driver(self):
        return GriptapeCloudStructureRunDriver(
            base_url="https://cloud-foo.griptape.ai",
            api_key="foo bar",
            structure_id="1",
            env={"key": "value"},
            structure_run_wait_time_interval=0,
        )

    def test_run(self, driver):
        mock_on_event = Mock()
        EventBus.add_event_listener(EventListener(on_event=mock_on_event))
        result = driver.run(TextArtifact("foo bar"))

        events = mock_on_event.call_args[0]
        assert len(events) == 1
        assert events[0].type == "FinishStructureRunEvent"

        assert isinstance(result, ErrorArtifact)
        assert result.value == "foo bar"

    def test_async_run(self, driver):
        driver.async_run = True
        result = driver.run(TextArtifact("foo bar"))
        assert isinstance(result, InfoArtifact)
        assert result.value == "Run started successfully"

    def test_run_timeout(self, driver, mocker):
        mocker.patch(
            "requests.get",
            return_value=Mock(
                json=Mock(
                    return_value={
                        "events": [],
                    }
                )
            ),
        )

        with pytest.raises(TimeoutError):
            driver.run(TextArtifact("foo bar"))
