import pytest
from schema import Schema

from griptape.engines import JsonExtractionEngine
from griptape.structures import Agent
from griptape.tasks import JsonExtractionTask
from tests.mocks.mock_prompt_driver import MockPromptDriver


class TestJsonExtractionTask:
    @pytest.fixture()
    def task(self):
        return JsonExtractionTask("foo", args={"template_schema": Schema({"foo": "bar"}).json_schema("TemplateSchema")})

    def test_run(self, task, mock_config):
        mock_config.drivers.prompt_driver.mock_output = (
            '[{"test_key_1": "test_value_1"}, {"test_key_2": "test_value_2"}]'
        )
        agent = Agent()

        agent.add_task(task)

        result = task.run()

        assert len(result.value) == 2
        assert result.value[0].value == '{"test_key_1": "test_value_1"}'
        assert result.value[1].value == '{"test_key_2": "test_value_2"}'

    def test_config_extraction_engine(self, task):
        Agent().add_task(task)

        assert isinstance(task.extraction_engine, JsonExtractionEngine)
        assert isinstance(task.extraction_engine.prompt_driver, MockPromptDriver)
