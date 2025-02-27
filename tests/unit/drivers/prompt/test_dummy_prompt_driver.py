import pytest

from griptape.drivers.prompt.dummy import DummyPromptDriver
from griptape.exceptions import DummyError


class TestDummyPromptDriver:
    @pytest.fixture()
    def prompt_driver(self):
        return DummyPromptDriver()

    def test_init(self, prompt_driver):
        assert prompt_driver

    def test_try_run(self, prompt_driver):
        with pytest.raises(DummyError):
            prompt_driver.try_run("prompt-stack")

    def test_try_stream_run(self, prompt_driver):
        with pytest.raises(DummyError):
            prompt_driver.try_run("prompt-stack")
