from unittest.mock import Mock

import pytest

from griptape.drivers.text_to_speech.openai import AzureOpenAiTextToSpeechDriver


class TestAzureOpenAiTextToSpeechDriver:
    @pytest.fixture()
    def mock_speech_create(self, mocker):
        mock_speech_create = mocker.patch("openai.AzureOpenAI").return_value.audio.speech.create
        mock_function = Mock(arguments='{"foo": "bar"}', id="mock-id")
        mock_function.name = "MockTool_test"
        mock_speech_create.return_value = Mock(
            content=b"speech",
        )

        return mock_speech_create

    def test_init(self):
        assert AzureOpenAiTextToSpeechDriver(azure_endpoint="foobar", azure_deployment="foobar")
        assert AzureOpenAiTextToSpeechDriver(azure_endpoint="foobar").azure_deployment == "tts"

    def test_run_text_to_audio(self, mock_speech_create):
        driver = AzureOpenAiTextToSpeechDriver(azure_endpoint="foobar")
        output = driver.run_text_to_audio(["foo", "bar"])
        mock_speech_create.assert_called_once_with(
            input="foo. bar",
            model=driver.model,
            response_format=driver.format,
            voice=driver.voice,
        )
        assert output.value == b"speech"
