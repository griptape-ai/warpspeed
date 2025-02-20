from unittest.mock import ANY

import pytest

from griptape.artifacts import ActionArtifact, AudioArtifact, TextArtifact
from griptape.common import ActionCallDeltaMessageContent, AudioDeltaMessageContent, TextDeltaMessageContent
from griptape.drivers.prompt.perplexity_sonar import PerplexitySonarPromptDriver
from tests.unit.drivers.prompt.test_openai_chat_prompt_driver import TestOpenAiChatPromptDriverFixtureMixin


class TestPerplexitySonarPromptDriver(TestOpenAiChatPromptDriverFixtureMixin):
    def test_init(self):
        assert PerplexitySonarPromptDriver(api_key="foo", model="gpt-4")

    @pytest.mark.parametrize("use_native_tools", [True, False])
    @pytest.mark.parametrize("structured_output_strategy", ["native", "tool"])
    @pytest.mark.parametrize("modalities", [["text"], ["text", "audio"], ["audio"]])
    def test_try_run(
        self,
        mock_chat_completion_create,
        prompt_stack,
        messages,
        use_native_tools,
        structured_output_strategy,
        modalities,
    ):
        # Given
        driver = PerplexitySonarPromptDriver(
            model="sonar-pro",
            use_native_tools=use_native_tools,
            structured_output_strategy=structured_output_strategy,
            modalities=modalities,
            extra_params={"foo": "bar"},
        )

        # When
        message = driver.try_run(prompt_stack)

        # Then
        mock_chat_completion_create.assert_called_once_with(
            model=driver.model,
            temperature=driver.temperature,
            user=driver.user,
            messages=messages,
            modalities=modalities,
            **{
                "audio": driver.audio,
            }
            if "audio" in driver.modalities
            else {},
            seed=driver.seed,
            **{
                "parallel_tool_calls": driver.parallel_tool_calls,
            }
            if prompt_stack.tools and driver.use_native_tools
            else {},
            **{
                "tools": self.OPENAI_TOOLS,
                "tool_choice": "required" if structured_output_strategy == "tool" else driver.tool_choice,
            }
            if use_native_tools
            else {},
            **{
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "Output",
                        "schema": self.OPENAI_STRUCTURED_OUTPUT_SCHEMA,
                        "strict": True,
                    },
                }
            }
            if structured_output_strategy == "native"
            else {},
            foo="bar",
        )
        assert isinstance(message.value[0], TextArtifact)
        assert message.value[0].value == "model-output"
        assert isinstance(message.value[1], AudioArtifact)
        assert message.value[1].value == b"assistant-audio-data"
        assert message.value[1].format == "wav"
        assert message.value[1].meta == {
            "audio_id": "audio-id",
            "transcript": "assistant-audio-transcription",
            "expires_at": ANY,
        }
        assert isinstance(message.value[2], ActionArtifact)
        assert message.value[2].value.tag == "mock-id"
        assert message.value[2].value.name == "MockTool"
        assert message.value[2].value.path == "test"
        assert message.value[2].value.input == {"foo": "bar"}

    @pytest.mark.parametrize("use_native_tools", [True, False])
    @pytest.mark.parametrize("structured_output_strategy", ["native", "tool"])
    @pytest.mark.parametrize("modalities", [["text"], ["text", "audio"], ["audio"]])
    def test_try_stream_run(
        self,
        mock_chat_completion_stream_create,
        prompt_stack,
        messages,
        use_native_tools,
        structured_output_strategy,
        modalities,
    ):
        # Given
        driver = PerplexitySonarPromptDriver(
            model="sonar-pro",
            stream=True,
            modalities=modalities,
            use_native_tools=use_native_tools,
            structured_output_strategy=structured_output_strategy,
            extra_params={"foo": "bar"},
        )

        # When
        stream = driver.try_stream(prompt_stack)
        event = next(stream)

        # Then
        mock_chat_completion_stream_create.assert_called_once_with(
            model=driver.model,
            temperature=driver.temperature,
            user=driver.user,
            **{
                "audio": driver.audio,
            }
            if "audio" in driver.modalities
            else {},
            stream=True,
            messages=messages,
            modalities=modalities,
            **{
                "tools": self.OPENAI_TOOLS,
                "tool_choice": "required" if structured_output_strategy == "tool" else driver.tool_choice,
            }
            if use_native_tools
            else {},
            seed=driver.seed,
            **{
                "parallel_tool_calls": driver.parallel_tool_calls,
            }
            if prompt_stack.tools and driver.use_native_tools
            else {},
            **{
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "Output",
                        "schema": self.OPENAI_STRUCTURED_OUTPUT_SCHEMA,
                        "strict": True,
                    },
                }
            }
            if structured_output_strategy == "native"
            else {},
            foo="bar",
            stream_options={"include_usage": True},
        )

        assert isinstance(event.content, TextDeltaMessageContent)
        assert event.content.text == "model-output"

        event = next(stream)
        assert isinstance(event.content, ActionCallDeltaMessageContent)
        assert event.content.tag == "mock-id"
        assert event.content.name == "MockTool"
        assert event.content.path == "test"

        event = next(stream)
        assert isinstance(event.content, ActionCallDeltaMessageContent)
        assert event.content.partial_input == '{"foo": "bar"}'

        event = next(stream)
        assert event.usage.input_tokens == 5
        assert event.usage.output_tokens == 10

        event = next(stream)
        assert isinstance(event.content, AudioDeltaMessageContent)
        assert event.content.id == "audio-id"

        event = next(stream)
        assert isinstance(event.content, AudioDeltaMessageContent)
        assert event.content.data == "YXNzaXN0YW50LWF1ZGlvLWRhdGE="

        event = next(stream)
        assert isinstance(event.content, AudioDeltaMessageContent)
        assert event.content.expires_at == ANY
        assert event.content.transcript == "assistant-audio-transcription"
