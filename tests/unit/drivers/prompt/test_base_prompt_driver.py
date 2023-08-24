from griptape.core import PromptStack
from tests.mocks.mock_prompt_driver import MockPromptDriver
from tests.mocks.mock_failing_prompt_driver import MockFailingPromptDriver
from griptape.artifacts import ErrorArtifact, TextArtifact
from griptape.tasks import PromptTask
from griptape.structures import Pipeline


class TestBasePromptDriver:
    def test_run_retries_success(self):
        driver = MockPromptDriver(max_attempts=1)
        pipeline = Pipeline(prompt_driver=driver)

        pipeline.add_task(
            PromptTask("test")
        )

        assert isinstance(pipeline.run().output, TextArtifact)

    def test_run_retries_failure(self):
        driver = MockFailingPromptDriver(max_failures=2, max_attempts=1)
        pipeline = Pipeline(prompt_driver=driver)

        pipeline.add_task(
            PromptTask("test")
        )

        assert isinstance(pipeline.run().output, ErrorArtifact)

    def test_token_count(self):
        assert MockPromptDriver().token_count(
            PromptStack(inputs=[PromptStack.Input("foobar", role=PromptStack.USER_ROLE)])
        ) == 4

    def test_max_output_tokens(self):
        assert MockPromptDriver().max_output_tokens("foobar") == 4087
        assert MockPromptDriver(max_tokens=100).max_output_tokens("foobar") == 100

    def test_prompt_stack_to_string(self):
        assert MockPromptDriver().prompt_stack_to_string(
            PromptStack(inputs=[PromptStack.Input("foobar", role=PromptStack.USER_ROLE)])
        ) == "User: foobar"

    def test_custom_prompt_stack_to_string(self):
        assert MockPromptDriver(
            prompt_stack_to_string=lambda stack: f"Foo: {stack.inputs[0].content}"
        ).prompt_stack_to_string(
            PromptStack(inputs=[PromptStack.Input("foobar", role=PromptStack.USER_ROLE)])
        ) == "Foo: foobar"
