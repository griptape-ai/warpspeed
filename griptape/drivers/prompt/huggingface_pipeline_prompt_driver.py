from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from attrs import Attribute, Factory, define, field

from griptape.artifacts import TextArtifact
from griptape.common import DeltaMessage, Message, PromptStack, TextMessageContent, observable
from griptape.configs import Defaults
from griptape.drivers.prompt import BasePromptDriver
from griptape.tokenizers import HuggingFaceTokenizer
from griptape.utils import import_optional_dependency
from griptape.utils.decorators import lazy_property

if TYPE_CHECKING:
    from collections.abc import Iterator

    from transformers import TextGenerationPipeline

    from griptape.drivers.prompt.base_prompt_driver import StructuredOutputStrategy

logger = logging.getLogger(Defaults.logging_config.logger_name)


@define
class HuggingFacePipelinePromptDriver(BasePromptDriver):
    """Hugging Face Pipeline Prompt Driver.

    Attributes:
        model: Hugging Face Hub model name.
    """

    max_tokens: int = field(default=250, kw_only=True, metadata={"serializable": True})
    model: str = field(kw_only=True, metadata={"serializable": True})
    tokenizer: HuggingFaceTokenizer = field(
        default=Factory(
            lambda self: HuggingFaceTokenizer(model=self.model, max_output_tokens=self.max_tokens),
            takes_self=True,
        ),
        kw_only=True,
    )
    structured_output_strategy: StructuredOutputStrategy = field(
        default="rule", kw_only=True, metadata={"serializable": True}
    )
    _pipeline: Optional[TextGenerationPipeline] = field(
        default=None, kw_only=True, alias="pipeline", metadata={"serializable": False}
    )

    @structured_output_strategy.validator  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
    def validate_structured_output_strategy(self, _: Attribute, value: str) -> str:
        if value in ("native", "tool"):
            raise ValueError(f"{__class__.__name__} does not support `{value}` structured output strategy.")

        return value

    @lazy_property()
    def pipeline(self) -> TextGenerationPipeline:
        return import_optional_dependency("transformers").pipeline(
            task="text-generation",
            model=self.model,
            max_new_tokens=self.max_tokens,
            tokenizer=self.tokenizer.tokenizer,
        )

    @observable
    def try_run(self, prompt_stack: PromptStack) -> Message:
        messages = self._prompt_stack_to_messages(prompt_stack)
        full_params = self._base_params(prompt_stack)
        logger.debug(
            (
                messages,
                full_params,
            )
        )

        result = self.pipeline(messages, **full_params)
        logger.debug(result)

        if isinstance(result, list):
            if len(result) == 1:
                generated_text = result[0]["generated_text"][-1]["content"]

                input_tokens = len(self.__prompt_stack_to_tokens(prompt_stack))
                output_tokens = len(self.tokenizer.tokenizer.encode(generated_text))  # pyright: ignore[reportArgumentType]

                return Message(
                    content=[TextMessageContent(TextArtifact(generated_text))],
                    role=Message.ASSISTANT_ROLE,
                    usage=Message.Usage(input_tokens=input_tokens, output_tokens=output_tokens),
                )
            else:
                raise Exception("completion with more than one choice is not supported yet")
        else:
            raise Exception("invalid output format")

    @observable
    def try_stream(self, prompt_stack: PromptStack) -> Iterator[DeltaMessage]:
        raise NotImplementedError("streaming is not supported")

    def prompt_stack_to_string(self, prompt_stack: PromptStack) -> str:
        return self.tokenizer.tokenizer.decode(self.__prompt_stack_to_tokens(prompt_stack))  # pyright: ignore[reportArgumentType]

    def _base_params(self, prompt_stack: PromptStack) -> dict:
        return {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "do_sample": True,
            **self.extra_params,
        }

    def _prompt_stack_to_messages(self, prompt_stack: PromptStack) -> list[dict]:
        messages = []

        for message in prompt_stack.messages:
            messages.append({"role": message.role, "content": message.to_text()})

        return messages

    def __prompt_stack_to_tokens(self, prompt_stack: PromptStack) -> list[int]:
        messages = self._prompt_stack_to_messages(prompt_stack)
        tokens = self.tokenizer.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)

        if isinstance(tokens, list):
            return tokens  # pyright: ignore[reportReturnType] According to the [docs](https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template), the return type is List[int].
        else:
            raise ValueError("Invalid output type.")
