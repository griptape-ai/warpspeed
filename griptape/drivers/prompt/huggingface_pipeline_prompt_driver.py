from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from attrs import Factory, define, field

from griptape.artifacts import TextArtifact
from griptape.common import DeltaMessage, MessageStack, Message, TextMessageContent
from griptape.drivers import BasePromptDriver
from griptape.tokenizers import HuggingFaceTokenizer
from griptape.utils import import_optional_dependency

if TYPE_CHECKING:
    from transformers import TextGenerationPipeline


@define
class HuggingFacePipelinePromptDriver(BasePromptDriver):
    """
    Attributes:
        params: Custom model run parameters.
        model: Hugging Face Hub model name.

    """

    max_tokens: int = field(default=250, kw_only=True, metadata={"serializable": True})
    model: str = field(kw_only=True, metadata={"serializable": True})
    params: dict = field(factory=dict, kw_only=True, metadata={"serializable": True})
    tokenizer: HuggingFaceTokenizer = field(
        default=Factory(
            lambda self: HuggingFaceTokenizer(model=self.model, max_output_tokens=self.max_tokens), takes_self=True
        ),
        kw_only=True,
    )
    pipe: TextGenerationPipeline = field(
        default=Factory(
            lambda self: import_optional_dependency("transformers").pipeline(
                "text-generation", model=self.model, max_new_tokens=self.max_tokens, tokenizer=self.tokenizer.tokenizer
            ),
            takes_self=True,
        )
    )

    def try_run(self, message_stack: MessageStack) -> Message:
        messages = self._message_stack_to_messages(message_stack)

        result = self.pipe(
            messages, max_new_tokens=self.max_tokens, temperature=self.temperature, do_sample=True, **self.params
        )

        if isinstance(result, list):
            if len(result) == 1:
                generated_text = result[0]["generated_text"][-1]["content"]

                input_tokens = len(self.__message_stack_to_tokens(message_stack))
                output_tokens = len(self.tokenizer.tokenizer.encode(generated_text))

                return Message(
                    content=[TextMessageContent(TextArtifact(generated_text))],
                    role=Message.ASSISTANT_ROLE,
                    usage=Message.Usage(input_tokens=input_tokens, output_tokens=output_tokens),
                )
            else:
                raise Exception("completion with more than one choice is not supported yet")
        else:
            raise Exception("invalid output format")

    def try_stream(self, message_stack: MessageStack) -> Iterator[DeltaMessage]:
        raise NotImplementedError("streaming is not supported")

    def message_stack_to_string(self, message_stack: MessageStack) -> str:
        return self.tokenizer.tokenizer.decode(self.__message_stack_to_tokens(message_stack))

    def _message_stack_to_messages(self, message_stack: MessageStack) -> list[dict]:
        messages = []

        for message in message_stack.messages:
            messages.append({"role": message.role, "content": message.to_text()})

        return messages

    def __message_stack_to_tokens(self, message_stack: MessageStack) -> list[int]:
        messages = self._message_stack_to_messages(message_stack)
        tokens = self.tokenizer.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)

        if isinstance(tokens, list):
            return tokens
        else:
            raise ValueError("Invalid output type.")
