from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Optional

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

    import boto3

    from griptape.common import PromptStack
    from griptape.drivers.prompt.base_prompt_driver import StructuredOutputStrategy

logger = logging.getLogger(Defaults.logging_config.logger_name)


@define
class AmazonSageMakerJumpstartPromptDriver(BasePromptDriver):
    session: boto3.Session = field(default=Factory(lambda: import_optional_dependency("boto3").Session()), kw_only=True)
    endpoint: str = field(kw_only=True, metadata={"serializable": True})
    custom_attributes: str = field(default="accept_eula=true", kw_only=True, metadata={"serializable": True})
    inference_component_name: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": True})
    stream: bool = field(default=False, kw_only=True, metadata={"serializable": True})
    max_tokens: int = field(default=250, kw_only=True, metadata={"serializable": True})
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
    _client: Optional[Any] = field(default=None, kw_only=True, alias="client", metadata={"serializable": False})

    @structured_output_strategy.validator  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
    def validate_structured_output_strategy(self, _: Attribute, value: str) -> str:
        if value != "rule":
            raise ValueError(f"{__class__.__name__} does not support `{value}` structured output strategy.")

        return value

    @lazy_property()
    def client(self) -> Any:
        return self.session.client("sagemaker-runtime")

    @stream.validator  # pyright: ignore[reportAttributeAccessIssue]
    def validate_stream(self, _: Attribute, stream: bool) -> None:  # noqa: FBT001
        if stream:
            raise ValueError("streaming is not supported")

    @observable
    def try_run(self, prompt_stack: PromptStack) -> Message:
        payload = {
            "inputs": self.prompt_stack_to_string(prompt_stack),
            "parameters": {**self._base_params(prompt_stack)},
        }
        logger.debug(payload)

        response = self.client.invoke_endpoint(
            EndpointName=self.endpoint,
            ContentType="application/json",
            Body=json.dumps(payload),
            CustomAttributes=self.custom_attributes,
            **(
                {"InferenceComponentName": self.inference_component_name}
                if self.inference_component_name is not None
                else {}
            ),
        )

        decoded_body = json.loads(response["Body"].read().decode("utf8"))
        logger.debug(decoded_body)

        if isinstance(decoded_body, list):
            if decoded_body:
                generated_text = decoded_body[0]["generated_text"]
            else:
                raise ValueError("model response is empty")
        else:
            generated_text = decoded_body["generated_text"]

        input_tokens = len(self.__prompt_stack_to_tokens(prompt_stack))
        output_tokens = len(self.tokenizer.tokenizer.encode(generated_text))  # pyright: ignore[reportArgumentType]

        return Message(
            content=[TextMessageContent(TextArtifact(generated_text))],
            role=Message.ASSISTANT_ROLE,
            usage=Message.Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        )

    @observable
    def try_stream(self, prompt_stack: PromptStack) -> Iterator[DeltaMessage]:
        raise NotImplementedError("streaming is not supported")

    def prompt_stack_to_string(self, prompt_stack: PromptStack) -> str:
        return self.tokenizer.tokenizer.decode(self.__prompt_stack_to_tokens(prompt_stack))  # pyright: ignore[reportArgumentType]

    def _base_params(self, prompt_stack: PromptStack) -> dict:
        return {
            "temperature": self.temperature,
            "max_new_tokens": self.max_tokens,
            "do_sample": True,
            "eos_token_id": self.tokenizer.tokenizer.eos_token_id,
            "stop_strings": self.tokenizer.stop_sequences,
            "return_full_text": False,
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
