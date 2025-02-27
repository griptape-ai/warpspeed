from __future__ import annotations

import logging
from typing import Optional

import tiktoken
from attrs import Factory, define, field

from griptape.tokenizers import BaseTokenizer


@define()
class OpenAiTokenizer(BaseTokenizer):
    DEFAULT_OPENAI_GPT_3_COMPLETION_MODEL = "gpt-3.5-turbo-instruct"
    DEFAULT_OPENAI_GPT_3_CHAT_MODEL = "gpt-3.5-turbo"
    DEFAULT_OPENAI_GPT_4_MODEL = "gpt-4o"
    DEFAULT_ENCODING = "cl100k_base"
    DEFAULT_MAX_TOKENS = 2049
    DEFAULT_MAX_OUTPUT_TOKENS = 4096
    TOKEN_OFFSET = 8

    # https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
    MODEL_PREFIXES_TO_MAX_INPUT_TOKENS = {
        "gpt-4o": 128000,
        "gpt-4-1106": 128000,
        "gpt-4-32k": 32768,
        "gpt-4": 8192,
        "gpt-3.5-turbo-16k": 16384,
        "gpt-3.5-turbo": 4096,
        "gpt-35-turbo-16k": 16384,
        "gpt-35-turbo": 4096,
        "text-embedding-ada-002": 8191,
        "text-embedding-ada-001": 2046,
        "text-embedding-3-small": 8191,
        "text-embedding-3-large": 8191,
    }

    MODEL_PREFIXES_TO_MAX_OUTPUT_TOKENS = {"gpt": 4096}

    EMBEDDING_MODELS = [
        "text-embedding-ada-002",
        "text-embedding-ada-001",
        "text-embedding-3-small",
        "text-embedding-3-large",
    ]

    _max_input_tokens: int = field(
        kw_only=True,
        default=Factory(lambda self: self._default_max_input_tokens(), takes_self=True),
        alias="max_input_tokens",
    )
    _max_output_tokens: int = field(
        kw_only=True,
        default=Factory(lambda self: self._default_max_output_tokens(), takes_self=True),
        alias="max_output_tokens",
    )

    @property
    def encoding(self) -> tiktoken.Encoding:
        try:
            return tiktoken.encoding_for_model(self.model)
        except KeyError:
            return tiktoken.get_encoding(self.DEFAULT_ENCODING)

    def _default_max_input_tokens(self) -> int:
        tokens = next((v for k, v in self.MODEL_PREFIXES_TO_MAX_INPUT_TOKENS.items() if self.model.startswith(k)), None)
        offset = 0 if self.model in self.EMBEDDING_MODELS else self.TOKEN_OFFSET

        return (tokens or self.DEFAULT_MAX_TOKENS) - offset

    def _default_max_output_tokens(self) -> int:
        tokens = next(
            (v for k, v in self.MODEL_PREFIXES_TO_MAX_OUTPUT_TOKENS.items() if self.model.startswith(k)),
            None,
        )

        if tokens is None:
            return self.DEFAULT_MAX_OUTPUT_TOKENS
        else:
            return tokens

    def count_tokens(self, text: str | list[dict], model: Optional[str] = None) -> int:  # noqa: C901
        """Handles the special case of ChatML.

        Implementation adopted from the official OpenAI notebook:
        https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb.
        """
        if isinstance(text, list):
            model = model or self.model

            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                logging.warning("model not found. Using cl100k_base encoding.")

                encoding = tiktoken.get_encoding("cl100k_base")

            if model in {
                "gpt-3.5-turbo-0613",
                "gpt-3.5-turbo-16k-0613",
                "gpt-4-0314",
                "gpt-4-32k-0314",
                "gpt-4-0613",
                "gpt-4-32k-0613",
                "gpt-4o-2024-05-13",
            }:
                tokens_per_message = 3
                tokens_per_name = 1
            elif model == "gpt-3.5-turbo-0301":
                # every message follows <|start|>{role/name}\n{content}<|end|>\n
                tokens_per_message = 4
                # if there's a name, the role is omitted
                tokens_per_name = -1
            elif "gpt-3.5-turbo" in model or "gpt-35-turbo" in model:
                logging.info("gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
                return self.count_tokens(text, model="gpt-3.5-turbo-0613")
            elif "gpt-4o" in model:
                logging.info("gpt-4o may update over time. Returning num tokens assuming gpt-4o-2024-05-13.")
                return self.count_tokens(text, model="gpt-4o-2024-05-13")
            elif "gpt-4" in model:
                logging.info("gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
                return self.count_tokens(text, model="gpt-4-0613")
            else:
                raise NotImplementedError(
                    f"token_count() is not implemented for model {model}. "
                    "See https://github.com/openai/openai-python/blob/main/chatml.md for "
                    "information on how messages are converted to tokens."
                )

            num_tokens = 0

            for message in text:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name

            # every reply is primed with <|start|>assistant<|message|>
            num_tokens += 3

            return num_tokens
        else:
            return len(self.encoding.encode(text, allowed_special=set(self.stop_sequences)))
