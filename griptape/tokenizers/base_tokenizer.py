from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from attrs import Factory, define, field

from griptape.utils.decorators import lazy_property


@define()
class BaseTokenizer(ABC):
    DEFAULT_MAX_INPUT_TOKENS = 4096
    DEFAULT_MAX_OUTPUT_TOKENS = 1000
    MODEL_PREFIXES_TO_MAX_INPUT_TOKENS = {}
    MODEL_PREFIXES_TO_MAX_OUTPUT_TOKENS = {}

    model: str = field(kw_only=True)
    stop_sequences: list[str] = field(default=Factory(list), kw_only=True)
    _max_input_tokens: Optional[int] = field(kw_only=True, default=None, alias="max_input_tokens")
    _max_output_tokens: Optional[int] = field(kw_only=True, default=None, alias="max_output_tokens")

    @lazy_property()
    def max_input_tokens(self) -> int:
        return self._default_max_input_tokens()

    @lazy_property()
    def max_output_tokens(self) -> int:
        return self._default_max_output_tokens()

    def __attrs_post_init__(self) -> None:
        if hasattr(self, "model"):
            if self.max_input_tokens is None:
                self.max_input_tokens = self._default_max_input_tokens()

            if self.max_output_tokens is None:
                self.max_output_tokens = self._default_max_output_tokens()

    def count_input_tokens_left(self, text: str) -> int:
        diff = self.max_input_tokens - self.count_tokens(text)

        if diff > 0:
            return diff
        else:
            return 0

    def count_output_tokens_left(self, text: str) -> int:
        diff = self.max_output_tokens - self.count_tokens(text)

        if diff > 0:
            return diff
        else:
            return 0

    @abstractmethod
    def count_tokens(self, text: str) -> int: ...

    def _default_max_input_tokens(self) -> int:
        tokens = next(
            (
                max_tokens
                for model_prefix, max_tokens in self.MODEL_PREFIXES_TO_MAX_INPUT_TOKENS.items()
                if model_prefix in self.model
            ),
            None,
        )

        if tokens is None:
            logging.warning(
                "Model %s not found in MODEL_PREFIXES_TO_MAX_INPUT_TOKENS, using default value of %s.",
                self.model,
                self.DEFAULT_MAX_INPUT_TOKENS,
            )
            return self.DEFAULT_MAX_INPUT_TOKENS
        else:
            return tokens

    def _default_max_output_tokens(self) -> int:
        tokens = next(
            (
                max_tokens
                for model_prefix, max_tokens in self.MODEL_PREFIXES_TO_MAX_OUTPUT_TOKENS.items()
                if model_prefix in self.model
            ),
            None,
        )

        if tokens is None:
            logging.warning(
                "Model %s not found in MODEL_PREFIXES_TO_MAX_OUTPUT_TOKENS, using default value of %s.",
                self.model,
                self.DEFAULT_MAX_OUTPUT_TOKENS,
            )
            return self.DEFAULT_MAX_OUTPUT_TOKENS
        else:
            return tokens
