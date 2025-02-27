from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import Factory, define, field

from griptape.tokenizers import BaseTokenizer
from griptape.utils import import_optional_dependency

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@define()
class HuggingFaceTokenizer(BaseTokenizer):
    tokenizer: PreTrainedTokenizerBase = field(
        default=Factory(
            lambda self: import_optional_dependency("transformers").AutoTokenizer.from_pretrained(self.model),
            takes_self=True,
        ),
        kw_only=True,
    )
    _max_input_tokens: int = field(
        default=Factory(lambda self: self.tokenizer.model_max_length, takes_self=True),
        kw_only=True,
        alias="max_input_tokens",
    )
    _max_output_tokens: int = field(default=4096, kw_only=True, alias="max_output_tokens")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))  # pyright: ignore[reportArgumentType]
