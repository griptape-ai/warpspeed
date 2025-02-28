from __future__ import annotations

from typing import Optional

from attrs import define, field

from griptape.exceptions import DummyError
from griptape.tokenizers import BaseTokenizer


@define
class DummyTokenizer(BaseTokenizer):
    model: Optional[str] = field(default=None, kw_only=True)
    _max_input_tokens: int = field(init=False, default=0, kw_only=True, alias="max_input_tokens")
    _max_output_tokens: int = field(init=False, default=0, kw_only=True, alias="max_output_tokens")

    def count_tokens(self, text: str) -> int:
        raise DummyError(__class__.__name__, "count_tokens")
