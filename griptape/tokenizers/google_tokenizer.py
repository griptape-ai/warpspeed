from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from attrs import define, field

from griptape.tokenizers import BaseTokenizer
from griptape.utils import import_optional_dependency
from griptape.utils.decorators import lazy_property

if TYPE_CHECKING:
    from google.generativeai.generative_models import GenerativeModel


@define()
class GoogleTokenizer(BaseTokenizer):
    MODEL_PREFIXES_TO_MAX_INPUT_TOKENS = {"gemini-1.5-pro": 2097152, "gemini": 1048576}
    MODEL_PREFIXES_TO_MAX_OUTPUT_TOKENS = {"gemini": 8192}

    api_key: str = field(kw_only=True, metadata={"serializable": True})
    _client: Optional[GenerativeModel] = field(
        default=None, kw_only=True, alias="client", metadata={"serializable": False}
    )

    @lazy_property()
    def client(self) -> GenerativeModel:
        genai = import_optional_dependency("google.generativeai")
        genai.configure(api_key=self.api_key)

        return genai.GenerativeModel(self.model)

    def count_tokens(self, text: str) -> int:
        return self.client.count_tokens(text).total_tokens
