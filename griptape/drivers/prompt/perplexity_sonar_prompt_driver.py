from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define, field

from griptape.drivers.prompt.openai import OpenAiChatPromptDriver

if TYPE_CHECKING:
    from griptape.common import PromptStack


@define
class PerplexitySonarPromptDriver(OpenAiChatPromptDriver):
    base_url: str = field(default="https://api.perplexity.ai", kw_only=True, metadata={"serializable": True})
    structured_output_strategy: str = field(default="native", kw_only=True, metadata={"serializable": True})

    def _base_params(self, prompt_stack: PromptStack) -> dict:
        params = super()._base_params(prompt_stack)

        if "stop" in params:
            del params["stop"]

        return params
