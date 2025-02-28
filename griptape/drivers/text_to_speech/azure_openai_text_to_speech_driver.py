from __future__ import annotations

from typing import Callable, Optional

import openai
from attrs import Factory, define, field

from griptape.drivers.text_to_speech.openai import OpenAiTextToSpeechDriver
from griptape.utils.decorators import lazy_property


@define
class AzureOpenAiTextToSpeechDriver(OpenAiTextToSpeechDriver):
    """Azure OpenAi Text to Speech Driver.

    Attributes:
        azure_deployment: An optional Azure OpenAi deployment id. Defaults to the model name.
        azure_endpoint: An Azure OpenAi endpoint.
        azure_ad_token: An optional Azure Active Directory token.
        azure_ad_token_provider: An optional Azure Active Directory token provider.
        api_version: An Azure OpenAi API version.
        client: An `openai.AzureOpenAI` client.
    """

    model: str = field(default="tts", kw_only=True, metadata={"serializable": True})
    azure_deployment: str = field(
        kw_only=True,
        default=Factory(lambda self: self.model, takes_self=True),
        metadata={"serializable": True},
    )
    azure_endpoint: str = field(kw_only=True, metadata={"serializable": True})
    azure_ad_token: Optional[str] = field(kw_only=True, default=None, metadata={"serializable": False})
    azure_ad_token_provider: Optional[Callable[[], str]] = field(
        kw_only=True,
        default=None,
        metadata={"serializable": False},
    )
    api_version: str = field(default="2024-07-01-preview", kw_only=True, metadata={"serializable": True})
    _client: Optional[openai.AzureOpenAI] = field(
        default=None, kw_only=True, alias="client", metadata={"serializable": False}
    )

    @lazy_property()
    def client(self) -> openai.AzureOpenAI:
        return openai.AzureOpenAI(
            organization=self.organization,
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.azure_deployment,
            azure_ad_token=self.azure_ad_token,
            azure_ad_token_provider=self.azure_ad_token_provider,
        )
