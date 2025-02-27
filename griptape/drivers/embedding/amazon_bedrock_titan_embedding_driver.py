from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

from attrs import Factory, define, field

from griptape.drivers.embedding import BaseEmbeddingDriver
from griptape.tokenizers.amazon_bedrock_tokenizer import AmazonBedrockTokenizer
from griptape.utils import import_optional_dependency
from griptape.utils.decorators import lazy_property

if TYPE_CHECKING:
    import boto3
    from mypy_boto3_bedrock import BedrockClient

    from griptape.tokenizers.base_tokenizer import BaseTokenizer


@define
class AmazonBedrockTitanEmbeddingDriver(BaseEmbeddingDriver):
    """Amazon Bedrock Titan Embedding Driver.

    Attributes:
        model: Embedding model name. Defaults to DEFAULT_MODEL.
        tokenizer: Optionally provide custom `BedrockTitanTokenizer`.
        session: Optionally provide custom `boto3.Session`.
        client: Optionally provide custom `bedrock-runtime` client.
    """

    DEFAULT_MODEL = "amazon.titan-embed-text-v1"

    model: str = field(default=DEFAULT_MODEL, kw_only=True, metadata={"serializable": True})
    session: boto3.Session = field(default=Factory(lambda: import_optional_dependency("boto3").Session()), kw_only=True)
    tokenizer: BaseTokenizer = field(
        default=Factory(lambda self: AmazonBedrockTokenizer(model=self.model), takes_self=True),
        kw_only=True,
    )
    _client: Optional[BedrockClient] = field(
        default=None, kw_only=True, alias="client", metadata={"serializable": False}
    )

    @lazy_property()
    def client(self) -> BedrockClient:
        return self.session.client("bedrock-runtime")

    def try_embed_chunk(self, chunk: str) -> list[float]:
        payload = {"inputText": chunk}

        response = self.client.invoke_model(
            body=json.dumps(payload),
            modelId=self.model,
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response.get("body").read())

        return response_body.get("embedding")
