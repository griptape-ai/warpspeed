from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import Factory, define, field

from griptape.configs.drivers import DriversConfig
from griptape.drivers.embedding.amazon_bedrock import AmazonBedrockTitanEmbeddingDriver
from griptape.drivers.image_generation.amazon_bedrock import AmazonBedrockImageGenerationDriver
from griptape.drivers.image_generation_model.bedrock_titan import BedrockTitanImageGenerationModelDriver
from griptape.drivers.prompt.amazon_bedrock import AmazonBedrockPromptDriver
from griptape.drivers.vector.local import LocalVectorStoreDriver
from griptape.utils import import_optional_dependency
from griptape.utils.decorators import lazy_property

if TYPE_CHECKING:
    import boto3


@define
class AmazonBedrockDriversConfig(DriversConfig):
    session: boto3.Session = field(
        default=Factory(lambda: import_optional_dependency("boto3").Session()),
        kw_only=True,
        metadata={"serializable": False},
    )

    @lazy_property()
    def prompt_driver(self) -> AmazonBedrockPromptDriver:
        return AmazonBedrockPromptDriver(session=self.session, model="anthropic.claude-3-7-sonnet-20250219-v1:0")

    @lazy_property()
    def embedding_driver(self) -> AmazonBedrockTitanEmbeddingDriver:
        return AmazonBedrockTitanEmbeddingDriver(session=self.session, model="amazon.titan-embed-text-v2:0")

    @lazy_property()
    def image_generation_driver(self) -> AmazonBedrockImageGenerationDriver:
        return AmazonBedrockImageGenerationDriver(
            session=self.session,
            model="amazon.titan-image-generator-v2:0",
            image_generation_model_driver=BedrockTitanImageGenerationModelDriver(),
        )

    @lazy_property()
    def vector_store_driver(self) -> LocalVectorStoreDriver:
        return LocalVectorStoreDriver(
            embedding_driver=AmazonBedrockTitanEmbeddingDriver(
                session=self.session, model="amazon.titan-embed-text-v2:0"
            )
        )
