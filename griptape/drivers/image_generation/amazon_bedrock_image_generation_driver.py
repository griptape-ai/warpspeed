from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Optional

from attrs import define, field, Factory

from griptape.artifacts import ImageArtifact
from griptape.drivers import BaseMultiModelImageGenerationDriver
from griptape.utils import import_optional_dependency

if TYPE_CHECKING:
    import boto3


@define
class AmazonBedrockImageGenerationDriver(BaseMultiModelImageGenerationDriver):
    """Driver for image generation models provided by Amazon Bedrock.

    Attributes:
        model: Bedrock model ID.
        session: boto3 session.
        bedrock_client: Bedrock runtime client.
        image_width: Width of output images. Defaults to 512 and must be a multiple of 64.
        image_height: Height of output images. Defaults to 512 and must be a multiple of 64.
        seed: Optionally provide a consistent seed to generation requests, increasing consistency in output.
    """

    session: boto3.Session = field(default=Factory(lambda: import_optional_dependency("boto3").Session()), kw_only=True)
    bedrock_client: Any = field(
        default=Factory(lambda self: self.session.client(service_name="bedrock-runtime"), takes_self=True)
    )
    image_width: int = field(default=512, kw_only=True, metadata={"serializable": True})
    image_height: int = field(default=512, kw_only=True, metadata={"serializable": True})
    seed: Optional[int] = field(default=None, kw_only=True, metadata={"serializable": True})

    def try_text_to_image(self, prompts: list[str], negative_prompts: Optional[list[str]] = None) -> ImageArtifact:
        request = self.image_generation_model_driver.text_to_image_request_parameters(
            prompts, self.image_width, self.image_height, negative_prompts=negative_prompts, seed=self.seed
        )

        image_bytes = self._make_request(request)

        return ImageArtifact(
            prompt=", ".join(prompts),
            value=image_bytes,
            format="png",
            width=self.image_width,
            height=self.image_height,
            model=self.model,
        )

    def try_image_variation(
        self, prompts: list[str], image: ImageArtifact, negative_prompts: Optional[list[str]] = None
    ) -> ImageArtifact:
        request = self.image_generation_model_driver.image_variation_request_parameters(
            prompts, image=image, negative_prompts=negative_prompts, seed=self.seed
        )

        image_bytes = self._make_request(request)

        return ImageArtifact(
            prompt=", ".join(prompts),
            value=image_bytes,
            format="png",
            width=image.width,
            height=image.height,
            model=self.model,
        )

    def try_image_inpainting(
        self,
        prompts: list[str],
        image: ImageArtifact,
        mask: ImageArtifact,
        negative_prompts: Optional[list[str]] = None,
    ) -> ImageArtifact:
        request = self.image_generation_model_driver.image_inpainting_request_parameters(
            prompts, image=image, mask=mask, negative_prompts=negative_prompts, seed=self.seed
        )

        image_bytes = self._make_request(request)

        return ImageArtifact(
            prompt=", ".join(prompts),
            value=image_bytes,
            format="png",
            width=image.width,
            height=image.height,
            model=self.model,
        )

    def try_image_outpainting(
        self,
        prompts: list[str],
        image: ImageArtifact,
        mask: ImageArtifact,
        negative_prompts: Optional[list[str]] = None,
    ) -> ImageArtifact:
        request = self.image_generation_model_driver.image_outpainting_request_parameters(
            prompts, image=image, mask=mask, negative_prompts=negative_prompts, seed=self.seed
        )

        image_bytes = self._make_request(request)

        return ImageArtifact(
            prompt=", ".join(prompts),
            value=image_bytes,
            format="png",
            width=image.width,
            height=image.height,
            model=self.model,
        )

    def _make_request(self, request: dict) -> bytes:
        response = self.bedrock_client.invoke_model(
            body=json.dumps(request), modelId=self.model, accept="application/json", contentType="application/json"
        )

        response_body = json.loads(response.get("body").read())

        try:
            image_bytes = self.image_generation_model_driver.get_generated_image(response_body)
        except Exception as e:
            raise ValueError(f"Inpainting generation failed: {e}") from e

        return image_bytes
