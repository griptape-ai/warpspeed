from abc import ABC, abstractmethod
from typing import Any, Optional

from attrs import define
from PIL.Image import Image


@define
class BaseDiffusionPipelineImageGenerationModelDriver(ABC):
    @abstractmethod
    def prepare_pipeline(self, model: str, device: Optional[str]) -> Any: ...

    @abstractmethod
    def make_image_param(self, image: Optional[Image]) -> Optional[dict[str, Image]]: ...

    @abstractmethod
    def make_additional_params(self, negative_prompts: Optional[list[str]], device: Optional[str]) -> dict: ...

    @abstractmethod
    def get_output_image_dimensions(self) -> Optional[tuple[int, int]]: ...
