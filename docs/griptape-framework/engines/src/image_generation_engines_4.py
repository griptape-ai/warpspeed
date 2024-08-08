from griptape.drivers import AmazonBedrockImageGenerationDriver, BedrockStableDiffusionImageGenerationModelDriver
from griptape.engines import InpaintingImageGenerationEngine
from griptape.loaders import ImageLoader

# Create a driver configured to use Stable Diffusion via Bedrock.
driver = AmazonBedrockImageGenerationDriver(
    image_generation_model_driver=BedrockStableDiffusionImageGenerationModelDriver(),
    model="stability.stable-diffusion-xl-v1",
)

# Create an engine configured to use the driver.
engine = InpaintingImageGenerationEngine(
    image_generation_driver=driver,
)

with open("tests/resources/mountain.png", "rb") as f:
    image_artifact = ImageLoader().load(f.read())

with open("tests/resources/mountain-mask.png", "rb") as f:
    mask_artifact = ImageLoader().load(f.read())

engine.run(
    prompts=["A photo of a castle built into the side of a mountain"],
    image=image_artifact,
    mask=mask_artifact,
)
