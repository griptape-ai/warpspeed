from griptape.drivers.image_generation.amazon_bedrock import AmazonBedrockImageGenerationDriver
from griptape.drivers.image_generation_model.bedrock_titan import BedrockTitanImageGenerationModelDriver
from griptape.structures import Agent
from griptape.tools import PromptImageGenerationTool

model_driver = BedrockTitanImageGenerationModelDriver()

driver = AmazonBedrockImageGenerationDriver(
    image_generation_model_driver=model_driver,
    model="amazon.titan-image-generator-v2:0",
)


agent = Agent(
    tools=[
        PromptImageGenerationTool(image_generation_driver=driver),
    ]
)

agent.run("Generate a watercolor painting of a dog riding a skateboard")
