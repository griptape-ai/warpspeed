import os

from pydantic import BaseModel

from griptape.drivers.prompt.perplexity_sonar import PerplexitySonarPromptDriver
from griptape.rules import Rule
from griptape.structures import Agent
from griptape.tasks import PromptTask


class Output(BaseModel):
    final_answer: str


agent = Agent(
    tasks=[
        PromptTask(
            prompt_driver=PerplexitySonarPromptDriver(
                model="sonar-pro", api_key=os.environ["PERPLEXITY_SONAR_API_KEY"]
            ),
            output_schema=Output,
            rules=[
                Rule("Be precise and concise"),
            ],
        )
    ],
)

agent.run("How many stars are there in our galaxy?")
