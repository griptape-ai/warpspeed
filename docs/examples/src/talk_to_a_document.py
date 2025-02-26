import base64
import logging

import requests

from griptape.artifacts import GenericArtifact, TextArtifact
from griptape.configs import Defaults
from griptape.configs.logging import TruncateLoggingFilter
from griptape.drivers.prompt.anthropic import AnthropicPromptDriver
from griptape.structures import Agent
from griptape.tasks.base_task import BaseTask
from griptape.tasks.prompt_task import PromptTask

# Truncate logs to 100 characters to avoid printing the entire document
truncate_log_filter = TruncateLoggingFilter(max_log_length=100)


def on_before_run(_: BaseTask) -> None:
    logging.getLogger(Defaults.logging_config.logger_name).addFilter(truncate_log_filter)


def on_after_run(_: BaseTask) -> None:
    logging.getLogger(Defaults.logging_config.logger_name).removeFilter(truncate_log_filter)


doc_bytes = requests.get("https://arxiv.org/pdf/1706.03762.pdf").content

agent = Agent(
    tasks=[
        PromptTask(
            prompt_driver=AnthropicPromptDriver(model="claude-3-7-sonnet-latest", max_attempts=0),
            on_before_run=on_before_run,
            on_after_run=on_after_run,
            input=[
                GenericArtifact(
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": base64.b64encode(doc_bytes).decode("utf-8"),
                        },
                    }
                ),
                TextArtifact("{{ args[0] }}"),
            ],
        )
    ],
)

agent.run("What is the title and who are the authors of this paper?")
