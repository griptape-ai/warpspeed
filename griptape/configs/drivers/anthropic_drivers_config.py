from attrs import define

from griptape.configs.drivers import DriversConfig
from griptape.drivers.prompt.anthropic import AnthropicPromptDriver
from griptape.utils.decorators import lazy_property


@define
class AnthropicDriversConfig(DriversConfig):
    @lazy_property()
    def prompt_driver(self) -> AnthropicPromptDriver:
        return AnthropicPromptDriver(model="claude-3-7-sonnet-latest")
