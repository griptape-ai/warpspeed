from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from attrs import define, field

from griptape.mixins.singleton_mixin import SingletonMixin
from griptape.utils.decorators import lazy_property

from .base_config import BaseConfig
from .logging.logging_config import LoggingConfig

if TYPE_CHECKING:
    from .drivers.base_drivers_config import BaseDriversConfig


@define(kw_only=True)
class _DefaultsConfig(BaseConfig, SingletonMixin):
    _logging_config: Optional[LoggingConfig] = field(default=None)
    _drivers_config: Optional[BaseDriversConfig] = field(default=None)

    @lazy_property()
    def logging_config(self) -> LoggingConfig:
        return LoggingConfig()

    @lazy_property()
    def drivers_config(self) -> BaseDriversConfig:
        from griptape.configs.drivers.openai_drivers_config import OpenAiDriversConfig

        return OpenAiDriversConfig()


Defaults = _DefaultsConfig()
