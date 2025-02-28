from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable, Optional, Union

from attrs import Attribute, define, evolve, field

from griptape.artifacts.text_artifact import TextArtifact
from griptape.common import observable
from griptape.configs import Defaults
from griptape.structures import Structure
from griptape.tasks import PromptTask
from griptape.utils.decorators import lazy_property

if TYPE_CHECKING:
    from pydantic import BaseModel
    from schema import Schema

    from griptape.artifacts import BaseArtifact
    from griptape.drivers.prompt import BasePromptDriver
    from griptape.tasks import BaseTask
    from griptape.tools import BaseTool


@define
class Agent(Structure):
    input: Union[str, list, tuple, BaseArtifact, Callable[[BaseTask], BaseArtifact]] = field(
        default=lambda task: task.full_context["args"][0] if task.full_context["args"] else TextArtifact(value=""),
    )
    _stream: Optional[bool] = field(default=None, kw_only=True, alias="stream")
    _prompt_driver: Optional[BasePromptDriver] = field(default=None, kw_only=True, alias="prompt_driver")
    output_schema: Optional[Union[Schema, type[BaseModel]]] = field(default=None, kw_only=True)
    tools: list[BaseTool] = field(factory=list, kw_only=True)
    max_meta_memory_entries: Optional[int] = field(default=20, kw_only=True)
    fail_fast: bool = field(default=False, kw_only=True)
    _tasks: list[Union[BaseTask, list[BaseTask]]] = field(
        factory=list, kw_only=True, alias="tasks", metadata={"serializable": True}
    )

    @lazy_property()
    def prompt_driver(self) -> BasePromptDriver:
        return evolve(Defaults.drivers_config.prompt_driver, stream=self.stream)

    @lazy_property()
    def stream(self) -> bool:
        return Defaults.drivers_config.prompt_driver.stream

    @fail_fast.validator  # pyright: ignore[reportAttributeAccessIssue]
    def validate_fail_fast(self, _: Attribute, fail_fast: bool) -> None:  # noqa: FBT001
        if fail_fast:
            raise ValueError("Agents cannot fail fast, as they can only have 1 task.")

    @_prompt_driver.validator  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
    def validate_prompt_driver(self, _: Attribute, prompt_driver: Optional[BasePromptDriver]) -> None:  # noqa: FBT001
        if prompt_driver is not None and self.stream is not None:
            warnings.warn(
                "`Agent.prompt_driver` is set, but `Agent.stream` was provided. `Agent.stream` will be ignored. This will be an error in the future.",
                UserWarning,
                stacklevel=2,
            )

    @_tasks.validator  # pyright: ignore[reportAttributeAccessIssue]
    def validate_tasks(self, _: Attribute, tasks: list) -> None:
        if tasks and self.prompt_driver is not None:
            warnings.warn(
                "`Agent.tasks` is set, but `Agent.prompt_driver` was provided. `Agent.prompt_driver` will be ignored. This will be an error in the future.",
                UserWarning,
                stacklevel=2,
            )

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()

        if len(self.tasks) == 0:
            self._init_task()

    @property
    def task(self) -> BaseTask:
        return self.tasks[0]

    def add_task(self, task: BaseTask) -> BaseTask:
        self._tasks.clear()

        task.preprocess(self)

        self._tasks.append(task)

        return task

    def add_tasks(self, *tasks: BaseTask | list[BaseTask]) -> list[BaseTask]:
        if len(tasks) > 1:
            raise ValueError("Agents can only have one task.")
        return super().add_tasks(*tasks)

    @observable
    def try_run(self, *args) -> Agent:
        self.task.run()

        return self

    def _init_task(self) -> None:
        task = PromptTask(
            self.input,
            prompt_driver=self.prompt_driver,
            tools=self.tools,
            output_schema=self.output_schema,
            max_meta_memory_entries=self.max_meta_memory_entries,
        )

        self.add_task(task)
