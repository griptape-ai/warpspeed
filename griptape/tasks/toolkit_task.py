from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from attrs import define

from griptape.tasks import PromptTask

if TYPE_CHECKING:
    from griptape.artifacts import ErrorArtifact, GenericArtifact, JsonArtifact, TextArtifact


@define
class ToolkitTask(PromptTask):
    def try_run(self) -> TextArtifact | GenericArtifact | JsonArtifact | ErrorArtifact:
        warnings.warn(
            "`ToolkitTask` is deprecated and will be removed in a future release. `PromptTask` is a drop-in replacement.",
            DeprecationWarning,
            stacklevel=2,
        )

        return super().try_run()
