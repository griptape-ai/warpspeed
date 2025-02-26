import os
import subprocess
from pathlib import Path

import pytest

SKIP_FILES = [
    "docs/griptape-framework/tools/official-tools/src/computer_tool_1.py",
    "docs/examples/src/load_query_and_chat_marqo_1.py",
    "docs/griptape-framework/drivers/src/embedding_drivers_2.py",
    "docs/griptape-framework/drivers/src/embedding_drivers_6.py",
    "docs/griptape-framework/drivers/src/embedding_drivers_7.py",
    "docs/griptape-framework/drivers/src/image_generation_drivers_7.py",
    "docs/griptape-framework/drivers/src/image_generation_drivers_8.py",
    "docs/griptape-framework/drivers/src/image_generation_drivers_9.py",
    "docs/griptape-framework/drivers/src/prompt_drivers_4.py",
    "docs/griptape-framework/drivers/src/prompt_drivers_10.py",
    "docs/griptape-framework/drivers/src/prompt_drivers_12.py",
    "docs/griptape-framework/drivers/src/prompt_drivers_14.py",
    "docs/griptape-framework/drivers/src/observability_drivers_1.py",
    "docs/griptape-framework/drivers/src/observability_drivers_2.py",
    "docs/griptape-framework/structures/src/observability_1.py",
    "docs/griptape-framework/structures/src/observability_2.py",
    "docs/griptape-framework/data/src/loaders_9.py",
]


def discover_python_files(directory):
    """Recursively find all .py files in the specified directory."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                python_files.append(
                    pytest.param(path, marks=pytest.mark.skipif(path in SKIP_FILES, reason="Skip file"))
                )
    return python_files


@pytest.mark.parametrize("python_file", discover_python_files("docs"))
def test_python_file_execution(python_file):
    """Run each Python file using Poetry. If it executes successfully, copy the logs to /tmp/logs."""
    result = subprocess.run(
        ["uv", "run", "python", python_file],
        capture_output=True,
        text=True,
        input="Hi\nexit\n",
    )

    assert result.returncode == 0
    assert "ERROR" not in result.stdout

    _save_log_file(python_file, result.stdout)


def _save_log_file(python_file: str, log_output: str) -> None:
    """Given a logs path and log output, save the log output to the logs path."""
    logs_path = Path("/tmp/logs") / Path(python_file).with_suffix(".txt").name
    logs_path.parent.mkdir(parents=True, exist_ok=True)

    if log_output:
        logs_path.write_text(log_output)
