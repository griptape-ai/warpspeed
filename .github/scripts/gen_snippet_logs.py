from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler

logging.basicConfig(level=logging.DEBUG, handlers=[RichHandler()])
logger = logging.getLogger(__name__)


def _convert_md_file_snippets(md_file_path: Path, updated_md_content: str, changed_snippet_path: Path) -> None:
    """Update the snippet in the markdown file with the logs.

    Converts this:
    ```python
    --8<-- "docs/griptape-framework/data/src/loaders_9.py"`
    ```
    === "Code"
        ```python
        --8<-- "docs/griptape-framework/data/src/loaders_9.py"`
        ```
    === "Logs"
        ```text
        --8<-- "docs/griptape-framework/data/logs/loaders_9.txt"`
        ```

    To this:

    Args:
        md_file_path (Path): The path to the markdown file.
        updated_md_content (str): The content of the markdown file.
        changed_snippet_path (Path): The path to the snippet that was changed.

    Returns:
        None
    """
    full_code_block = f'```python\n--8<-- "{changed_snippet_path}"\n```'
    logs_path = __snippet_path_to_logs_path(Path(changed_snippet_path))

    updated_block = f"""\
=== "Code"
    ```python
    --8<-- "{changed_snippet_path}"
    ```

=== "Logs"
    ```text
    --8<-- "{logs_path}"
    ```
"""

    updated_md_content = updated_md_content.replace(full_code_block, updated_block)
    md_file_path.write_text(updated_md_content, encoding="utf-8")
    logger.debug("Updated snippet in %s", md_file_path)


def _get_logs_for_snippet_path(snippet_path: Path, logs_dir: Optional[Path] = None) -> Optional[str]:
    """Get the logs for the snippet if they exist.

    Args:
        snippet_path (Path): The path to the snippet.
        logs_dir (Optional[Path], optional): The directory to look for logs in. Defaults to /tmp/logs.

    Returns:
        Optional[str]: The logs for the snippet if they exist.
    """
    if logs_dir is None:
        logs_dir = Path("/tmp/logs")
    logs_path = logs_dir / snippet_path.with_suffix(".txt").name

    if logs_path.exists():
        return logs_path.read_text(encoding="utf-8")
    return None


def _save_logs_for_snippet(snippet_path: Path, logs: str) -> None:
    """Save the logs for the snippet to its logs directory.

    Args:
        snippet_path (Path): The path to the snippet.
        logs (str): The logs for the snippet.

    Returns:
        None
    """
    logs_path = __snippet_path_to_logs_path(Path(snippet_path))

    logs_path.parent.mkdir(parents=True, exist_ok=True)

    logs_path.write_text(logs, encoding="utf-8")
    logger.debug("Saved logs for %s to %s", snippet_path, logs_path)


def __snippet_path_to_logs_path(snippet_path: Path) -> Path:
    """Convert a snippet path to a logs path.

    Replaces the `src` directory with `logs` and changes the extension to `.txt`.
    i.e. `src/tasks_11.py` -> `logs/tasks_11.txt`

    Args:
        snippet_path (Path): The path to the snippet.

    Returns:
        Path: The path to the logs for the snippet.

    """
    return Path(str(snippet_path).replace("src", "logs").replace(".py", ".txt"))


if __name__ == "__main__":
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "./docs"
    changed_snippets = sys.argv[2:] if len(sys.argv) > 2 else []

    changed_snippet_paths = [Path(snippet) for snippet in changed_snippets]

    # TODO: Replace, this should only run once to bootstrap the logs
    if not changed_snippet_paths:
        changed_snippet_paths = Path(root_dir).rglob("*.py")

    for changed_snippet_path in changed_snippet_paths:
        logger.debug("Changed snippet %s", changed_snippet_path)
        logs = _get_logs_for_snippet_path(changed_snippet_path)
        if logs:
            logger.debug("Found logs for %s", changed_snippet_path)
            for md_path in Path(root_dir).rglob("*.md"):
                md_content = md_path.read_text(encoding="utf-8")
                if str(changed_snippet_path) in md_content:
                    _convert_md_file_snippets(md_path, md_content, changed_snippet_path)
                    _save_logs_for_snippet(changed_snippet_path, logs)
