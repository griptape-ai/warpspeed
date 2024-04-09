from __future__ import annotations
import logging
import os
from pathlib import Path
from attr import define, field, Factory
from griptape.artifacts import ErrorArtifact, InfoArtifact, ListArtifact, BaseArtifact, TextArtifact
from griptape.artifacts.blob_artifact import BlobArtifact
from griptape.tools import BaseTool
from griptape.utils.decorators import activity
from griptape.loaders import BaseLoader, PdfLoader, CsvLoader, TextLoader, ImageLoader
from schema import Schema, Literal
from typing import Optional, Any


@define
class FileManager(BaseTool):
    """
    FileManager is a tool that can be used to load and save files.

    Attributes:
        workdir: The absolute directory to load files from and save files to.
        loaders: Dictionary of file extensions and matching loaders to use when loading files in load_files_from_disk.
        default_loader: The loader to use when loading files in load_files_from_disk without any matching loader in `loaders`.
        save_file_encoding: The encoding to use when saving files to disk.
    """

    workdir: str = field(default=Factory(lambda: os.getcwd()), kw_only=True)
    default_loader: Optional[BaseLoader] = field(default=None, kw_only=True)
    loaders: dict[str, BaseLoader] = field(
        default=Factory(
            lambda: {
                "pdf": PdfLoader(),
                "csv": CsvLoader(),
                "txt": TextLoader(),
                "html": TextLoader(),
                "json": TextLoader(),
                "yaml": TextLoader(),
                "xml": TextLoader(),
                "png": ImageLoader(),
                "jpg": ImageLoader(),
                "jpeg": ImageLoader(),
                "webp": ImageLoader(),
                "gif": ImageLoader(),
                "bmp": ImageLoader(),
                "tiff": ImageLoader(),
            }
        ),
        kw_only=True,
    )
    save_file_encoding: Optional[str] = field(default=None, kw_only=True)

    @workdir.validator  # pyright: ignore
    def validate_workdir(self, _, workdir: str) -> None:
        if not Path(workdir).is_absolute():
            raise ValueError("workdir has to be absolute absolute")

    @activity(
        config={
            "description": "Can be used to list files on disk",
            "schema": Schema(
                {Literal("path", description="Relative path in the POSIX format. For example, 'foo/bar'"): str}
            ),
        }
    )
    def list_files_from_disk(self, params: dict) -> TextArtifact | ErrorArtifact:
        path = params["values"]["path"].lstrip("/")
        full_path = Path(os.path.join(self.workdir, path))

        if os.path.exists(full_path):
            entries = os.listdir(full_path)

            return TextArtifact("\n".join([e for e in entries]))
        else:
            return ErrorArtifact("Path not found")

    @activity(
        config={
            "description": "Can be used to load files from disk",
            "schema": Schema(
                {
                    Literal(
                        "paths",
                        description="Relative paths to files to be loaded in the POSIX format. For example, ['foo/bar/file.txt']",
                    ): list
                }
            ),
        }
    )
    def load_files_from_disk(self, params: dict) -> ListArtifact | ErrorArtifact:
        artifacts = []

        for path in params["values"]["paths"]:
            path = path.lstrip("/")
            full_path = Path(os.path.join(self.workdir, path))
            extension = path.split(".")[-1]
            loader = self.loaders.get(extension) or self.default_loader
            with open(full_path, "rb") as file:
                content = file.read()

            if loader:
                result = loader.load(content)
            else:
                result = BlobArtifact(content)

            if isinstance(result, list):
                artifacts.extend(result)
            elif isinstance(result, BaseArtifact):
                artifacts.append(result)
            else:
                logging.warning(f"Unknown loader return type for file {path}")

        return ListArtifact(artifacts)

    @activity(
        config={
            "description": "Can be used to save memory artifacts to disk",
            "schema": Schema(
                {
                    Literal(
                        "dir_name",
                        description="Relative destination path name on disk in the POSIX format. For example, 'foo/bar'",
                    ): str,
                    Literal("file_name", description="Destination file name. For example, 'baz.txt'"): str,
                    "memory_name": str,
                    "artifact_namespace": str,
                }
            ),
        }
    )
    def save_memory_artifacts_to_disk(self, params: dict) -> ErrorArtifact | InfoArtifact:
        memory = self.find_input_memory(params["values"]["memory_name"])
        artifact_namespace = params["values"]["artifact_namespace"]
        dir_name = params["values"]["dir_name"]
        file_name = params["values"]["file_name"]

        if memory:
            list_artifact = memory.load_artifacts(artifact_namespace)

            if len(list_artifact) == 0:
                return ErrorArtifact("no artifacts found")
            elif len(list_artifact) == 1:
                try:
                    self._save_to_disk(os.path.join(self.workdir, dir_name, file_name), list_artifact.value[0].value)

                    return InfoArtifact("saved successfully")
                except Exception as e:
                    return ErrorArtifact(f"error writing file to disk: {e}")
            else:
                try:
                    for a in list_artifact.value:
                        self._save_to_disk(os.path.join(self.workdir, dir_name, f"{a.name}-{file_name}"), a.to_text())

                    return InfoArtifact("saved successfully")
                except Exception as e:
                    return ErrorArtifact(f"error writing file to disk: {e}")
        else:
            return ErrorArtifact("memory not found")

    @activity(
        config={
            "description": "Can be used to save content to a file",
            "schema": Schema(
                {
                    Literal(
                        "path",
                        description="Destination file path on disk in the POSIX format. For example, 'foo/bar/baz.txt'",
                    ): str,
                    "content": str,
                }
            ),
        }
    )
    def save_content_to_file(self, params: dict) -> ErrorArtifact | InfoArtifact:
        content = params["values"]["content"]
        new_path = params["values"]["path"].lstrip("/")
        full_path = os.path.join(self.workdir, new_path)

        try:
            self._save_to_disk(full_path, content)

            return InfoArtifact("saved successfully")
        except Exception as e:
            return ErrorArtifact(f"error writing file to disk: {e}")

    def _save_to_disk(self, path: str, value: Any) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as file:
            if isinstance(value, str):
                if self.save_file_encoding:
                    file.write(value.encode(self.save_file_encoding))
                else:
                    file.write(value.encode())
            else:
                file.write(value)
