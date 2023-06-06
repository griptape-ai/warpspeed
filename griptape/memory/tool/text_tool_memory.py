from typing import Union, Optional
from attr import define, field, Factory
from schema import Schema, Literal
from griptape.artifacts import BaseArtifact, TextArtifact, ErrorArtifact, InfoArtifact, ListArtifact
from griptape.core.decorators import activity
from griptape.engines import VectorQueryEngine
from griptape.memory.tool import BaseToolMemory


@define
class TextToolMemory(BaseToolMemory):
    allowlist: Optional[list[str]] = field(
        default=Factory(lambda: ["save"]),
        kw_only=True
    )
    query_engine: VectorQueryEngine = field(
        kw_only=True,
        default=Factory(lambda: VectorQueryEngine())
    )
    top_n: int = field(default=5, kw_only=True)

    def process_output(self, tool_activity: callable, artifact: BaseArtifact) -> Union[InfoArtifact, ErrorArtifact]:
        from griptape.utils import J2

        namespace = artifact.id if artifact else None

        if isinstance(artifact, TextArtifact):
            self.query_engine.vector_driver.upsert_text_artifact(artifact, namespace=namespace)
        elif isinstance(artifact, ListArtifact):
            [
                self.query_engine.vector_driver.upsert_text_artifact(
                    a, namespace=namespace
                ) for a in artifact.value if isinstance(a, TextArtifact)
            ]

        if namespace:
            output = J2("memory/tool/text.j2").render(
                memory_name=self.name,
                tool_name=tool_activity.__self__.name,
                activity_name=tool_activity.name,
                artifact_namespace=namespace
            )

            return InfoArtifact(output)
        else:
            return artifact

    def load_namespace_artifacts(self, namespace: str) -> list[BaseArtifact]:
        return [
            BaseArtifact.from_json(e.meta["artifact"])
            for e in self.query_engine.vector_driver.load_vectors(namespace)
        ]

    @activity(config={
        "description": "Can be used to save artifact values",
        "schema": Schema({
            "artifact_value": str
        }),
    })
    def save(self, params: dict) -> Union[InfoArtifact, ErrorArtifact]:
        artifact = TextArtifact(params["values"]["artifact_value"])
        namespace = self.query_engine.vector_driver.upsert_text_artifact(artifact, namespace=artifact.id)

        return InfoArtifact(f"Value was successfully stored with the following namespace: {namespace}")

    @activity(config={
        "description": "Can be used to load artifact values",
        "schema": Schema({
            "artifact_namespace": str
        })
    })
    def load(self, params: dict) -> ListArtifact:
        namespace = params["values"]["artifact_namespace"]

        return ListArtifact.from_list(
            [
                BaseArtifact.from_json(e.meta["artifact"])
                for e in self.query_engine.vector_driver.load_vectors(namespace)
            ]
        )

    @activity(config={
        "description": "Can be used to search artifact values",
        "schema": Schema({
            Literal(
                "query",
                description="Search query"
            ): str,
            "artifact_namespace": str
        })
    })
    def query(self, params: dict) -> TextArtifact:
        query = params["values"]["query"]
        namespace = params["values"]["namespace"]

        return self.query_engine.query(query, namespace=namespace)
