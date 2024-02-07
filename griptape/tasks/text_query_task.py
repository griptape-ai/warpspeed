from attr import define, field, Factory
from typing import Optional
from griptape.artifacts import TextArtifact
from griptape.engines import VectorQueryEngine
from griptape.loaders import TextLoader
from griptape.tasks import BaseTextInputTask


@define
class TextQueryTask(BaseTextInputTask):
    query_engine: VectorQueryEngine = field(kw_only=True)
    loader: TextLoader = field(default=Factory(lambda: TextLoader()), kw_only=True)
    namespace: Optional[str] = field(default=None, kw_only=True)
    top_n: Optional[int] = field(default=None, kw_only=True)
    preamble: Optional[str] = field(default=None, kw_only=True)

    def run(self) -> TextArtifact:
        return self.query_engine.query(
            self.input.to_text(),
            namespace=self.namespace,
            rulesets=self.all_rulesets,
            top_n=self.top_n,
            preamble=self.preamble,
        )
