from griptape.drivers import LocalVectorStoreDriver
from griptape.engines import VectorQueryEngine, PromptSummaryEngine, CsvExtractionEngine, JsonExtractionEngine
from griptape.memory import ToolMemory
from griptape.memory.tool.storage import TextToolMemoryStorage, BlobToolMemoryStorage
from tests.mocks.mock_embedding_driver import MockEmbeddingDriver
from tests.mocks.mock_prompt_driver import MockPromptDriver


def text_tool_memory_storage():
    return TextToolMemoryStorage(
        query_engine=VectorQueryEngine(
            vector_store_driver=LocalVectorStoreDriver(
                embedding_driver=MockEmbeddingDriver()
            ),
            prompt_driver=MockPromptDriver()
        ),
        summary_engine=PromptSummaryEngine(
            prompt_driver=MockPromptDriver()
        ),
        csv_extraction_engine=CsvExtractionEngine(
            prompt_driver=MockPromptDriver()
        ),
        json_extraction_engine=JsonExtractionEngine(
            prompt_driver=MockPromptDriver()
        )
    )


def text_tool_memory(name):
    return ToolMemory(
        name=name,
        memory_storage=[
            text_tool_memory_storage(),
            BlobToolMemoryStorage()
        ]
    )
