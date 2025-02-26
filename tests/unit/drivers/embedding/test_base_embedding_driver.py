from unittest.mock import patch

import pytest

from griptape.artifacts import TextArtifact
from griptape.artifacts.image_artifact import ImageArtifact
from griptape.artifacts.list_artifact import ListArtifact
from tests.mocks.mock_embedding_driver import MockEmbeddingDriver


class TestBaseEmbeddingDriver:
    @pytest.fixture()
    def driver(self):
        return MockEmbeddingDriver()

    def test_embed_text_artifact(self, driver):
        embedding = driver.embed_text_artifact(TextArtifact("foobar"))

        assert embedding == [0, 1]

    def test_embed_string(self, driver):
        embedding = driver.embed_string("foobar")

        assert embedding == [0, 1]

    @pytest.mark.parametrize(
        ("value", "expected_output"),
        [
            ("foobar", [0, 1]),
            ("foobar" * 5000, [0, 1]),
            (b"foobar", [0, 1]),
            (TextArtifact("foobar"), [0, 1]),
            (ImageArtifact(b"foobar", format="png", width=100, height=100), [0, 1]),
            (
                ListArtifact([TextArtifact("foobar"), ImageArtifact(b"foobar", format="png", width=100, height=100)]),
                [0, 1],
            ),
        ],
    )
    def test_embed(self, driver, value, expected_output):
        embedding = driver.embed(value)

        assert embedding == expected_output

    def test_no_tokenizer(self, driver):
        driver.tokenizer = None

        embedding = driver.embed_string("foobar")

        assert embedding == [0, 1]

    @patch.object(MockEmbeddingDriver, "try_embed_chunk")
    def test_embed_string_throws_when_retries_exhausted(self, try_embed_chunk, driver):
        try_embed_chunk.side_effect = Exception("nope")

        with pytest.raises(Exception) as e:
            driver.embed_string("foobar")

        assert e.value.args[0] == "nope"
