from contextlib import nullcontext

import pytest

from griptape.drivers.embedding.ollama import OllamaEmbeddingDriver


class TestOllamaEmbeddingDriver:
    @pytest.fixture(autouse=True)
    def mock_client(self, mocker):
        mock_client = mocker.patch("ollama.Client")

        mock_client.return_value.embeddings.return_value = {"embedding": [0, 1, 0]}

        return mock_client

    def test_init(self):
        assert OllamaEmbeddingDriver(model="foo")

    @pytest.mark.parametrize(
        ("chunk", "expected_output", "expected_error"),
        [
            ("foobar", [0, 1, 0], nullcontext()),
            (
                b"foobar",
                [],
                pytest.raises(ValueError, match="OllamaEmbeddingDriver does not support embedding bytes."),
            ),
        ],
    )
    def test_try_embed_chunk(self, chunk, expected_output, expected_error):
        with expected_error:
            assert OllamaEmbeddingDriver(model="foo").try_embed_chunk(chunk) == expected_output
