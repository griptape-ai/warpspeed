from contextlib import nullcontext
from unittest.mock import Mock

import pytest

from griptape.drivers.embedding.cohere import CohereEmbeddingDriver


class TestCohereEmbeddingDriver:
    @pytest.fixture(autouse=True)
    def mock_client(self, mocker):
        mock_client = mocker.patch("cohere.Client").return_value

        mock_client.embed.return_value = Mock(embeddings=[[0, 1, 0]])

        return mock_client

    def test_init(self):
        assert CohereEmbeddingDriver(model="embed-english-v3.0", api_key="bar", input_type="search_document")

    @pytest.mark.parametrize(
        ("chunk", "expected_output", "expected_error"),
        [
            ("foobar", [0, 1, 0], nullcontext()),
            (
                b"foobar",
                [],
                pytest.raises(ValueError, match="CohereEmbeddingDriver does not support embedding bytes."),
            ),
        ],
    )
    def test_try_embed_chunk(self, chunk, expected_output, expected_error):
        with expected_error:
            assert (
                CohereEmbeddingDriver(
                    model="embed-english-v3.0", api_key="bar", input_type="search_document"
                ).try_embed_chunk(chunk)
                == expected_output
            )
