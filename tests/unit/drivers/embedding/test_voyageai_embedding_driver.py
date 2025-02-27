from contextlib import nullcontext
from unittest.mock import Mock

import pytest

from griptape.drivers.embedding.voyageai import VoyageAiEmbeddingDriver


class TestVoyageAiEmbeddingDriver:
    @pytest.fixture(autouse=True)
    def mock_client(self, mocker):
        mock_client = mocker.patch("voyageai.Client")
        mock_client.return_value.embed.return_value = Mock(embeddings=[[0, 1, 0]])

        return mock_client

    def test_init(self):
        assert VoyageAiEmbeddingDriver()

    @pytest.mark.parametrize(
        ("chunk", "expected_output", "expected_error"),
        [
            ("foobar", [0, 1, 0], nullcontext()),
            (
                b"foobar",
                [],
                pytest.raises(ValueError, match="VoyageAiEmbeddingDriver does not support embedding bytes."),
            ),
        ],
    )
    def test_try_embed_chunk(self, chunk, expected_output, expected_error):
        with expected_error:
            assert VoyageAiEmbeddingDriver().try_embed_chunk(chunk) == expected_output
