from contextlib import nullcontext
from unittest.mock import Mock

import pytest

from griptape.drivers.embedding.huggingface_hub_embedding_driver import HuggingFaceHubEmbeddingDriver


class TestHuggingFaceHubEmbeddingDriver:
    @pytest.fixture(autouse=True)
    def mock_client(self, mocker):
        mock_client = mocker.patch("huggingface_hub.InferenceClient").return_value

        mock_response = Mock()
        mock_response.flatten().tolist.return_value = [0, 1, 0]
        mock_client.feature_extraction.return_value = mock_response
        return mock_client

    def test_init(self):
        assert HuggingFaceHubEmbeddingDriver(model="embed-english-v3.0", api_token="foo")

    @pytest.mark.parametrize(
        ("chunk", "expected_output", "expected_error"),
        [
            ("foobar", [0, 1, 0], nullcontext()),
            (
                b"foobar",
                [],
                pytest.raises(ValueError, match="HuggingFaceHubEmbeddingDriver does not support embedding bytes."),
            ),
        ],
    )
    def test_try_embed_chunk(self, chunk, expected_output, expected_error):
        with expected_error:
            assert (
                HuggingFaceHubEmbeddingDriver(model="embed-english-v3.0", api_token="bar").try_embed_chunk(chunk)
                == expected_output
            )
