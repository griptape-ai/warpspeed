from contextlib import nullcontext
from unittest import mock

import pytest

from griptape.drivers.embedding.amazon_bedrock import AmazonBedrockTitanEmbeddingDriver


class TestAmazonBedrockTitanEmbeddingDriver:
    @pytest.fixture(autouse=True)
    def _mock_session(self, mocker):
        fake_embeddings = '{"embedding": [0, 1, 0]}'

        mock_session_class = mocker.patch("boto3.Session")

        mock_session_object = mock.Mock()
        mock_client = mock.Mock()
        mock_response = mock.Mock()

        mock_response.get().read.return_value = fake_embeddings
        mock_client.invoke_model.return_value = mock_response
        mock_session_object.client.return_value = mock_client
        mock_session_class.return_value = mock_session_object

    def test_init(self):
        assert AmazonBedrockTitanEmbeddingDriver()

    @pytest.mark.parametrize(
        ("chunk", "expected_output", "expected_error"),
        [
            ("foobar", [0, 1, 0], nullcontext()),
            (
                b"foobar",
                [],
                pytest.raises(ValueError, match="AmazonBedrockTitanEmbeddingDriver does not support embedding bytes."),
            ),
        ],
    )
    def test_try_embed_chunk(self, chunk, expected_output, expected_error):
        with expected_error:
            assert AmazonBedrockTitanEmbeddingDriver().try_embed_chunk(chunk) == expected_output
