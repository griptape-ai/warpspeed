from contextlib import nullcontext
from unittest.mock import Mock

import pytest

from griptape.drivers.embedding.openai import AzureOpenAiEmbeddingDriver


class TestAzureOpenAiEmbeddingDriver:
    @pytest.fixture(autouse=True)
    def mock_openai(self, mocker):
        mock_chat_create = mocker.patch("openai.AzureOpenAI").return_value.embeddings.create

        mock_embedding = Mock()
        mock_embedding.embedding = [0, 1, 0]
        mock_response = Mock()
        mock_response.data = [mock_embedding]

        mock_chat_create.return_value = mock_response

        return mock_chat_create

    @pytest.fixture()
    def driver(self):
        return AzureOpenAiEmbeddingDriver(azure_endpoint="foobar", model="gpt-4", azure_deployment="foobar")

    def test_init(self, driver):
        assert driver
        assert AzureOpenAiEmbeddingDriver(azure_endpoint="foobar", model="gpt-4").azure_deployment == "gpt-4"

    @pytest.mark.parametrize(
        ("chunk", "expected_output", "expected_error"),
        [
            ("foobar", [0, 1, 0], nullcontext()),
            (
                b"foobar",
                [],
                pytest.raises(ValueError, match="AzureOpenAiEmbeddingDriver does not support embedding bytes."),
            ),
        ],
    )
    def test_embed_chunk(self, driver, chunk, expected_output, expected_error):
        with expected_error:
            assert driver.try_embed_chunk(chunk) == expected_output
