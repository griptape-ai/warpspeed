from unittest import mock

import pytest

from griptape.drivers.embedding.amazon_sagemaker_jumpstart import AmazonSageMakerJumpstartEmbeddingDriver
from griptape.tokenizers.openai_tokenizer import OpenAiTokenizer


class TestAmazonSageMakerJumpstartEmbeddingDriver:
    @pytest.fixture(autouse=True)
    def mock_client(self, mocker):
        mock_session_class = mocker.patch("boto3.Session")
        mock_session_object = mock.Mock()
        mock_client = mock.Mock()
        mock_response = mock.Mock()

        mock_client.invoke_endpoint.return_value = mock_response
        mock_session_object.client.return_value = mock_client
        mock_session_class.return_value = mock_session_object

        return mock_response

    def test_init(self):
        assert AmazonSageMakerJumpstartEmbeddingDriver(
            endpoint="test-endpoint",
            model="test-endpoint",
            tokenizer=OpenAiTokenizer(model=OpenAiTokenizer.DEFAULT_OPENAI_GPT_3_CHAT_MODEL),
        )

    def test_try_embed_chunk(self, mock_client):
        mock_client.get().read.return_value = b'{"embedding": [[0, 1, 0]]}'
        assert AmazonSageMakerJumpstartEmbeddingDriver(
            endpoint="test-endpoint",
            model="test-model",
            tokenizer=OpenAiTokenizer(model=OpenAiTokenizer.DEFAULT_OPENAI_GPT_3_CHAT_MODEL),
        ).try_embed_chunk("foobar") == [0, 1, 0]

        mock_client.get().read.return_value = b'{"embedding": [0, 2, 0]}'
        assert AmazonSageMakerJumpstartEmbeddingDriver(
            endpoint="test-endpoint",
            model="test-model",
            tokenizer=OpenAiTokenizer(model=OpenAiTokenizer.DEFAULT_OPENAI_GPT_3_CHAT_MODEL),
        ).try_embed_chunk("foobar") == [0, 2, 0]

        mock_client.get().read.return_value = b'{"embedding": []}'
        with pytest.raises(ValueError, match="model response is empty"):
            assert AmazonSageMakerJumpstartEmbeddingDriver(
                endpoint="test-endpoint",
                model="test-model",
                tokenizer=OpenAiTokenizer(model=OpenAiTokenizer.DEFAULT_OPENAI_GPT_3_CHAT_MODEL),
            ).try_embed_chunk("foobar") == [0, 2, 0]

        mock_client.get().read.return_value = b"{}"
        with pytest.raises(ValueError, match="invalid response from model"):
            assert AmazonSageMakerJumpstartEmbeddingDriver(
                endpoint="test-endpoint",
                model="test-model",
                tokenizer=OpenAiTokenizer(model=OpenAiTokenizer.DEFAULT_OPENAI_GPT_3_CHAT_MODEL),
            ).try_embed_chunk("foobar") == [0, 2, 0]

        with pytest.raises(
            ValueError, match="AmazonSageMakerJumpstartEmbeddingDriver does not support embedding bytes."
        ):
            assert (
                AmazonSageMakerJumpstartEmbeddingDriver(
                    endpoint="test-endpoint",
                    model="test-model",
                    tokenizer=OpenAiTokenizer(model=OpenAiTokenizer.DEFAULT_OPENAI_GPT_3_CHAT_MODEL),
                ).try_embed_chunk(b"foobar")
                == []
            )
