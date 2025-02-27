from unittest.mock import Mock

import pytest

from griptape.tokenizers import AnthropicTokenizer


class TestAnthropicTokenizer:
    @pytest.fixture()
    def tokenizer(self, request):
        return AnthropicTokenizer(model=request.param)

    @pytest.fixture(autouse=True)
    def mock_client(self, mocker):
        mock_client = mocker.patch("anthropic.Anthropic")

        mock_client.return_value = Mock(
            beta=Mock(messages=Mock(count_tokens=Mock(return_value=Mock(input_tokens=5, output_tokens=10))))
        )

        return mock_client

    @pytest.mark.parametrize(
        ("tokenizer", "expected"),
        [("claude-2.1", 5), ("claude-2.0", 5), ("claude-3-opus", 5), ("claude-3-sonnet", 5), ("claude-3-haiku", 5)],
        indirect=["tokenizer"],
    )
    def test_token_count(self, tokenizer, expected):
        assert tokenizer.count_tokens("foo bar huzzah") == expected

    @pytest.mark.parametrize(
        ("tokenizer", "expected"),
        [
            ("claude-2.0", 99995),
            ("claude-2.1", 199995),
            ("claude-3-opus", 199995),
            ("claude-3-sonnet", 199995),
            ("claude-3-haiku", 199995),
        ],
        indirect=["tokenizer"],
    )
    def test_input_tokens_left(self, tokenizer, expected):
        assert tokenizer.count_input_tokens_left("foo bar huzzah") == expected

    @pytest.mark.parametrize(
        ("tokenizer", "expected"),
        [
            ("claude-2.0", 4091),
            ("claude-2.1", 4091),
            ("claude-3-opus", 4091),
            ("claude-3-sonnet", 4091),
            ("claude-3-haiku", 4091),
        ],
        indirect=["tokenizer"],
    )
    def test_output_tokens_left(self, tokenizer, expected):
        assert tokenizer.count_output_tokens_left("foo bar huzzah") == expected
