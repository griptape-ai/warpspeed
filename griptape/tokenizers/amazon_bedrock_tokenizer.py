from __future__ import annotations

from attrs import define, field

from griptape.tokenizers.base_tokenizer import BaseTokenizer


@define()
class AmazonBedrockTokenizer(BaseTokenizer):
    MODEL_PREFIXES_TO_MAX_INPUT_TOKENS = {
        "anthropic.claude-3": 200000,
        "anthropic.claude-v2:1": 200000,
        "anthropic.claude": 100000,
        "cohere.command-r": 128000,
        "cohere.embed": 512,
        "cohere.command": 4000,
        "cohere": 1024,
        "ai21": 8192,
        "meta.llama3-8b-instruct": 8000,
        "meta.llama3-70b-instruct": 8000,
        "meta.llama3-2-1b-instruct": 131000,
        "meta.llama3-2-3b-instruct": 131000,
        "meta.llama3": 128000,
        "mistral.large-2407": 128000,
        "mistral.mistral": 32000,
        "mistral.mixtral": 32000,
        "amazon.nova-micro-v1": 128000,
        "amazon.nova": 300000,
        "amazon.titan-text-express-v1": 8000,
        "amazon.titan-text-lite-v1": 4000,
        "amazon.titan-text-premier-v1": 32000,
    }
    MODEL_PREFIXES_TO_MAX_OUTPUT_TOKENS = {
        "anthropic.claude-3-7": 8192,
        "anthropic.claude-3-5": 8192,
        "anthropic.claude": 4096,
        "cohere": 4096,
        "ai21.j2": 8191,
        "meta": 2048,
        "amazon.titan-text-lite": 4096,
        "amazon.titan-text-express": 8192,
        "amazon.titan-text-premier": 3072,
        "amazon.nova": 5000,
        "mistral.mistral": 8192,
        "mistral.mixtral": 4096,
    }

    model: str = field(kw_only=True)
    characters_per_token: int = field(default=4, kw_only=True)

    def count_tokens(self, text: str) -> int:
        return (len(text) + self.characters_per_token - 1) // self.characters_per_token
