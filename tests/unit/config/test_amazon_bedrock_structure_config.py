from pytest import fixture
from griptape.config import AmazonBedrockStructureConfig
from tests.utils.aws import mock_aws_credentials


class TestAmazonBedrockStructureConfig:
    @fixture(autouse=True)
    def run_before_and_after_tests(self):
        mock_aws_credentials()

    @fixture
    def config(self):
        return AmazonBedrockStructureConfig()

    def test_to_dict(self, config):
        assert config.to_dict() == {
            "global_drivers": {
                "conversation_memory_driver": None,
                "embedding_driver": {
                    "model": "amazon.titan-embed-text-v1",
                    "type": "AmazonBedrockTitanEmbeddingDriver",
                },
                "image_generation_driver": {
                    "image_generation_model_driver": {
                        "cfg_scale": 7,
                        "outpainting_mode": "PRECISE",
                        "quality": "standard",
                        "type": "BedrockTitanImageGenerationModelDriver",
                    },
                    "image_height": 512,
                    "image_width": 512,
                    "model": "amazon.titan-image-generator-v1",
                    "seed": None,
                    "type": "AmazonBedrockImageGenerationDriver",
                },
                "image_query_driver": {
                    "type": "AmazonBedrockImageQueryDriver",
                    "model": "anthropic.claude-3-sonnet-20240229-v1:0",
                    "max_output_tokens": 256,
                    "image_query_model_driver": {"type": "BedrockClaudeImageQueryModelDriver"},
                },
                "prompt_driver": {
                    "max_tokens": None,
                    "model": "anthropic.claude-3-sonnet-20240229-v1:0",
                    "prompt_model_driver": {"type": "BedrockClaudePromptModelDriver", "top_k": 250, "top_p": 0.999},
                    "stream": False,
                    "temperature": 0.1,
                    "type": "AmazonBedrockPromptDriver",
                },
                "type": "StructureGlobalDriversConfig",
                "vector_store_driver": {
                    "embedding_driver": {
                        "model": "amazon.titan-embed-text-v1",
                        "type": "AmazonBedrockTitanEmbeddingDriver",
                    },
                    "type": "LocalVectorStoreDriver",
                },
            },
            "type": "AmazonBedrockStructureConfig",
            "task_memory": {
                "type": "StructureTaskMemoryConfig",
                "query_engine": {
                    "type": "StructureTaskMemoryQueryEngineConfig",
                    "prompt_driver": {
                        "type": "AmazonBedrockPromptDriver",
                        "temperature": 0.1,
                        "max_tokens": None,
                        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
                        "prompt_model_driver": {"type": "BedrockClaudePromptModelDriver", "top_k": 250, "top_p": 0.999},
                        "stream": False,
                    },
                    "vector_store_driver": {
                        "type": "LocalVectorStoreDriver",
                        "embedding_driver": {
                            "type": "AmazonBedrockTitanEmbeddingDriver",
                            "model": "amazon.titan-embed-text-v1",
                        },
                    },
                },
                "extraction_engine": {
                    "type": "StructureTaskMemoryExtractionEngineConfig",
                    "csv": {
                        "type": "StructureTaskMemoryExtractionEngineCsvConfig",
                        "prompt_driver": {
                            "type": "AmazonBedrockPromptDriver",
                            "temperature": 0.1,
                            "max_tokens": None,
                            "model": "anthropic.claude-3-sonnet-20240229-v1:0",
                            "prompt_model_driver": {
                                "type": "BedrockClaudePromptModelDriver",
                                "top_k": 250,
                                "top_p": 0.999,
                            },
                            "stream": False,
                        },
                    },
                    "json": {
                        "type": "StructureTaskMemoryExtractionEngineJsonConfig",
                        "prompt_driver": {
                            "type": "AmazonBedrockPromptDriver",
                            "temperature": 0.1,
                            "max_tokens": None,
                            "model": "anthropic.claude-3-sonnet-20240229-v1:0",
                            "prompt_model_driver": {
                                "type": "BedrockClaudePromptModelDriver",
                                "top_k": 250,
                                "top_p": 0.999,
                            },
                            "stream": False,
                        },
                    },
                },
                "summary_engine": {
                    "type": "StructureTaskMemorySummaryEngineConfig",
                    "prompt_driver": {
                        "type": "AmazonBedrockPromptDriver",
                        "temperature": 0.1,
                        "max_tokens": None,
                        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
                        "prompt_model_driver": {"type": "BedrockClaudePromptModelDriver", "top_k": 250, "top_p": 0.999},
                        "stream": False,
                    },
                },
            },
        }

    def test_from_dict(self, config):
        assert AmazonBedrockStructureConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()
