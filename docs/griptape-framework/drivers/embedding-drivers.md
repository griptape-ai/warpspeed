---
search:
  boost: 2
---

## Overview

Embeddings in Griptape are multidimensional representations of text data. Embeddings carry semantic information, which makes them useful for extracting relevant chunks from large bodies of text for search and querying.

Griptape provides a way to build Embedding Drivers that are reused in downstream framework components. Every Embedding Driver has two basic methods that can be used to generate embeddings:

- [embed_text_artifact()](../../reference/griptape/drivers/embedding/base_embedding_driver.md#griptape.drivers.embedding.base_embedding_driver.BaseEmbeddingDriver.embed_text_artifact) for [TextArtifact](../../reference/griptape/artifacts/text_artifact.md)s.
- [embed_string()](../../reference/griptape/drivers/embedding/base_embedding_driver.md#griptape.drivers.embedding.base_embedding_driver.BaseEmbeddingDriver.embed_string) for any string.

You can optionally provide a [Tokenizer](../misc/tokenizers.md) via the [tokenizer](../../reference/griptape/drivers/embedding/base_embedding_driver.md#griptape.drivers.embedding.base_embedding_driver.BaseEmbeddingDriver.tokenizer) field to have the Driver automatically chunk the input text to fit into the token limit.

## Embedding Drivers

### OpenAI

The [OpenAiEmbeddingDriver](../../reference/griptape/drivers/embedding/openai_embedding_driver.md) uses the [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings).

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/embedding_drivers_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/embedding_drivers_1.txt"
    ```


```
[0.0017853748286142945, 0.006118456833064556, -0.005811543669551611]
```

### OpenAI Compatible

Many services such as [LMStudio](https://lmstudio.ai/) and [OhMyGPT](https://www.ohmygpt.com/) provide OpenAI-compatible APIs. You can use the [OpenAiEmbeddingDriver](../../reference/griptape/drivers/embedding/openai_embedding_driver.md) to interact with these services.
Simply set the `base_url` to the service's API endpoint and the `model` to the model name. If the service requires an API key, you can set it in the `api_key` field.

```python
--8<-- "docs/griptape-framework/drivers/src/embedding_drivers_2.py"
```

!!! tip

    Make sure to include `v1` at the end of the `base_url` to match the OpenAI API endpoint.

### Azure OpenAI

The [AzureOpenAiEmbeddingDriver](../../reference/griptape/drivers/embedding/azure_openai_embedding_driver.md) uses the same parameters as [OpenAiEmbeddingDriver](../../reference/griptape/drivers/embedding/openai_embedding_driver.md)
with updated defaults.

### Bedrock Titan

!!! info

    This driver requires the `drivers-embedding-amazon-bedrock` [extra](../index.md#extras).

The [AmazonBedrockTitanEmbeddingDriver](../../reference/griptape/drivers/embedding/amazon_bedrock_titan_embedding_driver.md) uses the [Amazon Bedrock Embeddings API](https://docs.aws.amazon.com/bedrock/latest/userguide/embeddings.html).

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/embedding_drivers_3.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/embedding_drivers_3.txt"
    ```


```
[-0.234375, -0.024902344, -0.14941406]
```

### Google

!!! info

    This driver requires the `drivers-embedding-google` [extra](../index.md#extras).

The [GoogleEmbeddingDriver](../../reference/griptape/drivers/embedding/google_embedding_driver.md) uses the [Google Embeddings API](https://ai.google.dev/tutorials/python_quickstart#use_embeddings).

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/embedding_drivers_4.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/embedding_drivers_4.txt"
    ```


```
[0.0588633, 0.0033929371, -0.072810836]
```

### Hugging Face Hub

!!! info

    This driver requires the `drivers-embedding-huggingface` [extra](../index.md#extras).

The [HuggingFaceHubEmbeddingDriver](../../reference/griptape/drivers/embedding/huggingface_hub_embedding_driver.md) connects to the [Hugging Face Hub API](https://huggingface.co/docs/hub/api). It supports models with the following tasks:

- feature-extraction

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/embedding_drivers_5.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/embedding_drivers_5.txt"
    ```


### Ollama

!!! info

    This driver requires the `drivers-embedding-ollama` [extra](../index.md#extras).

The [OllamaEmbeddingDriver](../../reference/griptape/drivers/embedding/ollama_embedding_driver.md) uses the [Ollama Embeddings API](https://ollama.com/blog/embedding-models).

```python
--8<-- "docs/griptape-framework/drivers/src/embedding_drivers_6.py"
```

### Amazon SageMaker Jumpstart

The [AmazonSageMakerJumpstartEmbeddingDriver](../../reference/griptape/drivers/embedding/amazon_sagemaker_jumpstart_embedding_driver.md) uses the [Amazon SageMaker Endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html) to generate embeddings on AWS.

!!! info

    This driver requires the `drivers-embedding-amazon-sagemaker` [extra](../index.md#extras).

```python
--8<-- "docs/griptape-framework/drivers/src/embedding_drivers_7.py"
```

### VoyageAI

The [VoyageAiEmbeddingDriver](../../reference/griptape/drivers/embedding/voyageai_embedding_driver.md) uses the [VoyageAI Embeddings API](https://www.voyageai.com/).

!!! info

    This driver requires the `drivers-embedding-voyageai` [extra](../index.md#extras).

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/embedding_drivers_8.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/embedding_drivers_8.txt"
    ```


### Cohere

The [CohereEmbeddingDriver](../../reference/griptape/drivers/embedding/cohere_embedding_driver.md) uses the [Cohere Embeddings API](https://docs.cohere.com/docs/embeddings).

!!! info

    This driver requires the `drivers-embedding-cohere` [extra](../index.md#extras).

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/embedding_drivers_9.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/embedding_drivers_9.txt"
    ```


### Override Default Structure Embedding Driver

Here is how you can override the Embedding Driver that is used by default in Structures.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/embedding_drivers_10.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/embedding_drivers_10.txt"
    ```

