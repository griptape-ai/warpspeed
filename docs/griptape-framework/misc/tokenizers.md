---
search:
  boost: 2
---

## Overview

Tokenizers are used throughout Griptape to calculate the number of [tokens](https://learn.microsoft.com/en-us/semantic-kernel/prompt-engineering/tokens) in a piece of text.
They are particularly useful for ensuring that the LLM token limits are not exceeded.

Tokenizers are a low level abstraction that you will rarely interact with directly.

## Tokenizers

### OpenAI

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/misc/src/tokenizers_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/misc/logs/tokenizers_1.txt"
    ```


### Cohere

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/misc/src/tokenizers_2.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/misc/logs/tokenizers_2.txt"
    ```


### Anthropic

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/misc/src/tokenizers_3.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/misc/logs/tokenizers_3.txt"
    ```


### Google

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/misc/src/tokenizers_4.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/misc/logs/tokenizers_4.txt"
    ```


### Hugging Face

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/misc/src/tokenizers_5.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/misc/logs/tokenizers_5.txt"
    ```


### Amazon Bedrock

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/misc/src/tokenizers_6.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/misc/logs/tokenizers_6.txt"
    ```


### Grok

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/misc/src/tokenizers_grok.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/misc/logs/tokenizers_grok.txt"
    ```


### Simple

Not all LLM providers have a public tokenizer API. In this case, you can use the `SimpleTokenizer` to count tokens based on a simple heuristic.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/misc/src/tokenizers_7.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/misc/logs/tokenizers_7.txt"
    ```

