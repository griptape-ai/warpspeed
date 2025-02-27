---
search:
  boost: 2
---

## Overview

Griptape exposes a global singleton, [Defaults](../../reference/griptape/configs/defaults_config.md), which can be used to access and modify the default configurations of the framework.

To update the default configurations, simply update the fields on the `Defaults` object.
Framework objects will be created with the currently set default configurations, but you can always override at the individual class level.

```python
--8<-- "docs/griptape-framework/structures/src/config_defaults.py"
```

### Drivers Configs

The [DriversConfig](../../reference/griptape/configs/drivers/drivers_config.md) class allows for the customization of Structures within Griptape, enabling specific settings such as Drivers to be defined for Tasks.

Griptape provides predefined [DriversConfig](../../reference/griptape/configs/drivers/drivers_config.md)'s for widely used services that provide APIs for most Driver types Griptape offers.

`DriversConfig`s can be used as a Python Context Manager using the `with` statement to temporarily change the default configurations for a block of code.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/drivers_config_with.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/drivers_config_with.txt"
    ```


#### OpenAI

The [OpenAI Driver config](../../reference/griptape/configs/drivers/openai_drivers_config.md) provides default Drivers for OpenAI's APIs. This is the default config for all Structures.

```python
--8<-- "docs/griptape-framework/structures/src/drivers_config_1.py"
```

#### Azure OpenAI

The [Azure OpenAI Driver config](../../reference/griptape/configs/drivers/azure_openai_drivers_config.md) provides default Drivers for Azure's OpenAI APIs.

```python
--8<-- "docs/griptape-framework/structures/src/drivers_config_2.py"
```

#### Amazon Bedrock

The [Amazon Bedrock Driver config](../../reference/griptape/configs/drivers/amazon_bedrock_drivers_config.md) provides default Drivers for Amazon Bedrock's APIs.

```python
--8<-- "docs/griptape-framework/structures/src/drivers_config_3.py"
```

#### Google

The [Google Driver config](../../reference/griptape/configs/drivers/google_drivers_config.md) provides default Drivers for Google's Gemini APIs.

```python
--8<-- "docs/griptape-framework/structures/src/drivers_config_4.py"
```

#### Anthropic

The [Anthropic Driver config](../../reference/griptape/configs/drivers/anthropic_drivers_config.md) provides default Drivers for Anthropic's APIs.

!!! info

    Anthropic does not provide an embeddings API which means you will need to use another service for embeddings.
    To override the default embedding driver, see: [Override Default Structure Embedding Driver](../drivers/embedding-drivers.md#override-default-structure-embedding-driver).

```python
--8<-- "docs/griptape-framework/structures/src/drivers_config_5.py"
```

#### Cohere

The [Cohere Driver config](../../reference/griptape/configs/drivers/cohere_drivers_config.md) provides default Drivers for Cohere's APIs.

```python
--8<-- "docs/griptape-framework/structures/src/drivers_config_6.py"
```

#### Custom

You can create your own [DriversConfig](../../reference/griptape/configs/drivers/drivers_config.md) by overriding relevant Drivers.
The [DriversConfig](../../reference/griptape/configs/drivers/drivers_config.md) class includes "Dummy" Drivers for all types, which throw a [DummyError](../../reference/griptape/exceptions/dummy_exception.md) if invoked without being overridden.
This approach ensures that you are informed through clear error messages if you attempt to use Structures without proper Driver configurations.

```python
--8<-- "docs/griptape-framework/structures/src/drivers_config_7.py"
```

### Logging Config

Griptape provides a predefined [LoggingConfig](../../reference/griptape/configs/logging/logging_config.md)'s for easily customizing the logging events that the framework emits. In order to customize the logger, the logger can be fetched by using the `Defaults.logging.logger_name`.

```python
--8<-- "docs/griptape-framework/structures/src/logging_config.py"
```

#### Debug Logs

You can enable debug logs to view more granular information such as request/response payloads.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/debug_logs.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/debug_logs.txt"
    ```


```
[10/09/24 15:30:04] INFO     PromptTask 75ef1747a5824bc8ac838f3081aeb57d
                             Input: Hello world!
                    DEBUG    {
                               "model": "gpt-4o",
                               "temperature": 0.1,
                               "user": "",
                               "seed": null,
                               "messages": [
                                 {
                                   "role": "user",
                                   "content": "Hello world!"
                                 }
                               ]
                             }
[10/09/24 15:30:05] DEBUG    {
                               "id": "chatcmpl-AGZTwg4T4YikR2KjF3AMIRxlIfcKa",
                               "choices": [
                                 {
                                   "finish_reason": "stop",
                                   "index": 0,
                                   "logprobs": null,
                                   "message": {
                                     "content": "Hello! How can I assist you today?",
                                     "refusal": null,
                                     "role": "assistant",
                                     "function_call": null,
                                     "tool_calls": null
                                   }
                                 }
                               ],
                               "created": 1728513004,
                               "model": "gpt-4o-2024-08-06",
                               "object": "chat.completion",
                               "service_tier": null,
                               "system_fingerprint": "fp_2f406b9113",
                               "usage": {
                                 "completion_tokens": 9,
                                 "prompt_tokens": 10,
                                 "total_tokens": 19,
                                 "prompt_tokens_details": {
                                   "cached_tokens": 0
                                 },
                                 "completion_tokens_details": {
                                   "reasoning_tokens": 0
                                 }
                               }
                             }
                    INFO     PromptTask 75ef1747a5824bc8ac838f3081aeb57d
                             Output: Hello! How can I assist you today?

```

### Loading/Saving Configs

You can serialize and deserialize Driver Configs using the [to_json()](../../reference/griptape/mixins/serializable_mixin.md#griptape.mixins.serializable_mixin.SerializableMixin.to_json) and [from_json()](../../reference/griptape/mixins/serializable_mixin.md#griptape.mixins.serializable_mixin.SerializableMixin.from_json) methods.

```python
--8<-- "docs/griptape-framework/structures/src/drivers_config_8.py"
```
