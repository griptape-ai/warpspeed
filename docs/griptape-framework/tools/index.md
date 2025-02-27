# Tools in Griptape

## Overview

One of the most powerful features of Griptape is the ability to use tools that interact with the outside world. Tools give the LLM abilities to invoke APIs, reference data sets, and expand their capabilities beyond simple text generation.

Many of our [Prompt Drivers](../drivers/prompt-drivers.md) leverage the native function calling built into LLMs. For models that do not support this, Griptape provides its own implementation using the [ReAct](https://arxiv.org/abs/2210.03629) technique.

You can switch between these strategies by setting `use_native_tools` to `True` (LLM-native tool calling) or `False` (Griptape tool calling) on your [Prompt Driver](../drivers/prompt-drivers.md).

## Griptape Tools

Griptape tools are special Python classes that LLMs can use to accomplish specific goals. A tool consists of multiple "activities," each represented by a function decorated with `@activity`. This decorator provides context to the LLM through descriptions and defines the input schema that the LLM must follow.

When a function is decorated with `@activity`, the decorator injects keyword arguments into the function according to the schema. Additionally, Griptape provides two special keyword arguments:

- `params: dict`
- `values: dict`

!!! info

    If your schema defines any parameters named `params` or `values`, they will be overwritten by the Griptape-provided arguments.

Here is an example of a custom tool for generating a random number:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/src/index_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/logs/index_1.txt"
    ```


## Tool Output and Task Memory

Output artifacts from all tool activities (except `InfoArtifact` and `ErrorArtifact`) are stored in short-term `TaskMemory`. To disable this behavior, set the `off_prompt` tool parameter to `False`.

## Using Tools in Pipelines

Griptape provides a set of official tools for accessing and processing data. You can also [build your own tools](./custom-tools/index.md).

Here is an example of a Pipeline using tools:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/src/index_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/logs/index_1.txt"
    ```


```
[08/12/24 15:18:19] INFO     PromptTask 48ac0486e5374e1ea53e8d2b955e511f
                             Input: Load https://www.griptape.ai, summarize it, and store it in griptape.txt
[08/12/24 15:18:20] INFO     Subtask 3b8365c077ae4a7e94087bfeff7a858c
                             Actions: [
                               {
                                 "tag": "call_P6vaURTXfiYBJZolTkUSRHRc",
                                 "name": "WebScraperTool",
                                 "path": "get_content",
                                 "input": {
                                   "values": {
                                     "url": "https://www.griptape.ai"
                                   }
                                 }
                               }
                             ]
                    INFO     Subtask 3b8365c077ae4a7e94087bfeff7a858c
                             Response: Output of "WebScraperTool.get_content" was stored in memory with memory_name "TaskMemory" and artifact_namespace
                             "301e546f4450489ea4680645297092a2"
[08/12/24 15:18:21] INFO     Subtask 930e9ca52e4140a48cce1e47368d45be
                             Actions: [
                               {
                                 "tag": "call_0VOTEvinRer7rG4oEirBYcow",
                                 "name": "PromptSummaryTool",
                                 "path": "summarize",
                                 "input": {
                                   "values": {
                                     "summary": {
                                       "memory_name": "TaskMemory",
                                       "artifact_namespace": "301e546f4450489ea4680645297092a2"
                                     }
                                   }
                                 }
                               }
                             ]
[08/12/24 15:18:24] INFO     Subtask 930e9ca52e4140a48cce1e47368d45be
                             Response: Griptape offers a comprehensive solution for building, deploying, and scaling AI applications in the cloud. It provides developers
                             with a framework and cloud services to create retrieval-driven AI-powered applications without needing extensive knowledge of AI or prompt
                             engineering. The Griptape Framework allows developers to build business logic using Python, ensuring better security, performance, and
                             cost-efficiency. Griptape Cloud handles infrastructure management, enabling seamless deployment and scaling of applications. Key features
                             include automated data preparation (ETL), retrieval as a service (RAG), and a structure runtime (RUN) for building AI agents, pipelines, and
                             workflows. Griptape also offers solutions for custom projects, turnkey SaaS offerings, and finished applications.
[08/12/24 15:18:27] INFO     Subtask d0f22504f576401f8d7e8ea78270a376
                             Actions: [
                               {
                                 "tag": "call_zdUe2vdR0DCfR6LKcxjI6ayb",
                                 "name": "FileManagerTool",
                                 "path": "save_content_to_file",
                                 "input": {
                                   "values": {
                                     "path": "griptape.txt",
                                     "content": "Griptape offers a comprehensive solution for building, deploying, and scaling AI applications in the cloud. It provides
                             developers with a framework and cloud services to create retrieval-driven AI-powered applications without needing extensive knowledge of AI or
                             prompt engineering. The Griptape Framework allows developers to build business logic using Python, ensuring better security, performance, and
                             cost-efficiency. Griptape Cloud handles infrastructure management, enabling seamless deployment and scaling of applications. Key features
                             include automated data preparation (ETL), retrieval as a service (RAG), and a structure runtime (RUN) for building AI agents, pipelines, and
                             workflows. Griptape also offers solutions for custom projects, turnkey SaaS offerings, and finished applications."
                                   }
                                 }
                               }
                             ]
                    INFO     Subtask d0f22504f576401f8d7e8ea78270a376
                             Response: Successfully saved file
[08/12/24 15:18:28] INFO     PromptTask 48ac0486e5374e1ea53e8d2b955e511f
                             Output: The content from https://www.griptape.ai has been summarized and stored in griptape.txt.
                    INFO     PromptTask 4a9c59b1c06d4c549373d243a12f1285
                             Input: Say the following in spanish: The content from https://www.griptape.ai has been summarized and stored in griptape.txt.
                    INFO     PromptTask 4a9c59b1c06d4c549373d243a12f1285
                             Output: El contenido de https://www.griptape.ai ha sido resumido y almacenado en griptape.txt.
```
