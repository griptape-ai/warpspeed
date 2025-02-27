The Griptape framework provides developers with the ability to create AI systems that operate across two dimensions: **predictability** and **creativity**.

For **predictability**, Griptape enforces structures like sequential pipelines, DAG-based workflows, and long-term memory. To facilitate creativity, Griptape safely prompts LLMs with tools and short-term memory connecting them to external APIs and data stores. The framework allows developers to transition between those two dimensions effortlessly based on their use case.

Griptape not only helps developers harness the potential of LLMs but also enforces trust boundaries, schema validation, and tool activity-level permissions. By doing so, Griptape maximizes LLMs’ reasoning while adhering to strict policies regarding their capabilities.

Griptape’s design philosophy is based on the following tenets:

1. **Modularity and composability**: All framework primitives are useful and usable on their own in addition to being easy to plug into each other.
1. **Technology-agnostic**: Griptape is designed to work with any capable LLM, data store, and backend through the abstraction of drivers.
1. **Keep data off prompt by default**: When working with data through loaders and tools, Griptape aims to keep it off prompt by default, making it easy to work with big data securely and with low latency.
1. **Minimal prompt engineering**: It’s much easier to reason about code written in Python, not natural languages. Griptape aims to default to Python in most cases unless absolutely necessary.

## Quick Start

### OpenAI API Key

First, configure an OpenAI client by [getting an API key](https://platform.openai.com/account/api-keys) and adding it to your environment as `OPENAI_API_KEY`.
By default, Griptape uses [OpenAI Completions API](https://platform.openai.com/docs/guides/completion) to execute LLM prompts, but other LLMs can be configured with the use of [Prompt Drivers](./drivers/prompt-drivers.md).

### Using pip

Install **griptape**:

```
pip install "griptape[all]" -U
```

### Using Uv

To get started with Griptape using [uv](https://docs.astral.sh/uv/), first create a new `uv` project from the terminal:

```
uv init griptape-quickstart
```

Change your working directory to the new `griptape-quickstart` directory created by `uv` and add the the `griptape` dependency.

```
uv add "griptape[all]"
```

### Extras

The `[all]` [extra](https://peps.python.org/pep-0508/#extras) ensures that you have access to the entire range of functionalities that Griptape offers.
This comprehensive installation is recommended for newcomers to get the complete Griptape experience.

However, if you wish to optimize the installation size or only require specific functionalities, you have two main options:

1. Core Dependencies: These are the foundational dependencies that enable Griptape to function with most of its default settings.
1. Extras: These are additional, vendor-specific drivers integrated within the Griptape framework. If a particular Driver mandates an extra, it will be explicitly highlighted in the documentation.

To install just the core dependencies:

```
uv add griptape
```

To install specific extras (e.g., drivers for [AnthropicPromptDriver](./drivers/prompt-drivers.md#anthropic) and [PineconeVectorStoreDriver](./drivers/vector-store-drivers.md#pinecone)):

```
uv add "griptape[drivers-prompt-anthropic,drivers-vector-pinecone]"
```

For a comprehensive list of extras, please refer to the `[project.optional-dependencies]` section of Griptape's [pyproject.toml](https://github.com/griptape-ai/griptape/blob/main/pyproject.toml).

## Build a Simple Agent

With Griptape, you can create *structures*, such as [Agents](./structures/agents.md), [Pipelines](./structures/pipelines.md), and [Workflows](./structures/workflows.md), that are composed of different types of tasks. First, let's build a simple Agent that we can interact with through a chat based interface.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/src/index_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/logs/index_1.txt"
    ```


Run this script in your IDE and you'll be presented with a `Q:` prompt where you can interact with your model.

```
Q: Write me a haiku about griptape
processing...
[09/08/23 09:52:45] INFO     PromptTask d4302227570e4a978ed79e7e0444337b
                             Input: Write me a haiku about griptape
[09/08/23 09:52:48] INFO     PromptTask d4302227570e4a978ed79e7e0444337b
                             Output: Griptape rough and true,
                             Skateboard's trusty, silent guide,
                             In each ride, we're glued.
A: Griptape rough and true,
Skateboard's trusty, silent guide,
In each ride, we're glued.
Q:
```

If you want to skip the chat interface and load an initial prompt, you can do so using the `.run()` method:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/src/index_2.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/logs/index_2.txt"
    ```


Agents on their own are fun, but let's add some capabilities to them using Griptape Tools.

### Build a Simple Agent with Tools

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/src/index_3.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/logs/index_3.txt"
    ```


Here is the chain of thought from the Agent. Notice where it realizes it can use the tool you just injected to do the calculation.[^1]

```
[07/23/24 10:47:38] INFO     PromptTask 6a51060d1fb74e57840a91aa319f26dc
                             Input: what is 7^12
[07/23/24 10:47:39] INFO     Subtask 0c984616fd2345a7b48a0b0d692daa3c
                             Actions: [
                               {
                                 "tag": "call_RTRm7JLFV0F73dCVPmoWVJqO",
                                 "name": "CalculatorTool",
                                 "path": "calculate",
                                 "input": {
                                   "values": {
                                     "expression": "7**12"
                                   }
                                 }
                               }
                             ]
                    INFO     Subtask 0c984616fd2345a7b48a0b0d692daa3c
                             Response: 13841287201
[07/23/24 10:47:40] INFO     PromptTask 6a51060d1fb74e57840a91aa319f26dc
                             Output: 13,841,287,201
Answer: 13,841,287,201
```

## Build a Simple Pipeline

Agents are great for getting started, but they are intentionally limited to a single task. Pipelines, however, allow us to define any number of tasks to run in sequence. Let's define a simple two-task Pipeline that uses tools and memory:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/src/index_4.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/logs/index_4.txt"
    ```


```
[08/12/24 14:50:28] INFO     PromptTask 19dcf6020968468a91aa8a93c2a3f645
                             Input: Load https://www.griptape.ai, summarize it, and store it in griptape.txt
[08/12/24 14:50:30] INFO     Subtask a685799379c5421b91768353fc219939
                             Actions: [
                               {
                                 "tag": "call_YL5Ozd9WUtag4ykR5Agm12Ce",
                                 "name": "WebScraperTool",
                                 "path": "get_content",
                                 "input": {
                                   "values": {
                                     "url": "https://www.griptape.ai"
                                   }
                                 }
                               }
                             ]
[08/12/24 14:50:31] INFO     Subtask a685799379c5421b91768353fc219939
                             Response: Output of "WebScraperTool.get_content" was stored in memory with memory_name "TaskMemory" and artifact_namespace
                             "6be3a2e0494841fda966b98bec9ffccb"
[08/12/24 14:50:33] INFO     Subtask 1cf0c19843aa4fada5745c4a82eb4237
                             Actions: [
                               {
                                 "tag": "call_ElTYTPeocOU62I0VjzRqmfoF",
                                 "name": "PromptSummaryTool",
                                 "path": "summarize",
                                 "input": {
                                   "values": {
                                     "summary": {
                                       "memory_name": "TaskMemory",
                                       "artifact_namespace": "6be3a2e0494841fda966b98bec9ffccb"
                                     }
                                   }
                                 }
                               }
                             ]
[08/12/24 14:50:35] INFO     Subtask 1cf0c19843aa4fada5745c4a82eb4237
                             Response: Griptape offers a comprehensive solution for building, deploying, and scaling AI applications in the cloud. It provides developers
                             with a framework and cloud services to create retrieval-driven AI-powered applications. The Griptape Framework allows developers to build
                             business logic using Python, ensuring better security, performance, and cost-efficiency. It simplifies the creation of Gen AI Agents, Systems of
                             Agents, Pipelines, Workflows, and RAG implementations without needing extensive knowledge of Gen AI or Prompt Engineering.

                             Griptape Cloud handles infrastructure management, offering services like ETL pipelines for data preparation, Retrieval as a Service (RAG) for
                             generating answers and summaries, and a Structure Runtime (RUN) for building AI agents and workflows. This enables seamless scaling and
                             integration with client applications, catering to custom projects, turnkey SaaS offerings, and finished apps.
[08/12/24 14:50:38] INFO     Subtask aaaeca1a089844d4915d065deb3c00cf
                             Actions: [
                               {
                                 "tag": "call_eKvIUIw45aRYKDBpT1gGKc9b",
                                 "name": "FileManagerTool",
                                 "path": "save_content_to_file",
                                 "input": {
                                   "values": {
                                     "path": "griptape.txt",
                                     "content": "Griptape offers a comprehensive solution for building, deploying, and scaling AI applications in the cloud. It provides
                             developers with a framework and cloud services to create retrieval-driven AI-powered applications. The Griptape Framework allows developers to
                             build business logic using Python, ensuring better security, performance, and cost-efficiency. It simplifies the creation of Gen AI Agents,
                             Systems of Agents, Pipelines, Workflows, and RAG implementations without needing extensive knowledge of Gen AI or Prompt
                             Engineering.\n\nGriptape Cloud handles infrastructure management, offering services like ETL pipelines for data preparation, Retrieval as a
                             Service (RAG) for generating answers and summaries, and a Structure Runtime (RUN) for building AI agents and workflows. This enables seamless
                             scaling and integration with client applications, catering to custom projects, turnkey SaaS offerings, and finished apps."
                                   }
                                 }
                               }
                             ]
                    INFO     Subtask aaaeca1a089844d4915d065deb3c00cf
                             Response: Successfully saved file
[08/12/24 14:50:39] INFO     PromptTask 19dcf6020968468a91aa8a93c2a3f645
                             Output: The content from https://www.griptape.ai has been summarized and stored in griptape.txt.
                    INFO     PromptTask dbbb38f144f445db896dc12854f17ad3
                             Input: Say the following in spanish: The content from https://www.griptape.ai has been summarized and stored in griptape.txt.
[08/12/24 14:50:42] INFO     PromptTask dbbb38f144f445db896dc12854f17ad3
                             Output: El contenido de https://www.griptape.ai ha sido resumido y almacenado en griptape.txt.
```

[^1]: In some cases a model might be capable of basic arithmetic. For example, gpt-3.5 returns the correct numeric answer but in an odd format.
