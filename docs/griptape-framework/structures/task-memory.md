---
search:
  boost: 2
---

## Overview

Task Memory is a powerful feature of Griptape that allows you to control where the data returned by [Tools](../tools/index.md) is stored. This is useful in the following scenarios:

- **Security requirements**: many organizations don't want data to leave their cloud for regulatory and security reasons.
- **Long textual content**: when textual content returned by Tools can't fit in the token limit, it's often useful to perform actions on it as a separate operation, not through the main LLM.
- **Non-textual content**: Tools can generate images, videos, PDFs, and other non-textual content that can be stored in Task Memory and acted upon later by other Tools.

!!! tip

    Running into issue with Task Memory? Check out the [Task Memory Considerations](#task-memory-considerations) section for some common pitfalls.

## Off Prompt

You can enable or disable sending a Tool's results to Task Memory with the `off_prompt` parameter. By default, all Tools have `off_prompt` set to `False` making this an opt-in feature.
When `off_prompt` is set to `True`, the Tool will store its output in Task Memory. When `off_prompt` is set to `False`, the Tool will return its output directly to the LLM.

Lets look at a simple example where `off_prompt` is set to `False`:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/task_memory_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/task_memory_1.txt"
    ```


```
[04/26/24 13:06:42] INFO     PromptTask 36b9dea13b9d479fb752014f41dca54c
                             Input: What is the square root of 12345?
[04/26/24 13:06:48] INFO     Subtask a88c0feeaef6493796a9148ed68c9caf
                             Thought: To find the square root of 12345, I can use the CalculatorTool action with the expression "12345 ** 0.5".
                             Actions: [{"name": "CalculatorTool", "path": "calculate", "input": {"values": {"expression": "12345 ** 0.5"}}, "tag": "sqrt_12345"}]
                    INFO     Subtask a88c0feeaef6493796a9148ed68c9caf
                             Response: 111.1080555135405
[04/26/24 13:06:49] INFO     PromptTask 36b9dea13b9d479fb752014f41dca54c
                             Output: The square root of 12345 is approximately 111.108.
```

Since the result of the CalculatorTool Tool is neither sensitive nor too large, we can set `off_prompt` to `False` and not use Task Memory.

Let's explore what happens when `off_prompt` is set to `True`:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/task_memory_2.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/task_memory_2.txt"
    ```


```
[04/26/24 13:07:02] INFO     PromptTask ecbb788d9830491ab72a8a2bbef5fb0a
                             Input: What is the square root of 12345?
[04/26/24 13:07:10] INFO     Subtask 4700dc0c2e934d1a9af60a28bd770bc6
                             Thought: To find the square root of a number, we can use the CalculatorTool action with the expression "sqrt(12345)". However, the CalculatorTool
                             action only supports basic arithmetic operations and does not support the sqrt function. Therefore, we need to use the equivalent expression
                             for square root which is raising the number to the power of 0.5.
                             Actions: [{"name": "CalculatorTool", "path": "calculate", "input": {"values": {"expression": "12345**0.5"}}, "tag": "sqrt_calculation"}]
                    INFO     Subtask 4700dc0c2e934d1a9af60a28bd770bc6
                             Response: Output of "CalculatorTool.calculate" was stored in memory with memory_name "TaskMemory" and artifact_namespace
                             "6be74c5128024c0588eb9bee1fdb9aa5"
[04/26/24 13:07:16] ERROR    Subtask ecbb788d9830491ab72a8a2bbef5fb0a
                             Invalid action JSON: Or({Literal("name", description=""): 'CalculatorTool', Literal("path", description="Can be used for computing simple
                             numerical or algebraic calculations in Python"): 'calculate', Literal("input", description=""): {'values': Schema({Literal("expression",
                             description="Arithmetic expression parsable in pure Python. Single line only. Don't use variables. Don't use any imports or external
                             libraries"): <class 'str'>})}, Literal("tag", description="Unique tag name for action execution."): <class 'str'>}) did not validate {'name':
                             'Memory', 'path': 'get', 'input': {'memory_name': 'TaskMemory', 'artifact_namespace': '6be74c5128024c0588eb9bee1fdb9aa5'}, 'tag':
                             'get_sqrt_result'}
                             Key 'name' error:
                             'CalculatorTool' does not match 'Memory'
...Output truncated for brevity...
```

When we set `off_prompt` to `True`, the Agent does not function as expected, even generating an error. This is because the Calculator output is being stored in Task Memory but the Agent has no way to access it.
To fix this, we need a [Tool that can read from Task Memory](#tools-that-can-read-from-task-memory) such as the `PromptSummaryTool`.
This is an example of [not providing a Task Memory compatible Tool](#not-providing-a-task-memory-compatible-tool).

## Prompt Summary Tool

The [PromptSummaryTool](../tools/official-tools/index.md#prompt-summary) is a Tool that allows an Agent to summarize the Artifacts in Task Memory. It has the following methods:

Let's add `PromptSummaryTool` to the Agent and run the same task.
Note that on the `PromptSummaryTool` we've set `off_prompt` to `False` so that the results of the query can be returned directly to the LLM.
If we had kept it as `True`, the results would have been stored back Task Memory which would've put us back to square one. See [Task Memory Looping](#task-memory-looping) for more information on this scenario.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/task_memory_3.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/task_memory_3.txt"
    ```


```
[08/12/24 14:54:04] INFO     PromptTask f7ebd8acc3d64e3ca9db82ef9ec4e65f
                             Input: What is the square root of 12345?
[08/12/24 14:54:05] INFO     Subtask 777693d039e74ed288f663742fdde2ea
                             Actions: [
                               {
                                 "tag": "call_DXSs19G27VOV7EmP3PoRwGZI",
                                 "name": "Calculator",
                                 "path": "calculate",
                                 "input": {
                                   "values": {
                                     "expression": "12345 ** 0.5"
                                   }
                                 }
                               }
                             ]
                    INFO     Subtask 777693d039e74ed288f663742fdde2ea
                             Response: Output of "Calculator.calculate" was stored in memory with memory_name "TaskMemory" and artifact_namespace
                             "370853a8937f4dd7a9e923254459cff2"
[08/12/24 14:54:06] INFO     Subtask c8394ca51f1f4ae1b715618a2c5c8120
                             Actions: [
                               {
                                 "tag": "call_qqpsWEvAUGIcPLrwAHGuH6o3",
                                 "name": "PromptSummaryTool",
                                 "path": "summarize",
                                 "input": {
                                   "values": {
                                     "summary": {
                                       "memory_name": "TaskMemory",
                                       "artifact_namespace": "370853a8937f4dd7a9e923254459cff2"
                                     }
                                   }
                                 }
                               }
                             ]
[08/12/24 14:54:07] INFO     Subtask c8394ca51f1f4ae1b715618a2c5c8120
                             Response: The text contains a single numerical value: 111.1080555135405.
[08/12/24 14:54:08] INFO     PromptTask f7ebd8acc3d64e3ca9db82ef9ec4e65f
                             Output: The square root of 12345 is approximately 111.108.
```

While this fixed the problem, it took a handful more steps than when we just had `CalculatorTool()`. Something like a basic calculation is an instance of where [Task Memory may not be necessary](#task-memory-may-not-be-necessary).
Let's look at a more complex example where Task Memory shines.

## Large Data

Let's say we want to query the contents of a very large webpage.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/task_memory_4.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/task_memory_4.txt"
    ```


When running this example, we get the following error:

```
[04/26/24 13:20:02] ERROR    PromptTask 67e2f907f95d4850ae79f9da67df54c1
                             Error code: 400 - {'error': {'message': "This model's maximum context length is 8192 tokens. However, your messages resulted in 73874 tokens.
                             Please reduce the length of the messages.", 'type': 'invalid_request_error', 'param': 'messages', 'code': 'context_length_exceeded'}}
```

This is because the content of the webpage is too large to fit in the LLM's input token limit. We can fix this by storing the content in Task Memory, and then querying it with the `QueryTool`.
Note that we're setting `off_prompt` to `False` on the `QueryTool` so that the _queried_ content can be returned directly to the LLM.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/task_memory_5.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/task_memory_5.txt"
    ```


And now we get the expected output:

```
[08/12/24 14:56:18] INFO     PromptTask d3ce58587dc944b0a30a205631b82944
                             Input: According to this page https://en.wikipedia.org/wiki/Elden_Ring, how many copies of Elden Ring have been sold?
[08/12/24 14:56:20] INFO     Subtask 494850ec40fe474c83d48b5620c5dcbb
                             Actions: [
                               {
                                 "tag": "call_DGsOHC4AVxhV7RPVA7q3rATX",
                                 "name": "WebScraperTool",
                                 "path": "get_content",
                                 "input": {
                                   "values": {
                                     "url": "https://en.wikipedia.org/wiki/Elden_Ring"
                                   }
                                 }
                               }
                             ]
[08/12/24 14:56:25] INFO     Subtask 494850ec40fe474c83d48b5620c5dcbb
                             Response: Output of "WebScraperTool.get_content" was stored in memory with memory_name "TaskMemory" and artifact_namespace
                             "b9f53d6d9b35455aaf4d99719c1bfffa"
[08/12/24 14:56:26] INFO     Subtask 8669ee523bb64550850566011bcd14e2
                             Actions: [
                               {
                                 "tag": "call_DGsOHC4AVxhV7RPVA7q3rATX",
                                 "name": "QueryTool",
                                 "path": "search",
                                 "input": {
                                   "values": {
                                     "query": "number of copies sold",
                                     "content": {
                                       "memory_name": "TaskMemory",
                                       "artifact_namespace": "b9f53d6d9b35455aaf4d99719c1bfffa"
                                     }
                                   }
                                 }
                               }
                             ]
[08/12/24 14:56:29] INFO     Subtask 8669ee523bb64550850566011bcd14e2
                             Response: "Elden Ring" sold 13.4 million copies worldwide by the end of March 2022 and 25 million by June 2024. The downloadable content (DLC)
                             "Shadow of the Erdtree" sold five million copies within three days of its release.
[08/12/24 14:56:30] INFO     PromptTask d3ce58587dc944b0a30a205631b82944
                             Output: Elden Ring sold 13.4 million copies worldwide by the end of March 2022 and 25 million by June 2024.
```

## Sensitive Data

Because Task Memory splits up the storage and retrieval of data, you can use different models for each step.

Here is an example where we use GPT-4 to orchestrate the Tools and store the data in Task Memory, and Anthropic's Claude 3 Haiku model to query the raw content.
In this example, GPT-4 _never_ sees the contents of the page, only that it was stored in Task Memory. Even the query results generated by the Haiku model are stored in Task Memory so that the `FileManagerTool` can save the results to disk without GPT-4 ever seeing them.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/task_memory_6.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/task_memory_6.txt"
    ```


```
[08/12/24 14:55:21] INFO     PromptTask 329b1abc760e4d30bbf23e349451d930
                             Input: Use this page https://en.wikipedia.org/wiki/Elden_Ring to find how many copies of Elden Ring have been sold, and then save the result to
                             a file.
[08/12/24 14:55:23] INFO     Subtask 26205b5623174424b618abafd886c4d8
                             Actions: [
                               {
                                 "tag": "call_xMK0IyFZFbjlTapK7AA6kbNq",
                                 "name": "WebScraperTool",
                                 "path": "get_content",
                                 "input": {
                                   "values": {
                                     "url": "https://en.wikipedia.org/wiki/Elden_Ring"
                                   }
                                 }
                               }
                             ]
[08/12/24 14:55:28] INFO     Subtask 26205b5623174424b618abafd886c4d8
                             Response: Output of "WebScraperTool.get_content" was stored in memory with memory_name "TaskMemory" and artifact_namespace
                             "44b8f230645148d0b8d44354c0f2df5b"
[08/12/24 14:55:31] INFO     Subtask d8b4cf297a0d4d9db04e4f8e63b746c8
                             Actions: [
                               {
                                 "tag": "call_Oiqq6oI20yqmdNrH9Mawb2fS",
                                 "name": "QueryTool",
                                 "path": "search",
                                 "input": {
                                   "values": {
                                     "query": "copies sold",
                                     "content": {
                                       "memory_name": "TaskMemory",
                                       "artifact_namespace": "44b8f230645148d0b8d44354c0f2df5b"
                                     }
                                   }
                                 }
                               }
                             ]
[08/12/24 14:55:34] INFO     Subtask d8b4cf297a0d4d9db04e4f8e63b746c8
                             Response: Output of "QueryTool.search" was stored in memory with memory_name "TaskMemory" and artifact_namespace
                             "fd828ddd629e4974a7837f9dfde65954"
[08/12/24 14:55:38] INFO     Subtask 7aafcb3fb0d845858e2fcf9b8dc8a7ec
                             Actions: [
                               {
                                 "tag": "call_nV1DIPAEhUEAVMCjXND0pKoS",
                                 "name": "FileManagerTool",
                                 "path": "save_memory_artifacts_to_disk",
                                 "input": {
                                   "values": {
                                     "dir_name": "results",
                                     "file_name": "elden_ring_sales.txt",
                                     "memory_name": "TaskMemory",
                                     "artifact_namespace": "fd828ddd629e4974a7837f9dfde65954"
                                   }
                                 }
                               }
                             ]
                    INFO     Subtask 7aafcb3fb0d845858e2fcf9b8dc8a7ec
                             Response: Successfully saved memory artifacts to disk
[08/12/24 14:55:40] INFO     PromptTask 329b1abc760e4d30bbf23e349451d930
                             Output: Successfully saved the number of copies sold of Elden Ring to a file named "elden_ring_sales.txt" in the "results" directory.
```

## Tools That Can Read From Task Memory

As seen in the previous example, certain Tools are designed to read directly from Task Memory. This means that you can use these Tools to interact with the data stored in Task Memory without needing to pass it through the LLM.

Today, these include:

- [PromptSummaryTool](../tools/official-tools/index.md#prompt-summary)
- [ExtractionTool](../tools/official-tools/index.md#extraction)
- [RagTool](../tools/official-tools/index.md#rag)
- [FileManagerTool](../tools/official-tools/index.md#file-manager)
- [ImageQueryTool](../tools/official-tools/index.md#image-query)

## Task Memory Considerations

Task Memory is a powerful feature of Griptape, but with great power comes great responsibility. Here are some things to keep in mind when using Task Memory:

### Tool Return Types

Griptape will only store Artifacts in Task Memory that have been explicitly defined in the `artifact_storages` parameter of the `TaskMemory` object.
If you try to store an Artifact that is not defined in `artifact_storages`, Griptape will raise an error. The exception to this is `InfoArtifact`s and `ErrorArtifact`s. Griptape will never store these Artifacts store in Task Memory.
By default, Griptape will store `TextArtifact`'s, `BlobArtifact`'s in Task Memory. Additionally, Griptape will also store the elements of `ListArtifact`'s as long as they are of a supported Artifact type.

### Not Providing a Task Memory Compatible Tool

When using Task Memory, make sure that you have at least one Tool that can read from Task Memory. If you don't, the data stored in Task Memory will be inaccessible to the Agent and it may hallucinate Tool Activities.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/task_memory_7.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/task_memory_7.txt"
    ```


### Task Memory Looping

An improper configuration of Tools can lead to the LLM using the Tools in a loop. For example, if you have a Tool that stores data in Task Memory and another Tool that queries that data from Task Memory ([Tools That Can Read From Task Memory](#tools-that-can-read-from-task-memory)), make sure that the query Tool does not store the data back in Task Memory.
This can create a loop where the same data is stored and queried over and over again.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/task_memory_8.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/task_memory_8.txt"
    ```


### Task Memory May Not Be Necessary

Task Memory may not be necessary for all use cases. If the data returned by a Tool is not sensitive, not too large, and does not need to be acted upon by another Tool, you can leave the default of `off_prompt` to `False` and return the data directly to the LLM.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/task_memory_9.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/task_memory_9.txt"
    ```

