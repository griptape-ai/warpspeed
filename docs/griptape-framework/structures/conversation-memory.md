---
search:
  boost: 2
---

## Overview

You can use Conversation Memory to give Griptape Structures the ability to keep track of the conversation across runs. All structures are created with [ConversationMemory](../../reference/griptape/memory/structure/conversation_memory.md) by default.

### Example

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/conversation_memory_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/conversation_memory_1.txt"
    ```


```
[09/19/23 14:21:07] INFO     PromptTask 3e64ca5d5f634a11957cbf46adce251a
                             Input: My favorite animal is a Liger.
[09/19/23 14:21:13] INFO     PromptTask 3e64ca5d5f634a11957cbf46adce251a
                             Output: That's fascinating! Ligers, a hybrid offspring of a male lion and a female tiger, are indeed unique and
                             interesting animals. They are known to be the largest of all big cats. Do you have a particular reason why you
                             like them so much?
                    INFO     PromptTask 3e64ca5d5f634a11957cbf46adce251a
                             Input: What is my favorite animal?
[09/19/23 14:21:15] INFO     PromptTask 3e64ca5d5f634a11957cbf46adce251a
                             Output: Your favorite animal is a Liger, as you previously mentioned.
```

You can disable conversation memory in any structure by setting it to `None`:

```python
--8<-- "docs/griptape-framework/structures/src/conversation_memory_2.py"
```

### Interaction With Structures

#### Per Structure

By default, Conversation Memory [Runs](../../reference/griptape/memory/structure/run.md) are created for each run of the structure. Griptape takes the Structure's [input_task](../../reference/griptape/structures/structure.md#griptape.structures.structure.Structure.input_task)'s input and the [output_task](../../reference/griptape/structures/structure.md#griptape.structures.structure.Structure.output_task)'s output, storing them in the Run. Tasks that are neither the input task nor the output task are not stored in the Run.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/conversation_memory_per_structure.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/conversation_memory_per_structure.txt"
    ```


In this example, the `improve` Task is "forgotten" after the Structure's run is finished. This approach allows you to perform intermediary work within a Structure without it being stored in, and potentially cluttering, Conversation Memory.

#### Per Task

You can change when Conversation Memory Runs are created by modifying [Structure.conversation_memory_strategy](../../reference/griptape/structures/structure.md#griptape.structures.structure.Structure.conversation_memory_strategy) from the default `per_structure` to `per_task`.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/conversation_memory_per_task.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/conversation_memory_per_task.txt"
    ```


Now, each _Task_ creates a Conversation Memory Run when it runs. This eliminates the need to feed the output of Tasks into each other using context variables like `{{ parent_output }}` since the output of the previous Task is stored in Conversation Memory and loaded when the next Task runs.

To blend the two approaches, you can disable Conversation Memory on individual tasks by setting [PromptTask.conversation_memory](../../reference/griptape/tasks/prompt_task.md#griptape.tasks.prompt_task.PromptTask.conversation_memory) to `None`.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/conversation_memory_per_task_with_disabled.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/conversation_memory_per_task_with_disabled.txt"
    ```


## Types of Memory

Griptape provides several types of Conversation Memory to fit various use-cases.

### Conversation Memory

[ConversationMemory](../../reference/griptape/memory/structure/conversation_memory.md) will keep track of the full task input and output for all runs.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/conversation_memory_3.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/conversation_memory_3.txt"
    ```


You can set the [max_runs](../../reference/griptape/memory/structure/base_conversation_memory.md#griptape.memory.structure.base_conversation_memory.BaseConversationMemory.max_runs) parameter to limit how many runs are kept in memory.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/conversation_memory_4.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/conversation_memory_4.txt"
    ```


### Summary Conversation Memory

[SummaryConversationMemory](../../reference/griptape/memory/structure/summary_conversation_memory.md) will progressively summarize task input and output of runs.

You can choose to offset which runs are summarized with the
[offset](../../reference/griptape/memory/structure/summary_conversation_memory.md#griptape.memory.structure.summary_conversation_memory.SummaryConversationMemory.offset) parameter.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/conversation_memory_5.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/conversation_memory_5.txt"
    ```

