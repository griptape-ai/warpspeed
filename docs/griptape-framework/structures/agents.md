---
search:
  boost: 2
---

## Overview

An [Agent](../../reference/griptape/structures/agent.md) is the quickest way to get started with Griptape.
Agents take in [tools](../../reference/griptape/structures/agent.md#griptape.structures.agent.Agent.tools) and [input](../../reference/griptape/structures/agent.md#griptape.structures.agent.Agent.input)
directly, which the agent uses to add a [Prompt Task](./tasks.md#prompt-task).

You can access the final output of the Agent by using the [output](../../reference/griptape/structures/structure.md#griptape.structures.structure.Structure.output) attribute.

### Agent Tools

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/agents_2.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/agents_2.txt"
    ```


```
[07/23/24 10:53:41] INFO     PromptTask 487db777bc014193ba90b061451b69a6
                             Input: Calculate the following: what's 13^7?
[07/23/24 10:53:42] INFO     Subtask 126cefa3ac5347b88495e25af52f3268
                             Actions: [
                               {
                                 "tag": "call_ZSCH6vNoycOgtPJH2DL2U9ji",
                                 "name": "CalculatorTool",
                                 "path": "calculate",
                                 "input": {
                                   "values": {
                                     "expression": "13**7"
                                   }
                                 }
                               }
                             ]
                    INFO     Subtask 126cefa3ac5347b88495e25af52f3268
                             Response: 62748517
[07/23/24 10:53:43] INFO     PromptTask 487db777bc014193ba90b061451b69a6
                             Output: 62,748,517
Answer: 62,748,517
```

### Agent Input

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/agents_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/agents_1.txt"
    ```


```
[09/08/23 10:10:24] INFO     PromptTask e70fb08090b24b91a9307fa83479e851
                             Input: Write me a haiku about Skateboards and Programming
[09/08/23 10:10:28] INFO     PromptTask e70fb08090b24b91a9307fa83479e851
                             Output: Code on wheels in flight,
                             Skateboards meet algorithms bright,
                             In binary, we ignite.
```
