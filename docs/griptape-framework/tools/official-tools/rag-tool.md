# Rag Tool

The [RagTool](../../../reference/griptape/tools/rag/tool.md) enables LLMs to query modular RAG engines.

Here is an example of how it can be used with a local vector store driver:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/official-tools/src/rag_tool_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/official-tools/logs/rag_tool_1.txt"
    ```


```
[07/11/24 13:30:43] INFO     PromptTask a6d057d5c71d4e9cb6863a2adb64b76c
                             Input: what is Griptape?
[07/11/24 13:30:44] INFO     Subtask 8fd89ed9eefe49b8892187f2fca3890a
                             Actions: [
                               {
                                 "tag": "call_4MaDzOuKnWAs2gmhK3KJhtjI",
                                 "name": "RagTool",
                                 "path": "search",
                                 "input": {
                                   "values": {
                                     "query": "What is Griptape?"
                                   }
                                 }
                               }
                             ]
[07/11/24 13:30:49] INFO     Subtask 8fd89ed9eefe49b8892187f2fca3890a
                             Response: Griptape builds AI-powered applications that connect securely to your enterprise data and APIs. Griptape Agents provide incredible
                             power and flexibility when working with large language models.
                    INFO     PromptTask a6d057d5c71d4e9cb6863a2adb64b76c
                             Output: Griptape builds AI-powered applications that connect securely to your enterprise data and APIs. Griptape Agents provide incredible
                             power and flexibility when working with large language models.
```
