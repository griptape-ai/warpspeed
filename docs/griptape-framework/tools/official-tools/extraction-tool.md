The [ExractionTool](../../../reference/griptape/tools/extraction/tool.md) enables LLMs to extract structured text from unstructured data.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/official-tools/src/extraction_tool_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/official-tools/logs/extraction_tool_1.txt"
    ```


```
[08/12/24 15:58:03] INFO     PromptTask 43b3d209a83c470d8371b7ef4af175b4
                             Input: Load https://griptape.ai and extract key info
[08/12/24 15:58:05] INFO     Subtask 6a9a63802faf4717bab24bbbea2cb49b
                             Actions: [
                               {
                                 "tag": "call_SgrmWdXaYTQ1Cz9iB0iIZSYD",
                                 "name": "WebScraperTool",
                                 "path": "get_content",
                                 "input": {
                                   "values": {
                                     "url": "https://griptape.ai"
                                   }
                                 }
                               }
                             ]
[08/12/24 15:58:06] INFO     Subtask 6a9a63802faf4717bab24bbbea2cb49b
                             Response: Output of "WebScraperTool.get_content" was stored in memory with memory_name "TaskMemory" and artifact_namespace
                             "bf1c865b82554c9e896cb514bb86844c"
[08/12/24 15:58:07] INFO     Subtask c06388d6079541d5aaff25c30e322c51
                             Actions: [
                               {
                                 "tag": "call_o3MrpM01OnhCfpxsMe85tpDF",
                                 "name": "ExtractionTool",
                                 "path": "extract_json",
                                 "input": {
                                   "values": {
                                     "data": {
                                       "memory_name": "TaskMemory",
                                       "artifact_namespace": "bf1c865b82554c9e896cb514bb86844c"
                                     }
                                   }
                                 }
                               }
                             ]
[08/12/24 15:58:11] INFO     Subtask c06388d6079541d5aaff25c30e322c51
                             Response: {"company_name": "Griptape", "industry": "AI Applications", "product_features": ["Turn any developer into an AI developer.", "Build
                             your business logic using predictable, programmable python.", "Off-Prompt\u2122 for better security, performance, and lower costs.", "Deploy and
                             run the ETL, RAG, and structures you developed.", "Simple API abstractions.", "Skip the infrastructure management.", "Scale seamlessly with
                             workload requirements.", "Clean and clear abstractions for building Gen AI Agents, Systems of Agents, Pipelines, Workflows, and RAG
                             implementations.", "Build ETL pipelines to prep data for secure LLM access.", "Compose retrieval patterns for fast, accurate, detailed
                             information.", "Write agents, pipelines, and workflows to integrate business logic.", "Automated Data Prep (ETL): Connect any data source,
                             extract, prep/transform, and load into a vector database index.", "Retrieval as a Service (RAG): Generate answers, summaries, and details from
                             your own data with ready-made or custom retrieval patterns.", "Structure Runtime (RUN): Build AI agents, pipelines, and workflows for real-time
                             interfaces, transactional processes, and batch workloads."]}
[08/12/24 15:58:14] INFO     PromptTask 43b3d209a83c470d8371b7ef4af175b4
                             Output: Extracted key information from Griptape's website.
```
