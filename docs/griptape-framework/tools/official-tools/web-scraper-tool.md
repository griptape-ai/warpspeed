# Web Scraper Tool

This tool enables LLMs to scrape web pages for full text, summaries, authors, titles, and keywords. It can also execute search queries to answer specific questions about the page. This tool uses OpenAI APIs for some of its activities, so in order to use it provide a valid API key in `openai_api_key`.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/official-tools/src/web_scraper_tool_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/official-tools/logs/web_scraper_tool_1.txt"
    ```


```
[08/12/24 15:32:08] INFO     PromptTask b14a4305365f4b17a4dcf235f84397e2
                             Input: Based on https://www.griptape.ai/, tell me what griptape is
[08/12/24 15:32:10] INFO     Subtask bf396977ea634eb28f55388d3f828f5d
                             Actions: [
                               {
                                 "tag": "call_ExEzJDZuBfnsa9pZMSr6mtsS",
                                 "name": "WebScraperTool",
                                 "path": "get_content",
                                 "input": {
                                   "values": {
                                     "url": "https://www.griptape.ai/"
                                   }
                                 }
                               }
                             ]
                    INFO     Subtask bf396977ea634eb28f55388d3f828f5d
                             Response: Output of "WebScraperTool.get_content" was stored in memory with memory_name "TaskMemory" and artifact_namespace
                             "a55c85bf1aa944d5b69bbe8d61382179"
[08/12/24 15:32:11] INFO     Subtask 31852039bd274b71bf46feaf22b68112
                             Actions: [
                               {
                                 "tag": "call_6Dovx2GKE2GLjaYIuwXvBxVn",
                                 "name": "PromptSummaryTool",
                                 "path": "summarize",
                                 "input": {
                                   "values": {
                                     "summary": {
                                       "memory_name": "TaskMemory",
                                       "artifact_namespace": "a55c85bf1aa944d5b69bbe8d61382179"
                                     }
                                   }
                                 }
                               }
                             ]
[08/12/24 15:32:15] INFO     Subtask 31852039bd274b71bf46feaf22b68112
                             Response: Griptape offers a comprehensive solution for building, deploying, and scaling AI applications in the cloud. It provides developers
                             with a framework and cloud services to create retrieval-driven AI-powered applications without needing extensive knowledge in AI or prompt
                             engineering.

                             **Griptape Framework:**
                             - Enables developers to build AI applications using Python.
                             - Offers better security, performance, and cost-efficiency with Off-Prompt™ technology.
                             - Facilitates the creation of Gen AI Agents, Systems of Agents, Pipelines, Workflows, and RAG implementations.

                             **Griptape Cloud:**
                             - Simplifies deployment and execution of ETL, RAG, and other structures.
                             - Provides API abstractions and eliminates the need for infrastructure management.
                             - Supports seamless scaling to accommodate growing workloads.

                             **Solutions & Applications:**
                             - Custom project development.
                             - Turnkey SaaS offerings for non-tech businesses.
                             - Ready-made apps and options to offer apps to customers.

                             **Key Features:**
                             - Automated Data Prep (ETL): Connect, extract, transform, and load data into a vector database index.
                             - Retrieval as a Service (RAG): Generate answers, summaries, and details from your data using customizable retrieval patterns.
                             - Structure Runtime (RUN): Build and integrate AI agents, pipelines, and workflows into client applications.
[08/12/24 15:32:21] INFO     PromptTask b14a4305365f4b17a4dcf235f84397e2
                             Output: Griptape is a comprehensive solution designed to facilitate the building, deploying, and scaling of AI applications in the cloud. It
                             provides developers with a framework and cloud services that simplify the creation of retrieval-driven AI-powered applications, even for those
                             without extensive AI or prompt engineering expertise.

                             ### Key Components of Griptape:

                             1. **Griptape Framework:**
                                - **Development:** Allows developers to build AI applications using Python.
                                - **Technology:** Utilizes Off-Prompt™ technology for enhanced security, performance, and cost-efficiency.
                                - **Capabilities:** Supports the creation of Gen AI Agents, Systems of Agents, Pipelines, Workflows, and Retrieval-Augmented Generation (RAG)
                             implementations.

                             2. **Griptape Cloud:**
                                - **Deployment:** Simplifies the deployment and execution of ETL (Extract, Transform, Load), RAG, and other structures.
                                - **API Abstractions:** Provides API abstractions to eliminate the need for infrastructure management.
                                - **Scalability:** Supports seamless scaling to accommodate growing workloads.

                             ### Solutions & Applications:
                             - **Custom Projects:** Development of tailored AI solutions.
                             - **Turnkey SaaS:** Ready-to-use SaaS offerings for non-technical businesses.
                             - **Ready-made Apps:** Pre-built applications and options to offer apps to customers.

                             ### Key Features:
                             - **Automated Data Prep (ETL):** Connects, extracts, transforms, and loads data into a vector database index.
                             - **Retrieval as a Service (RAG):** Generates answers, summaries, and details from data using customizable retrieval patterns.
                             - **Structure Runtime (RUN):** Facilitates the building and integration of AI agents, pipelines, and workflows into client applications.

                             In summary, Griptape provides a robust platform for developing and managing AI applications, making it accessible for developers and businesses
                             to leverage AI technology effectively.
```
