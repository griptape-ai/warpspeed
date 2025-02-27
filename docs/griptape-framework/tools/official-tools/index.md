---
search:
  boost: 2
---

## Overview

Griptape provides a set of official Tools for common tasks. In general, Griptape-included Tools are designed to be general-purpose and Driver-based for easy integration with different backends.

## Tools

### Audio Transcription

This Tool enables [Agents](../../../griptape-framework/structures/agents.md) to transcribe speech from text using [Audio Transcription Drivers](../../../reference/griptape/drivers/audio_transcription/base_audio_transcription_driver.md).

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/official-tools/src/audio_transcription_tool_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/official-tools/logs/audio_transcription_tool_1.txt"
    ```


### Calculator

This tool enables LLMs to make simple calculations.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/official-tools/src/calculator_tool_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/official-tools/logs/calculator_tool_1.txt"
    ```


```
[09/08/23 14:23:51] INFO     Task bbc6002a5e5b4655bb52b6a550a1b2a5              
                             Input: What is 10 raised to the power of 5?        
[09/08/23 14:23:56] INFO     Subtask 3e9211a0f44c4277812ae410c43adbc9           
                             Thought: The question is asking for the result of  
                             10 raised to the power of 5. This is a mathematical
                             operation that can be performed using the          
                             CalculatorTool tool.                                   
                                                                                
                             Action: {"name": "CalculatorTool",     
                             "path": "calculate", "input": {"values":       
                             {"expression": "10**5"}}}                          
                    INFO     Subtask 3e9211a0f44c4277812ae410c43adbc9           
                             Response: 100000                                
[09/08/23 14:23:58] INFO     Task bbc6002a5e5b4655bb52b6a550a1b2a5              
                             Output: 10 raised to the power of 5 is 100000.  
```

### Computer

This tool enables LLMs to execute Python code and run shell commands inside a Docker container. You have to have the Docker daemon running in order for this tool to work.

You can specify a local working directory and environment variables during tool initialization:

```python
--8<-- "docs/griptape-framework/tools/official-tools/src/computer_tool_1.py"
```

```
❮ uv run python src/docs/task-memory.py
[08/12/24 15:13:56] INFO     PromptTask 203ee958d1934811afe0bb86fb246e86
                             Input: Make 2 files and then list the files in the current directory
[08/12/24 15:13:58] INFO     Subtask eb4e843b6f37498f9f0e85ada68114ac
                             Actions: [
                               {
                                 "tag": "call_S17vPQsMCqWY1Lt5x8NtDnTK",
                                 "name": "Computer",
                                 "path": "execute_command",
                                 "input": {
                                   "values": {
                                     "command": "touch file1.txt file2.txt"
                                   }
                                 }
                               }
                             ]
                    INFO     Subtask eb4e843b6f37498f9f0e85ada68114ac
                             Response: Tool returned an empty value
[08/12/24 15:13:59] INFO     Subtask 032770e7697d44f6a0c8559bfea60420
                             Actions: [
                               {
                                 "tag": "call_n61SVDYUGWTt681BaDSaHgt1",
                                 "name": "Computer",
                                 "path": "execute_command",
                                 "input": {
                                   "values": {
                                     "command": "ls"
                                   }
                                 }
                               }
                             ]
                    INFO     Subtask 032770e7697d44f6a0c8559bfea60420
                             Response: file1.txt
                             file2.txt
[08/12/24 15:14:00] INFO     PromptTask 203ee958d1934811afe0bb86fb246e86
                             Output: file1.txt, file2.txt
```

### Date Time

This tool enables LLMs to get current date and time.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/official-tools/src/date_time_tool_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/official-tools/logs/date_time_tool_1.txt"
    ```


```
[09/11/23 15:26:02] INFO     Task d0bf49dacd8849e695494578a333f6cc              
                             Input: {'description': 'What is the current date   
                             and time?'}                                        
[09/11/23 15:26:06] INFO     Subtask 1c6c8d43926d4eff81992886301d5655           
                             Thought: The user wants to know the current date   
                             and time. I can use the DateTime tool with the     
                             get_current_datetime activity to find this         
                             information.                                       
                                                                                
                             Action: {"name": "DateTime",       
                             "path": "get_current_datetime"}                
                    INFO     Subtask 1c6c8d43926d4eff81992886301d5655           
                             Response: 2023-09-11 15:26:06.767997            
[09/11/23 15:26:08] INFO     Task d0bf49dacd8849e695494578a333f6cc              
                             Output: The current date and time is September 11, 
                             2023, 15:26:06.
```

### Email

The [EmailTool](../../../reference/griptape/tools/email/tool.md) enables LLMs to send emails.

```python
--8<-- "docs/griptape-framework/tools/official-tools/src/email_tool_1.py"
```

For debugging purposes, you can run a local SMTP server that the LLM can send emails to:

```shell
python -m smtpd -c DebuggingServer -n localhost:1025
```

### Extraction

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

### File Manager

This tool enables LLMs to save and load files.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/official-tools/src/file_manager_tool_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/official-tools/logs/file_manager_tool_1.txt"
    ```


```
[09/12/23 12:07:56] INFO     Task 16a1ce1847284ae3805485bad7d99116              
                             Input: Can you get me the sample1.txt file?        
[09/12/23 12:08:04] INFO     Subtask ddcf48d970ce4edbbc22a46b2f83ec4f           
                             Thought: The user wants the content of the file    
                             named "sample1.txt". I can use the FileManagerTool tool
                             with the activity "load_files_from_disk" to load   
                             the file from the disk.                            
                                                                                
                             Action: {"name": "FileManagerTool",    
                             "path": "load_files_from_disk", "input":       
                             {"values": {"paths": ["sample1.txt"]}}}            
                    INFO     Subtask ddcf48d970ce4edbbc22a46b2f83ec4f           
                             Response:                                       
                             [BlobArtifact(id='a715cc1bc6724bf28566a5b3c343b6ed'
                             , name='sample1.txt', type='BlobArtifact',         
                             value=b'This is the content of sample1.txt',       
                             dir='')]                                           
[09/12/23 12:08:10] INFO     Task 16a1ce1847284ae3805485bad7d99116              
                             Output: The content of the file "sample1.txt" is   
                             "This is the content of sample1.txt". 
```

### Griptape Cloud Tool

The [GriptapeCloudToolTool](../../../reference/griptape/tools/griptape_cloud_tool/tool.md) integrates with Griptape Cloud's hosted Tools.

**Note:** This tool requires a [Tool](https://cloud.griptape.ai/tools) hosted in Griptape Cloud and an [API Key](https://cloud.griptape.ai/configuration/api-keys) for access.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/official-tools/src/griptape_cloud_tool_tool.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/official-tools/logs/griptape_cloud_tool_tool.txt"
    ```


### Image Query

This tool allows Agents to execute natural language queries on the contents of images using multimodal models.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/official-tools/src/image_query_tool_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/official-tools/logs/image_query_tool_1.txt"
    ```


### Inpainting Image Generation

This tool allows LLMs to generate images using inpainting, where an input image is altered within the area specified by a mask image according to a prompt. The input and mask images can be provided either by their file path or by their [Task Memory](../../../griptape-framework/structures/task-memory.md) references.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/official-tools/src/inpainting_image_generation_tool_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/official-tools/logs/inpainting_image_generation_tool_1.txt"
    ```


### Outpainting Image Generation

This tool allows LLMs to generate images using outpainting, where an input image is altered outside of the area specified by a mask image according to a prompt. The input and mask images can be provided either by their file path or by their [Task Memory](../../../griptape-framework/structures/task-memory.md) references.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/official-tools/src/outpainting_image_generation_tool_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/official-tools/logs/outpainting_image_generation_tool_1.txt"
    ```


### Prompt Image Generation

This tool allows LLMs to generate images from a text prompt.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/official-tools/src/prompt_image_generation_tool_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/official-tools/logs/prompt_image_generation_tool_1.txt"
    ```


### Prompt Summary

The [PromptSummaryTool](../../../reference/griptape/tools/prompt_summary/tool.md) enables LLMs summarize text data.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/official-tools/src/prompt_summary_tool_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/official-tools/logs/prompt_summary_tool_1.txt"
    ```


````
[08/12/24 15:54:46] INFO     PromptTask 8be73eb542c44418ba880399044c017a
                             Input: How can I build Neovim from source for MacOS according to this https://github.com/neovim/neovim/blob/master/BUILD.md
[08/12/24 15:54:47] INFO     Subtask cd362a149e1d400997be93c1342d1663
                             Actions: [
                               {
                                 "tag": "call_DGsOHC4AVxhV7RPVA7q3rATX",
                                 "name": "WebScraperTool",
                                 "path": "get_content",
                                 "input": {
                                   "values": {
                                     "url": "https://github.com/neovim/neovim/blob/master/BUILD.md"
                                   }
                                 }
                               }
                             ]
[08/12/24 15:54:49] INFO     Subtask cd362a149e1d400997be93c1342d1663
                             Response: Output of "WebScraperTool.get_content" was stored in memory with memory_name "TaskMemory" and artifact_namespace
                             "990b689c57de4581b8715963c0aecfe3"
[08/12/24 15:54:50] INFO     Subtask 919a4a9eb900439ab9bfbf6e921feba3
                             Actions: [
                               {
                                 "tag": "call_DK3a4MYoElJbaCrUJekBReIc",
                                 "name": "PromptSummaryTool",
                                 "path": "summarize",
                                 "input": {
                                   "values": {
                                     "summary": {
                                       "memory_name": "TaskMemory",
                                       "artifact_namespace": "990b689c57de4581b8715963c0aecfe3"
                                     }
                                   }
                                 }
                               }
                             ]
[08/12/24 15:54:56] INFO     Subtask 919a4a9eb900439ab9bfbf6e921feba3
                             Response: The text provides detailed instructions for building and installing Neovim from source on various operating systems, including Linux,
                             BSD, Windows, and macOS. Key points include:

                             1. **Pre-requisites**: Ensure all build dependencies are installed.
                             2. **Cloning and Building**:
                                - Clone the Neovim repository.
                                - Use `make` with appropriate `CMAKE_BUILD_TYPE` (Release, Debug, RelWithDebInfo).
                                - For stable releases, checkout the stable branch.
                                - Use `gmake` on BSD and follow specific instructions for Windows using MSVC.
                             3. **Installation**:
                                - Default install location is `/usr/local`.
                                - On Debian/Ubuntu, build a DEB package for clean installation.
                                - Custom install locations can be set using `CMAKE_INSTALL_PREFIX`.
                             4. **Dependencies**:
                                - Third-party dependencies are automatically downloaded.
                                - Use `Ninja` and `ccache` for faster builds.
                             5. **Platform-Specific Instructions**:
                                - Detailed steps for building on Windows using Visual Studio.
                                - Instructions for building on macOS using Homebrew or MacPorts.
                                - Specific commands for building on NixOS, Cygwin, and MSYS2.
                             6. **Debugging and Development**:
                                - Use `RelWithDebInfo` for optimized builds with debug info.
                                - Instructions for using debuggers and handling build cache.
                             7. **Localization**:
                                - Commands for checking and updating translations.

                             The text also includes troubleshooting tips and links to additional resources for further guidance.
[08/12/24 15:55:01] INFO     PromptTask 8be73eb542c44418ba880399044c017a
                             Output: To build Neovim from source on macOS, follow these steps:

                             1. **Install Dependencies**:
                                Ensure you have the necessary build dependencies installed. You can use Homebrew to install them:
                                ```sh
                                brew install ninja libtool automake cmake pkg-config gettext curl
                                ```

                             2. **Clone the Neovim Repository**:
                                Clone the Neovim repository from GitHub:
                                ```sh
                                git clone https://github.com/neovim/neovim.git
                                cd neovim
                                ```

                             3. **Checkout the Stable Branch (Optional)**:
                                If you want to build the stable release, checkout the stable branch:
                                ```sh
                                git checkout stable
                                ```

                             4. **Build Neovim**:
                                Use `make` to build Neovim. You can specify the build type (Release, Debug, RelWithDebInfo):
                                ```sh
                                make CMAKE_BUILD_TYPE=Release
                                ```

                             5. **Install Neovim**:
                                After building, install Neovim. The default install location is `/usr/local`:
                                ```sh
                                sudo make install
                                ```

                             By following these steps, you should be able to build and install Neovim from source on macOS. For more detailed instructions and
                             troubleshooting tips, refer to the [BUILD.md](https://github.com/neovim/neovim/blob/master/BUILD.md) file in the Neovim repository.
````

### Query

The [QueryTool](../../../reference/griptape/tools/query/tool.md) enables Agents to query unstructured data for specific information.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/official-tools/src/query_tool_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/official-tools/logs/query_tool_1.txt"
    ```


```
[08/12/24 15:49:23] INFO     PromptTask a88abda2e5324bdf81a3e2b99c26b9df
                             Input: Tell me about the architecture as described here: https://neovim.io/doc/user/vim_diff.html
[08/12/24 15:49:24] INFO     Subtask 3dc9910bcac44c718b3aedd6222e372a
                             Actions: [
                               {
                                 "tag": "call_VY4r5YRc2QDjtBvn89z5PH8E",
                                 "name": "WebScraperTool",
                                 "path": "get_content",
                                 "input": {
                                   "values": {
                                     "url": "https://neovim.io/doc/user/vim_diff.html"
                                   }
                                 }
                               }
                             ]
[08/12/24 15:49:25] INFO     Subtask 3dc9910bcac44c718b3aedd6222e372a
                             Response: Output of "WebScraperTool.get_content" was stored in memory with memory_name "TaskMemory" and artifact_namespace
                             "bec6deeac5f84e369c41210e67905415"
[08/12/24 15:49:26] INFO     Subtask f41d2189ecff4458acb8e6dadb5b13aa
                             Actions: [
                               {
                                 "tag": "call_GtBICZi6oIeL85Aj7q5szul9",
                                 "name": "QueryTool",
                                 "path": "query",
                                 "input": {
                                   "values": {
                                     "query": "architecture",
                                     "content": {
                                       "memory_name": "TaskMemory",
                                       "artifact_namespace": "bec6deeac5f84e369c41210e67905415"
                                     }
                                   }
                                 }
                               }
                             ]
[08/12/24 15:49:33] INFO     Subtask f41d2189ecff4458acb8e6dadb5b13aa
                             Response: The architecture of Neovim (Nvim) is designed to improve stability, performance, and extensibility. Here are some key points about
                             Nvim's architecture:

                             1. **Decoupled UI**: The Nvim UI is decoupled from the core editor. All UIs, including the built-in TUI (terminal user interface), are plugins
                             that connect to a Nvim server. Multiple Nvim UI clients can connect to the same Nvim editor server.

                             2. **External Plugins**: External plugins run in separate processes, which improves stability and allows those plugins to work without blocking
                             the editor. Even "legacy" Python and Ruby plugins, which use the old Vim interfaces, run out-of-process, so they cannot crash Nvim.

                             3. **Libuv**: Platform and I/O facilities are built upon libuv. Nvim benefits from libuv features and bug fixes, and other projects benefit from
                             improvements to libuv by Nvim developers.

                             4. **Robust API**: Nvim has a robust API, which is used instead of exposing internal test functions like Vim's `test_autochdir()`,
                             `test_settime()`, etc.

                             5. **Feature Inclusion**: Nvim always includes all features, in contrast to Vim, which ships various combinations of 100+ optional features.
                             This reduces the surface area for bugs and removes a common source of confusion and friction for users.

                             6. **External Plugins and Extensions**: Nvim avoids features that cannot be provided on all platforms, delegating those to external
                             plugins/extensions.

                             These architectural decisions make Nvim more stable, extensible, and user-friendly compared to traditional Vim.
[08/12/24 15:49:37] INFO     PromptTask a88abda2e5324bdf81a3e2b99c26b9df
                             Output: The architecture of Neovim (Nvim) is designed to enhance stability, performance, and extensibility. Here are the key points:

                             1. **Decoupled UI**: The user interface (UI) is separated from the core editor. All UIs, including the built-in terminal user interface (TUI),
                             are plugins that connect to a Nvim server. This allows multiple UI clients to connect to the same Nvim editor server.

                             2. **External Plugins**: Plugins run in separate processes, which improves stability and prevents them from blocking the editor. Even older
                             Python and Ruby plugins run out-of-process, ensuring they cannot crash Nvim.

                             3. **Libuv**: Nvim's platform and I/O facilities are built on libuv, benefiting from its features and bug fixes. Improvements made by Nvim
                             developers to libuv also benefit other projects.

                             4. **Robust API**: Nvim provides a robust API, avoiding the need to expose internal test functions like Vim does.

                             5. **Feature Inclusion**: Unlike Vim, which ships with various combinations of optional features, Nvim includes all features by default. This
                             reduces bugs and user confusion.

                             6. **External Plugins and Extensions**: Nvim delegates features that cannot be provided on all platforms to external plugins/extensions.

                             These architectural choices make Nvim more stable, extensible, and user-friendly compared to traditional Vim.
```

### Rag

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

### Rest Api

This tool enables LLMs to call REST APIs.

The [RestApiTool](../../../reference/griptape/tools/rest_api/tool.md) tool uses the following parameters:

The following example is built using [https://jsonplaceholder.typicode.com/guide/](https://jsonplaceholder.typicode.com/guide/).

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/official-tools/src/rest_api_tool_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/official-tools/logs/rest_api_tool_1.txt"
    ```


### Sql

This tool enables LLMs to execute SQL statements via [SQLAlchemy](https://www.sqlalchemy.org/). Depending on your underlying SQL engine, [configure](https://docs.sqlalchemy.org/en/20/core/engines.html) your `engine_url` and give the LLM a hint about what engine you are using via `engine_name`, so that it can create engine-specific statements.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/official-tools/src/sql_tool_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/official-tools/logs/sql_tool_1.txt"
    ```


```
[08/12/24 14:59:31] INFO     PromptTask e302f7315d1a4f939e0125103ff4f09f
                             Input: SELECT * FROM people;
[08/12/24 14:59:34] INFO     Subtask 809d1a281b85447f90706d431b77b845
                             Actions: [
                               {
                                 "tag": "call_dCxHWwPwgmDvDKVd3QeOzyuT",
                                 "name": "SqlClient",
                                 "path": "execute_query",
                                 "input": {
                                   "values": {
                                     "sql_query": "SELECT * FROM people"
                                   }
                                 }
                               }
                             ]
[08/12/24 14:59:35] INFO     Subtask 809d1a281b85447f90706d431b77b845
                             Response: 1,Lee,Andrews,"Engineer, electrical"

                             2,Michael,Woods,"Therapist, art"

                             3,Joshua,Allen,"Therapist, sports"

                             4,Eric,Foster,English as a second language teacher

                             5,John,Daniels,Printmaker

                             6,Matthew,Barton,Podiatrist

                             7,Audrey,Wilson,IT technical support officer

                             8,Leah,Knox,"Social research officer, government"

                             9,David,Macdonald,Public relations account executive

                             10,Erica,Ramos,"Accountant, chartered public finance"
[08/12/24 14:59:43] INFO     PromptTask e302f7315d1a4f939e0125103ff4f09f
                             Output:
                             1. Lee Andrews - Engineer, electrical
                             2. Michael Woods - Therapist, art
                             3. Joshua Allen - Therapist, sports
                             4. Eric Foster - English as a second language teacher
                             5. John Daniels - Printmaker
                             6. Matthew Barton - Podiatrist
                             7. Audrey Wilson - IT technical support officer
                             8. Leah Knox - Social research officer, government
                             9. David Macdonald - Public relations account executive
                             10. Erica Ramos - Accountant, chartered public finance
```

### Structure Run

The [StructureRunTool](../../../reference/griptape/tools/structure_run/tool.md) Tool provides a way to run Structures via a Tool.
It requires you to provide a [Structure Run Driver](../../../griptape-framework/drivers/structure-run-drivers.md) to run the Structure in the desired environment.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/official-tools/src/structure_run_tool_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/official-tools/logs/structure_run_tool_1.txt"
    ```


```
[05/02/24 13:50:03] INFO     PromptTask 4e9458375bda4fbcadb77a94624ed64c
                             Input: what is modular RAG?
[05/02/24 13:50:10] INFO     Subtask 5ef2d72028fc495aa7faf6f46825b004
                             Thought: To answer this question, I need to run a search for the term "modular RAG". I will use the StructureRunTool action to execute a
                             search structure.
                             Actions: [
                               {
                                 "name": "StructureRunTool",
                                 "path": "run_structure",
                                 "input": {
                                   "values": {
                                     "args": "modular RAG"
                                   }
                                 },
                                 "tag": "search_modular_RAG"
                               }
                             ]
[05/02/24 13:50:36] INFO     Subtask 5ef2d72028fc495aa7faf6f46825b004
                             Response: {'id': '87fa21aded76416e988f8bf39c19760b', 'name': '87fa21aded76416e988f8bf39c19760b', 'type': 'TextArtifact', 'value': 'Modular
                             Retrieval-Augmented Generation (RAG) is an advanced approach that goes beyond the traditional RAG paradigms, offering enhanced adaptability
                             and versatility. It involves incorporating diverse strategies to improve its components by adding specialized modules for retrieval and
                             processing capabilities. The Modular RAG framework allows for module substitution or reconfiguration to address specific challenges, expanding
                             flexibility by integrating new modules or adjusting interaction flow among existing ones. This approach supports both sequential processing
                             and integrated end-to-end training across its components, illustrating progression and refinement within the RAG family.'}
[05/02/24 13:50:44] INFO     PromptTask 4e9458375bda4fbcadb77a94624ed64c
                             Output: Modular Retrieval-Augmented Generation (RAG) is an advanced approach that goes beyond the traditional RAG paradigms, offering enhanced
                             adaptability and versatility. It involves incorporating diverse strategies to improve its components by adding specialized modules for
                             retrieval and processing capabilities. The Modular RAG framework allows for module substitution or reconfiguration to address specific
                             challenges, expanding flexibility by integrating new modules or adjusting interaction flow among existing ones. This approach supports both
                             sequential processing and integrated end-to-end training across its components, illustrating progression and refinement within the RAG family.
```

### Text To Speech

This Tool enables LLMs to synthesize speech from text using [Text to Speech Drivers](../../../reference/griptape/drivers/text_to_speech/index.md).

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/official-tools/src/text_to_speech_tool_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/official-tools/logs/text_to_speech_tool_1.txt"
    ```


### Variation Image Generation

This Tool allows LLMs to generate variations of an input image from a text prompt. The input image can be provided either by its file path or by its [Task Memory](../../../griptape-framework/structures/task-memory.md) reference.

#### Referencing an Image by File Path

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/official-tools/src/variation_image_generation_tool_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/official-tools/logs/variation_image_generation_tool_1.txt"
    ```


#### Referencing an Image in Task Memory

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/official-tools/src/variation_image_generation_tool_2.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/official-tools/logs/variation_image_generation_tool_2.txt"
    ```


### Vector Store

The [VectorStoreTool](../../../reference/griptape/tools/vector_store/tool.md) enables LLMs to query vector stores.

Here is an example of how it can be used with a local vector store driver:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/official-tools/src/vector_store_tool_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/official-tools/logs/vector_store_tool_1.txt"
    ```


### Web Scraper

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

### Web Search

This tool enables LLMs to search the web.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/official-tools/src/web_search_tool_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/official-tools/logs/web_search_tool_1.txt"
    ```


```
[09/08/23 15:37:25] INFO     Task 2cf557f7f7cd4a20a7fa2f0c46af2f71              
                             Input: Tell me how photosynthesis works            
[09/08/23 15:37:32] INFO     Subtask d023ef9f41d142229513510cf4f47afe           
                             Thought: I know that photosynthesis is a process   
                             used by plants and other organisms to convert light
                             energy into chemical energy that can later be      
                             released to fuel the organisms' activities.        
                             However, to provide a detailed explanation, I will 
                             need to conduct a web search.                      
                                                                                
                             Action: {"name": "WebSearch",      
                             "path": "search", "input": {"values": {"query":
                             "How does photosynthesis work?"}}}                 
                    INFO     Subtask d023ef9f41d142229513510cf4f47afe           
                             Response: {'url':                               
                             'https://www.nationalgeographic.org/encyclopedia/ph
                             otosynthesis/', 'title': 'Photosynthesis',         
                             'description': 'Jul 15, 2022 ... Photosynthesis is 
                             the process by which plants use sunlight, water,   
                             and carbon dioxide to create oxygen and energy in  
                             the form of sugar.'}                               
                             {'url':                                            
                             'https://www.snexplores.org/article/explainer-how-p
                             hotosynthesis-works', 'title': 'Explainer: How     
                             photosynthesis works', 'description': 'Oct 28, 2020
                             ... Photosynthesis is the process of creating sugar
                             and oxygen from carbon dioxide, water and sunlight.
                             It happens through a long series of                
                             chemical\xa0...'}                                  
                             {'url':                                            
                             'https://www.sciencefocus.com/nature/how-does-photo
                             synthesis-work', 'title': 'Photosynthesis: What is 
                             it and how does it work? - BBC Science ...',       
                             'description': "Jul 27, 2022 ... Photosynthesis is 
                             the process by which carbohydrate molecules are    
                             synthesised. It's used by plants, algae and certain
                             bacteriato turn sunlight,\xa0..."}                 
                             {'url': 'https://oregonforests.org/photosynthesis',
                             'title': 'Photosynthesis | OregonForests',         
                             'description': "Here's how it works: Tree and plant
                             roots absorb water, as well as minerals and        
                             nutrients, from the soil. At the same time, the    
                             leaves or needles absorb carbon\xa0..."}           
                             {'url':                                            
                             'https://ssec.si.edu/stemvisions-blog/what-photosyn
                             thesis', 'title': 'What is Photosynthesis |        
                             Smithsonian Science Education Center',             
                             'description': 'Apr 12, 2017 ... Rather, plants use
                             sunlight, water, and the gases in the air to make  
                             glucose, which is a form of sugar that plants need 
                             to survive. This process\xa0...'}                  
[09/08/23 15:37:50] INFO     Task 2cf557f7f7cd4a20a7fa2f0c46af2f71              
                             Output: Photosynthesis is the process by which     
                             plants, algae, and certain bacteria convert light  
                             energy, usually from the sun, into chemical energy 
                             in the form of glucose or sugar. This process      
                             involves several steps:                            
                                                                                
                             1. Absorption of light: The process begins when    
                             light is absorbed by proteins containing           
                             chlorophylls (pigments) present in chloroplasts.   
                                                                                
                             2. Conversion of light energy to chemical energy:  
                             The absorbed light energy is used to convert carbon
                             dioxide from the atmosphere and water from the soil
                             into glucose. This conversion process occurs       
                             through a series of chemical reactions known as the
                             light-dependent reactions and the Calvin cycle.    
                                                                                
                             3. Release of oxygen: As a byproduct of these      
                             reactions, oxygen is produced and released into the
                             atmosphere.                                        
                                                                                
                             4. Use of glucose: The glucose produced is used by 
                             the plant for growth and development. It can also  
                             be stored for later use.                           
                                                                                
                             In summary, photosynthesis is a vital process for  
                             life on Earth as it is the primary source of oxygen
                             in the atmosphere and forms the basis of the food  
                             chain.      
```

Extra schema properties can be added to the Tool to allow for more customization if the Driver supports it.
In this example, we add a `sort` property to the `search` Activity which will be added as a [Google custom search query parameter](https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list).

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/official-tools/src/web_search_tool_2.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/official-tools/logs/web_search_tool_2.txt"
    ```

