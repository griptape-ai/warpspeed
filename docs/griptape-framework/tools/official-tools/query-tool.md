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
