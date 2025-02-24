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
