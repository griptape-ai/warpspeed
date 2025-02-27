## Overview

Building your own tools is easy with Griptape!

Tools are nothing more than Python classes that inherit from [BaseTool](../../../reference/griptape/tools/base_tool.md).
Each method in the class is decorated with an [activity](../../../reference/griptape/utils/decorators.md#griptape.utils.decorators.activity) decorator which informs the LLM how and when it should use that Tool Activity.

## Random Number Generator Tool

Here is a simple random number generator Tool:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/custom-tools/src/index_2.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/custom-tools/logs/index_2.txt"
    ```


Check out other [Griptape Tools](https://github.com/griptape-ai/griptape/tree/main/griptape/tools) to learn more about tool implementation details.

## Tool Dependencies

Each Tool can also have its own dependencies. You can specify them in a `requirements.txt` file in the tool directory and Griptape will install them during Tool execution.
To start, create a directory for your Tool inside your project. The directory must have the following structure:

- `tool.py` file with a tool Python class.
- `requirements.txt` file with tool Python dependencies.

That's it! Import and use your Tool in your project as you would with any other Griptape Tool.
