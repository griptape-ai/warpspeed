# Inpainting Image Generation Tool

This tool allows LLMs to generate images using inpainting, where an input image is altered within the area specified by a mask image according to a prompt. The input and mask images can be provided either by their file path or by their [Task Memory](../../../griptape-framework/structures/task-memory.md) references.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/official-tools/src/inpainting_image_generation_tool_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/official-tools/logs/inpainting_image_generation_tool_1.txt"
    ```

