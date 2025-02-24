# Variation Image Generation Tool

This Tool allows LLMs to generate variations of an input image from a text prompt. The input image can be provided either by its file path or by its [Task Memory](../../../griptape-framework/structures/task-memory.md) reference.

## Referencing an Image by File Path

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/official-tools/src/variation_image_generation_tool_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/official-tools/logs/variation_image_generation_tool_1.txt"
    ```


## Referencing an Image in Task Memory

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/official-tools/src/variation_image_generation_tool_2.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/official-tools/logs/variation_image_generation_tool_2.txt"
    ```

