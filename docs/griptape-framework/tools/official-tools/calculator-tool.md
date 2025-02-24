# Calculator Tool

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
