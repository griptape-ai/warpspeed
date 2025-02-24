# File Manager Tool

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
