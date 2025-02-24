# Sql Tool

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
