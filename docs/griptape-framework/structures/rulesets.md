---
search:
  boost: 2
---

## Overview

A [Ruleset](../../reference/griptape/rules/ruleset.md) can be used to define [Rule](../../reference/griptape/rules/base_rule.md)s for [Structures](../structures/agents.md) and [Tasks](../structures/tasks.md). Griptape places Rules into the LLM's system prompt for strong control over the output.

## Types of Rules

### Rule

[Rule](../../reference/griptape/rules/base_rule.md)s shape the LLM's behavior by defining specific guidelines or instructions for how it should interpret and respond to inputs. Rules can be used to modify language style, tone, or even behavior based on what you define.

!!! tip

    Avoid writing large amounts of text in a single Rule.
    Breaking down your Rules generally helps the LLM follow them more effectively. Additionally, it makes it easier to evaluate the Rule's effectiveness using tools like the [Eval Engine](../engines/eval-engines.md).
    If you have an existing system prompt, consider [overriding the default system prompt](#overriding-system-prompts) instead.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/basic_rule.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/basic_rule.txt"
    ```


```
[09/10/24 14:41:52] INFO     PromptTask b7b23a88ea9e4cd0befb7e7a4ed596b0
                             Input: Hi there! How are you?
                    INFO     PromptTask b7b23a88ea9e4cd0befb7e7a4ed596b0
                             Output: Ahoy, matey! I be doing just fine, thank ye fer askin'. How be the winds blowin' in yer sails today?
```

### Json Schema

!!! tip

    [Structured Output](../drivers/prompt-drivers.md#structured-output) provides a more robust solution for having the LLM generate structured output.

[JsonSchemaRule](../../reference/griptape/rules/json_schema_rule.md)s defines a structured format for the LLM's output by providing a JSON schema.
This is particularly useful when you need the LLM to return well-formed data, such as JSON objects, with specific fields and data types.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/json_schema_rule.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/json_schema_rule.txt"
    ```


```
[09/10/24 14:44:53] INFO     PromptTask fb26dd41803443c0b51c3d861626e07a
                             Input: What is the sentiment of this message?: 'I am so happy!'
[09/10/24 14:44:54] INFO     PromptTask fb26dd41803443c0b51c3d861626e07a
                             Output: {
                               "answer": "The sentiment of the message is positive.",
                               "relevant_emojis": ["😊", "😃"]
                             }
```

Although Griptape leverages the `schema` library, you're free to use any JSON schema generation library to define your schema!

For example, using `pydantic`:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/json_schema_rule_pydantic.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/json_schema_rule_pydantic.txt"
    ```


```
[09/11/24 09:45:58] INFO     PromptTask eae43f52829c4289a6cca9ee7950e075
                             Input: What is the sentiment of this message?: 'I am so happy!'
                    INFO     PromptTask eae43f52829c4289a6cca9ee7950e075
                             Output: {
                               "answer": "The sentiment of the message is positive.",
                               "relevant_emojis": ["😊", "😄"]
                             }
answer='The sentiment of the message is positive.' relevant_emojis=['😊', '😄']
```

## Structure

### Rulesets

You can define a Ruleset at the Structure level if you need to have certain behaviors across all Tasks.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/rulesets_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/rulesets_1.txt"
    ```


```
[09/29/23 13:44:35] INFO     PromptTask 0ecf932b1602493781485de37028f1df
                             Input: Respond to this user's question: How do I do a kickflip?
[09/29/23 13:44:41] INFO     PromptTask 0ecf932b1602493781485de37028f1df
                             Output: Hello! This is Todd from SkaterWorld, Inc. To do a kickflip, you'll need to place
                             your back foot on the tail and your front foot across the skateboard. Push down on the tail
                             while dragging the edge of your front foot up the board. Then, jump and flick your front foot
                             out to the side. Practice makes perfect, so keep trying!
                    INFO     PromptTask 1f7f5c0af17240dc8cb785d7efbdbfb6
                             Input: Extract keywords from this response: Hello! This is Todd from SkaterWorld, Inc. To do
                             a kickflip, you'll need to place your back foot on the tail and your front foot across the
                             skateboard. Push down on the tail while dragging the edge of your front foot up the board.
                             Then, jump and flick your front foot out to the side. Practice makes perfect, so keep trying!
[09/29/23 13:44:46] INFO     PromptTask 1f7f5c0af17240dc8cb785d7efbdbfb6
                             Output: The keywords from the response are: Todd, SkaterWorld, Inc., kickflip, back foot,
                             tail, front foot, skateboard, push down, dragging, edge, board, jump, flick, side, practice,
                             trying.
```

### Rules

You can pass [rules](../../reference/griptape/mixins/rule_mixin.md#griptape.mixins.rule_mixin.RuleMixin.rules) directly to the Structure to have a Ruleset created for you.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/rulesets_2.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/rulesets_2.txt"
    ```


```
[09/29/23 13:31:41] INFO     PromptTask 51c0030b7a854ae5a9bef4595014915c
                             Input: Respond to this question from the user: 'How do I bake a cake?'
[09/29/23 13:31:45] INFO     PromptTask 51c0030b7a854ae5a9bef4595014915c
                             Output: 📖🥣🥚🥛🍚🧈🍰🔥⏲️👩‍🍳🎂
                    INFO     PromptTask 9ea16d8e79d84cbab9823e234b51f013
                             Input: How would you rate your response (1-5)? 1 being bad, 5 being good. Response:
                             '📖🥣🥚🥛🍚🧈🍰🔥⏲️👩‍🍳🎂'
[09/29/23 13:31:46] INFO     PromptTask 9ea16d8e79d84cbab9823e234b51f013
                             Output: 👍👍👍👍👍
```

## Task

### Rulesets

You can define a Ruleset at the Task level if you need to have different behaviors per Task.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/rulesets_3.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/rulesets_3.txt"
    ```


```
[09/29/23 13:39:22] INFO     PromptTask 0950581dd35e403c9fc51d246861bfc9
                             Input: Respond to the following prompt: I love skateboarding!
[09/29/23 13:39:24] INFO     PromptTask 0950581dd35e403c9fc51d246861bfc9
                             Output: THAT'S AWESOME! 😃 KEEP UP THE GOOD WORK AND ALWAYS STAY SAFE! 🛹👍👏
                    INFO     PromptTask 325f9b7acaca47a2a097d322288e1bfa
                             Input: Determine the sentiment of the following text: THAT'S AWESOME! 😃 KEEP UP THE GOOD
                             WORK AND ALWAYS STAY SAFE! 🛹👍👏
[09/29/23 13:39:26] INFO     PromptTask 325f9b7acaca47a2a097d322288e1bfa
                             Output: Thê sêntimênt ôf thê têxt is pôsitivê.
```

### Rules

You can pass [rules](../../reference/griptape/mixins/rule_mixin.md#griptape.mixins.rule_mixin.RuleMixin.rules) directly to the Task to have a Ruleset created for you.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/rulesets_4.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/rulesets_4.txt"
    ```


```
[09/25/23 16:29:05] INFO     PromptTask d1cc2c0b780d4b32b6309ceab11173f4
                             Input: How are you?
[09/25/23 16:29:07] INFO     PromptTask d1cc2c0b780d4b32b6309ceab11173f4
                             Output: {
                               "emoji_response": "😊👍"
                             }
```

## Overriding System Prompts

While Rulesets are the preferred way to steer LLM output, sometimes you may want to fully override the system prompt.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/generate_system_prompt.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/generate_system_prompt.txt"
    ```


Note that when overriding the system prompt, it is your responsibility to integrate anything that goes in [by default](https://github.com/griptape-ai/griptape/blob/6b31c129fc19a9ba4bdb205ad9e2a40aef9b121f/griptape/tasks/prompt_task.py?plain=1#L216-L221).
You can achieve this by appending the default system prompt to your custom prompt like so:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/generate_system_prompt_with_rules.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/generate_system_prompt_with_rules.txt"
    ```

