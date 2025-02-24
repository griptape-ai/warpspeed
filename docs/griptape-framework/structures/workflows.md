---
search:
  boost: 2
---

## Overview

A [Workflow](../../reference/griptape/structures/workflow.md) is a non-sequential DAG that can be used for complex concurrent scenarios with tasks having multiple inputs.

You can access the final output of the Workflow by using the [output](../../reference/griptape/structures/structure.md#griptape.structures.structure.Structure.output) attribute.

## Context

Workflows have access to the following [context](../../reference/griptape/structures/workflow.md#griptape.structures.workflow.Workflow.context) variables in addition to the [base context](./tasks.md#context):

- `task_outputs`: dictionary containing mapping of all task IDs to their outputs.
- `parent_outputs`: dictionary containing mapping of parent task IDs to their outputs.
- `parents_output_text`: string containing the concatenated outputs of all parent tasks.
- `parents`: dictionary containing mapping of parent task IDs to their task objects.
- `children`: dictionary containing mapping of child task IDs to their task objects.

## Workflow

Let's build a simple workflow. Let's say, we want to write a story in a fantasy world with some unique characters. We could setup a workflow that generates a world based on some keywords. Then we pass the world description to any number of child tasks that create characters. Finally, the last task pulls in information from all parent tasks and writes up a short story.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/workflows_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/workflows_1.txt"
    ```


Note that we use the `StructureVisualizer` to get a visual representation of the workflow. If we visit the printed url, it should look like this:

![Workflow](https://mermaid.ink/img/Z3JhcGggVEQ7OwoJd29ybGQtLT4gc3RvcnkgJiBzY290dHkgJiBhbm5pZTsKCXNjb3R0eS0tPiBzdG9yeTsKCWFubmllLS0+IHN0b3J5Ow==)

!!! Info

    Output edited for brevity

```
[09/08/23 10:26:21] INFO     PromptTask world
                             Input: Create a fictional world based on the following key words fantasy, ocean, tidal lock
[09/08/23 10:27:11] INFO     PromptTask world
                             Output: Welcome to the world of "Oceanus Fantasia", a realm where fantasy and oceanic wonders intertwine in a
                             unique celestial phenomenon known as tidal lock.

                             Oceanus Fantasia is a vast, water-dominated planet, with its surface covered by a seemingly endless ocean. The
                             ocean is not just water, but a magical liquid that sparkles with all the colors of the rainbow under the planet's
                             twin suns. This magical ocean is home to a myriad of fantastical creatures, from the tiny luminescent pixie-fish
                             to the colossal, wise leviathans that roam the depths.

                             ...

                             In Oceanus Fantasia, magic and technology coexist. The inhabitants use their understanding of the magical ocean
                             and its creatures to power their cities, heal their sick, and even control the weather. The ocean's magic is a
                             part of them, and they are a part of the ocean.

                             Welcome to Oceanus Fantasia, where the ocean's tide writes the story of a world in perpetual twilight, a world of
                             magic, mystery, and endless adventure.
                    INFO     PromptTask scotty
                             Input: Based on the following world description create a character named Scotty:
                             Welcome to the world of "Oceanus Fantasia", a realm where fantasy and oceanic wonders intertwine in a unique
                             celestial phenomenon known as tidal lock.

                             ...

                             In Oceanus Fantasia, magic and technology coexist. The inhabitants use their understanding of the magical ocean
                             and its creatures to power their cities, heal their sick, and even control the weather. The ocean's magic is a
                             part of them, and they are a part of the ocean.

                             Welcome to Oceanus Fantasia, where the ocean's tide writes the story of a world in perpetual twilight, a world of
                             magic, mystery, and endless adventure.
                    INFO     PromptTask annie
                             Input: Based on the following world description create a character named Annie:
                             Welcome to the world of "Oceanus Fantasia", a realm where fantasy and oceanic wonders intertwine in a unique
                             celestial phenomenon known as tidal lock.

                             ...

                             Welcome to Oceanus Fantasia, where the ocean's tide writes the story of a world in perpetual twilight, a world of
                             magic, mystery, and endless adventure.
[09/08/23 10:27:47] INFO     PromptTask scotty
                             Output: Character Name: Scotty

                             Scotty is a Tide Whisperer, one of the mysterious seers who reside in the Twilight Zone of Oceanus Fantasia. He
                             is a tall, slender figure with a complexion that reflects the perpetual twilight of his home, a mix of the golden
                             hue of the Solaris and the luminescent glow of the Lunarians. His hair, a cascade of silver waves, mirrors the
                             ever-changing colors of the magical ocean. His eyes, a deep sea-green, hold a depth that seems to echo the
                             vastness of the ocean itself.

                             ...

                             Scotty embodies the spirit of Oceanus Fantasia, a world of magic, mystery, and endless adventure. He is a beacon
                             of hope and wisdom in a world divided by light and darkness, a testament to the unity and harmony that can exist
                             in diversity.
[09/08/23 10:28:47] INFO     PromptTask annie
                             Output: Character Name: Annie

                             Annie is a young, vibrant inhabitant of the Twilight Zone in Oceanus Fantasia. She is a Tide Whisperer, a seer
                             who can read the future in the ebb and flow of the magical tides. Her skin is a soft, iridescent hue, a blend of
                             the golden tones of the Solaris people and the luminescent glow of the Lunarians, reflecting her unique position
                             between the two cultures.

                             ...

                             Annie's best friend is a luminescent pixie-fish named Lumi. Lumi is a constant companion and often assists Annie
                             in her tide readings. Together, they represent the spirit of Oceanus Fantasia - a world of magic, mystery, and
                             endless adventure.
                    INFO     PromptTask story
                             Input: Based on the following description of the world and characters, write a short story:
                             Welcome to the world of "Oceanus Fantasia", a realm where fantasy and oceanic wonders intertwine in a unique
                             celestial phenomenon known as tidal lock.

                             ...

                             Annie's best friend is a luminescent pixie-fish named Lumi. Lumi is a constant companion and often assists Annie
                             in her tide readings. Together, they represent the spirit of Oceanus Fantasia - a world of magic, mystery, and
                             endless adventure.
[09/08/23 10:29:47] INFO     PromptTask story
                             Output: In the world of Oceanus Fantasia, where the ocean's tide writes the story of a world in perpetual
                             twilight, lived Scotty and Annie, the Tide Whisperers. They were the bridge between the Solaris Kingdom and the
                             Lunarian Clan, the mediators of the perpetual day and eternal night.
                             ...
                             Scotty and Annie, with their deep understanding of the magical ocean and its creatures, continued to guide the
                             inhabitants of Oceanus Fantasia. They embodied the spirit of their world, a world of magic, mystery, and endless
                             adventure. They were the beacon of hope and wisdom in a world divided by light and darkness, a testament to the
                             unity and harmony that can exist in diversity.
```

### Declarative vs Imperative Syntax

The above example showed how to create a workflow using the declarative syntax via the `parent_ids` init param, but there are a number of declarative and imperative options for you to choose between. There is no functional difference, they merely exist to allow you to structure your code as is most readable for your use case. Possibilities are illustrated below.

Declaratively specify parents (same as above example):

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/workflows_2.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/workflows_2.txt"
    ```


Declaratively specify children:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/workflows_3.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/workflows_3.txt"
    ```


Declaratively specifying a mix of parents and children:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/workflows_4.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/workflows_4.txt"
    ```


Imperatively specify parents:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/workflows_5.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/workflows_5.txt"
    ```


Imperatively specify children:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/workflows_6.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/workflows_6.txt"
    ```


Imperatively specify a mix of parents and children:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/workflows_7.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/workflows_7.txt"
    ```


Or even mix imperative and declarative:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/workflows_8.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/workflows_8.txt"
    ```


### Insert Parallel Tasks

`Workflow.insert_tasks()` provides a convenient way to insert parallel tasks between parents and children.

!!! info

    By default, all children are removed from the parent task and all parent tasks are removed from the child task. If you want to keep these parent-child relationships, then set the `preserve_relationship` parameter to `True`.

Imperatively insert parallel tasks between a parent and child:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/structures/src/workflows_9.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/structures/logs/workflows_9.txt"
    ```


output:

```
[06/18/24 09:52:21] INFO     PromptTask animal
                             Input: Name an animal
[06/18/24 09:52:22] INFO     PromptTask animal
                             Output: elephant
                    INFO     PromptTask adjective
                             Input: Describe elephant with an adjective
                    INFO     PromptTask color
                             Input: Describe elephant with a color
                    INFO     PromptTask color
                             Output: gray
                    INFO     PromptTask adjective
                             Output: majestic
                    INFO     PromptTask new-animal
                             Input: Name an animal described as:
                             majestic
                             gray
[06/18/24 09:52:23] INFO     PromptTask new-animal
                             Output: elephant
```

### Bitshift Composition

Task relationships can also be set up with the Python bitshift operators `>>` and `<<`. The following statements are all functionally equivalent:

```python
task1 >> task2
task1.add_child(task2)

task2 << task1
task2.add_parent(task1)

task3 >> [task4, task5]
task3.add_children([task4, task5])
```

When using the bitshift to compose operators, the relationship is set in the direction that the bitshift operator points.
For example, `task1 >> task2` means that `task1` runs first and `task2` runs second.
Multiple operators can be composed – keep in mind the chain is executed left-to-right and the rightmost object is always returned. For example:

```python
task1 >> task2 >> task3 << task4
```

is equivalent to:

```python
task1.add_child(task2)
task2.add_child(task3)
task3.add_parent(task4)
```
