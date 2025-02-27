---
search:
  boost: 2
---

## Overview

You can configure the global [EventBus](../../reference/griptape/events/event_bus.md) with [EventListener](../../reference/griptape/events/event_listener.md)s to listen for various framework events.
See [Event Listener Drivers](../drivers/event-listener-drivers.md) for examples on forwarding events to external services.

## Specific Event Types

You can listen to specific event types:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/misc/src/events_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/misc/logs/events_1.txt"
    ```


```
<class 'griptape.events.start_task_event.StartTaskEvent'>
[09/08/23 10:51:16] INFO     PromptTask a20c236d1d86480fb14ae976e6cf8983
                             Input: tell me about griptape
<class 'griptape.events.start_prompt_event.StartPromptEvent'>
<class 'griptape.events.finish_prompt_event.FinishPromptEvent'>
[09/08/23 10:51:27] INFO     PromptTask a20c236d1d86480fb14ae976e6cf8983
                             Output: Griptape is a gritty, sandpaper-like material that is applied to the top of a skateboard deck. It
                             provides traction and grip, allowing skateboarders to keep their feet firmly on the board and perform tricks more
                             easily. Griptape comes in different colors and designs, but the most common type is black. It's adhesive on one
                             side so it can stick to the skateboard. Over time, griptape can wear down and need to be replaced to maintain
                             optimal performance. It's an essential component for skateboarding and other similar sports.
<class 'griptape.events.finish_task_event.FinishTaskEvent'>
```

## All Event Types

Or listen to all events:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/misc/src/events_2.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/misc/logs/events_2.txt"
    ```


```
Handler 1 <class 'griptape.events.start_structure_run_event.StartStructureRunEvent'>
Handler 2 <class 'griptape.events.start_structure_run_event.StartStructureRunEvent'>
Handler 1 <class 'griptape.events.start_task_event.StartTaskEvent'>
Handler 2 <class 'griptape.events.start_task_event.StartTaskEvent'>
[10/26/23 11:49:29] INFO     PromptTask 20e3ef1f8856453ebabc84863ac36784
                             Input: tell me about griptape
Handler 1 <class 'griptape.events.start_prompt_event.StartPromptEvent'>
Handler 2 <class 'griptape.events.start_prompt_event.StartPromptEvent'>
Handler 1 <class 'griptape.events.finish_prompt_event.FinishPromptEvent'>
Handler 2 <class 'griptape.events.finish_prompt_event.FinishPromptEvent'>
[10/26/23 11:49:55] INFO     PromptTask 20e3ef1f8856453ebabc84863ac36784
                             Output: Griptape is a gritty, sandpaper-like material that is applied to the top of
                             a skateboard, longboard, or scooter deck to provide traction between the rider's
                             feet and the deck. It is an essential component for performing tricks and
                             maintaining control of the board or scooter.

                             Griptape is typically black, but it comes in a variety of colors and designs. It is
                             made by embedding an abrasive material (similar to sand) into a tough,
                             weather-resistant adhesive. The adhesive side is then applied to the deck, while
                             the abrasive side faces up to provide grip.

                             The grip provided by the griptape allows riders to keep their footing on the board,
                             especially during tricks where the board is flipped or spun. It also helps riders
                             control the board better during downhill rides or sharp turns.

                             Over time, griptape can wear down and lose its effectiveness, at which point it can
                             be removed and replaced. It's an essential part of skateboarding equipment and
                             plays a significant role in the sport's safety and performance.
Handler 1 <class 'griptape.events.finish_task_event.FinishTaskEvent'>
Handler 2 <class 'griptape.events.finish_task_event.FinishTaskEvent'>
Handler 1 <class 'griptape.events.finish_structure_run_event.FinishStructureRunEvent'>
Handler 2 <class 'griptape.events.finish_structure_run_event.FinishStructureRunEvent'>
```

## Stream Iterator

You can use `Structure.run_stream()` for streaming Events from the `Structure` in the form of an iterator.

!!! tip

    Set `stream=True` on your [Prompt Driver](../drivers/prompt-drivers.md) in order to receive completion chunk events.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/misc/src/events_streaming.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/misc/logs/events_streaming.txt"
    ```


## Context Managers

You can also use [EventListener](../../reference/griptape/events/event_listener.md)s as a Python Context Manager.
The `EventListener` will automatically be added and removed from the [EventBus](../../reference/griptape/events/event_bus.md) when entering and exiting the context.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/misc/src/events_context.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/misc/logs/events_context.txt"
    ```


## Streaming

You can use the [BaseChunkEvent](../../reference/griptape/events/base_chunk_event.md) to stream the completion results from Prompt Drivers.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/misc/src/events_3.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/misc/logs/events_3.txt"
    ```


You can also use the [TextChunkEvent](../../reference/griptape/events/text_chunk_event.md) and [ActionChunkEvent](../../reference/griptape/events/action_chunk_event.md) to further differentiate the different types of chunks for more customized output.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/misc/src/events_chunk_stream.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/misc/logs/events_chunk_stream.txt"
    ```


If you want Griptape to handle the chunk events for you, use the [Stream](../../reference/griptape/utils/stream.md) utility to automatically wrap
[BaseChunkEvent](../../reference/griptape/events/base_chunk_event.md)s in a Python iterator.

The `Stream` utility does not automatically enable streaming on the Drivers that produce `BaseChunkEvent`s.
Make sure to enable streaming on the Drivers or else `Stream` will yield no iterations.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/misc/src/events_4.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/misc/logs/events_4.txt"
    ```


## Counting Tokens

To count tokens, you can use Event Listeners and the [TokenCounter](../../reference/griptape/utils/token_counter.md) util:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/misc/src/events_5.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/misc/logs/events_5.txt"
    ```


```
[09/25/23 16:32:41] INFO     PromptTask c93569eb1d264675b52bef184b269621
                             Input: tell me about large language models
[09/25/23 16:33:01] INFO     PromptTask c93569eb1d264675b52bef184b269621
                             Output: Large language models are a type of artificial intelligence model that are trained on
                             a vast amount of text data. They are designed to generate human-like text based on the input
                             they are given. These models can answer questions, write essays, summarize texts, translate
                             languages, and even generate creative content like poetry or stories.

                             One of the most well-known large language models is GPT-3, developed by OpenAI. It has 175
                             billion machine learning parameters and was trained on hundreds of gigabytes of text.

                             These models work by predicting the probability of a word given the previous words used in
                             the text. They don't understand text in the way humans do, but they can generate coherent and
                             contextually relevant sentences by learning patterns in the data they were trained on.

                             However, they also have limitations. For instance, they can sometimes generate incorrect or
                             nonsensical responses. They can also be sensitive to slight changes in input phrasing, and
                             they don't have the ability to fact-check information or access real-time data, so they can't
                             provide up-to-date information or verify the truthfulness of their outputs. They also don't
                             have a sense of ethics or morality, so they rely on guidelines and safety measures put in
                             place by their developers.
total tokens: 273
```

## Inspecting Payloads

You can use the [StartPromptEvent](../../reference/griptape/events/start_prompt_event.md) to inspect the Prompt Stack and final prompt string before it is sent to the LLM.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/misc/src/events_6.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/misc/logs/events_6.txt"
    ```


```
...
Prompt Stack Messages:
system:
user: Write me a poem.
Final Prompt String:


User: Write me a poem.

Assistant:
...
```

## `EventListenerDriver.on_event` Return Value Behavior

The value that gets returned from the [`EventListener.on_event`](../../reference/griptape/events/event_listener.md#griptape.events.event_listener.EventListener.on_event) will determine what gets sent to the `event_listener_driver`.

### `EventListener.on_event` is None

By default, the `EventListener.on_event` function is `None`. Any events that the `EventListener` is listening for will get sent to the `event_listener_driver` as-is.

### Return `BaseEvent` or `dict`

You can return a `BaseEvent` or `dict` object from `EventListener.on_event`, and it will get sent to the `event_listener_driver`.

### Return `None`

You can return `None` in the on_event function to prevent the event from getting sent to the `event_listener_driver`.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/misc/src/events_no_publish.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/misc/logs/events_no_publish.txt"
    ```

