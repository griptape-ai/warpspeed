---
search:
  boost: 2
---

## Overview

Event Listener Drivers are used to send Griptape [Events](../misc/events.md) to external services.

You can instantiate Drivers and pass them to Event Listeners in your Structure:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/event_listener_drivers_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/event_listener_drivers_1.txt"
    ```


Or use them independently:

```python
--8<-- "docs/griptape-framework/drivers/src/event_listener_drivers_2.py"
```

## Event Listener Drivers

Griptape offers the following Event Listener Drivers for forwarding Griptape Events.

### Amazon SQS

!!! info

    This driver requires the `drivers-event-listener-amazon-sqs` [extra](../index.md#extras).

The [AmazonSqsEventListenerDriver](../../reference/griptape/drivers/event_listener/amazon_sqs_event_listener_driver.md) sends Events to an [Amazon SQS](https://aws.amazon.com/sqs/) queue.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/event_listener_drivers_3.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/event_listener_drivers_3.txt"
    ```


### AWS IoT

!!! info

    This driver requires the `drivers-event-listener-amazon-iot` [extra](../index.md#extras).

The [AwsIotCoreEventListenerDriver](../../reference/griptape/drivers/event_listener/aws_iot_core_event_listener_driver.md) sends Events to the [AWS IoT Message Broker](https://aws.amazon.com/iot-core/).

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/event_listener_drivers_4.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/event_listener_drivers_4.txt"
    ```


### Griptape Cloud

The [GriptapeCloudEventListenerDriver](../../reference/griptape/drivers/event_listener/griptape_cloud_event_listener_driver.md) sends Events to [Griptape Cloud](https://www.griptape.ai/cloud).

!!! note

    This Driver is required when using the Griptape Cloud Managed Structures feature. For local development, you can use the [Skatepark Emulator](https://github.com/griptape-ai/griptape-cli?tab=readme-ov-file#skatepark-emulator).

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/event_listener_drivers_5.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/event_listener_drivers_5.txt"
    ```


### Webhook Event Listener Driver

The [WebhookEventListenerDriver](../../reference/griptape/drivers/event_listener/webhook_event_listener_driver.md) sends Events to any [Webhook](https://en.wikipedia.org/wiki/Webhook) URL.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/event_listener_drivers_6.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/event_listener_drivers_6.txt"
    ```


### Pusher

!!! info

    This driver requires the `drivers-event-listener-pusher` [extra](../index.md#extras).

The [PusherEventListenerDriver](../../reference/griptape/drivers/event_listener/pusher_event_listener_driver.md) sends Events to [Pusher](https://pusher.com).

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/event_listener_drivers_7.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/event_listener_drivers_7.txt"
    ```

