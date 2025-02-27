---
search:
  boost: 2
---

## Overview

Structure Run Drivers can be used to run Griptape Structures in a variety of runtime environments.
When combined with the [Structure Run Task](../../griptape-framework/structures/tasks.md#structure-run-task) or [Structure Run Tool](../../griptape-framework/tools/official-tools/index.md#structure-run) you can create complex, multi-agent pipelines that span multiple runtime environments.

## Structure Run Drivers

### Local

The [LocalStructureRunDriver](../../reference/griptape/drivers/structure_run/local_structure_run_driver.md) is used to run Griptape Structures in the same runtime environment as the code that is running the Structure.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/structure_run_drivers_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/structure_run_drivers_1.txt"
    ```


### Griptape Cloud

The [GriptapeCloudStructureRunDriver](../../reference/griptape/drivers/structure_run/griptape_cloud_structure_run_driver.md) is used to run Griptape Structures in the Griptape Cloud.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/structure_run_drivers_2.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/structure_run_drivers_2.txt"
    ```

