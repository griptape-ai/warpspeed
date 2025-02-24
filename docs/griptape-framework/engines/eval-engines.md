---
search:
  boost: 2
---

## Overview

The [Eval Engine](../../reference/griptape/engines/eval/index.md) is used to evaluate the performance of an LLM's output against a given input. The engine returns a score between 0 and 1, along with a reason for the score.

Eval Engines require either [criteria](../../reference/griptape/engines/eval/eval_engine.md#griptape.engines.eval.eval_engine.EvalEngine.criteria) or [evaluation_steps](../../reference/griptape/engines/eval/eval_engine.md#griptape.engines.eval.eval_engine.EvalEngine.evaluation_steps) to be set.
If `criteria` is set, Griptape will generate `evaluation_steps` for you. This is useful for getting started, but you may to explicitly set `evaluation_steps` for more complex evaluations.
Either `criteria` or `evaluation_steps` must be set, but not both.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/engines/src/eval_engines_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/engines/logs/eval_engines_1.txt"
    ```


```
Eval Steps [
  "Compare the actual output to the expected output to identify any discrepancies.",
  "Verify the factual accuracy of the actual output by cross-referencing with the expected output.",
  "Assess whether the actual output meets the criteria outlined in the expected output.",
  "Determine if any information in the actual output contradicts the expected output."
]
Score: 1.0
Reason: The actual output 'Glass' matches the expected output 'Glass', with no discrepancies or contradictions.
```
