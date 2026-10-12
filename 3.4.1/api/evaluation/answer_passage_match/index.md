# dspy.evaluate.answer_passage_match

## `dspy.evaluate.answer_passage_match(example, pred, trace=None)`

Return True if any passage in `pred.context` contains the answer(s).

Strings are normalized (and passages also use DPR normalization internally).

Parameters:

| Name      | Type | Description                                                                | Default    |
| --------- | ---- | -------------------------------------------------------------------------- | ---------- |
| `example` |      | dspy.Example object with field answer (str or list[str]).                  | *required* |
| `pred`    |      | dspy.Prediction object with field context (list[str]) containing passages. | *required* |
| `trace`   |      | Unused; reserved for compatibility.                                        | `None`     |

Returns:

| Name   | Type | Description                                                         |
| ------ | ---- | ------------------------------------------------------------------- |
| `bool` |      | True if any passage contains any reference answer; otherwise False. |

Examples:

```
import dspy

example = dspy.Example(answer="Eiffel Tower")
pred = dspy.Prediction(context=["The Eiffel Tower is in Paris.", "..."])

answer_passage_match(example, pred)  # True
```

Source code in `dspy/evaluate/metrics.py`

````
def answer_passage_match(example, pred, trace=None):
    """Return True if any passage in `pred.context` contains the answer(s).

    Strings are normalized (and passages also use DPR normalization internally).

    Args:
        example: `dspy.Example` object with field `answer` (str or list[str]).
        pred: `dspy.Prediction` object with field `context` (list[str]) containing passages.
        trace: Unused; reserved for compatibility.

    Returns:
        bool: True if any passage contains any reference answer; otherwise False.

    Examples:
        ```python
        import dspy

        example = dspy.Example(answer="Eiffel Tower")
        pred = dspy.Prediction(context=["The Eiffel Tower is in Paris.", "..."])

        answer_passage_match(example, pred)  # True
        ```
    """
    if isinstance(example.answer, str):
        return _passage_match(pred.context, [example.answer])
    elif isinstance(example.answer, list):
        return _passage_match(pred.context, example.answer)

    raise ValueError(f"Invalid answer type: {type(example.answer)}")
````
