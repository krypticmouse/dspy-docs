# dspy.BootstrapFewShot

## `dspy.BootstrapFewShot(metric=None, metric_threshold=None, teacher_settings: dict | None = None, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1, max_errors=None)`

Bases: `Teleprompter`

A Teleprompter class that composes a set of demos/examples to go into a predictor's prompt. These demos come from a combination of labeled examples in the training set, and bootstrapped demos.

Each bootstrap round copies the LM with a new `rollout_id` at `temperature=1.0` to bypass caches and gather diverse traces.

Parameters:

| Name                     | Type            | Description                                                                                                                                                | Default |
| ------------------------ | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `metric`                 | `Callable`      | A function that compares an expected value and predicted value, outputting the result of that comparison.                                                  | `None`  |
| `metric_threshold`       | `float`         | If the metric yields a numerical value, then check it against this threshold when deciding whether or not to accept a bootstrap example. Defaults to None. | `None`  |
| `teacher_settings`       | `dict`          | Settings for the teacher model. Defaults to None.                                                                                                          | `None`  |
| `max_bootstrapped_demos` | `int`           | Maximum number of bootstrapped demonstrations to include. Defaults to 4.                                                                                   | `4`     |
| `max_labeled_demos`      | `int`           | Maximum number of labeled demonstrations to include. Defaults to 16.                                                                                       | `16`    |
| `max_rounds`             | `int`           | Number of iterations to attempt generating the required bootstrap examples. If unsuccessful after max_rounds, the program ends. Defaults to 1.             | `1`     |
| `max_errors`             | `Optional[int]` | Maximum number of errors until program ends. If None, inherits from dspy.settings.max_errors.                                                              | `None`  |

Source code in `dspy/teleprompt/bootstrap.py`

```
def __init__(
    self,
    metric=None,
    metric_threshold=None,
    teacher_settings: dict | None = None,
    max_bootstrapped_demos=4,
    max_labeled_demos=16,
    max_rounds=1,
    max_errors=None,
):
    """A Teleprompter class that composes a set of demos/examples to go into a predictor's prompt.
    These demos come from a combination of labeled examples in the training set, and bootstrapped demos.

    Each bootstrap round copies the LM with a new ``rollout_id`` at ``temperature=1.0`` to
    bypass caches and gather diverse traces.

    Args:
        metric (Callable): A function that compares an expected value and predicted value,
            outputting the result of that comparison.
        metric_threshold (float, optional): If the metric yields a numerical value, then check it
            against this threshold when deciding whether or not to accept a bootstrap example.
            Defaults to None.
        teacher_settings (dict, optional): Settings for the `teacher` model.
            Defaults to None.
        max_bootstrapped_demos (int): Maximum number of bootstrapped demonstrations to include.
            Defaults to 4.
        max_labeled_demos (int): Maximum number of labeled demonstrations to include.
            Defaults to 16.
        max_rounds (int): Number of iterations to attempt generating the required bootstrap
            examples. If unsuccessful after `max_rounds`, the program ends. Defaults to 1.
        max_errors (Optional[int]): Maximum number of errors until program ends.
            If ``None``, inherits from ``dspy.settings.max_errors``.
    """
    self.metric = metric
    self.metric_threshold = metric_threshold
    self.teacher_settings = {} if teacher_settings is None else teacher_settings

    self.max_bootstrapped_demos = max_bootstrapped_demos
    self.max_labeled_demos = max_labeled_demos
    self.max_rounds = max_rounds
    self.max_errors = max_errors
    self.error_count = 0
    self.error_lock = threading.Lock()
```

### Functions

#### `compile(student, *, teacher=None, trainset)`

Source code in `dspy/teleprompt/bootstrap.py`

```
def compile(self, student, *, teacher=None, trainset):
    self.trainset = trainset

    self._prepare_student_and_teacher(student, teacher)
    self._prepare_predictor_mappings()
    self._bootstrap()

    self.student = self._train()
    self.student._compiled = True

    return self.student
```

#### `get_params() -> dict[str, Any]`

Get the parameters of the teleprompter.

Returns:

| Type             | Description                         |
| ---------------- | ----------------------------------- |
| `dict[str, Any]` | The parameters of the teleprompter. |

Source code in `dspy/teleprompt/teleprompt.py`

```
def get_params(self) -> dict[str, Any]:
    """
    Get the parameters of the teleprompter.

    Returns:
        The parameters of the teleprompter.
    """
    return self.__dict__
```

:::
