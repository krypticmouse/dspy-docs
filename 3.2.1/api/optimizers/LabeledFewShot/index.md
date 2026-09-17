# dspy.LabeledFewShot

## `dspy.LabeledFewShot(k=16)`

Bases: `Teleprompter`

Source code in `dspy/teleprompt/vanilla.py`

```
def __init__(self, k=16):
    self.k = k
```

### Functions

#### `compile(student, *, trainset, sample=True)`

Source code in `dspy/teleprompt/vanilla.py`

```
def compile(self, student, *, trainset, sample=True):
    self.student = student.reset_copy()
    self.trainset = trainset

    if len(self.trainset) == 0:
        return self.student

    rng = random.Random(0)

    for predictor in self.student.predictors():
        if sample:
            predictor.demos = rng.sample(self.trainset, min(self.k, len(self.trainset)))
        else:
            predictor.demos = self.trainset[: min(self.k, len(self.trainset))]

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
