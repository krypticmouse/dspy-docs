# dspy.Ensemble

## `dspy.Ensemble(*, reduce_fn=None, size=None, deterministic=False)`

Bases: `Teleprompter`

A common reduce_fn is dspy.majority.

Source code in `dspy/teleprompt/ensemble.py`

```
def __init__(self, *, reduce_fn=None, size=None, deterministic=False):
    """A common reduce_fn is dspy.majority."""

    assert deterministic is False, "TODO: Implement example hashing for deterministic ensemble."

    self.reduce_fn = reduce_fn
    self.size = size
    self.deterministic = deterministic
```

### Methods:

#### `compile(programs)`

Source code in `dspy/teleprompt/ensemble.py`

```
def compile(self, programs):
    size = self.size
    reduce_fn = self.reduce_fn

    import dspy

    class EnsembledProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.programs = programs

        def forward(self, *args, **kwargs):
            programs = random.sample(self.programs, size) if size else self.programs
            outputs = [prog(*args, **kwargs) for prog in programs]

            if reduce_fn:
                return reduce_fn(outputs)

            return outputs

    return EnsembledProgram()
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
