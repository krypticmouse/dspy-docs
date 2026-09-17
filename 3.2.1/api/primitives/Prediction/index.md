# dspy.Prediction

## `dspy.Prediction(*args, **kwargs)`

Bases: `Example`

A prediction object that contains the output of a DSPy module.

Prediction inherits from Example.

To allow feedback-augmented scores, Prediction supports comparison operations (\<, >, \<=, >=) for Predictions with a `score` field. The comparison operations compare the 'score' values as floats. For equality comparison, Predictions are equal if their underlying data stores are equal (inherited from Example).

Arithmetic operations (+, /, etc.) are also supported for Predictions with a 'score' field, operating on the score value.

Source code in `dspy/primitives/prediction.py`

```
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    del self._demos
    del self._input_keys

    self._completions = None
    self._lm_usage = None
```

### Functions

#### `copy(**kwargs)`

Return a shallow copy, optionally overriding fields.

Parameters:

| Name       | Type | Description                            | Default |
| ---------- | ---- | -------------------------------------- | ------- |
| `**kwargs` |      | Fields to add or override in the copy. | `{}`    |

Examples:

```
>>> import dspy
>>> ex = dspy.Example(question="Why?", answer="Because.")
>>> ex.copy(answer="No reason.")
Example({'question': 'Why?', 'answer': 'No reason.'}) (input_keys=None)
```

Source code in `dspy/primitives/example.py`

```
def copy(self, **kwargs):
    """Return a shallow copy, optionally overriding fields.

    Args:
        **kwargs: Fields to add or override in the copy.

    Examples:
        >>> import dspy
        >>> ex = dspy.Example(question="Why?", answer="Because.")
        >>> ex.copy(answer="No reason.")
        Example({'question': 'Why?', 'answer': 'No reason.'}) (input_keys=None)
    """
    return type(self)(base=self, **kwargs)
```

#### `from_completions(list_or_dict, signature=None)`

Source code in `dspy/primitives/prediction.py`

```
@classmethod
def from_completions(cls, list_or_dict, signature=None):
    obj = cls()
    obj._completions = Completions(list_or_dict, signature=signature)
    obj._store = {k: v[0] for k, v in obj._completions.items()}

    return obj
```

#### `get(key, default=None)`

Return the value for `key`, or `default` if the field doesn't exist.

Parameters:

| Name      | Type | Description                          | Default    |
| --------- | ---- | ------------------------------------ | ---------- |
| `key`     |      | Field name to look up.               | *required* |
| `default` |      | Value to return when key is missing. | `None`     |

Examples:

```
>>> import dspy
>>> ex = dspy.Example(name="Alice")
>>> ex.get("name")
'Alice'
>>> ex.get("city", "Unknown")
'Unknown'
```

Source code in `dspy/primitives/example.py`

```
def get(self, key, default=None):
    """Return the value for `key`, or `default` if the field doesn't exist.

    Args:
        key: Field name to look up.
        default: Value to return when `key` is missing.

    Examples:
        >>> import dspy
        >>> ex = dspy.Example(name="Alice")
        >>> ex.get("name")
        'Alice'
        >>> ex.get("city", "Unknown")
        'Unknown'
    """
    return self._store.get(key, default)
```

#### `get_lm_usage()`

Source code in `dspy/primitives/prediction.py`

```
def get_lm_usage(self):
    return self._lm_usage
```

#### `inputs()`

Return a new `Example` containing only the input fields.

Requires `with_inputs` to have been called first.

Raises:

| Type         | Description                                    |
| ------------ | ---------------------------------------------- |
| `ValueError` | If with_inputs was not called on this example. |

Examples:

```
>>> import dspy
>>> ex = dspy.Example(question="Why?", answer="Because.").with_inputs("question")
>>> ex.inputs()
Example({'question': 'Why?'}) (input_keys={'question'})
```

Source code in `dspy/primitives/example.py`

```
def inputs(self):
    """Return a new `Example` containing only the input fields.

    Requires `with_inputs` to have been called first.

    Raises:
        ValueError: If `with_inputs` was not called on this example.

    Examples:
        >>> import dspy
        >>> ex = dspy.Example(question="Why?", answer="Because.").with_inputs("question")
        >>> ex.inputs()
        Example({'question': 'Why?'}) (input_keys={'question'})
    """
    if self._input_keys is None:
        raise ValueError("Inputs have not been set for this example. Use `example.with_inputs()` to set them.")

    # return items that are in input_keys
    d = {key: self._store[key] for key in self._store if key in self._input_keys}
    # return type(self)(d)
    new_instance = type(self)(base=d)
    new_instance._input_keys = self._input_keys  # Preserve input_keys in new instance
    return new_instance
```

#### `items(include_dspy=False)`

Return `(field_name, value)` pairs, like `dict.items()`.

Parameters:

| Name           | Type | Description                                            | Default |
| -------------- | ---- | ------------------------------------------------------ | ------- |
| `include_dspy` |      | If True, include internal fields prefixed with dspy\_. | `False` |

Examples:

```
>>> import dspy
>>> dspy.Example(question="Why?", answer="Because.").items()
[('question', 'Why?'), ('answer', 'Because.')]
```

Source code in `dspy/primitives/example.py`

```
def items(self, include_dspy=False):
    """Return `(field_name, value)` pairs, like `dict.items()`.

    Args:
        include_dspy: If `True`, include internal fields prefixed
            with `dspy_`.

    Examples:
        >>> import dspy
        >>> dspy.Example(question="Why?", answer="Because.").items()
        [('question', 'Why?'), ('answer', 'Because.')]
    """
    return [(k, v) for k, v in self._store.items() if not k.startswith("dspy_") or include_dspy]
```

#### `keys(include_dspy=False)`

Return field names, like `dict.keys()`.

Parameters:

| Name           | Type | Description                                                                           | Default |
| -------------- | ---- | ------------------------------------------------------------------------------------- | ------- |
| `include_dspy` |      | If True, include internal fields prefixed with dspy\_. Normally you can ignore these. | `False` |

Examples:

```
>>> import dspy
>>> dspy.Example(question="Why?", answer="Because.").keys()
['question', 'answer']
```

Source code in `dspy/primitives/example.py`

```
def keys(self, include_dspy=False):
    """Return field names, like `dict.keys()`.

    Args:
        include_dspy: If `True`, include internal fields prefixed
            with `dspy_`. Normally you can ignore these.

    Examples:
        >>> import dspy
        >>> dspy.Example(question="Why?", answer="Because.").keys()
        ['question', 'answer']
    """
    return [k for k in self._store.keys() if not k.startswith("dspy_") or include_dspy]
```

#### `labels()`

Return a new `Example` containing only the label (non-input) fields.

Requires `with_inputs` to have been called first, since labels are everything that is *not* an input.

Examples:

```
>>> import dspy
>>> ex = dspy.Example(question="Why?", answer="Because.").with_inputs("question")
>>> ex.labels()
Example({'answer': 'Because.'}) (input_keys=None)
```

Source code in `dspy/primitives/example.py`

```
def labels(self):
    """Return a new `Example` containing only the label (non-input) fields.

    Requires `with_inputs` to have been called first, since labels are
    everything that is *not* an input.

    Examples:
        >>> import dspy
        >>> ex = dspy.Example(question="Why?", answer="Because.").with_inputs("question")
        >>> ex.labels()
        Example({'answer': 'Because.'}) (input_keys=None)
    """
    # return items that are NOT in input_keys
    input_keys = self.inputs().keys()
    d = {key: self._store[key] for key in self._store if key not in input_keys}
    return type(self)(d)
```

#### `set_lm_usage(value)`

Source code in `dspy/primitives/prediction.py`

```
def set_lm_usage(self, value):
    self._lm_usage = value
```

#### `toDict()`

Convert to a plain dictionary, recursively serializing nested objects.

Nested `Example` objects, Pydantic models, lists, and dicts are converted so the result is JSON-friendly.

Examples:

```
>>> import dspy
>>> dspy.Example(question="Why?", answer="Because.").toDict()
{'question': 'Why?', 'answer': 'Because.'}
```

Source code in `dspy/primitives/example.py`

```
def toDict(self):  # noqa: N802
    """Convert to a plain dictionary, recursively serializing nested objects.

    Nested `Example` objects, Pydantic models, lists, and dicts are
    converted so the result is JSON-friendly.

    Examples:
        >>> import dspy
        >>> dspy.Example(question="Why?", answer="Because.").toDict()
        {'question': 'Why?', 'answer': 'Because.'}
    """
    def convert_to_serializable(value):
        if hasattr(value, "toDict"):
            return value.toDict()
        elif isinstance(value, BaseModel):
            # Handle Pydantic models (e.g., dspy.History)
            return value.model_dump()
        elif isinstance(value, list):
            return [convert_to_serializable(item) for item in value]
        elif isinstance(value, dict):
            return {k: convert_to_serializable(v) for k, v in value.items()}
        else:
            return value

    serializable_store = {}
    for k, v in self._store.items():
        serializable_store[k] = convert_to_serializable(v)

    return serializable_store
```

#### `values(include_dspy=False)`

Return field values, like `dict.values()`.

Parameters:

| Name           | Type | Description                                            | Default |
| -------------- | ---- | ------------------------------------------------------ | ------- |
| `include_dspy` |      | If True, include internal fields prefixed with dspy\_. | `False` |

Examples:

```
>>> import dspy
>>> dspy.Example(question="Why?", answer="Because.").values()
['Why?', 'Because.']
```

Source code in `dspy/primitives/example.py`

```
def values(self, include_dspy=False):
    """Return field values, like `dict.values()`.

    Args:
        include_dspy: If `True`, include internal fields prefixed
            with `dspy_`.

    Examples:
        >>> import dspy
        >>> dspy.Example(question="Why?", answer="Because.").values()
        ['Why?', 'Because.']
    """
    return [v for k, v in self._store.items() if not k.startswith("dspy_") or include_dspy]
```

#### `with_inputs(*keys)`

Mark which fields are inputs and return a new `Example`.

Fields not listed here are treated as labels (expected outputs). DSPy optimizers and evaluators use this split: they pass `example.inputs()` to your program and compare the output against `example.labels()`.

Parameters:

| Name    | Type | Description                | Default |
| ------- | ---- | -------------------------- | ------- |
| `*keys` |      | Names of the input fields. | `()`    |

Returns:

| Type | Description                                     |
| ---- | ----------------------------------------------- |
|      | A copy of this Example with the input keys set. |

Examples:

```
>>> import dspy
>>> ex = dspy.Example(question="Why?", answer="Because.").with_inputs("question")
>>> ex.inputs().keys()
['question']
>>> ex.labels().keys()
['answer']
```

Source code in `dspy/primitives/example.py`

```
def with_inputs(self, *keys):
    """Mark which fields are inputs and return a new `Example`.

    Fields not listed here are treated as labels (expected outputs).
    DSPy optimizers and evaluators use this split: they pass
    `example.inputs()` to your program and compare the output against
    `example.labels()`.

    Args:
        *keys: Names of the input fields.

    Returns:
        A copy of this `Example` with the input keys set.

    Examples:
        >>> import dspy
        >>> ex = dspy.Example(question="Why?", answer="Because.").with_inputs("question")
        >>> ex.inputs().keys()
        ['question']
        >>> ex.labels().keys()
        ['answer']
    """
    copied = self.copy()
    copied._input_keys = set(keys)
    return copied
```

#### `without(*keys)`

Return a copy with the specified fields removed.

Parameters:

| Name    | Type | Description          | Default |
| ------- | ---- | -------------------- | ------- |
| `*keys` |      | Field names to drop. | `()`    |

Examples:

```
>>> import dspy
>>> ex = dspy.Example(question="Why?", answer="Because.", source="web")
>>> ex.without("source")
Example({'question': 'Why?', 'answer': 'Because.'}) (input_keys=None)
```

Source code in `dspy/primitives/example.py`

```
def without(self, *keys):
    """Return a copy with the specified fields removed.

    Args:
        *keys: Field names to drop.

    Examples:
        >>> import dspy
        >>> ex = dspy.Example(question="Why?", answer="Because.", source="web")
        >>> ex.without("source")
        Example({'question': 'Why?', 'answer': 'Because.'}) (input_keys=None)
    """
    copied = self.copy()
    for key in keys:
        del copied[key]
    return copied
```
