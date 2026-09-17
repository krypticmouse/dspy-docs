# dspy.Example

## `dspy.Example(base=None, **kwargs)`

A flexible data container for DSPy examples and training data.

The `Example` class is the standard data format used in DSPy evaluation and optimization.

Key features

- Dictionary-like access patterns (item access, iteration, etc.)
- Flexible initialization from dictionaries, other `Example` instances, or keyword arguments
- Input/output field separation for training data
- Serialization support for saving/loading `Example` instances
- Immutable operations that return new `Example` instances

Examples:

````
Basic usage with keyword arguments:

```python
import dspy

# Create an example with input and output fields
example = dspy.Example(
    question="What is the capital of France?",
    answer="Paris",
)
print(example.question)  # "What is the capital of France?"
print(example.answer)   # "Paris"
````

Initialize from a dictionary:

```python
data = {"question": "What is 2+2?", "answer": "4"}
example = dspy.Example(data)
print(example["question"])  # "What is 2+2?"
```

Copy from another Example:

```python
original = dspy.Example(question="Hello", answer="World")
copy = dspy.Example(original)
print(copy.question)  # "Hello"
```

Working with input/output separation:

```python
# Mark which fields are inputs for training
example = dspy.Example(
    question="What is the weather?",
    answer="It's sunny",
).with_inputs("question")

# Get only input fields
inputs = example.inputs()
print(inputs.question)  # "What is the weather?"

# Get only output fields (labels)
labels = example.labels()
print(labels.answer)  # "It's sunny"
```

Dictionary-like operations:

```python
example = dspy.Example(name="Alice", age=30)

# Check if key exists
if "name" in example:
    print("Name field exists")

# Get with default value
city = example.get("city", "Unknown")
print(city)  # "Unknown"
```

```

Initialize an Example instance.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `base` |  | Optional base data source. Can be: - Another Example instance (copies its data) - A dictionary (copies its key-value pairs) - None (creates empty Example) | `None` |
| `**kwargs` |  | Additional key-value pairs to store in the Example. | `{}` |

Source code in `dspy/primitives/example.py`

```

def __init__(self, base=None, \*\*kwargs): """Initialize an Example instance.

```
Args:
    base: Optional base data source. Can be:
        - Another Example instance (copies its data)
        - A dictionary (copies its key-value pairs)
        - None (creates empty Example)
    **kwargs: Additional key-value pairs to store in the Example.
"""
# Internal storage and other attributes
self._store = {}
self._demos = []
self._input_keys = None

# Initialize from a base Example if provided
if base and isinstance(base, type(self)):
    self._store = base._store.copy()

# Initialize from a dict if provided
elif base and isinstance(base, dict):
    self._store = base.copy()

# Update with provided kwargs
self._store.update(kwargs)
```

```

### Functions

#### `copy(**kwargs)`

Source code in `dspy/primitives/example.py`

```

def copy(self, \*\*kwargs): return type(self)(base=self, \*\*kwargs)

```

#### `get(key, default=None)`

Source code in `dspy/primitives/example.py`

```

def get(self, key, default=None): return self.\_store.get(key, default)

```

#### `inputs()`

Source code in `dspy/primitives/example.py`

```

def inputs(self): if self.\_input_keys is None: raise ValueError("Inputs have not been set for this example. Use `example.with_inputs()` to set them.")

```
# return items that are in input_keys
d = {key: self._store[key] for key in self._store if key in self._input_keys}
# return type(self)(d)
new_instance = type(self)(base=d)
new_instance._input_keys = self._input_keys  # Preserve input_keys in new instance
return new_instance
```

```

#### `items(include_dspy=False)`

Source code in `dspy/primitives/example.py`

```

def items(self, include_dspy=False): return \[(k, v) for k, v in self._store.items() if not k.startswith("dspy_") or include_dspy\]

```

#### `keys(include_dspy=False)`

Source code in `dspy/primitives/example.py`

```

def keys(self, include_dspy=False): return \[k for k in self._store.keys() if not k.startswith("dspy_") or include_dspy\]

```

#### `labels()`

Source code in `dspy/primitives/example.py`

```

def labels(self): # return items that are NOT in input_keys input_keys = self.inputs().keys() d = {key: self.\_store[key] for key in self.\_store if key not in input_keys} return type(self)(d)

```

#### `toDict()`

Source code in `dspy/primitives/example.py`

```

def toDict(self): # noqa: N802 def convert_to_serializable(value): if hasattr(value, "toDict"): return value.toDict() elif isinstance(value, list): return [convert_to_serializable(item) for item in value] elif isinstance(value, dict): return {k: convert_to_serializable(v) for k, v in value.items()} else: return value

```
serializable_store = {}
for k, v in self._store.items():
    serializable_store[k] = convert_to_serializable(v)

return serializable_store
```

```

#### `values(include_dspy=False)`

Source code in `dspy/primitives/example.py`

```

def values(self, include_dspy=False): return \[v for k, v in self._store.items() if not k.startswith("dspy_") or include_dspy\]

```

#### `with_inputs(*keys)`

Source code in `dspy/primitives/example.py`

```

def with_inputs(self, \*keys): copied = self.copy() copied.\_input_keys = set(keys) return copied

```

#### `without(*keys)`

Source code in `dspy/primitives/example.py`

```

def without(self, \*keys): copied = self.copy() for key in keys: del copied[key] return copied

````

:::```
````
