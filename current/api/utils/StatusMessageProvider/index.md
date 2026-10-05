# dspy.streaming.StatusMessageProvider

## `dspy.streaming.StatusMessageProvider`

Provides customizable status message streaming for DSPy programs.

This class serves as a base for creating custom status message providers. Users can subclass and override its methods to define specific status messages for different stages of program execution, each method must return a string.

Examples:

```
class MyStatusMessageProvider(StatusMessageProvider):
    def lm_start_status_message(self, instance, inputs):
        return f"Calling LM with inputs {inputs}..."

    def module_end_status_message(self, outputs):
        return f"Module finished with output: {outputs}!"

program = dspy.streamify(dspy.Predict("q->a"), status_message_provider=MyStatusMessageProvider())
```

### Methods:

#### `lm_end_status_message(outputs: Any)`

Status message after a `dspy.LM` is called.

Source code in `dspy/streaming/messages.py`

```
def lm_end_status_message(self, outputs: Any):
    """Status message after a `dspy.LM` is called."""
    pass
```

#### `lm_start_status_message(instance: Any, inputs: dict[str, Any])`

Status message before a `dspy.LM` is called.

Source code in `dspy/streaming/messages.py`

```
def lm_start_status_message(self, instance: Any, inputs: dict[str, Any]):
    """Status message before a `dspy.LM` is called."""
    pass
```

#### `module_end_status_message(outputs: Any)`

Status message after a `dspy.Module` or `dspy.Predict` is called.

Source code in `dspy/streaming/messages.py`

```
def module_end_status_message(self, outputs: Any):
    """Status message after a `dspy.Module` or `dspy.Predict` is called."""
    pass
```

#### `module_start_status_message(instance: Any, inputs: dict[str, Any])`

Status message before a `dspy.Module` or `dspy.Predict` is called.

Source code in `dspy/streaming/messages.py`

```
def module_start_status_message(self, instance: Any, inputs: dict[str, Any]):
    """Status message before a `dspy.Module` or `dspy.Predict` is called."""
    pass
```

#### `tool_end_status_message(outputs: Any)`

Status message after a `dspy.Tool` is called.

Source code in `dspy/streaming/messages.py`

```
def tool_end_status_message(self, outputs: Any):
    """Status message after a `dspy.Tool` is called."""
    return "Tool calling finished! Querying the LLM with tool calling results..."
```

#### `tool_start_status_message(instance: Any, inputs: dict[str, Any])`

Status message before a `dspy.Tool` is called.

Source code in `dspy/streaming/messages.py`

```
def tool_start_status_message(self, instance: Any, inputs: dict[str, Any]):
    """Status message before a `dspy.Tool` is called."""
    return f"Calling tool {instance.name}..."
```
