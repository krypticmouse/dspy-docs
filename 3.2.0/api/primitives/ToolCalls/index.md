# dspy.ToolCalls

## `dspy.ToolCalls`

Bases: `Type`

### Functions

#### `adapt_to_native_lm_feature(signature: type[Signature], field_name: str, lm: BaseLM, lm_kwargs: dict[str, Any]) -> type[Signature]`

Adapt the custom type to the native LM feature if possible.

When the LM and configuration supports the related native LM feature, e.g., native tool calling, native reasoning, etc., we adapt the signature and `lm_kwargs` to enable the native LM feature.

Parameters:

| Name         | Type              | Description                                                                                   | Default    |
| ------------ | ----------------- | --------------------------------------------------------------------------------------------- | ---------- |
| `signature`  | `type[Signature]` | The DSPy signature for the LM call.                                                           | *required* |
| `field_name` | `str`             | The name of the field in the signature to adapt to the native LM feature.                     | *required* |
| `lm`         | `BaseLM`          | The LM instance.                                                                              | *required* |
| `lm_kwargs`  | `dict[str, Any]`  | The keyword arguments for the LM call, subject to in-place updates if adaptation if required. | *required* |

Returns:

| Type              | Description                                                                                        |
| ----------------- | -------------------------------------------------------------------------------------------------- |
| `type[Signature]` | The adapted signature. If the custom type is not natively supported by the LM, return the original |
| `type[Signature]` | signature.                                                                                         |

Source code in `dspy/adapters/types/base_type.py`

```
@classmethod
def adapt_to_native_lm_feature(
    cls,
    signature: type["Signature"],
    field_name: str,
    lm: BaseLM,
    lm_kwargs: dict[str, Any],
) -> type["Signature"]:
    """Adapt the custom type to the native LM feature if possible.

    When the LM and configuration supports the related native LM feature, e.g., native tool calling, native
    reasoning, etc., we adapt the signature and `lm_kwargs` to enable the native LM feature.

    Args:
        signature: The DSPy signature for the LM call.
        field_name: The name of the field in the signature to adapt to the native LM feature.
        lm: The LM instance.
        lm_kwargs: The keyword arguments for the LM call, subject to in-place updates if adaptation if required.

    Returns:
        The adapted signature. If the custom type is not natively supported by the LM, return the original
        signature.
    """
    return signature
```

#### `description() -> str`

Source code in `dspy/adapters/types/tool.py`

```
@classmethod
def description(cls) -> str:
    return (
        "Tool calls information, including the name of the tools and the arguments to be passed to it. "
        "Arguments must be provided in JSON format."
    )
```

#### `extract_custom_type_from_annotation(annotation)`

Extract all custom types from the annotation.

This is used to extract all custom types from the annotation of a field, while the annotation can have arbitrary level of nesting. For example, we detect `Tool` is in `list[dict[str, Tool]]`.

Source code in `dspy/adapters/types/base_type.py`

```
@classmethod
def extract_custom_type_from_annotation(cls, annotation):
    """Extract all custom types from the annotation.

    This is used to extract all custom types from the annotation of a field, while the annotation can
    have arbitrary level of nesting. For example, we detect `Tool` is in `list[dict[str, Tool]]`.
    """
    # Direct match. Nested type like `list[dict[str, Event]]` passes `isinstance(annotation, type)` in python 3.10
    # while fails in python 3.11. To accommodate users using python 3.10, we need to capture the error and ignore it.
    try:
        if isinstance(annotation, type) and issubclass(annotation, cls):
            return [annotation]
    except TypeError:
        pass

    origin = get_origin(annotation)
    if origin is None:
        return []

    result = []
    # Recurse into all type args
    for arg in get_args(annotation):
        result.extend(cls.extract_custom_type_from_annotation(arg))

    return result
```

#### `format() -> list[dict[str, Any]]`

Source code in `dspy/adapters/types/tool.py`

```
def format(self) -> list[dict[str, Any]]:
    # The tool_call field is compatible with OpenAI's tool calls schema.
    return {
        "tool_calls": [tool_call.format() for tool_call in self.tool_calls],
    }
```

#### `from_dict_list(tool_calls_dicts: list[dict[str, Any]]) -> ToolCalls`

Convert a list of dictionaries to a ToolCalls instance.

Parameters:

| Name        | Type | Description                                                                       | Default    |
| ----------- | ---- | --------------------------------------------------------------------------------- | ---------- |
| `dict_list` |      | A list of dictionaries, where each dictionary should have 'name' and 'args' keys. | *required* |

Returns:

| Type        | Description           |
| ----------- | --------------------- |
| `ToolCalls` | A ToolCalls instance. |

Examples:

````
```python
tool_calls_dict = [
    {"name": "search", "args": {"query": "hello"}},
    {"name": "translate", "args": {"text": "world"}}
]
tool_calls = ToolCalls.from_dict_list(tool_calls_dict)
````

```

Source code in `dspy/adapters/types/tool.py`

```

@classmethod def from_dict_list(cls, tool_calls_dicts: list\[dict[str, Any]\]) -> "ToolCalls": """Convert a list of dictionaries to a ToolCalls instance.

````
Args:
    dict_list: A list of dictionaries, where each dictionary should have 'name' and 'args' keys.

Returns:
    A ToolCalls instance.

Examples:

    ```python
    tool_calls_dict = [
        {"name": "search", "args": {"query": "hello"}},
        {"name": "translate", "args": {"text": "world"}}
    ]
    tool_calls = ToolCalls.from_dict_list(tool_calls_dict)
    ```
"""
tool_calls = [cls.ToolCall(**item) for item in tool_calls_dicts]
return cls(tool_calls=tool_calls)
````

```

#### `is_streamable() -> bool`

Whether the custom type is streamable.

Source code in `dspy/adapters/types/base_type.py`

```

@classmethod def is_streamable(cls) -> bool: """Whether the custom type is streamable.""" return False

```

#### `parse_lm_response(response: str | dict[str, Any]) -> Optional[Type]`

Parse a LM response into the custom type.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `response` | `str | dict[str, Any]` | A LM response. | *required* |

Returns:

| Type | Description |
| --- | --- |
| `Optional[Type]` | A custom type object. |

Source code in `dspy/adapters/types/base_type.py`

```

@classmethod def parse_lm_response(cls, response: str | dict[str, Any]) -> Optional\["Type"\]: """Parse a LM response into the custom type.

```
Args:
    response: A LM response.

Returns:
    A custom type object.
"""
return None
```

```

#### `parse_stream_chunk(chunk: ModelResponseStream) -> Optional[Type]`

Parse a stream chunk into the custom type.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `chunk` | `ModelResponseStream` | A stream chunk. | *required* |

Returns:

| Type | Description |
| --- | --- |
| `Optional[Type]` | A custom type object or None if the chunk is not for this custom type. |

Source code in `dspy/adapters/types/base_type.py`

```

@classmethod def parse_stream_chunk(cls, chunk: "ModelResponseStream") -> Optional\["Type"\]: """ Parse a stream chunk into the custom type.

```
Args:
    chunk: A stream chunk.

Returns:
    A custom type object or None if the chunk is not for this custom type.
"""
return None
```

```

#### `serialize_model()`

Source code in `dspy/adapters/types/base_type.py`

```

@pydantic.model_serializer() def serialize_model(self): formatted = self.format() if isinstance(formatted, list): return ( f"{CUSTOM_TYPE_START_IDENTIFIER}{json.dumps(formatted, ensure_ascii=False)}{CUSTOM_TYPE_END_IDENTIFIER}" ) return formatted

```

#### `validate_input(data: Any)`

Source code in `dspy/adapters/types/tool.py`

```

@pydantic.model_validator(mode="before") @classmethod def validate_input(cls, data: Any): if isinstance(data, cls): return data

```
# Handle case where data is a list of dicts with "name" and "args" keys
if isinstance(data, list) and all(
    isinstance(item, dict) and "name" in item and "args" in item for item in data
):
    return {"tool_calls": [cls.ToolCall(**item) for item in data]}
# Handle case where data is a dict
elif isinstance(data, dict):
    if "tool_calls" in data:
        # Handle case where data is a dict with "tool_calls" key
        tool_calls_data = data["tool_calls"]
        if isinstance(tool_calls_data, list):
            return {
                "tool_calls": [
                    cls.ToolCall(**item) if isinstance(item, dict) else item for item in tool_calls_data
                ]
            }
    elif "name" in data and "args" in data:
        # Handle case where data is a dict with "name" and "args" keys
        return {"tool_calls": [cls.ToolCall(**data)]}

raise ValueError(f"Received invalid value for `dspy.ToolCalls`: {data}")
```

```
```
