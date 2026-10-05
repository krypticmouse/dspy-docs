# dspy.Tool

## `dspy.Tool(func: Callable, name: str | None = None, desc: str | None = None, args: dict[str, Any] | None = None, arg_types: dict[str, Any] | None = None, arg_desc: dict[str, str] | None = None)`

Bases: `Type`

Tool class.

This class is used to simplify the creation of tools for tool calling (function calling) in LLMs. Only supports functions for now.

Initialize the Tool class.

Users can choose to specify the `name`, `desc`, `args`, and `arg_types`, or let the `dspy.Tool` automatically infer the values from the function. For values that are specified by the user, automatic inference will not be performed on them.

Parameters:

| Name        | Type                       | Description                                                                                                              | Default    |
| ----------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ---------- |
| `func`      | `Callable`                 | The actual function that is being wrapped by the tool.                                                                   | *required* |
| `name`      | `Optional[str]`            | The name of the tool. Defaults to None.                                                                                  | `None`     |
| `desc`      | `Optional[str]`            | The description of the tool. Defaults to None.                                                                           | `None`     |
| `args`      | `Optional[dict[str, Any]]` | The args and their schema of the tool, represented as a dictionary from arg name to arg’s json schema. Defaults to None. | `None`     |
| `arg_types` | `Optional[dict[str, Any]]` | The argument types of the tool, represented as a dictionary from arg name to the type of the argument. Defaults to None. | `None`     |
| `arg_desc`  | `Optional[dict[str, str]]` | Descriptions for each arg, represented as a dictionary from arg name to description string. Defaults to None.            | `None`     |

Examples:

```
def foo(x: int, y: str = "hello"):
    return str(x) + y

tool = Tool(foo)
print(tool.args)
# Expected output: {'x': {'type': 'integer'}, 'y': {'type': 'string', 'default': 'hello'}}
```

Source code in `dspy/adapters/types/tool.py`

````
def __init__(
    self,
    func: Callable,
    name: str | None = None,
    desc: str | None = None,
    args: dict[str, Any] | None = None,
    arg_types: dict[str, Any] | None = None,
    arg_desc: dict[str, str] | None = None,
):
    """Initialize the Tool class.

    Users can choose to specify the `name`, `desc`, `args`, and `arg_types`, or let the `dspy.Tool`
    automatically infer the values from the function. For values that are specified by the user, automatic inference
    will not be performed on them.

    Args:
        func (Callable): The actual function that is being wrapped by the tool.
        name (Optional[str], optional): The name of the tool. Defaults to None.
        desc (Optional[str], optional): The description of the tool. Defaults to None.
        args (Optional[dict[str, Any]], optional): The args and their schema of the tool, represented as a
            dictionary from arg name to arg's json schema. Defaults to None.
        arg_types (Optional[dict[str, Any]], optional): The argument types of the tool, represented as a dictionary
            from arg name to the type of the argument. Defaults to None.
        arg_desc (Optional[dict[str, str]], optional): Descriptions for each arg, represented as a
            dictionary from arg name to description string. Defaults to None.

    Examples:

    ```python
    def foo(x: int, y: str = "hello"):
        return str(x) + y

    tool = Tool(foo)
    print(tool.args)
    # Expected output: {'x': {'type': 'integer'}, 'y': {'type': 'string', 'default': 'hello'}}
    ```
    """
    super().__init__(func=func, name=name, desc=desc, args=args, arg_types=arg_types, arg_desc=arg_desc)
    self._parse_function(func, arg_desc)
````

### Methods:

#### `__call__(**kwargs)`

Source code in `dspy/adapters/types/tool.py`

```
@with_callbacks
def __call__(self, **kwargs):
    parsed_kwargs = self._validate_and_parse_args(**kwargs)
    result = self.func(**parsed_kwargs)
    if asyncio.iscoroutine(result):
        if settings.allow_tool_async_sync_conversion:
            return self._run_async_in_sync(result)
        else:
            raise ValueError(
                "You are calling `__call__` on an async tool, please use `acall` instead or enable "
                "async-to-sync conversion with `dspy.configure(allow_tool_async_sync_conversion=True)` "
                "or `with dspy.context(allow_tool_async_sync_conversion=True):`."
            )
    return result
```

#### `acall(**kwargs)`

Source code in `dspy/adapters/types/tool.py`

```
@with_callbacks
async def acall(self, **kwargs):
    parsed_kwargs = self._validate_and_parse_args(**kwargs)
    result = self.func(**parsed_kwargs)
    if asyncio.iscoroutine(result):
        return await result
    else:
        # We should allow calling a sync tool in the async path.
        return result
```

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

Description of the custom type

Source code in `dspy/adapters/types/base_type.py`

```
@classmethod
def description(cls) -> str:
    """Description of the custom type"""
    return ""
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

#### `format()`

Source code in `dspy/adapters/types/tool.py`

```
def format(self):
    return str(self)
```

#### `format_as_litellm_function_call()`

Source code in `dspy/adapters/types/tool.py`

```
def format_as_litellm_function_call(self):
    return {
        "type": "function",
        "function": {
            "name": self.name,
            "description": self.desc,
            "parameters": {
                "type": "object",
                "properties": self.args,
                "required": [k for k in self.args if "default" not in self.args[k]],
            },
        },
    }
```

#### `from_langchain(tool: BaseTool) -> Tool`

Build a DSPy tool from a LangChain tool.

Parameters:

| Name   | Type       | Description                    | Default    |
| ------ | ---------- | ------------------------------ | ---------- |
| `tool` | `BaseTool` | The LangChain tool to convert. | *required* |

Returns:

| Type   | Description    |
| ------ | -------------- |
| `Tool` | A Tool object. |

Examples:

```
import asyncio
import dspy
from langchain.tools import tool as lc_tool

@lc_tool
def add(x: int, y: int):
    "Add two numbers together."
    return x + y

dspy_tool = dspy.Tool.from_langchain(add)

async def run_tool():
    return await dspy_tool.acall(x=1, y=2)

print(asyncio.run(run_tool()))
# 3
```

Source code in `dspy/adapters/types/tool.py`

````
@classmethod
def from_langchain(cls, tool: "BaseTool") -> "Tool":
    """
    Build a DSPy tool from a LangChain tool.

    Args:
        tool: The LangChain tool to convert.

    Returns:
        A Tool object.

    Examples:

    ```python
    import asyncio
    import dspy
    from langchain.tools import tool as lc_tool

    @lc_tool
    def add(x: int, y: int):
        "Add two numbers together."
        return x + y

    dspy_tool = dspy.Tool.from_langchain(add)

    async def run_tool():
        return await dspy_tool.acall(x=1, y=2)

    print(asyncio.run(run_tool()))
    # 3
    ```
    """
    from dspy.utils.langchain_tool import convert_langchain_tool

    return convert_langchain_tool(tool)
````

#### `from_mcp_tool(session: _MCPToolClient, tool: mcp.types.Tool, *, result_mode: Literal['text', 'structured'] = 'text') -> Tool`

Build a DSPy tool from an MCP tool and a compatible MCP client.

Parameters:

| Name          | Type                            | Description                                                                                                                                                            | Default    |
| ------------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `session`     | `_MCPToolClient`                | An MCP client or session with an async call_tool method.                                                                                                               | *required* |
| `tool`        | `Tool`                          | The MCP tool to convert.                                                                                                                                               | *required* |
| `result_mode` | `Literal['text', 'structured']` | "text" preserves DSPy’s existing text/non-text conversion. "structured" returns MCP structured content exactly when present, with the existing conversion as fallback. | `'text'`   |

Returns:

| Type   | Description    |
| ------ | -------------- |
| `Tool` | A Tool object. |

Source code in `dspy/adapters/types/tool.py`

```
@classmethod
def from_mcp_tool(
    cls,
    session: _MCPToolClient,
    tool: "mcp.types.Tool",
    *,
    result_mode: Literal["text", "structured"] = "text",
) -> "Tool":
    """
    Build a DSPy tool from an MCP tool and a compatible MCP client.

    Args:
        session: An MCP client or session with an async ``call_tool`` method.
        tool: The MCP tool to convert.
        result_mode: ``"text"`` preserves DSPy's existing text/non-text
            conversion. ``"structured"`` returns MCP structured content
            exactly when present, with the existing conversion as fallback.

    Returns:
        A Tool object.
    """
    from dspy.utils.mcp import convert_mcp_tool

    return convert_mcp_tool(session, tool, result_mode=result_mode)
```

#### `is_streamable() -> bool`

Whether the custom type is streamable.

Source code in `dspy/adapters/types/base_type.py`

```
@classmethod
def is_streamable(cls) -> bool:
    """Whether the custom type is streamable."""
    return False
```

#### `parse_lm_response(response: str | dict[str, Any]) -> Optional[Type]`

Parse a LM response into the custom type.

Parameters:

| Name       | Type  | Description      | Default        |
| ---------- | ----- | ---------------- | -------------- |
| `response` | \`str | dict[str, Any]\` | A LM response. |

Returns:

| Type             | Description           |
| ---------------- | --------------------- |
| `Optional[Type]` | A custom type object. |

Source code in `dspy/adapters/types/base_type.py`

```
@classmethod
def parse_lm_response(cls, response: str | dict[str, Any]) -> Optional["Type"]:
    """Parse a LM response into the custom type.

    Args:
        response: A LM response.

    Returns:
        A custom type object.
    """
    return None
```

#### `parse_stream_chunk(chunk: ModelResponseStream) -> Optional[Type]`

Parse a stream chunk into the custom type.

Parameters:

| Name    | Type                  | Description     | Default    |
| ------- | --------------------- | --------------- | ---------- |
| `chunk` | `ModelResponseStream` | A stream chunk. | *required* |

Returns:

| Type             | Description                                                            |
| ---------------- | ---------------------------------------------------------------------- |
| `Optional[Type]` | A custom type object or None if the chunk is not for this custom type. |

Source code in `dspy/adapters/types/base_type.py`

```
@classmethod
def parse_stream_chunk(cls, chunk: "ModelResponseStream") -> Optional["Type"]:
    """
    Parse a stream chunk into the custom type.

    Args:
        chunk: A stream chunk.

    Returns:
        A custom type object or None if the chunk is not for this custom type.
    """
    return None
```

#### `serialize_model()`

Source code in `dspy/adapters/types/base_type.py`

```
@pydantic.model_serializer()
def serialize_model(self):
    formatted = self.format()
    if isinstance(formatted, list):
        return (
            f"{CUSTOM_TYPE_START_IDENTIFIER}{json.dumps(formatted, ensure_ascii=False)}{CUSTOM_TYPE_END_IDENTIFIER}"
        )
    return formatted
```
