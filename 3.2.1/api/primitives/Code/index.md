# dspy.Code

## `dspy.Code`

Bases: `Type`

Code type in DSPy.

This type is useful for code generation and code analysis.

Example 1: dspy.Code as output type in code generation:

```
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


class CodeGeneration(dspy.Signature):
    '''Generate python code to answer the question.'''

    question: str = dspy.InputField(description="The question to answer")
    code: dspy.Code["java"] = dspy.OutputField(description="The code to execute")


predict = dspy.Predict(CodeGeneration)

result = predict(question="Given an array, find if any of the two numbers sum up to 10")
print(result.code)
```

Example 2: dspy.Code as input type in code analysis:

```
import dspy
import inspect

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class CodeAnalysis(dspy.Signature):
    '''Analyze the time complexity of the function.'''

    code: dspy.Code["python"] = dspy.InputField(description="The function to analyze")
    result: str = dspy.OutputField(description="The time complexity of the function")


predict = dspy.Predict(CodeAnalysis)


def sleepsort(x):
    import time

    for i in x:
        time.sleep(i)
        print(i)

result = predict(code=inspect.getsource(sleepsort))
print(result.result)
```

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

Source code in `dspy/adapters/types/code.py`

````
@classmethod
def description(cls) -> str:
    return (
        "Code represented in a string, specified in the `code` field. If this is an output field, the code "
        f"field should follow the markdown code block format, e.g. \n```{cls.language.lower()}\n{{code}}\n```"
        f"\nProgramming language: {cls.language}"
    )
````

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

Source code in `dspy/adapters/types/code.py`

```
def format(self):
    return f"{self.code}"
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

Override to bypass the \<> and \<> tags.

Source code in `dspy/adapters/types/code.py`

```
@pydantic.model_serializer()
def serialize_model(self):
    """Override to bypass the <<CUSTOM-TYPE-START-IDENTIFIER>> and <<CUSTOM-TYPE-END-IDENTIFIER>> tags."""
    return self.format()
```

#### `validate_input(data: Any)`

Source code in `dspy/adapters/types/code.py`

```
@pydantic.model_validator(mode="before")
@classmethod
def validate_input(cls, data: Any):
    if isinstance(data, cls):
        return data

    if isinstance(data, str):
        return {"code": _filter_code(data)}

    if isinstance(data, dict):
        if "code" not in data:
            raise ValueError("`code` field is required for `dspy.Code`")
        if not isinstance(data["code"], str):
            raise ValueError(f"`code` field must be a string, but received type: {type(data['code'])}")
        return {"code": _filter_code(data["code"])}

    raise ValueError(f"Received invalid value for `dspy.Code`: {data}")
```
