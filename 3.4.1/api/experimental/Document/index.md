# dspy.experimental.Document

## `dspy.experimental.Document`

Bases: `Type`

A document type for providing content that can be cited by language models.

This type represents documents that can be passed to language models for citation-enabled responses, particularly useful with Anthropic’s Citations API. Documents include the content and metadata that helps the LM understand and reference the source material.

Attributes:

| Name         | Type                                       | Description                                                  |
| ------------ | ------------------------------------------ | ------------------------------------------------------------ |
| `data`       | `str`                                      | The text content of the document                             |
| `title`      | \`str                                      | None\`                                                       |
| `media_type` | `Literal['text/plain', 'application/pdf']` | MIME type of the document content (defaults to “text/plain”) |
| `context`    | \`str                                      | None\`                                                       |

Examples:

```
import dspy
from dspy.signatures import Signature
from dspy.experimental import Document, Citations

class AnswerWithSources(Signature):
    '''Answer questions using provided documents with citations.'''
    documents: list[Document] = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    citations: Citations = dspy.OutputField()

# Create documents
docs = [
    Document(
        data="The Earth orbits the Sun in an elliptical path.",
        title="Basic Astronomy Facts"
    ),
    Document(
        data="Water boils at 100°C at standard atmospheric pressure.",
        title="Physics Fundamentals",
    )
]

# Use with a citation-supporting model
lm = dspy.LM("anthropic/claude-opus-4-1-20250805")
predictor = dspy.Predict(AnswerWithSources)
result = predictor(documents=docs, question="What temperature does water boil?", lm=lm)
print(result.citations)
```

### Methods:

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

Description of the document type for use in prompts.

Source code in `dspy/adapters/types/document.py`

```
@classmethod
def description(cls) -> str:
    """Description of the document type for use in prompts."""
    return (
        "A document containing text content that can be referenced and cited. "
        "Include the full text content and optionally a title for proper referencing."
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

Format document for LM consumption.

Returns:

| Type                   | Description                                                                                      |
| ---------------------- | ------------------------------------------------------------------------------------------------ |
| `list[dict[str, Any]]` | A list containing the document block in the format expected by citation-enabled language models. |

Source code in `dspy/adapters/types/document.py`

```
def format(self) -> list[dict[str, Any]]:
    """Format document for LM consumption.

    Returns:
        A list containing the document block in the format expected by citation-enabled language models.
    """
    # PDFs use a base64 source; only text/plain uses a text source.
    source_type = "base64" if self.media_type == "application/pdf" else "text"
    document_block = {
        "type": "document",
        "source": {
            "type": source_type,
            "media_type": self.media_type,
            "data": self.data
        },
        "citations": {"enabled": True}
    }

    if self.title:
        document_block["title"] = self.title

    if self.context:
        document_block["context"] = self.context

    return [document_block]
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

#### `validate_input(data: Any)`

Source code in `dspy/adapters/types/document.py`

```
@pydantic.model_validator(mode="before")
@classmethod
def validate_input(cls, data: Any):
    if isinstance(data, cls):
        return data

    # Handle case where data is just a string (data only)
    if isinstance(data, str):
        return {"data": data}

    # Handle case where data is a dict
    elif isinstance(data, dict):
        return data

    raise ValueError(f"Received invalid value for `Document`: {data}")
```
