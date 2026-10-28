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

#### `format() -> list[TextPart | DocumentPart]`

Render the document as lm15 content parts.

Plain-text documents become text the model reads directly, framed by their title and context. PDFs become one document part. Provider-side citation marking is not part of lm15’s document part yet, so citations are produced by parsing the model’s answer rather than a native API.

Source code in `dspy/adapters/types/document.py`

```
def format(self) -> list[TextPart | DocumentPart]:
    """Render the document as lm15 content parts.

    Plain-text documents become text the model reads directly, framed by
    their title and context. PDFs become one document part. Provider-side
    citation marking is not part of lm15's document part yet, so citations
    are produced by parsing the model's answer rather than a native API.
    """
    header = []
    if self.title:
        header.append(f"Title: {self.title}")
    if self.context:
        header.append(f"Context: {self.context}")
    parts = [TextPart("\n".join(header) + "\n")] if header else []
    if self.media_type == "application/pdf":
        parts.append(document(data=self.data, media_type=self.media_type))
    else:
        parts.append(TextPart(self.data))
    return parts
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

#### `parse_lm_response(response: Response) -> Optional[Type]`

Read the native representation of this type out of an lm15 `Response`.

Parameters:

| Name       | Type       | Description                     | Default    |
| ---------- | ---------- | ------------------------------- | ---------- |
| `response` | `Response` | The model’s dspy.lm15.Response. | *required* |

Returns:

| Type             | Description                                                             |
| ---------------- | ----------------------------------------------------------------------- |
| `Optional[Type]` | A custom type object, or None when the response carries nothing for it. |

Source code in `dspy/adapters/types/base_type.py`

```
@classmethod
def parse_lm_response(cls, response: Response) -> Optional["Type"]:
    """Read the native representation of this type out of an lm15 `Response`.

    Args:
        response: The model's `dspy.lm15.Response`.

    Returns:
        A custom type object, or None when the response carries nothing for it.
    """
    return None
```

#### `parse_stream_chunk(chunk) -> Optional[Type]`

Parse a stream chunk into the custom type.

Parameters:

| Name    | Type | Description                                              | Default    |
| ------- | ---- | -------------------------------------------------------- | ---------- |
| `chunk` |      | A listener-facing stream chunk (chunk.choices[0].delta). | *required* |

Returns:

| Type             | Description                                                            |
| ---------------- | ---------------------------------------------------------------------- |
| `Optional[Type]` | A custom type object or None if the chunk is not for this custom type. |

Source code in `dspy/adapters/types/base_type.py`

```
@classmethod
def parse_stream_chunk(cls, chunk) -> Optional["Type"]:
    """
    Parse a stream chunk into the custom type.

    Args:
        chunk: A listener-facing stream chunk (`chunk.choices[0].delta`).

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
        if not all(_is_part(part) for part in formatted):
            raise TypeFormatError(
                f"{type(self).__name__}.format() must return a string or dspy.lm15 content parts; "
                "OpenAI-style content dictionaries are no longer accepted (DSPy 3.5)."
            )
        payload = json.dumps([part_to_dict(part) for part in formatted], ensure_ascii=False)
        return f"{CUSTOM_TYPE_START_IDENTIFIER}{payload}{CUSTOM_TYPE_END_IDENTIFIER}"
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

```
  March 11, 2026
```
