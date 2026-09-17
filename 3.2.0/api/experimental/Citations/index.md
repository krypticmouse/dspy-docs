# dspy.experimental.Citations

## `dspy.experimental.Citations`

Bases: `Type`

Citations extracted from an LM response with source references.

This type represents citations returned by language models that support citation extraction, particularly Anthropic's Citations API through LiteLLM. Citations include the quoted text and source information.

Examples:

```
import os
import dspy
from dspy.signatures import Signature
from dspy.experimental import Citations, Document
os.environ["ANTHROPIC_API_KEY"] = "YOUR_ANTHROPIC_API_KEY"

class AnswerWithSources(Signature):
    '''Answer questions using provided documents with citations.'''
    documents: list[Document] = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    citations: Citations = dspy.OutputField()

# Create documents to provide as sources
docs = [
    Document(
        data="The Earth orbits the Sun in an elliptical path.",
        title="Basic Astronomy Facts"
    ),
    Document(
        data="Water boils at 100°C at standard atmospheric pressure.",
        title="Physics Fundamentals",
        metadata={"author": "Dr. Smith", "year": 2023}
    )
]

# Use with a model that supports citations like Claude
lm = dspy.LM("anthropic/claude-opus-4-1-20250805")
predictor = dspy.Predict(AnswerWithSources)
result = predictor(documents=docs, question="What temperature does water boil?", lm=lm)

for citation in result.citations.citations:
    print(citation.format())
```

### Functions

#### `adapt_to_native_lm_feature(signature, field_name, lm, lm_kwargs) -> bool`

Source code in `dspy/adapters/types/citation.py`

```
@classmethod
def adapt_to_native_lm_feature(cls, signature, field_name, lm, lm_kwargs) -> bool:
    if lm.model.startswith("anthropic/"):
        return signature.delete(field_name)
    return signature
```

#### `description() -> str`

Description of the citations type for use in prompts.

Source code in `dspy/adapters/types/citation.py`

```
@classmethod
def description(cls) -> str:
    """Description of the citations type for use in prompts."""
    return (
        "Citations with quoted text and source references. "
        "Include the exact text being cited and information about its source."
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

Format citations as a list of dictionaries.

Source code in `dspy/adapters/types/citation.py`

```
def format(self) -> list[dict[str, Any]]:
    """Format citations as a list of dictionaries."""
    return [citation.format() for citation in self.citations]
```

#### `from_dict_list(citations_dicts: list[dict[str, Any]]) -> Citations`

Convert a list of dictionaries to a Citations instance.

Parameters:

| Name              | Type                   | Description                                                                                                                                 | Default    |
| ----------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `citations_dicts` | `list[dict[str, Any]]` | A list of dictionaries, where each dictionary should have 'cited_text' key and 'document_index', 'start_char_index', 'end_char_index' keys. | *required* |

Returns:

| Type        | Description           |
| ----------- | --------------------- |
| `Citations` | A Citations instance. |

Examples:

```
citations_dict = [
    {
        "cited_text": "The sky is blue",
        "document_index": 0,
        "document_title": "Weather Guide",
        "start_char_index": 0,
        "end_char_index": 15,
        "supported_text": "The sky was blue yesterday."
    }
]
citations = Citations.from_dict_list(citations_dict)
```

Source code in `dspy/adapters/types/citation.py`

````
@classmethod
def from_dict_list(cls, citations_dicts: list[dict[str, Any]]) -> "Citations":
    """Convert a list of dictionaries to a Citations instance.

    Args:
        citations_dicts: A list of dictionaries, where each dictionary should have 'cited_text' key
            and 'document_index', 'start_char_index', 'end_char_index' keys.

    Returns:
        A Citations instance.

    Examples:
        ```python
        citations_dict = [
            {
                "cited_text": "The sky is blue",
                "document_index": 0,
                "document_title": "Weather Guide",
                "start_char_index": 0,
                "end_char_index": 15,
                "supported_text": "The sky was blue yesterday."
            }
        ]
        citations = Citations.from_dict_list(citations_dict)
        ```
    """
    citations = [cls.Citation(**item) for item in citations_dicts]
    return cls(citations=citations)
````

#### `is_streamable() -> bool`

Whether the Citations type is streamable.

Source code in `dspy/adapters/types/citation.py`

```
@classmethod
def is_streamable(cls) -> bool:
    """Whether the Citations type is streamable."""
    return True
```

#### `parse_lm_response(response: str | dict[str, Any]) -> Optional[Citations]`

Parse a LM response into Citations.

Parameters:

| Name       | Type  | Description      | Default                                       |
| ---------- | ----- | ---------------- | --------------------------------------------- |
| `response` | \`str | dict[str, Any]\` | A LM response that may contain citation data. |

Returns:

| Type                  | Description                                                   |
| --------------------- | ------------------------------------------------------------- |
| `Optional[Citations]` | A Citations object if citation data is found, None otherwise. |

Source code in `dspy/adapters/types/citation.py`

```
@classmethod
def parse_lm_response(cls, response: str | dict[str, Any]) -> Optional["Citations"]:
    """Parse a LM response into Citations.

    Args:
        response: A LM response that may contain citation data.

    Returns:
        A Citations object if citation data is found, None otherwise.
    """
    if isinstance(response, dict):
        # Check if the response contains citations in the expected format
        if "citations" in response:
            citations_data = response["citations"]
            if isinstance(citations_data, list):
                return cls.from_dict_list(citations_data)

    return None
```

#### `parse_stream_chunk(chunk) -> Optional[Citations]`

Parse a stream chunk into Citations.

Parameters:

| Name    | Type | Description                 | Default    |
| ------- | ---- | --------------------------- | ---------- |
| `chunk` |      | A stream chunk from the LM. | *required* |

Returns:

| Type                  | Description                                                             |
| --------------------- | ----------------------------------------------------------------------- |
| `Optional[Citations]` | A Citations object if the chunk contains citation data, None otherwise. |

Source code in `dspy/adapters/types/citation.py`

```
@classmethod
def parse_stream_chunk(cls, chunk) -> Optional["Citations"]:
    """
    Parse a stream chunk into Citations.

    Args:
        chunk: A stream chunk from the LM.

    Returns:
        A Citations object if the chunk contains citation data, None otherwise.
    """
    try:
        # Check if the chunk has citation data in provider_specific_fields
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "provider_specific_fields") and delta.provider_specific_fields:
                citation_data = delta.provider_specific_fields.get("citation")
                if citation_data:
                    return cls.from_dict_list([citation_data])
    except Exception:
        pass
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

Source code in `dspy/adapters/types/citation.py`

```
@pydantic.model_validator(mode="before")
@classmethod
def validate_input(cls, data: Any):
    if isinstance(data, cls):
        return data

    # Handle case where data is a list of dicts with citation info
    if isinstance(data, list) and all(isinstance(item, dict) and "cited_text" in item for item in data):
        return {"citations": [cls.Citation(**item) for item in data]}

    # Handle case where data is a dict
    elif isinstance(data, dict):
        if "citations" in data:
            # Handle case where data is a dict with "citations" key
            citations_data = data["citations"]
            if isinstance(citations_data, list):
                return {
                    "citations": [
                        cls.Citation(**item) if isinstance(item, dict) else item for item in citations_data
                    ]
                }
        elif "cited_text" in data:
            # Handle case where data is a single citation dict
            return {"citations": [cls.Citation(**data)]}

    raise ValueError(f"Received invalid value for `Citations`: {data}")
```
