# dspy.Image

## `dspy.Image(url: Any = None, *, download: bool = False, verify: bool = True, **data)`

Bases: `Type`

Create an Image.

#### Parameters

url: The image source. Supported values include

```
- ``str``: HTTP(S)/GS URL or local file path
- ``bytes``: raw image bytes
- ``PIL.Image.Image``: a PIL image instance
- ``dict`` with a single ``{"url": value}`` entry (legacy form)
- already encoded data URI
```

download

Whether remote URLs should be downloaded to infer their MIME type.

verify

Whether to verify SSL certificates when downloading images from URLs. Set to False for self-signed certificates. Default is True.

Any additional keyword arguments are passed to :class:`pydantic.BaseModel`.

Source code in `dspy/adapters/types/image.py`

```
def __init__(self, url: Any = None, *, download: bool = False, verify: bool = True, **data):
    """Create an Image.

    Parameters
    ----------
    url:
        The image source. Supported values include

        - ``str``: HTTP(S)/GS URL or local file path
        - ``bytes``: raw image bytes
        - ``PIL.Image.Image``: a PIL image instance
        - ``dict`` with a single ``{"url": value}`` entry (legacy form)
        - already encoded data URI

    download:
        Whether remote URLs should be downloaded to infer their MIME type.

    verify:
        Whether to verify SSL certificates when downloading images from URLs.
        Set to False for self-signed certificates. Default is True.

    Any additional keyword arguments are passed to :class:`pydantic.BaseModel`.
    """

    if url is not None and "url" not in data:
        # Support a positional argument while allowing ``url=`` in **data.
        if isinstance(url, dict) and set(url.keys()) == {"url"}:
            # Legacy dict form from previous model validator.
            data["url"] = url["url"]
        else:
            # ``url`` may be a string, bytes, or a PIL image.
            data["url"] = url

    if "url" in data:
        # Normalize any accepted input into a base64 data URI or plain URL.
        data["url"] = encode_image(data["url"], download_images=download, verify=verify)

    # Delegate the rest of initialization to pydantic's BaseModel.
    super().__init__(**data)
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

#### `format() -> list[dict[str, Any]] | str`

Source code in `dspy/adapters/types/image.py`

```
@lru_cache(maxsize=32)
def format(self) -> list[dict[str, Any]] | str:
    try:
        image_url = encode_image(self.url)
    except Exception as e:
        raise ValueError(f"Failed to format image for DSPy: {e}")
    return [{"type": "image_url", "image_url": {"url": image_url}}]
```

#### `from_PIL(pil_image)`

Source code in `dspy/adapters/types/image.py`

```
@classmethod
def from_PIL(cls, pil_image):  # noqa: N802
    warnings.warn(
        "Image.from_PIL is deprecated; use Image(pil_image) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return cls(pil_image)
```

#### `from_file(file_path: str)`

Source code in `dspy/adapters/types/image.py`

```
@classmethod
def from_file(cls, file_path: str):
    warnings.warn(
        "Image.from_file is deprecated; use Image(file_path) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return cls(file_path)
```

#### `from_url(url: str, download: bool = False)`

Source code in `dspy/adapters/types/image.py`

```
@classmethod
def from_url(cls, url: str, download: bool = False):
    warnings.warn(
        "Image.from_url is deprecated; use Image(url) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return cls(url, download=download)
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
