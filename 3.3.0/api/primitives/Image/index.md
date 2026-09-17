# dspy.Image

## `dspy.Image(source: Any = None, /, *, download: Any = _UNSET, verify: Any = _UNSET, **data)`

Bases: `Type`

Create an Image.

#### Parameters

source: The positional-only image source. Supported values include

```
- ``str``: HTTP(S)/GS URL or an encoded data URI
- ``bytes``: raw image bytes
- ``PIL.Image.Image``: a PIL image instance
- ``dict`` with a single ``{"url": value}`` entry (legacy form)
- already encoded data URI
```

download, verify: Deprecated. Use :meth:`from_url` (which accepts `verify`) to download a remote image, or :meth:`from_path` for a local file. Accepted for one release with a warning: `download=True` eagerly fetches an HTTP(S) `source`.

Any additional keyword arguments are passed to :class:`pydantic.BaseModel`.

Ordinary construction never touches the filesystem or network. Local files and remote resources must be loaded explicitly with :meth:`from_path` and :meth:`from_url`. The deprecated positional `Image(url, download=True)` compatibility call also downloads.

Source code in `dspy/adapters/types/image.py`

```
def __init__(self, source: Any = None, /, *, download: Any = _UNSET, verify: Any = _UNSET, **data):
    """Create an Image.

    Parameters
    ----------
    source:
        The positional-only image source. Supported values include

        - ``str``: HTTP(S)/GS URL or an encoded data URI
        - ``bytes``: raw image bytes
        - ``PIL.Image.Image``: a PIL image instance
        - ``dict`` with a single ``{"url": value}`` entry (legacy form)
        - already encoded data URI

    download, verify:
        Deprecated. Use :meth:`from_url` (which accepts ``verify``) to download a
        remote image, or :meth:`from_path` for a local file. Accepted for one release
        with a warning: ``download=True`` eagerly fetches an HTTP(S) ``source``.

    Any additional keyword arguments are passed to :class:`pydantic.BaseModel`.

    Ordinary construction never touches the filesystem or network. Local files and remote
    resources must be loaded explicitly with :meth:`from_path` and :meth:`from_url`. The
    deprecated positional ``Image(url, download=True)`` compatibility call also downloads.
    """

    download_requested = download is not _UNSET
    verify_requested = verify is not _UNSET
    if (download_requested or verify_requested) and source is None:
        # `download`/`verify` are a compatibility shim for the positional constructor
        # `Image(url, download=True)`. They must never be honored through the validation
        # path: pydantic routes dict data into `__init__`, so an untrusted value such as
        # `{"url": "http://169.254.169.254/...", "download": true}` would otherwise trigger
        # a server-side fetch during output parsing. Requiring a positional source keeps the
        # shim reachable only from direct developer construction.
        raise TypeError(
            "`download` and `verify` are only valid with a positional image source; "
            "use Image.from_url(url, verify=...) to download a remote image."
        )
    if download_requested or verify_requested:
        warnings.warn(
            "The `download` and `verify` arguments to Image() are deprecated and will be removed in "
            "3.4. Use Image.from_url(url, verify=...) to download a remote image, or "
            "Image.from_path(path) for a local file.",
            DeprecationWarning,
            stacklevel=2,
        )

    if source is not None and "url" in data:
        raise TypeError("Image received both `source` and `url`; provide only one image source")

    if source is not None:
        # Support a positional source while retaining the Pydantic ``url`` field.
        if isinstance(source, dict) and set(source.keys()) == {"url"}:
            # Legacy dict form from previous model validator.
            data["url"] = source["url"]
        else:
            data["url"] = source

    url = data.get("url")
    if download_requested and download and isinstance(url, str) and _is_http_url(url):
        # Legacy download=True path: eagerly fetch the remote URL.
        data["url"] = _encode_image_from_url(url, verify=verify if verify_requested else True)
    elif "url" in data:
        # Normalize any accepted input into a base64 data URI or plain URL.
        data["url"] = encode_image(data["url"])

    # Delegate the rest of initialization to pydantic's BaseModel.
    super().__init__(**data)
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
        "Image.from_PIL is deprecated and will be removed in 3.4; use Image(pil_image) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return cls(pil_image)
```

#### `from_file(file_path: str) -> Image`

Deprecated alias for :meth:`from_path`.

Source code in `dspy/adapters/types/image.py`

```
@classmethod
def from_file(cls, file_path: str) -> "Image":
    """Deprecated alias for :meth:`from_path`."""
    warnings.warn(
        "Image.from_file is deprecated and will be removed in 3.4; use Image.from_path instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return cls.from_path(file_path)
```

#### `from_path(file_path: str) -> Image`

Read a local file and encode it as a data URI.

Source code in `dspy/adapters/types/image.py`

```
@classmethod
def from_path(cls, file_path: str) -> "Image":
    """Read a local file and encode it as a data URI."""
    if not os.path.isfile(file_path):
        raise ValueError(f"File not found: {file_path}")
    return cls(_encode_image_from_file(file_path))
```

#### `from_url(url: str, verify: bool = True) -> Image`

Download an HTTP(S) resource and encode it as a data URI.

Security: this performs an explicit, caller-initiated fetch and applies no SSRF protection beyond requiring an HTTP(S) scheme. Like `requests.get`, it will reach private, loopback, or cloud-metadata hosts and follow redirects to them. When `url` is derived from untrusted input, the caller is responsible for validating the host against an allowlist before calling this method.

Source code in `dspy/adapters/types/image.py`

```
@classmethod
def from_url(cls, url: str, verify: bool = True) -> "Image":
    """Download an HTTP(S) resource and encode it as a data URI.

    Security: this performs an explicit, caller-initiated fetch and applies no
    SSRF protection beyond requiring an HTTP(S) scheme. Like ``requests.get``, it
    will reach private, loopback, or cloud-metadata hosts and follow redirects to
    them. When ``url`` is derived from untrusted input, the caller is responsible
    for validating the host against an allowlist before calling this method.
    """
    if not _is_http_url(url):
        raise ValueError(f"Image.from_url requires an HTTP(S) URL, received: {url}")
    return cls(_encode_image_from_url(url, verify=verify))
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
