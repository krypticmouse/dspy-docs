# dspy.Audio

## `dspy.Audio(*args, **data)`

Bases: `Type`

An audio input type for DSPy.

Construct from in-memory values only: a data URI string, raw `bytes` (with `audio_format=`), a numpy-like array, a `{"data", "audio_format"}` dict, or another `Audio`. Raw base64 is passed as `Audio(data=..., audio_format=...)`. Construction and adapter parsing never access the filesystem or network; use :meth:`from_path` to read a local file, :meth:`from_url` to download a remote resource, or :meth:`from_array` for array data.

Source code in `dspy/adapters/types/audio.py`

```
def __init__(self, *args, **data):
    if len(args) > 1:
        raise TypeError(f"Audio expected at most 1 positional argument, received {len(args)}")
    if args:
        if "data" in data:
            raise TypeError("Audio received data as both a positional and keyword argument")
        value = args[0]
        sampling_rate = data.pop("sampling_rate", None)
        audio_format = data.pop("audio_format", None)
        if audio_format is not None and _carries_own_format(value):
            raise TypeError(
                "Audio received audio_format alongside an input that already carries its format; provide only one"
            )
        if sampling_rate is not None and not hasattr(value, "shape"):
            raise TypeError("Audio received sampling_rate for a non-array input; it only applies to array data")
        normalized = encode_audio(value, sampling_rate=sampling_rate or 16000, format=audio_format or "wav")
        normalized.update(data)
        data = normalized
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

#### `format() -> list[AudioPart]`

Source code in `dspy/adapters/types/audio.py`

```
def format(self) -> list[AudioPart]:
    try:
        data = self.data
    except Exception as e:
        raise ValueError(f"Failed to format audio for DSPy: {e}")
    media_type = {"mp3": "audio/mpeg", "mpeg": "audio/mpeg"}.get(self.audio_format, f"audio/{self.audio_format}")
    return [audio(data=data, media_type=media_type)]
```

#### `from_array(array: Any, sampling_rate: int, format: str = 'wav') -> Audio`

Process numpy-like array and encode it as base64. Uses sampling rate and audio format for encoding.

Source code in `dspy/adapters/types/audio.py`

```
@classmethod
def from_array(cls, array: Any, sampling_rate: int, format: str = "wav") -> "Audio":
    """
    Process numpy-like array and encode it as base64. Uses sampling rate and audio format for encoding.
    """
    if not SF_AVAILABLE:
        raise ImportError("soundfile is required to process audio arrays.")

    byte_buffer = io.BytesIO()
    sf.write(
        byte_buffer,
        array,
        sampling_rate,
        format=format.upper(),
        subtype="PCM_16",
    )
    encoded_data = base64.b64encode(byte_buffer.getvalue()).decode("utf-8")
    return cls(data=encoded_data, audio_format=format)
```

#### `from_file(file_path: str) -> Audio`

Deprecated alias for :meth:`from_path`.

Source code in `dspy/adapters/types/audio.py`

```
@classmethod
def from_file(cls, file_path: str) -> "Audio":
    """Deprecated alias for :meth:`from_path`."""
    warnings.warn(
        "Audio.from_file is deprecated and will be removed in 3.4; use Audio.from_path instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return cls.from_path(file_path)
```

#### `from_path(file_path: str) -> Audio`

Read local audio file and encode it as base64.

Source code in `dspy/adapters/types/audio.py`

```
@classmethod
def from_path(cls, file_path: str) -> "Audio":
    """
    Read local audio file and encode it as base64.
    """
    if not os.path.isfile(file_path):
        raise ValueError(f"File not found: {file_path}")

    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("audio/"):
        raise ValueError(f"Unsupported MIME type for audio: {mime_type}")

    with open(file_path, "rb") as file:
        file_data = file.read()

    audio_format = mime_type.split("/")[1]

    audio_format = _normalize_audio_format(audio_format)

    encoded_data = base64.b64encode(file_data).decode("utf-8")
    return cls(data=encoded_data, audio_format=audio_format)
```

#### `from_url(url: str, verify: bool = True, timeout: float | None = 30.0) -> Audio`

Download an audio file from URL and encode it as base64.

Security: this performs an explicit, caller-initiated fetch and applies no SSRF protection beyond requiring an HTTP(S) scheme. Like `requests.get`, it will reach private, loopback, or cloud-metadata hosts and follow redirects to them. When `url` is derived from untrusted input, the caller is responsible for validating the host against an allowlist before calling this method.

Parameters:

| Name      | Type    | Description                                                             | Default                                                                                                                                                 |
| --------- | ------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `url`     | `str`   | The URL of the audio to download.                                       | *required*                                                                                                                                              |
| `verify`  | `bool`  | Whether to verify SSL certificates. Set to False for self-signed certs. | `True`                                                                                                                                                  |
| `timeout` | \`float | None\`                                                                  | Seconds to wait for the server before raising requests.exceptions.Timeout. None disables the timeout and can hang indefinitely on an unresponsive host. |

Source code in `dspy/adapters/types/audio.py`

```
@classmethod
def from_url(cls, url: str, verify: bool = True, timeout: float | None = 30.0) -> "Audio":
    """
    Download an audio file from URL and encode it as base64.

    Security: this performs an explicit, caller-initiated fetch and applies no
    SSRF protection beyond requiring an HTTP(S) scheme. Like ``requests.get``, it
    will reach private, loopback, or cloud-metadata hosts and follow redirects to
    them. When ``url`` is derived from untrusted input, the caller is responsible
    for validating the host against an allowlist before calling this method.

    Args:
        url: The URL of the audio to download.
        verify: Whether to verify SSL certificates. Set to False for self-signed certs.
        timeout: Seconds to wait for the server before raising
            ``requests.exceptions.Timeout``. ``None`` disables the timeout and can
            hang indefinitely on an unresponsive host.
    """
    parsed_url = urlparse(url)
    if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
        raise ValueError(f"Audio.from_url requires an HTTP(S) URL, received: {url}")
    response = requests.get(url, verify=verify, timeout=timeout)
    response.raise_for_status()
    mime_type = response.headers.get("Content-Type", "audio/wav")
    if not mime_type.startswith("audio/"):
        raise ValueError(f"Unsupported MIME type for audio: {mime_type}")
    audio_format = mime_type.split("/")[1]

    audio_format = _normalize_audio_format(audio_format)

    encoded_data = base64.b64encode(response.content).decode("utf-8")
    return cls(data=encoded_data, audio_format=audio_format)
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

#### `validate_input(values: Any) -> Any`

Validate input for Audio, expecting ‘data’ and ‘audio_format’ keys in dictionary.

Source code in `dspy/adapters/types/audio.py`

```
@pydantic.model_validator(mode="before")
@classmethod
def validate_input(cls, values: Any) -> Any:
    """
    Validate input for Audio, expecting 'data' and 'audio_format' keys in dictionary.
    """
    if isinstance(values, cls):
        return {"data": values.data, "audio_format": values.audio_format}
    return encode_audio(values)
```

```
  July 31, 2026
```
