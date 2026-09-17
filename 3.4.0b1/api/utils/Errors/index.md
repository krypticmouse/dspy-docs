# dspy errors

DSPy exposes structured exception classes for LM failures and adapter parsing failures. LM errors inherit from `dspy.LMError`, which inherits from `dspy.DSPyError` and carries metadata such as `model`, `provider`, `status`, `request_id`, and `retry_after` when available.

Use `dspy.LMError` to catch any LM call failure, or catch a concrete subclass when you need specific handling.

```
try:
    result = program(question="...")
except dspy.ContextWindowExceededError as e:
    print(f"Prompt was too long for {e.model}")
except dspy.LMRateLimitError as e:
    print(f"Rate limited; retry after {e.retry_after} seconds")
except dspy.LMError as e:
    print(f"LM failed with code={e.code}, provider={e.provider}, request_id={e.request_id}")
```

Use `dspy.is_retryable_lm_error(error)` to classify LM failures that are generally safe to retry: rate limits, timeouts, server errors, and transport errors. DSPy’s built-in `dspy.LM` delegates provider retries to LiteLLM, but callers can use this helper after retries are exhausted. Retryability is advisory: respect provider policy and `retry_after` when present.

```
try:
    result = program(question="...")
except dspy.LMError as e:
    if dspy.is_retryable_lm_error(e):
        # Schedule another attempt later.
        raise
    raise
```

## API Reference

### Classes

#### `DSPyError(message: str = '', *, code: str | None = None, model: str | None = None, provider: str | None = None, provider_code: str | None = None, status: int | None = None, request_id: str | None = None, retry_after: float | None = None)`

Bases: `Exception`

Base class for DSPy errors with structured metadata.

Parameters:

| Name            | Type    | Description                   | Default                                                      |
| --------------- | ------- | ----------------------------- | ------------------------------------------------------------ |
| `message`       | `str`   | Human-readable error message. | `''`                                                         |
| `code`          | \`str   | None\`                        | Stable DSPy error code. Defaults to the class code.          |
| `model`         | \`str   | None\`                        | Model identifier involved in the failure.                    |
| `provider`      | \`str   | None\`                        | Provider or backend that returned the error.                 |
| `provider_code` | \`str   | None\`                        | Provider-specific error code, when available.                |
| `status`        | \`int   | None\`                        | HTTP status code, when the error came from an HTTP response. |
| `request_id`    | \`str   | None\`                        | Provider request ID, when available.                         |
| `retry_after`   | \`float | None\`                        | Suggested retry delay in seconds, when available.            |

Source code in `dspy/utils/exceptions.py`

```
def __init__(
    self,
    message: str = "",
    *,
    code: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    provider_code: str | None = None,
    status: int | None = None,
    request_id: str | None = None,
    retry_after: float | None = None,
):
    self.message = message
    self.code = code or self.default_code
    self.model = model
    self.provider = provider
    self.provider_code = provider_code
    self.status = status
    self.request_id = request_id
    self.retry_after = retry_after

    prefix = f"[{model}] " if model else ""
    super().__init__(f"{prefix}{message}" if message else prefix.rstrip())
```

##### Attributes

###### `default_code: str | None = None`

###### `message = message`

###### `code = code or self.default_code`

###### `model = model`

###### `provider = provider`

###### `provider_code = provider_code`

###### `status = status`

###### `request_id = request_id`

###### `retry_after = retry_after`

##### Methods:

#### `LMError(message: str = '', *, code: str | None = None, model: str | None = None, provider: str | None = None, provider_code: str | None = None, status: int | None = None, request_id: str | None = None, retry_after: float | None = None)`

Bases: `DSPyError`

Base class for language model errors.

Catch this class for failures at DSPy’s engine and capability boundaries. Concrete subclasses identify local configuration, transport, authentication, rate limits, invalid requests, unsupported features, and provider failures. Invalid Python API arguments still raise TypeError/ValueError; missing dependencies raise ImportError. Cancellation and warning policies propagate.

Source code in `dspy/utils/exceptions.py`

```
def __init__(
    self,
    message: str = "",
    *,
    code: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    provider_code: str | None = None,
    status: int | None = None,
    request_id: str | None = None,
    retry_after: float | None = None,
):
    self.message = message
    self.code = code or self.default_code
    self.model = model
    self.provider = provider
    self.provider_code = provider_code
    self.status = status
    self.request_id = request_id
    self.retry_after = retry_after

    prefix = f"[{model}] " if model else ""
    super().__init__(f"{prefix}{message}" if message else prefix.rstrip())
```

##### Attributes

###### `default_code = 'lm_error'`

#### `LMTransportError(message: str = '', *, code: str | None = None, model: str | None = None, provider: str | None = None, provider_code: str | None = None, status: int | None = None, request_id: str | None = None, retry_after: float | None = None)`

Bases: `LMError`

The LM request failed before the provider returned a response.

This commonly represents network, DNS, TLS, connection-reset, or similar client-side transport failures.

Source code in `dspy/utils/exceptions.py`

```
def __init__(
    self,
    message: str = "",
    *,
    code: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    provider_code: str | None = None,
    status: int | None = None,
    request_id: str | None = None,
    retry_after: float | None = None,
):
    self.message = message
    self.code = code or self.default_code
    self.model = model
    self.provider = provider
    self.provider_code = provider_code
    self.status = status
    self.request_id = request_id
    self.retry_after = retry_after

    prefix = f"[{model}] " if model else ""
    super().__init__(f"{prefix}{message}" if message else prefix.rstrip())
```

##### Attributes

###### `default_code = 'transport'`

#### `LMConfigurationError(message: str = '', *, code: str | None = None, model: str | None = None, provider: str | None = None, provider_code: str | None = None, status: int | None = None, request_id: str | None = None, retry_after: float | None = None)`

Bases: `LMError`

The LM or provider client is not configured correctly.

Source code in `dspy/utils/exceptions.py`

```
def __init__(
    self,
    message: str = "",
    *,
    code: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    provider_code: str | None = None,
    status: int | None = None,
    request_id: str | None = None,
    retry_after: float | None = None,
):
    self.message = message
    self.code = code or self.default_code
    self.model = model
    self.provider = provider
    self.provider_code = provider_code
    self.status = status
    self.request_id = request_id
    self.retry_after = retry_after

    prefix = f"[{model}] " if model else ""
    super().__init__(f"{prefix}{message}" if message else prefix.rstrip())
```

##### Attributes

###### `default_code = 'configuration'`

#### `LMNotConfiguredError(message: str = '', *, code: str | None = None, model: str | None = None, provider: str | None = None, provider_code: str | None = None, status: int | None = None, request_id: str | None = None, retry_after: float | None = None)`

Bases: `LMConfigurationError`

The LM is missing required provider configuration or credentials.

Source code in `dspy/utils/exceptions.py`

```
def __init__(
    self,
    message: str = "",
    *,
    code: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    provider_code: str | None = None,
    status: int | None = None,
    request_id: str | None = None,
    retry_after: float | None = None,
):
    self.message = message
    self.code = code or self.default_code
    self.model = model
    self.provider = provider
    self.provider_code = provider_code
    self.status = status
    self.request_id = request_id
    self.retry_after = retry_after

    prefix = f"[{model}] " if model else ""
    super().__init__(f"{prefix}{message}" if message else prefix.rstrip())
```

##### Attributes

###### `default_code = 'not_configured'`

#### `LMUnsupportedFeatureError(message: str = '', *, features: list[str] | None = None, issues: list[str] | None = None, **kwargs: Any)`

Bases: `LMError`

The LM, provider, or DSPy provider wrapper does not support a requested feature.

Parameters:

| Name       | Type        | Description                                      | Default                                                                                                        |
| ---------- | ----------- | ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| `message`  | `str`       | Human-readable error message.                    | `''`                                                                                                           |
| `features` | \`list[str] | None\`                                           | Feature names that were requested but unavailable, such as "finetuning", "reinforce", or "structured_outputs". |
| `issues`   | \`list[str] | None\`                                           | Optional detailed reasons the requested feature could not be used.                                             |
| `**kwargs` | `Any`       | Structured error metadata accepted by DSPyError. | `{}`                                                                                                           |

Source code in `dspy/utils/exceptions.py`

```
def __init__(
    self,
    message: str = "",
    *,
    features: list[str] | None = None,
    issues: list[str] | None = None,
    **kwargs: Any,
):
    self.features = list(features or [])
    self.issues = list(issues or [])
    super().__init__(message, **kwargs)
```

##### Attributes

###### `default_code = 'unsupported_feature'`

###### `features = list(features or [])`

###### `issues = list(issues or [])`

##### Methods:

#### `LMProviderError(message: str = '', *, code: str | None = None, model: str | None = None, provider: str | None = None, provider_code: str | None = None, status: int | None = None, request_id: str | None = None, retry_after: float | None = None)`

Bases: `LMError`

The provider returned an error response.

Provider errors include structured metadata when available, such as HTTP `status`, provider `request_id`, provider-specific `provider_code`, and `retry_after` for rate limits.

Source code in `dspy/utils/exceptions.py`

```
def __init__(
    self,
    message: str = "",
    *,
    code: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    provider_code: str | None = None,
    status: int | None = None,
    request_id: str | None = None,
    retry_after: float | None = None,
):
    self.message = message
    self.code = code or self.default_code
    self.model = model
    self.provider = provider
    self.provider_code = provider_code
    self.status = status
    self.request_id = request_id
    self.retry_after = retry_after

    prefix = f"[{model}] " if model else ""
    super().__init__(f"{prefix}{message}" if message else prefix.rstrip())
```

##### Attributes

###### `default_code = 'provider'`

#### `LMUnexpectedError(message: str = '', *, code: str | None = None, model: str | None = None, provider: str | None = None, provider_code: str | None = None, status: int | None = None, request_id: str | None = None, retry_after: float | None = None)`

Bases: `LMError`

An unexpected failure occurred at the LM provider boundary.

DSPy raises this for an unclassified engine failure and preserves its original exception as the cause. Only the owning engine interprets SDK errors: arbitrary message text or status-like attributes do not make a custom engine failure retryable. This is not a provider-response error or a model-output parsing error.

Source code in `dspy/utils/exceptions.py`

```
def __init__(
    self,
    message: str = "",
    *,
    code: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    provider_code: str | None = None,
    status: int | None = None,
    request_id: str | None = None,
    retry_after: float | None = None,
):
    self.message = message
    self.code = code or self.default_code
    self.model = model
    self.provider = provider
    self.provider_code = provider_code
    self.status = status
    self.request_id = request_id
    self.retry_after = retry_after

    prefix = f"[{model}] " if model else ""
    super().__init__(f"{prefix}{message}" if message else prefix.rstrip())
```

##### Attributes

###### `default_code = 'unexpected'`

#### `LMAuthError(message: str = '', *, code: str | None = None, model: str | None = None, provider: str | None = None, provider_code: str | None = None, status: int | None = None, request_id: str | None = None, retry_after: float | None = None)`

Bases: `LMProviderError`

The provider rejected the request because authentication failed.

Source code in `dspy/utils/exceptions.py`

```
def __init__(
    self,
    message: str = "",
    *,
    code: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    provider_code: str | None = None,
    status: int | None = None,
    request_id: str | None = None,
    retry_after: float | None = None,
):
    self.message = message
    self.code = code or self.default_code
    self.model = model
    self.provider = provider
    self.provider_code = provider_code
    self.status = status
    self.request_id = request_id
    self.retry_after = retry_after

    prefix = f"[{model}] " if model else ""
    super().__init__(f"{prefix}{message}" if message else prefix.rstrip())
```

##### Attributes

###### `default_code = 'auth'`

#### `LMBillingError(message: str = '', *, code: str | None = None, model: str | None = None, provider: str | None = None, provider_code: str | None = None, status: int | None = None, request_id: str | None = None, retry_after: float | None = None)`

Bases: `LMProviderError`

The provider rejected the request because billing or quota failed.

Source code in `dspy/utils/exceptions.py`

```
def __init__(
    self,
    message: str = "",
    *,
    code: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    provider_code: str | None = None,
    status: int | None = None,
    request_id: str | None = None,
    retry_after: float | None = None,
):
    self.message = message
    self.code = code or self.default_code
    self.model = model
    self.provider = provider
    self.provider_code = provider_code
    self.status = status
    self.request_id = request_id
    self.retry_after = retry_after

    prefix = f"[{model}] " if model else ""
    super().__init__(f"{prefix}{message}" if message else prefix.rstrip())
```

##### Attributes

###### `default_code = 'billing'`

#### `LMRateLimitError(message: str = '', *, code: str | None = None, model: str | None = None, provider: str | None = None, provider_code: str | None = None, status: int | None = None, request_id: str | None = None, retry_after: float | None = None)`

Bases: `LMProviderError`

The provider rate-limited the request.

Check the `retry_after` attribute for a provider-suggested retry delay when one is available.

Source code in `dspy/utils/exceptions.py`

```
def __init__(
    self,
    message: str = "",
    *,
    code: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    provider_code: str | None = None,
    status: int | None = None,
    request_id: str | None = None,
    retry_after: float | None = None,
):
    self.message = message
    self.code = code or self.default_code
    self.model = model
    self.provider = provider
    self.provider_code = provider_code
    self.status = status
    self.request_id = request_id
    self.retry_after = retry_after

    prefix = f"[{model}] " if model else ""
    super().__init__(f"{prefix}{message}" if message else prefix.rstrip())
```

##### Attributes

###### `default_code = 'rate_limit'`

#### `LMInvalidRequestError(message: str = '', *, code: str | None = None, model: str | None = None, provider: str | None = None, provider_code: str | None = None, status: int | None = None, request_id: str | None = None, retry_after: float | None = None)`

Bases: `LMProviderError`

The provider rejected the request shape or resource.

Source code in `dspy/utils/exceptions.py`

```
def __init__(
    self,
    message: str = "",
    *,
    code: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    provider_code: str | None = None,
    status: int | None = None,
    request_id: str | None = None,
    retry_after: float | None = None,
):
    self.message = message
    self.code = code or self.default_code
    self.model = model
    self.provider = provider
    self.provider_code = provider_code
    self.status = status
    self.request_id = request_id
    self.retry_after = retry_after

    prefix = f"[{model}] " if model else ""
    super().__init__(f"{prefix}{message}" if message else prefix.rstrip())
```

##### Attributes

###### `default_code = 'invalid_request'`

#### `ContextWindowExceededError(*, model: str | None = None, message: str = 'Context window exceeded', **kwargs: Any)`

Bases: `LMInvalidRequestError`

Raised when the prompt exceeds the model’s context window.

Custom engines raise dspy.lm15.ContextLengthError; DSPy projects it to this public type. Legacy 3.4 LM subclasses may still raise this error directly. Modules such as ReAct catch it to shorten an overlong history; it does not trigger a generation retry or adapter-format fallback.

Parameters:

| Name       | Type  | Description                                                        | Default                                         |
| ---------- | ----- | ------------------------------------------------------------------ | ----------------------------------------------- |
| `model`    | \`str | None\`                                                             | The model identifier that rejected the request. |
| `message`  | `str` | Description of the error. Defaults to "Context window exceeded".   | `'Context window exceeded'`                     |
| `**kwargs` | `Any` | Structured error metadata such as provider, status, or request_id. | `{}`                                            |

Source code in `dspy/utils/exceptions.py`

```
def __init__(
    self,
    *,
    model: str | None = None,
    message: str = "Context window exceeded",
    **kwargs: Any,
):
    super().__init__(message, model=model, **kwargs)
```

##### Attributes

###### `default_code = 'context_window_exceeded'`

##### Methods:

#### `LMUnsupportedModelError(message: str = '', *, code: str | None = None, model: str | None = None, provider: str | None = None, provider_code: str | None = None, status: int | None = None, request_id: str | None = None, retry_after: float | None = None)`

Bases: `LMInvalidRequestError`

The requested model is unavailable or unsupported by the provider.

Source code in `dspy/utils/exceptions.py`

```
def __init__(
    self,
    message: str = "",
    *,
    code: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    provider_code: str | None = None,
    status: int | None = None,
    request_id: str | None = None,
    retry_after: float | None = None,
):
    self.message = message
    self.code = code or self.default_code
    self.model = model
    self.provider = provider
    self.provider_code = provider_code
    self.status = status
    self.request_id = request_id
    self.retry_after = retry_after

    prefix = f"[{model}] " if model else ""
    super().__init__(f"{prefix}{message}" if message else prefix.rstrip())
```

##### Attributes

###### `default_code = 'unsupported_model'`

#### `LMTimeoutError(message: str = '', *, code: str | None = None, model: str | None = None, provider: str | None = None, provider_code: str | None = None, status: int | None = None, request_id: str | None = None, retry_after: float | None = None)`

Bases: `LMProviderError`

The provider request timed out.

Source code in `dspy/utils/exceptions.py`

```
def __init__(
    self,
    message: str = "",
    *,
    code: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    provider_code: str | None = None,
    status: int | None = None,
    request_id: str | None = None,
    retry_after: float | None = None,
):
    self.message = message
    self.code = code or self.default_code
    self.model = model
    self.provider = provider
    self.provider_code = provider_code
    self.status = status
    self.request_id = request_id
    self.retry_after = retry_after

    prefix = f"[{model}] " if model else ""
    super().__init__(f"{prefix}{message}" if message else prefix.rstrip())
```

##### Attributes

###### `default_code = 'timeout'`

#### `LMServerError(message: str = '', *, code: str | None = None, model: str | None = None, provider: str | None = None, provider_code: str | None = None, status: int | None = None, request_id: str | None = None, retry_after: float | None = None)`

Bases: `LMProviderError`

The provider failed while handling the request.

Source code in `dspy/utils/exceptions.py`

```
def __init__(
    self,
    message: str = "",
    *,
    code: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    provider_code: str | None = None,
    status: int | None = None,
    request_id: str | None = None,
    retry_after: float | None = None,
):
    self.message = message
    self.code = code or self.default_code
    self.model = model
    self.provider = provider
    self.provider_code = provider_code
    self.status = status
    self.request_id = request_id
    self.retry_after = retry_after

    prefix = f"[{model}] " if model else ""
    super().__init__(f"{prefix}{message}" if message else prefix.rstrip())
```

##### Attributes

###### `default_code = 'server'`

#### `AdapterParseError(adapter_name: str, signature: Signature, lm_response: str, message: str | None = None, parsed_result: str | None = None)`

Bases: `DSPyError`

Raised when an adapter cannot parse an LM response into signature outputs.

Parameters:

| Name            | Type        | Description                                            | Default                                              |
| --------------- | ----------- | ------------------------------------------------------ | ---------------------------------------------------- |
| `adapter_name`  | `str`       | Name of the adapter that failed to parse the response. | *required*                                           |
| `signature`     | `Signature` | DSPy signature whose output fields were expected.      | *required*                                           |
| `lm_response`   | `str`       | Raw LM response text or representation being parsed.   | *required*                                           |
| `message`       | \`str       | None\`                                                 | Optional additional context about the parse failure. |
| `parsed_result` | \`str       | None\`                                                 | Partial parsed result, if any.                       |

Source code in `dspy/utils/exceptions.py`

```
def __init__(
    self,
    adapter_name: str,
    signature: Signature,
    lm_response: str,
    message: str | None = None,
    parsed_result: str | None = None,
):
    self.adapter_name = adapter_name
    self.signature = signature
    self.lm_response = lm_response
    self.parsed_result = parsed_result

    message = f"{message}\n\n" if message else ""
    message = (
        f"{message}"
        f"Adapter {adapter_name} failed to parse the LM response. \n\n"
        f"LM Response: {lm_response} \n\n"
        f"Expected to find output fields in the LM response: [{', '.join(signature.output_fields.keys())}] \n\n"
    )

    if parsed_result is not None:
        message += f"Actual output fields parsed from the LM response: [{', '.join(parsed_result.keys())}] \n\n"

    super().__init__(message)
```

##### Attributes

###### `default_code = 'adapter_parse_error'`

###### `adapter_name = adapter_name`

###### `signature = signature`

###### `lm_response = lm_response`

###### `parsed_result = parsed_result`

##### Methods:

### Functions:

#### `is_retryable_lm_error(error: Exception) -> bool`

Return whether an LM error is generally safe to retry.

DSPy owns retries; managed engines perform one attempt. This classification describes transient failures, not proof that the provider did no work. Never replay after visible stream output or a completed generation, and respect provider policy and valid `retry_after` hints. Network retries can still repeat a request already processed or billed by the provider.

Parameters:

| Name    | Type        | Description                | Default    |
| ------- | ----------- | -------------------------- | ---------- |
| `error` | `Exception` | The exception to classify. | *required* |

Source code in `dspy/utils/exceptions.py`

```
def is_retryable_lm_error(error: Exception) -> bool:
    """Return whether an LM error is generally safe to retry.

    DSPy owns retries; managed engines perform one attempt. This classification
    describes transient failures, not proof that the provider did no work.
    Never replay after visible stream output or a completed generation, and
    respect provider policy and valid `retry_after` hints. Network retries can
    still repeat a request already processed or billed by the provider.

    Args:
        error: The exception to classify.
    """
    return isinstance(error, _RETRYABLE_LM_ERRORS)
```
