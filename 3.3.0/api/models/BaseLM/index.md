# dspy.BaseLM

## `dspy.BaseLM(model, model_type='chat', temperature=None, max_tokens=None, cache=True, callbacks: list[BaseCallback] | None = None, num_retries: int = 3, **kwargs)`

Base class for DSPy language models.

Most users should use `dspy.LM`, which is a `BaseLM` subclass.

For advanced use cases, such as custom language model backends, users can subclass `BaseLM` and implement `forward()`.

DSPy is migrating `forward()` from the legacy OpenAI/LiteLLM-shaped contract to a typed DSPy contract. During this migration, subclasses should declare which contract they implement with `forward_contract`:

- `forward_contract = "typed_lm"`: implement `forward(request: dspy.LMRequest) -> dspy.LMResponse`. This is the preferred contract for new custom LMs.
- `forward_contract = "legacy"`: implement `forward(prompt=None, messages=None, **kwargs)` and return an OpenAI-like provider response. This remains the default during the migration.

`BaseLM.__call__()` is the compatibility boundary. In DSPy 3.3 and 3.4, ordinary calls preserve the legacy public return value, `list[str | dict]`:

```
outputs = lm("What is DSPy?")
outputs = lm(messages=[{"role": "user", "content": "What is DSPy?"}])
```

Calls can flow internally through the typed `LMRequest` / `LMResponse` path without changing the public return shape. The typed path is used when the caller passes an explicit `dspy.LMRequest`, when `dspy.context(experimental=True)` is active, or when the subclass declares `forward_contract = "typed_lm"`. It accepts richer direct-call inputs, including `dspy.System`, `dspy.User`, `dspy.Assistant`, `dspy.ToolResult`, content parts, and prior `dspy.LMResponse` objects. The public return value remains legacy outputs unless the caller explicitly opts into typed output with an `LMRequest` or `experimental=True`.

Example typed direct call:

```
with dspy.context(experimental=True):
    response = lm(
        dspy.System("You are concise."),
        dspy.User("What is DSPy?"),
    )
    print(response.text)
```

`LMResponse` is designed to feel familiar to users of the legacy output list while carrying substantially more structure, including typed outputs, usage, cache status, provider metadata, tool calls, reasoning, citations, and multimodal content.

LMs must be serializable as part of saved DSPy programs. The default `dump_state()` and `load_state()` implementations support subclasses whose persistent state is fully captured by `BaseLM.__init__()` arguments. If a subclass stores additional persistent state, override both methods.

Examples:

Preferred typed custom LM:

```
import dspy


class EchoLM(dspy.BaseLM):
    forward_contract = "typed_lm"

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        return dspy.LMResponse.from_text("hello", model=request.model)


lm = EchoLM(model="test/echo")

with dspy.context(experimental=True):
    response = lm(dspy.User("Say hello."))
    print(response.text)
```

Legacy custom LM for an OpenAI-like provider:

```
from openai import OpenAI

import dspy


class MyLegacyLM(dspy.BaseLM):
    forward_contract = "legacy"

    def forward(self, prompt=None, messages=None, **kwargs):
        client = OpenAI()
        return client.chat.completions.create(
            model=self.model,
            messages=messages or [{"role": "user", "content": prompt}],
            **self.kwargs,
            **kwargs,
        )


lm = MyLegacyLM(model="gpt-4o-mini")
dspy.configure(lm=lm)
print(dspy.Predict("q -> a")(q="Why did the chicken cross the kitchen?"))
```

Initialize a base language model.

Parameters:

| Name          | Type                 | Description                                                  | Default                                    |
| ------------- | -------------------- | ------------------------------------------------------------ | ------------------------------------------ |
| `model`       |                      | The model identifier.                                        | *required*                                 |
| `model_type`  |                      | The LM API type, such as "chat", "text", or "responses".     | `'chat'`                                   |
| `temperature` |                      | The default sampling temperature.                            | `None`                                     |
| `max_tokens`  |                      | The default maximum number of output tokens.                 | `None`                                     |
| `cache`       |                      | Whether requests should use DSPy’s cache by default.         | `True`                                     |
| `num_retries` | `int`                | The default number of provider request retries.              | `3`                                        |
| `callbacks`   | \`list[BaseCallback] | None\`                                                       | Optional instance-level callback handlers. |
| `**kwargs`    |                      | Additional default request parameters stored in self.kwargs. | `{}`                                       |

Source code in `dspy/clients/base_lm.py`

```
def __init__(
    self,
    model,
    model_type="chat",
    temperature=None,
    max_tokens=None,
    cache=True,
    callbacks: list[BaseCallback] | None = None,
    num_retries: int = 3,
    **kwargs,
):
    """Initialize a base language model.

    Args:
        model: The model identifier.
        model_type: The LM API type, such as `"chat"`, `"text"`, or
            `"responses"`.
        temperature: The default sampling temperature.
        max_tokens: The default maximum number of output tokens.
        cache: Whether requests should use DSPy's cache by default.
        num_retries: The default number of provider request retries.
        callbacks: Optional instance-level callback handlers.
        **kwargs: Additional default request parameters stored in
            `self.kwargs`.
    """
    self.model = model
    self.model_type = model_type
    self.cache = cache
    self.callbacks = list(callbacks or [])
    self.num_retries = num_retries
    self.kwargs = self._get_initial_kwargs(temperature=temperature, max_tokens=max_tokens, **kwargs)
    self.history = []
    self._warned_zero_temp_rollout = False
```

### Methods:

#### `__call__(*items: Any, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, request: LMRequest | None = None, **kwargs) -> LMResponse | list[dict[str, Any] | str]`

Call the language model synchronously.

The default call path preserves DSPy’s legacy behavior and returns `list[str | dict]`. Calls flow internally through `LMRequest` / `LMResponse` when either:

- `request=` is provided or the first positional argument is an `LMRequest`, or
- `dspy.context(experimental=True)` is active, or
- the subclass declares `forward_contract = "typed_lm"`.

In the typed request path, positional `items` are normalized with `LMRequest.from_call()`. This supports a single prompt string, direct message objects such as `dspy.System(...)`, `dspy.User(...)`, and `dspy.Assistant(...)`, `dspy.ToolResult(...)` tool messages, and prior `LMResponse` values as assistant turns.

Parameters:

| Name       | Type                     | Description                                                                                                                                                                                                                                       | Default                                                                                      |
| ---------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `*items`   | `Any`                    | Optional direct-call inputs. In the legacy path this may contain at most one prompt string. In the typed path it may contain normalized messages, message sequences, prior LMResponse values, or content parts accepted by LMRequest.from_call(). | `()`                                                                                         |
| `prompt`   | \`str                    | None\`                                                                                                                                                                                                                                            | Optional prompt string. Do not combine with positional prompt input.                         |
| `messages` | \`list\[dict[str, Any]\] | None\`                                                                                                                                                                                                                                            | Optional OpenAI-chat-shaped messages. Do not combine with items or prompt in the typed path. |
| `request`  | \`LMRequest              | None\`                                                                                                                                                                                                                                            | Optional explicit normalized request. Call kwargs override request config when provided.     |
| `**kwargs` |                          | Per-call generation parameters.                                                                                                                                                                                                                   | `{}`                                                                                         |

Returns:

| Type         | Description          |
| ------------ | -------------------- |
| \`LMResponse | list\[dict[str, Any] |
| \`LMResponse | list\[dict[str, Any] |

Source code in `dspy/clients/base_lm.py`

```
@with_callbacks
def __call__(
    self,
    *items: Any,
    prompt: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    request: LMRequest | None = None,
    **kwargs,
) -> LMResponse | list[dict[str, Any] | str]:
    """Call the language model synchronously.

    The default call path preserves DSPy's legacy behavior and returns `list[str | dict]`. Calls flow internally
    through `LMRequest` / `LMResponse` when either:

    - `request=` is provided or the first positional argument is an `LMRequest`, or
    - `dspy.context(experimental=True)` is active, or
    - the subclass declares `forward_contract = "typed_lm"`.

    In the typed request path, positional `items` are normalized with `LMRequest.from_call()`. This supports
    a single prompt string, direct message objects such as `dspy.System(...)`, `dspy.User(...)`, and
    `dspy.Assistant(...)`, `dspy.ToolResult(...)` tool messages, and prior `LMResponse` values as assistant turns.

    Args:
        *items: Optional direct-call inputs. In the legacy path this may contain at most one prompt string. In the
            typed path it may contain normalized messages, message sequences, prior `LMResponse` values, or content
            parts accepted by `LMRequest.from_call()`.
        prompt: Optional prompt string. Do not combine with positional prompt input.
        messages: Optional OpenAI-chat-shaped messages. Do not combine with `items` or `prompt` in the typed path.
        request: Optional explicit normalized request. Call kwargs override request config when provided.
        **kwargs: Per-call generation parameters.

    Returns:
        `LMResponse` for explicit `LMRequest` calls or `experimental=True`; otherwise DSPy's legacy list of output
        strings or dictionaries, even when a typed LM subclass uses the typed path internally.
    """
    return_typed_response, forward_contract, normalized_request = self._prepare_lm_call(
        items=items,
        prompt=prompt,
        messages=messages,
        request=request,
        kwargs=kwargs,
    )
    if normalized_request is None:
        return self._legacy_call_direct(*items, prompt=prompt, messages=messages, **kwargs)

    if forward_contract == "typed_lm":
        response = self.forward(normalized_request)
        response = self._finalize_lm_response(normalized_request, self._validate_typed_lm_response(response))
    else:
        response = self._legacy_forward_as_lm_response(normalized_request)
    if return_typed_response:
        return response
    return response.to_legacy_outputs()
```

#### `acall(*items: Any, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, request: LMRequest | None = None, **kwargs) -> LMResponse | list[dict[str, Any] | str]`

Asynchronously call the language model.

This is the async equivalent of `__call__()`. It preserves legacy outputs by default and returns `dspy.LMResponse` for explicit `LMRequest` calls or experimental direct calls.

Source code in `dspy/clients/base_lm.py`

```
@with_callbacks
async def acall(
    self,
    *items: Any,
    prompt: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    request: LMRequest | None = None,
    **kwargs,
) -> LMResponse | list[dict[str, Any] | str]:
    """Asynchronously call the language model.

    This is the async equivalent of `__call__()`. It preserves legacy outputs by default and returns
    `dspy.LMResponse` for explicit `LMRequest` calls or experimental direct calls.
    """
    return_typed_response, forward_contract, normalized_request = self._prepare_lm_call(
        items=items,
        prompt=prompt,
        messages=messages,
        request=request,
        kwargs=kwargs,
    )
    if normalized_request is None:
        return await self._legacy_acall_direct(*items, prompt=prompt, messages=messages, **kwargs)

    if forward_contract == "typed_lm":
        response = await self.aforward(normalized_request)
        response = self._finalize_lm_response(normalized_request, self._validate_typed_lm_response(response))
    else:
        response = await self._legacy_aforward_as_lm_response(normalized_request)
    if return_typed_response:
        return response
    return response.to_legacy_outputs()
```

#### `aforward(prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs)`

Async forward pass for the language model.

Subclasses that support async calls must implement this method according to `forward_contract`.

For `forward_contract = "legacy"`, implement `aforward(prompt=None, messages=None, **kwargs)` and return one of these OpenAI-like provider responses:

- [OpenAI response format](https://platform.openai.com/docs/api-reference/responses/object)
- [OpenAI chat completion format](https://platform.openai.com/docs/api-reference/chat/object)
- [OpenAI text completion format](https://platform.openai.com/docs/api-reference/completions/object)

For `forward_contract = "typed_lm"`, implement `aforward(request: dspy.LMRequest) -> dspy.LMResponse`.

Raises:

| Type      | Description                                                                                                                                                                                                                                                                                                                                                                                       |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LMError` | Base class for LM configuration, transport, provider, and unsupported-feature failures. Notable subclasses include dspy.ContextWindowExceededError for context-window failures, which adapters use to avoid inappropriate fallback retries when the prompt is too long. Each subclass should catch its provider’s native context-window error and re-raise it as dspy.ContextWindowExceededError. |

Source code in `dspy/clients/base_lm.py`

```
async def aforward(
    self,
    prompt: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    **kwargs
):
    """Async forward pass for the language model.

    Subclasses that support async calls must implement this method according to `forward_contract`.

    For `forward_contract = "legacy"`, implement
    `aforward(prompt=None, messages=None, **kwargs)` and return one of these OpenAI-like provider responses:

    - [OpenAI response format](https://platform.openai.com/docs/api-reference/responses/object)
    - [OpenAI chat completion format](https://platform.openai.com/docs/api-reference/chat/object)
    - [OpenAI text completion format](https://platform.openai.com/docs/api-reference/completions/object)

    For `forward_contract = "typed_lm"`, implement `aforward(request: dspy.LMRequest) -> dspy.LMResponse`.

    Raises:
        dspy.LMError: Base class for LM configuration, transport, provider,
            and unsupported-feature failures. Notable subclasses include
            `dspy.ContextWindowExceededError` for context-window failures,
            which adapters use to avoid inappropriate fallback retries when
            the prompt is too long. Each subclass should catch its
            provider's native context-window error and re-raise it as
            `dspy.ContextWindowExceededError`.
    """
    raise NotImplementedError("Subclasses must implement this method.")
```

#### `copy(**kwargs)`

Return a copy of the language model with updated parameters.

The default implementation makes a shallow runtime copy. Provider clients, sessions, and local model handles are preserved by reference. DSPy-owned mutable state is isolated for `history`, the `callbacks` list, and the `kwargs` dict. Other attributes are shared by reference. Subclasses with additional mutable DSPy-owned state should override this method.

Parameters:

| Name       | Type | Description                                                                                                                                                                                                                  | Default |
| ---------- | ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `**kwargs` |      | Attribute or request-parameter updates to apply to the copy. For example, lm.copy(rollout_id=1, temperature=1.0) returns an LM whose requests use a different rollout ID at non-zero temperature to bypass cache collisions. | `{}`    |

Returns:

| Type | Description           |
| ---- | --------------------- |
|      | A copied LM instance. |

Source code in `dspy/clients/base_lm.py`

```
def copy(self, **kwargs):
    """Return a copy of the language model with updated parameters.

    The default implementation makes a shallow runtime copy. Provider
    clients, sessions, and local model handles are preserved by reference.
    DSPy-owned mutable state is isolated for `history`, the `callbacks`
    list, and the `kwargs` dict. Other attributes are shared by reference.
    Subclasses with additional mutable DSPy-owned state should override this
    method.

    Args:
        **kwargs: Attribute or request-parameter updates to apply to the
            copy. For example, `lm.copy(rollout_id=1, temperature=1.0)`
            returns an LM whose requests use a different rollout ID at
            non-zero temperature to bypass cache collisions.

    Returns:
        A copied LM instance.
    """

    new_instance = copy_module.copy(self)
    new_instance.history = []
    new_instance.callbacks = list(getattr(self, "callbacks", []) or [])
    new_instance.kwargs = dict(getattr(self, "kwargs", {}) or {})

    for key, value in kwargs.items():
        if hasattr(new_instance, key):
            setattr(new_instance, key, value)
        if (key in new_instance.kwargs) or (not hasattr(self, key)):
            if value is None:
                new_instance.kwargs.pop(key, None)
            else:
                new_instance.kwargs[key] = value
    if hasattr(new_instance, "_warned_zero_temp_rollout"):
        new_instance._warned_zero_temp_rollout = False

    return new_instance
```

#### `dump_state() -> dict[str, Any]`

Return a sanitized reconstruction state for this LM.

Subclasses whose state is captured by `BaseLM.__init__` can use this default. Subclasses with extra persistent state should override both `dump_state` and `load_state`.

Returns:

| Type             | Description                                                     |
| ---------------- | --------------------------------------------------------------- |
| `dict[str, Any]` | A dictionary that can be passed to BaseLM.load_state. The state |
| `dict[str, Any]` | excludes API keys.                                              |

Source code in `dspy/clients/base_lm.py`

```
def dump_state(self) -> dict[str, Any]:
    """Return a sanitized reconstruction state for this LM.

    Subclasses whose state is captured by `BaseLM.__init__` can use this
    default. Subclasses with extra persistent state should override both
    `dump_state` and `load_state`.

    Returns:
        A dictionary that can be passed to `BaseLM.load_state`. The state
        excludes API keys.
    """
    filtered_kwargs = {key: value for key, value in self.kwargs.items() if key not in ("api_key", LM_CLASS_STATE_KEY)}
    return {
        LM_CLASS_STATE_KEY: f"{type(self).__module__}.{type(self).__qualname__}",
        "model": self.model,
        "model_type": self.model_type,
        "cache": self.cache,
        "num_retries": getattr(self, "num_retries", 3),
        **filtered_kwargs,
    }
```

#### `forward(prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs)`

Forward pass for the language model.

Subclasses must implement this method according to `forward_contract`.

For `forward_contract = "legacy"`, implement `forward(prompt=None, messages=None, **kwargs)` and return one of these OpenAI-like provider responses:

- [OpenAI response format](https://platform.openai.com/docs/api-reference/responses/object)
- [OpenAI chat completion format](https://platform.openai.com/docs/api-reference/chat/object)
- [OpenAI text completion format](https://platform.openai.com/docs/api-reference/completions/object)

For `forward_contract = "typed_lm"`, implement `forward(request: dspy.LMRequest) -> dspy.LMResponse`.

Raises:

| Type      | Description                                                                                                                                                                                                                                                                                                                                                                                       |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LMError` | Base class for LM configuration, transport, provider, and unsupported-feature failures. Notable subclasses include dspy.ContextWindowExceededError for context-window failures, which adapters use to avoid inappropriate fallback retries when the prompt is too long. Each subclass should catch its provider’s native context-window error and re-raise it as dspy.ContextWindowExceededError. |

Source code in `dspy/clients/base_lm.py`

```
def forward(
    self,
    prompt: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    **kwargs
):
    """Forward pass for the language model.

    Subclasses must implement this method according to `forward_contract`.

    For `forward_contract = "legacy"`, implement
    `forward(prompt=None, messages=None, **kwargs)` and return one of these OpenAI-like provider responses:

    - [OpenAI response format](https://platform.openai.com/docs/api-reference/responses/object)
    - [OpenAI chat completion format](https://platform.openai.com/docs/api-reference/chat/object)
    - [OpenAI text completion format](https://platform.openai.com/docs/api-reference/completions/object)

    For `forward_contract = "typed_lm"`, implement `forward(request: dspy.LMRequest) -> dspy.LMResponse`.

    Raises:
        dspy.LMError: Base class for LM configuration, transport, provider,
            and unsupported-feature failures. Notable subclasses include
            `dspy.ContextWindowExceededError` for context-window failures,
            which adapters use to avoid inappropriate fallback retries when
            the prompt is too long. Each subclass should catch its
            provider's native context-window error and re-raise it as
            `dspy.ContextWindowExceededError`.
    """
    raise NotImplementedError("Subclasses must implement this method.")
```

#### `inspect_history(n: int = 1, file: TextIO | None = None) -> None`

Source code in `dspy/clients/base_lm.py`

```
def inspect_history(self, n: int = 1, file: "TextIO | None" = None) -> None:
    pretty_print_history(self.history, n, file=file)
```

#### `load_state(state: dict[str, Any], *, allow_custom_lm_class: bool = False) -> BaseLM`

Reconstruct an LM from `dump_state` output.

Legacy states without a class marker load as `dspy.LM`. Custom LM classes must be importable by their module-qualified class path and are only loaded when `allow_custom_lm_class=True`.

Parameters:

| Name                    | Type             | Description                                                                                                     | Default    |
| ----------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------- | ---------- |
| `state`                 | `dict[str, Any]` | Serialized LM state produced by dump_state.                                                                     | *required* |
| `allow_custom_lm_class` | `bool`           | If True, allow importing and loading custom BaseLM subclasses recorded in state. Enable only for trusted state. | `False`    |

Returns:

| Type     | Description                    |
| -------- | ------------------------------ |
| `BaseLM` | The reconstructed LM instance. |

Raises:

| Type          | Description                                                               |
| ------------- | ------------------------------------------------------------------------- |
| `ValueError`  | If state references a custom LM class and allow_custom_lm_class is False. |
| `ImportError` | If the serialized LM class cannot be imported.                            |
| `TypeError`   | If the serialized class is not a BaseLM subclass.                         |

Source code in `dspy/clients/base_lm.py`

```
@classmethod
def load_state(cls, state: dict[str, Any], *, allow_custom_lm_class: bool = False) -> "BaseLM":
    """Reconstruct an LM from `dump_state` output.

    Legacy states without a class marker load as `dspy.LM`. Custom LM
    classes must be importable by their module-qualified class path and are
    only loaded when `allow_custom_lm_class=True`.

    Args:
        state: Serialized LM state produced by `dump_state`.
        allow_custom_lm_class: If True, allow importing and loading custom
            `BaseLM` subclasses recorded in `state`. Enable only for trusted
            state.

    Returns:
        The reconstructed LM instance.

    Raises:
        ValueError: If `state` references a custom LM class and
            `allow_custom_lm_class` is False.
        ImportError: If the serialized LM class cannot be imported.
        TypeError: If the serialized class is not a `BaseLM` subclass.
    """
    state = dict(state)
    class_path = state.pop(LM_CLASS_STATE_KEY, None)

    if cls is BaseLM:
        if class_path is None:
            # Legacy saved programs did not record the concrete LM class.
            from dspy.clients.lm import LM

            return LM(**state)

        if class_path != _BUILTIN_LM_CLASS_PATH and not allow_custom_lm_class:
            raise ValueError(
                f"Refusing to import custom serialized LM class `{class_path}`. "
                "Pass allow_unsafe_lm_state=True when loading trusted files to enable custom LM classes."
            )

        lm_cls = _import_lm_class(class_path)
        if not issubclass(lm_cls, BaseLM):
            raise TypeError(f"Serialized LM class `{class_path}` must be a subclass of dspy.BaseLM.")
        if "allow_custom_lm_class" in inspect.signature(lm_cls.load_state).parameters:
            return lm_cls.load_state(state, allow_custom_lm_class=allow_custom_lm_class)
        return lm_cls.load_state(state)

    return cls(**state)
```

#### `update_history(entry)`

Source code in `dspy/clients/base_lm.py`

```
def update_history(self, entry):
    if settings.disable_history:
        return

    # Global LM history
    if len(GLOBAL_HISTORY) >= MAX_HISTORY_SIZE:
        GLOBAL_HISTORY.pop(0)

    GLOBAL_HISTORY.append(entry)

    if settings.max_history_size == 0:
        return

    # dspy.LM.history
    if len(self.history) >= settings.max_history_size:
        self.history.pop(0)

    self.history.append(entry)

    # Per-module history
    caller_modules = settings.caller_modules or []
    for module in caller_modules:
        if len(module.history) >= settings.max_history_size:
            module.history.pop(0)
        module.history.append(entry)
```
