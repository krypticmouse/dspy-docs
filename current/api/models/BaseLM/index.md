# dspy.BaseLM

## `dspy.BaseLM(model, model_type='chat', temperature=None, max_tokens=None, cache=True, callbacks: list[BaseCallback] | None = None, num_retries: int = 3, **kwargs)`

Base class for DSPy language models.

Legacy subclasses implement forward(prompt=None, messages=None, \*\*kwargs) and optionally aforward with the same arguments. Ordinary calls return lists of strings or dictionaries. The built-in LM also accepts explicit dspy.lm15.Request calls. The experimental setting does not change outputs.

Implementing custom LMs through forward()/aforward() is deprecated. The old subclass interface remains supported throughout DSPy 3.4 and is scheduled for removal in 3.5. Implement an engine with complete(Request) -> Response and pass it to dspy.LM(engine=…) instead. LegacyEngine and AsyncLegacyEngine are transition wrappers for 3.4 only and are also scheduled for removal in 3.5. OpenAI-style messages= dictionaries are deprecated too: use lm15.Request and Message objects instead. lm(“hello”) remains a list-returning convenience. See the [migration guide](https://dspy.ai/community/normalized-lm-api-migration/#custom-engines-and-legacy-plugins).

Persistent custom state belongs in dump_state/load_state. Runtime clients are shared by copy(), while DSPy history, callbacks and kwargs are isolated.

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

#### `__call__(prompt=None, *, messages=None, **kwargs)`

Return legacy outputs, or one lm15.Response for an explicit Request.

Source code in `dspy/clients/base_lm.py`

```
@with_callbacks
def __call__(self, prompt=None, *, messages=None, **kwargs):
    """Return legacy outputs, or one lm15.Response for an explicit Request."""
    from dspy.clients.execution import execute, finalize, prepare

    call = prepare(self, prompt, messages, kwargs)
    if not call.managed:
        warn_legacy_lm()
    warn_openai_messages(self, messages)
    return finalize(self, call, execute(self, call))
```

#### `acall(prompt=None, *, messages=None, **kwargs)`

Async equivalent of **call**, with the same execution ownership.

Source code in `dspy/clients/base_lm.py`

```
@with_callbacks
async def acall(self, prompt=None, *, messages=None, **kwargs):
    """Async equivalent of __call__, with the same execution ownership."""
    import asyncio

    from dspy.clients.execution import aexecute, finalize, prepare

    # Canonical media snapshots may read local files. Context variables
    # propagate to the worker; callbacks/finalization stay on the caller.
    call = await asyncio.to_thread(prepare, self, prompt, messages, kwargs, asynchronous=True)
    if not call.managed:
        warn_legacy_lm()
    warn_openai_messages(self, messages)
    return finalize(self, call, await aexecute(self, call))
```

#### `aforward(prompt=None, messages=None, **kwargs)`

Asynchronously return an OpenAI-shaped provider response.

Source code in `dspy/clients/base_lm.py`

```
async def aforward(self, prompt=None, messages=None, **kwargs):
    """Asynchronously return an OpenAI-shaped provider response."""
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

#### `forward(prompt=None, messages=None, **kwargs)`

Return an OpenAI-shaped provider response for a legacy LM call.

Source code in `dspy/clients/base_lm.py`

```
def forward(self, prompt=None, messages=None, **kwargs):
    """Return an OpenAI-shaped provider response for a legacy LM call."""
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
