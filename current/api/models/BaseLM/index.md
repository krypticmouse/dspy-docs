# dspy.BaseLM

## `dspy.BaseLM(model: str, model_type: Literal['chat', 'text', 'responses'] = 'chat', temperature: float | None = None, max_tokens: int | None = None, cache: bool = True, callbacks: list[BaseCallback] | None = None, num_retries: int = 3, engine: Engine | None = None, async_engine: AsyncEngine | None = None, **kwargs: Any)`

Base class for DSPy language models.

Every LM executes through an *engine*: an object implementing `complete(Request) -> Response`, optionally `stream(Request)` yielding lm15 stream events, plus an async counterpart with `async complete` and an async `stream`. `dspy.LM` selects its built-in engines from the model string; any other LM supplies engines directly. DSPy’s LM layer owns response caching, retries, candidate fan-out, usage accounting, callbacks and history around that single-request contract, so engines never add a second layer of those.

Calls speak lm15 objects: `lm(request)` returns one `dspy.lm15.Response`. Adapters call the LM separately for each candidate. `lm("text")` remains a convenience that renders one user message and returns a list of outputs. OpenAI-style `messages=` dictionaries and `forward()` implementations were removed in DSPy 3.5; see the migration guide at https://dspy.ai/community/normalized-lm-api-migration/.

Persistent custom state belongs in dump_state/load_state. Runtime clients are shared by copy(), while DSPy history, callbacks and kwargs are isolated.

Initialize a base language model.

Parameters:

| Name           | Type                                   | Description                                                                                                               | Default                                                                                                                                                 |
| -------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`        | `str`                                  | The model identifier.                                                                                                     | *required*                                                                                                                                              |
| `model_type`   | `Literal['chat', 'text', 'responses']` | The LM API type, such as "chat", "text", or "responses".                                                                  | `'chat'`                                                                                                                                                |
| `temperature`  | \`float                                | None\`                                                                                                                    | The default sampling temperature.                                                                                                                       |
| `max_tokens`   | \`int                                  | None\`                                                                                                                    | The default maximum number of output tokens.                                                                                                            |
| `cache`        | `bool`                                 | Whether requests should use DSPy’s cache by default.                                                                      | `True`                                                                                                                                                  |
| `callbacks`    | \`list[BaseCallback]                   | None\`                                                                                                                    | Optional instance-level callback handlers.                                                                                                              |
| `num_retries`  | `int`                                  | The default number of engine attempts after a transient failure.                                                          | `3`                                                                                                                                                     |
| `engine`       | \`Engine                               | None\`                                                                                                                    | An object implementing complete(Request) -> Response. Engines are borrowed; DSPy never closes them.                                                     |
| `async_engine` | \`AsyncEngine                          | None\`                                                                                                                    | The async counterpart used by acall. Either engine may be omitted. Capability hints come from the sync engine when present, otherwise the async engine. |
| `**kwargs`     | `Any`                                  | Additional default generation options stored in self.kwargs, in the OpenAI vocabulary (top_p, stop, reasoning_effort, …). | `{}`                                                                                                                                                    |

Source code in `dspy/clients/base_lm.py`

```
def __init__(
    self,
    model: str,
    model_type: Literal["chat", "text", "responses"] = "chat",
    temperature: float | None = None,
    max_tokens: int | None = None,
    cache: bool = True,
    callbacks: list[BaseCallback] | None = None,
    num_retries: int = 3,
    engine: Engine | None = None,
    async_engine: AsyncEngine | None = None,
    **kwargs: Any,
) -> None:
    """Initialize a base language model.

    Args:
        model: The model identifier.
        model_type: The LM API type, such as `"chat"`, `"text"`, or
            `"responses"`.
        temperature: The default sampling temperature.
        max_tokens: The default maximum number of output tokens.
        cache: Whether requests should use DSPy's cache by default.
        callbacks: Optional instance-level callback handlers.
        num_retries: The default number of engine attempts after a
            transient failure.
        engine: An object implementing `complete(Request) -> Response`.
            Engines are borrowed; DSPy never closes them.
        async_engine: The async counterpart used by `acall`. Either engine
            may be omitted. Capability hints come from the sync engine
            when present, otherwise the async engine.
        **kwargs: Additional default generation options stored in
            `self.kwargs`, in the OpenAI vocabulary (`top_p`, `stop`,
            `reasoning_effort`, ...).
    """
    from dspy.clients.lm import LM, _refuse_client_settings

    # LM validates its effective selection before calling super (named
    # selections arrive here as None). Other LMs own custom engines only.
    if not isinstance(self, LM):
        _refuse_client_settings(engine, kwargs, where="dspy.BaseLM")
    validate_engines(engine, async_engine)
    self.model = model
    self.model_type = model_type
    self.cache = cache
    self.callbacks = list(callbacks or [])
    self.num_retries = num_retries
    self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
    self.history = []
    self._engine_spec: Engine | Literal["auto", "lm15", "litellm"] | None = engine
    self._async_engine_spec = async_engine
```

### Methods:

#### `__call__(prompt: Request | str | None = None, *, cache: bool | None = None, rollout_id: Any = None, **options: Any) -> Response | list[str | dict[str, Any] | None]`

Run one model call.

Parameters:

| Name         | Type      | Description                                                                                                                                                                                                                                               | Default                                                 |
| ------------ | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `prompt`     | \`Request | str                                                                                                                                                                                                                                                       | None\`                                                  |
| `cache`      | \`bool    | None\`                                                                                                                                                                                                                                                    | Override this LM’s response-cache setting for the call. |
| `rollout_id` | `Any`     | Distinguish otherwise identical cached requests.                                                                                                                                                                                                          | `None`                                                  |
| `**options`  | `Any`     | Generation options for a string prompt, in the OpenAI vocabulary (temperature, max_tokens, n, …), plus client settings (api_key, api_base, …) for this call. A Request accepts only the client settings; its config already holds the generation options. | `{}`                                                    |

Returns:

| Type       | Description |
| ---------- | ----------- |
| \`Response | list\[str   |

Source code in `dspy/clients/base_lm.py`

```
@with_callbacks
def __call__(
    self,
    prompt: Request | str | None = None,
    *,
    cache: bool | None = None,
    rollout_id: Any = None,
    **options: Any,
) -> Response | list[str | dict[str, Any] | None]:
    """Run one model call.

    Args:
        prompt: A `dspy.lm15.Request`, or a string rendered as one user
            message with this LM's generation defaults.
        cache: Override this LM's response-cache setting for the call.
        rollout_id: Distinguish otherwise identical cached requests.
        **options: Generation options for a string prompt, in the OpenAI
            vocabulary (`temperature`, `max_tokens`, `n`, ...), plus
            client settings (`api_key`, `api_base`, ...) for this call.
            A `Request` accepts only the client settings; its `config`
            already holds the generation options.

    Returns:
        The `Response` for a `Request`; a list of outputs for a string.
    """
    from dspy.clients.execution import execute, finalize, prepare

    call = prepare(self, prompt, options, cache=cache, rollout_id=rollout_id)
    return finalize(self, call, execute(self, call))
```

#### `acall(prompt: Request | str | None = None, *, cache: bool | None = None, rollout_id: Any = None, **options: Any) -> Response | list[str | dict[str, Any] | None]`

Async equivalent of `__call__`, with the same execution ownership.

Source code in `dspy/clients/base_lm.py`

```
@with_callbacks
async def acall(
    self,
    prompt: Request | str | None = None,
    *,
    cache: bool | None = None,
    rollout_id: Any = None,
    **options: Any,
) -> Response | list[str | dict[str, Any] | None]:
    """Async equivalent of `__call__`, with the same execution ownership."""
    import asyncio

    from dspy.clients.execution import aexecute, finalize, prepare

    # Canonical media snapshots may read local files. Context variables
    # propagate to the worker; callbacks/finalization stay on the caller.
    call = await asyncio.to_thread(prepare, self, prompt, options, cache=cache, rollout_id=rollout_id,
                                   asynchronous=True)
    return finalize(self, call, await aexecute(self, call))
```

#### `copy(**kwargs)`

Return a copy of the language model with updated parameters.

The default implementation makes a shallow runtime copy. Engines, provider clients, sessions, and local model handles are preserved by reference. DSPy-owned mutable state is isolated for `history`, the `callbacks` list, and the `kwargs` dict. Other attributes are shared by reference. Subclasses with additional mutable DSPy-owned state should override this method.

Supplying `engine` replaces the pair, dropping the old async engine unless `async_engine` is also supplied. Supplying only `async_engine` replaces or clears that side while preserving the sync engine.

Parameters:

| Name       | Type | Description                                                                                                                                                                                                                  | Default |
| ---------- | ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `**kwargs` |      | Attribute or generation-option updates to apply to the copy. For example, lm.copy(rollout_id=1, temperature=1.0) returns an LM whose requests use a different rollout ID at non-zero temperature to bypass cache collisions. | `{}`    |

Returns:

| Type | Description           |
| ---- | --------------------- |
|      | A copied LM instance. |

Source code in `dspy/clients/base_lm.py`

```
def copy(self, **kwargs):
    """Return a copy of the language model with updated parameters.

    The default implementation makes a shallow runtime copy. Engines,
    provider clients, sessions, and local model handles are preserved by
    reference. DSPy-owned mutable state is isolated for `history`, the
    `callbacks` list, and the `kwargs` dict. Other attributes are shared by
    reference. Subclasses with additional mutable DSPy-owned state should
    override this method.

    Supplying `engine` replaces the pair, dropping the old async engine
    unless `async_engine` is also supplied. Supplying only `async_engine`
    replaces or clears that side while preserving the sync engine.

    Args:
        **kwargs: Attribute or generation-option updates to apply to the
            copy. For example, `lm.copy(rollout_id=1, temperature=1.0)`
            returns an LM whose requests use a different rollout ID at
            non-zero temperature to bypass cache collisions.

    Returns:
        A copied LM instance.
    """

    from dspy.clients.lm import LM, _refuse_client_settings

    if "engine" in kwargs:
        spec = kwargs.pop("engine")
        async_spec = kwargs.pop("async_engine", None)
    else:
        spec = self.engine
        async_spec = kwargs.pop("async_engine", self.async_engine)
    self._check_engines(spec, async_spec)
    where = "LM.copy" if isinstance(self, LM) else "BaseLM.copy"
    _refuse_client_settings(spec, {**self.kwargs, **kwargs}, where=where)
    new_instance = copy_module.copy(self)
    new_instance._engine_spec = spec
    new_instance._async_engine_spec = async_spec
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
    state = {
        LM_CLASS_STATE_KEY: f"{type(self).__module__}.{type(self).__qualname__}",
        "model": self.model,
        "model_type": self.model_type,
        "cache": self.cache,
        "num_retries": getattr(self, "num_retries", 3),
        **filtered_kwargs,
    }
    from dspy.clients.lm import _dump_engine

    if isinstance(self.engine, str):
        if self.engine != "auto":
            state["engine"] = self.engine
    else:
        state["engine"] = _dump_engine(self.engine, "engine") if self.engine is not None else None
        if self.async_engine is not None:
            state["async_engine"] = _dump_engine(self.async_engine, "async_engine")
    return state
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
        if lm_cls is not BaseLM:
            if "allow_custom_lm_class" in inspect.signature(lm_cls.load_state).parameters:
                return lm_cls.load_state(state, allow_custom_lm_class=allow_custom_lm_class)
            return lm_cls.load_state(state)

    from dspy.clients.lm import _load_engine

    # LM.load_state already decodes its records before delegating here.
    for name in ("engine", "async_engine"):
        if isinstance(state.get(name), dict):
            state[name] = _load_engine(state[name], name, allow_custom_lm_class=allow_custom_lm_class)
    return cls(**state)
```

#### `update_history(entry)`

Source code in `dspy/clients/base_lm.py`

```
def update_history(self, entry):
    record_history(self, entry)
```

```
  October 28, 2026
```
