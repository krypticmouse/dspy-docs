# dspy.LM

## `dspy.LM(model: str, model_type: Literal['chat', 'text', 'responses'] = 'chat', temperature: float | None = None, max_tokens: int | None = None, cache: bool = True, callbacks: list[BaseCallback] | None = None, num_retries: int = 3, provider: Provider | None = None, finetuning_model: str | None = None, launch_kwargs: dict[str, Any] | None = None, train_kwargs: dict[str, Any] | None = None, use_developer_role: bool = False, engine: Any = 'auto', async_engine: Any = None, prompt_cache: CacheConfig | None = None, **kwargs)`

Bases: `BaseLM`

A language model supporting chat or text completion requests for use with DSPy modules.

Use lm(“hello”) for a list-returning convenience call, or pass an explicit dspy.lm15.Request to receive a dspy.lm15.Response. OpenAI-style messages= dictionaries are deprecated and scheduled for removal in DSPy 3.5. Adapters and custom engines must migrate to the canonical request/response contract. See https://dspy.ai/community/normalized-lm-api-migration/.

Create a new language model instance for use with DSPy modules and programs.

Parameters:

| Name               | Type                                   | Description                                                                                                                                                                                                                                                                                                                                                                                                                             | Default                                                                                                                                                                                                                                                                                                                                                         |
| ------------------ | -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`            | `str`                                  | The model to use. This should be a string of the form "llm_provider/llm_name" supported by LiteLLM. For example, "openai/gpt-4o".                                                                                                                                                                                                                                                                                                       | *required*                                                                                                                                                                                                                                                                                                                                                      |
| `model_type`       | `Literal['chat', 'text', 'responses']` | The type of the model, such as "chat", "text", or "responses".                                                                                                                                                                                                                                                                                                                                                                          | `'chat'`                                                                                                                                                                                                                                                                                                                                                        |
| `temperature`      | \`float                                | None\`                                                                                                                                                                                                                                                                                                                                                                                                                                  | The sampling temperature to use when generating responses.                                                                                                                                                                                                                                                                                                      |
| `max_tokens`       | \`int                                  | None\`                                                                                                                                                                                                                                                                                                                                                                                                                                  | The maximum number of tokens to generate per response.                                                                                                                                                                                                                                                                                                          |
| `cache`            | `bool`                                 | Whether to cache the model responses for reuse to improve performance and reduce costs.                                                                                                                                                                                                                                                                                                                                                 | `True`                                                                                                                                                                                                                                                                                                                                                          |
| `callbacks`        | \`list[BaseCallback]                   | None\`                                                                                                                                                                                                                                                                                                                                                                                                                                  | A list of callback functions to run before and after each request.                                                                                                                                                                                                                                                                                              |
| `num_retries`      | `int`                                  | The number of times to retry a request if it fails transiently due to network error, rate limiting, etc. Requests are retried with exponential backoff.                                                                                                                                                                                                                                                                                 | `3`                                                                                                                                                                                                                                                                                                                                                             |
| `engine`           | `Any`                                  | ‘auto’ prefers lm15 for representable requests; ‘litellm’ preserves the compatibility backend; ‘lm15’ refuses unsupported mappings rather than selecting LiteLLM. A custom engine implements complete(Request) -> Response and optionally stream(Request). Engines are borrowed and own their connection: api_key, api_base, timeout and the other client settings are refused with a custom engine, at construction and on every call. | `'auto'`                                                                                                                                                                                                                                                                                                                                                        |
| `async_engine`     | `Any`                                  | Async counterpart when supplying a custom engine object. The pair is one unit: copy(engine=…) replaces both unless async_engine= is given too.                                                                                                                                                                                                                                                                                          | `None`                                                                                                                                                                                                                                                                                                                                                          |
| `prompt_cache`     | \`CacheConfig                          | None\`                                                                                                                                                                                                                                                                                                                                                                                                                                  | Optional lm15 CacheConfig for provider-side prompt caching on ordinary calls. Separate from DSPy’s response cache. Requires native lm15 or a canonical custom engine; may incur cache-write/storage charges. A call-time value overrides this default, and None removes it. Explicit Request calls use only Request.config.cache. No cache resource is created. |
| `provider`         | \`Provider                             | None\`                                                                                                                                                                                                                                                                                                                                                                                                                                  | The training/launch provider. This does not select the inference engine.                                                                                                                                                                                                                                                                                        |
| `finetuning_model` | \`str                                  | None\`                                                                                                                                                                                                                                                                                                                                                                                                                                  | The model to finetune. In some providers, the models available for finetuning is different from the models available for inference.                                                                                                                                                                                                                             |
| `rollout_id`       |                                        | Optional integer used to differentiate cache entries for otherwise identical requests. Different values bypass DSPy’s caches while still caching future calls with the same inputs and rollout ID. Note that rollout_id only affects generation when temperature is non-zero. This argument is stripped before sending requests to the provider.                                                                                        | *required*                                                                                                                                                                                                                                                                                                                                                      |

Source code in `dspy/clients/lm.py`

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
    provider: Provider | None = None,
    finetuning_model: str | None = None,
    launch_kwargs: dict[str, Any] | None = None,
    train_kwargs: dict[str, Any] | None = None,
    use_developer_role: bool = False,
    engine: Any = "auto",
    async_engine: Any = None,
    prompt_cache: CacheConfig | None = None,
    **kwargs,
):
    """Create a new language model instance for use with DSPy modules and programs.

    Args:
        model: The model to use. This should be a string of the form
            `"llm_provider/llm_name"` supported by LiteLLM. For example,
            `"openai/gpt-4o"`.
        model_type: The type of the model, such as `"chat"`, `"text"`, or
            `"responses"`.
        temperature: The sampling temperature to use when generating responses.
        max_tokens: The maximum number of tokens to generate per response.
        cache: Whether to cache the model responses for reuse to improve performance
            and reduce costs.
        callbacks: A list of callback functions to run before and after each request.
        num_retries: The number of times to retry a request if it fails transiently due to
            network error, rate limiting, etc. Requests are retried with exponential
            backoff.
        engine: 'auto' prefers lm15 for representable requests; 'litellm' preserves the compatibility backend;
            'lm15' refuses unsupported mappings rather than selecting LiteLLM. A custom engine implements
            complete(Request) -> Response and optionally stream(Request). Engines are borrowed and own
            their connection: api_key, api_base, timeout and the other client settings are refused with a
            custom engine, at construction and on every call.
        async_engine: Async counterpart when supplying a custom engine object. The pair is one unit:
            copy(engine=...) replaces both unless async_engine= is given too.
        prompt_cache: Optional lm15 CacheConfig for provider-side prompt caching on ordinary calls.
            Separate from DSPy's response cache. Requires native lm15 or a canonical custom engine;
            may incur cache-write/storage charges. A call-time value overrides this default, and None
            removes it. Explicit Request calls use only Request.config.cache. No cache resource is created.
        provider: The training/launch provider. This does not select the inference engine.
        finetuning_model: The model to finetune. In some providers, the models available for finetuning is different
            from the models available for inference.
        rollout_id: Optional integer used to differentiate cache entries for otherwise
            identical requests. Different values bypass DSPy's caches while still caching
            future calls with the same inputs and rollout ID. Note that `rollout_id`
            only affects generation when `temperature` is non-zero. This argument is
            stripped before sending requests to the provider.
    """
    _check_engines(engine, async_engine)
    _refuse_client_settings(engine, kwargs)
    if isinstance(num_retries, bool) or not isinstance(num_retries, int) or num_retries < 0:
        raise ValueError("num_retries must be a nonnegative integer")
    if prompt_cache is not None:
        if not isinstance(prompt_cache, CacheConfig):
            raise TypeError("prompt_cache must be a dspy.lm15.CacheConfig or None")
        kwargs["prompt_cache"] = prompt_cache
    self._engine_spec = engine
    self._async_engine_spec = async_engine
    # The declared providers this LM routes with, bound now and kept for
    # its life (copies share them): selection, capabilities, pricing and
    # both engines read one tuple, so a later registration cannot split them.
    from dspy.lm15 import registered_providers

    self._providers = registered_providers()
    self._engine_store = {}
    self._engine_lock = threading.RLock()
    super().__init__(
        model=model,
        model_type=model_type,
        temperature=temperature,
        max_tokens=max_tokens,
        cache=cache,
        num_retries=num_retries,
        callbacks=callbacks,
        **kwargs,
    )

    self.provider = provider or self.infer_provider()
    self.finetuning_model = finetuning_model
    self.launch_kwargs = launch_kwargs or {}
    self.train_kwargs = train_kwargs or {}
    self.use_developer_role = use_developer_role

    self._warn_zero_temp_rollout(self.kwargs.get("temperature"), self.kwargs.get("rollout_id"))
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

Source code in `dspy/clients/lm.py`

```
async def aforward(self, prompt=None, messages=None, **kwargs):
    import asyncio

    from dspy.clients.execution import aexecute, prepare

    call = await asyncio.to_thread(prepare, self, prompt, messages, kwargs, asynchronous=True, direct=True)
    return (await aexecute(self, call)).provider_response()
```

#### `copy(**kwargs)`

Source code in `dspy/clients/lm.py`

```
def copy(self, **kwargs):
    if kwargs.get("prompt_cache") is not None and not isinstance(kwargs["prompt_cache"], CacheConfig):
        raise TypeError("prompt_cache must be a dspy.lm15.CacheConfig or None")
    if "engine" in kwargs:
        # The engine pair is one unit: a new engine drops the old async
        # counterpart unless a new one comes with it.
        spec = kwargs.pop("engine")
        async_spec = kwargs.pop("async_engine", None)
    else:
        spec = self._engine_spec
        async_spec = kwargs.pop("async_engine", self._async_engine_spec)
    _check_engines(spec, async_spec)
    _refuse_client_settings(spec, {**self.kwargs, **kwargs}, where="LM.copy")
    copied = super().copy(**kwargs)
    copied._engine_spec = spec
    copied._async_engine_spec = async_spec
    return copied
```

#### `dump_state()`

Return a sanitized reconstruction state for this LM.

A custom engine is recorded as its class path and its own `dump_state()`; the class must be importable by that path in the process that loads the state. A class defined in `__main__` (a script or notebook) loads only where `__main__` defines it again; for durable state define the engine in an importable module.

Returns:

| Type | Description                                             |
| ---- | ------------------------------------------------------- |
|      | A dictionary that can be passed to BaseLM.load_state to |
|      | reconstruct this LM. The state excludes API keys.       |

Source code in `dspy/clients/lm.py`

```
def dump_state(self):
    """Return a sanitized reconstruction state for this LM.

    A custom engine is recorded as its class path and its own
    ``dump_state()``; the class must be importable by that path in the
    process that loads the state. A class defined in ``__main__`` (a
    script or notebook) loads only where ``__main__`` defines it again;
    for durable state define the engine in an importable module.

    Returns:
        A dictionary that can be passed to `BaseLM.load_state` to
        reconstruct this `LM`. The state excludes API keys.
    """
    state = super().dump_state()
    if state.get("prompt_cache") is not None:
        from dspy._vendor.lm15.serde import cache_config_to_dict

        state["prompt_cache"] = cache_config_to_dict(state["prompt_cache"])
    if isinstance(self._engine_spec, str):
        if self._engine_spec != "auto":
            state["engine"] = self._engine_spec
    else:
        # A custom engine is saved as its class path and its own state;
        # loading it imports a class from the file, so it is gated like a
        # custom LM class (allow_unsafe_lm_state).
        state["engine"] = _dump_engine(self._engine_spec, "engine")
        if self._async_engine_spec is not None:
            state["async_engine"] = _dump_engine(self._async_engine_spec, "async_engine")
    state.update(
        {
            "finetuning_model": self.finetuning_model,
            "launch_kwargs": self.launch_kwargs,
            "train_kwargs": self.train_kwargs,
        }
    )
    if self.use_developer_role:
        state["use_developer_role"] = self.use_developer_role
    if _is_openai_reasoning_model(self.model) and "max_completion_tokens" in state:
        state["max_tokens"] = state.pop("max_completion_tokens")
    return state
```

#### `finetune(train_data: list[dict[str, Any]], train_data_format: TrainDataFormat | None, train_kwargs: dict[str, Any] | None = None) -> TrainingJob`

Source code in `dspy/clients/lm.py`

```
def finetune(
    self,
    train_data: list[dict[str, Any]],
    train_data_format: TrainDataFormat | None,
    train_kwargs: dict[str, Any] | None = None,
) -> TrainingJob:
    from dspy import settings as settings

    if not self.provider.finetunable:
        raise LMUnsupportedFeatureError(
            f"Provider {self.provider} does not support fine-tuning, please specify your provider by explicitly "
            "setting `provider` when creating the `dspy.LM` instance. For example, "
            "`dspy.LM('openai/gpt-4.1-mini-2025-04-14', provider=dspy.OpenAIProvider())`.",
            model=self.model,
            provider=self._provider_name,
            features=["finetuning"],
        )

    def thread_function_wrapper():
        return self._run_finetune_job(job)

    thread = threading.Thread(target=thread_function_wrapper)
    train_kwargs = train_kwargs or self.train_kwargs
    model_to_finetune = self.finetuning_model or self.model
    job = self.provider.TrainingJob(
        thread=thread,
        model=model_to_finetune,
        train_data=train_data,
        train_data_format=train_data_format,
        train_kwargs=train_kwargs,
    )
    thread.start()

    return job
```

#### `forward(prompt=None, messages=None, **kwargs)`

Compatibility forward entry point; public calls also record history.

Source code in `dspy/clients/lm.py`

```
def forward(self, prompt=None, messages=None, **kwargs):
    """Compatibility forward entry point; public calls also record history."""
    from dspy.clients.execution import execute, prepare

    return execute(self, prepare(self, prompt, messages, kwargs, direct=True)).provider_response()
```

#### `infer_provider() -> Provider`

Source code in `dspy/clients/lm.py`

```
def infer_provider(self) -> Provider:
    if OpenAIProvider.is_provider_model(self.model):
        return OpenAIProvider()
    return Provider()
```

#### `inspect_history(n: int = 1, file: TextIO | None = None) -> None`

Source code in `dspy/clients/base_lm.py`

```
def inspect_history(self, n: int = 1, file: "TextIO | None" = None) -> None:
    pretty_print_history(self.history, n, file=file)
```

#### `kill(launch_kwargs: dict[str, Any] | None = None)`

Source code in `dspy/clients/lm.py`

```
def kill(self, launch_kwargs: dict[str, Any] | None = None):
    self.provider.kill(self, launch_kwargs)
```

#### `launch(launch_kwargs: dict[str, Any] | None = None)`

Source code in `dspy/clients/lm.py`

```
def launch(self, launch_kwargs: dict[str, Any] | None = None):
    self.provider.launch(self, launch_kwargs)
```

#### `load_state(state: dict[str, Any], *, allow_custom_lm_class: bool = False)`

Source code in `dspy/clients/lm.py`

```
@classmethod
def load_state(cls, state: dict[str, Any], *, allow_custom_lm_class: bool = False):
    state = dict(state)
    if isinstance(state.get("prompt_cache"), dict):
        from dspy._vendor.lm15.serde import cache_config_from_dict

        state["prompt_cache"] = cache_config_from_dict(state["prompt_cache"])

    model = state.get("model")
    if isinstance(model, str) and _is_openai_reasoning_model(model) and "max_completion_tokens" in state:
        if "max_tokens" not in state:
            state["max_tokens"] = state["max_completion_tokens"]
        state.pop("max_completion_tokens")

    if not isinstance(state.get("engine", "auto"), str):
        state["engine"] = _load_engine(state["engine"], "engine", allow_custom_lm_class=allow_custom_lm_class)
        if state.get("async_engine") is not None:
            state["async_engine"] = _load_engine(
                state["async_engine"], "async_engine", allow_custom_lm_class=allow_custom_lm_class
            )
    elif state.get("async_engine") is not None:
        raise ValueError("Serialized async_engine without a custom engine")

    return super().load_state(state, allow_custom_lm_class=allow_custom_lm_class)
```

#### `reinforce(train_kwargs) -> ReinforceJob`

Source code in `dspy/clients/lm.py`

```
def reinforce(self, train_kwargs) -> ReinforceJob:
    # TODO(GRPO Team): Should we return an initialized job here?
    from dspy import settings as settings

    if not self.provider.reinforceable:
        raise LMUnsupportedFeatureError(
            f"Provider {self.provider} does not implement the reinforcement learning interface.",
            model=self.model,
            provider=self._provider_name,
            features=["reinforce"],
        )

    job = self.provider.ReinforceJob(lm=self, train_kwargs=train_kwargs)
    job.initialize()
    return job
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
