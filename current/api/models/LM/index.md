# dspy.LM

## `dspy.LM(model: str, model_type: Literal['chat', 'text', 'responses'] = 'chat', temperature: float | None = None, max_tokens: int | None = None, cache: bool = True, callbacks: list[BaseCallback] | None = None, num_retries: int = 3, provider: Provider | None = None, finetuning_model: str | None = None, launch_kwargs: dict[str, Any] | None = None, train_kwargs: dict[str, Any] | None = None, use_developer_role: bool = False, engine: Engine | Literal['auto', 'lm15', 'litellm'] | None = 'auto', async_engine: AsyncEngine | None = None, prompt_cache: CacheConfig | None = None, **kwargs)`

Bases: `BaseLM`

A language model supporting chat or text completion requests for use with DSPy modules.

`lm(request)` runs one `dspy.lm15.Request` and returns a `dspy.lm15.Response`; adapters call it once per candidate. `lm("hello")` is a convenience that renders one user message with this LM’s generation defaults and returns a list of outputs. Custom backends are engines passed with `engine=`; see https://dspy.ai/community/normalized-lm-api-migration/.

Create a new language model instance for use with DSPy modules and programs.

Parameters:

| Name               | Type                                   | Description                                                                                                                                                                                                                                                                                                                                      | Default                                                                                                                                                                                                                                                                                                                                                         |
| ------------------ | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`            | `str`                                  | The model to use. This should be a string of the form "provider/model" supported by lm15, a registered provider, or optional LiteLLM. For example, "openai/gpt-4o".                                                                                                                                                                              | *required*                                                                                                                                                                                                                                                                                                                                                      |
| `model_type`       | `Literal['chat', 'text', 'responses']` | The type of the model, such as "chat", "text", or "responses".                                                                                                                                                                                                                                                                                   | `'chat'`                                                                                                                                                                                                                                                                                                                                                        |
| `temperature`      | \`float                                | None\`                                                                                                                                                                                                                                                                                                                                           | The sampling temperature to use when generating responses.                                                                                                                                                                                                                                                                                                      |
| `max_tokens`       | \`int                                  | None\`                                                                                                                                                                                                                                                                                                                                           | The maximum number of tokens to generate per response.                                                                                                                                                                                                                                                                                                          |
| `cache`            | `bool`                                 | Whether to cache the model responses for reuse to improve performance and reduce costs.                                                                                                                                                                                                                                                          | `True`                                                                                                                                                                                                                                                                                                                                                          |
| `callbacks`        | \`list[BaseCallback]                   | None\`                                                                                                                                                                                                                                                                                                                                           | A list of callback functions to run before and after each request.                                                                                                                                                                                                                                                                                              |
| `num_retries`      | `int`                                  | The number of times to retry a request if it fails transiently due to network error, rate limiting, etc. Requests are retried with exponential backoff.                                                                                                                                                                                          | `3`                                                                                                                                                                                                                                                                                                                                                             |
| `engine`           | \`Engine                               | Literal['auto', 'lm15', 'litellm']                                                                                                                                                                                                                                                                                                               | None\`                                                                                                                                                                                                                                                                                                                                                          |
| `async_engine`     | \`AsyncEngine                          | None\`                                                                                                                                                                                                                                                                                                                                           | Async counterpart when supplying a custom engine object. The pair is one unit: copy(engine=…) replaces both unless async_engine= is given too. Pass engine=None for an async-only custom LM. Capability hints come from the sync engine when present, otherwise the async engine; each call mode requires its own engine.                                       |
| `prompt_cache`     | \`CacheConfig                          | None\`                                                                                                                                                                                                                                                                                                                                           | Optional lm15 CacheConfig for provider-side prompt caching on ordinary calls. Separate from DSPy’s response cache. Requires native lm15 or a canonical custom engine; may incur cache-write/storage charges. A call-time value overrides this default, and None removes it. Explicit Request calls use only Request.config.cache. No cache resource is created. |
| `provider`         | \`Provider                             | None\`                                                                                                                                                                                                                                                                                                                                           | The training/launch provider. This does not select the inference engine.                                                                                                                                                                                                                                                                                        |
| `finetuning_model` | \`str                                  | None\`                                                                                                                                                                                                                                                                                                                                           | The model to finetune. In some providers, the models available for finetuning is different from the models available for inference.                                                                                                                                                                                                                             |
| `rollout_id`       |                                        | Optional integer used to differentiate cache entries for otherwise identical requests. Different values bypass DSPy’s caches while still caching future calls with the same inputs and rollout ID. Note that rollout_id only affects generation when temperature is non-zero. This argument is stripped before sending requests to the provider. | *required*                                                                                                                                                                                                                                                                                                                                                      |

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
    engine: Engine | Literal["auto", "lm15", "litellm"] | None = "auto",
    async_engine: AsyncEngine | None = None,
    prompt_cache: CacheConfig | None = None,
    **kwargs,
):
    """Create a new language model instance for use with DSPy modules and programs.

    Args:
        model: The model to use. This should be a string of the form
            `"provider/model"` supported by lm15, a registered provider, or optional LiteLLM. For example,
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
            Pass engine=None for an async-only custom LM. Capability hints come from the sync
            engine when present, otherwise the async engine; each call mode requires its own engine.
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
    if not isinstance(model, str) or not model:
        raise ValueError("model must be a non-empty string")
    _, separator, rest = model.partition("/")
    if separator and not rest:
        # "openai/" selected LiteLLM silently and failed at call time.
        raise ValueError(f"model {model!r} has a provider prefix but no model id after it")
    if isinstance(num_retries, bool) or not isinstance(num_retries, int) or num_retries < 0:
        raise ValueError("num_retries must be a nonnegative integer")
    if prompt_cache is not None:
        if not isinstance(prompt_cache, CacheConfig):
            raise TypeError("prompt_cache must be a dspy.lm15.CacheConfig or None")
        kwargs["prompt_cache"] = prompt_cache
    # The declared providers this LM routes with, bound now and kept for
    # its life (copies share them): selection, capabilities, pricing and
    # both engines read one tuple, so a later registration cannot split them.
    from dspy.lm15 import registered_providers

    self._providers = registered_providers()
    self._engine_store = {}
    self._engine_lock = threading.RLock()
    self._warned_zero_temp_rollout = False
    super().__init__(
        model=model,
        model_type=model_type,
        temperature=temperature,
        max_tokens=max_tokens,
        cache=cache,
        num_retries=num_retries,
        callbacks=callbacks,
        engine=None if isinstance(engine, str) else engine,
        async_engine=async_engine,
        **kwargs,
    )
    if _is_openai_reasoning_model(self.model):
        if (temperature and temperature != 1.0) or (max_tokens and max_tokens < 16000):
            raise LMConfigurationError(
                "OpenAI's reasoning models require passing temperature=1.0 or None and max_tokens >= 16000 or None to "
                "`dspy.LM(...)`, e.g., dspy.LM('openai/gpt-5', temperature=1.0, max_tokens=16000)",
                model=self.model,
                provider=self._provider_name,
            )
        self.kwargs = dict(temperature=temperature, max_completion_tokens=max_tokens, **kwargs)
    if self.kwargs.get("rollout_id") is None:
        self.kwargs.pop("rollout_id", None)
    if isinstance(engine, str):
        self._engine_spec = engine

    self.provider = provider or self.infer_provider()
    self.finetuning_model = finetuning_model
    self.launch_kwargs = launch_kwargs or {}
    self.train_kwargs = train_kwargs or {}
    self.use_developer_role = use_developer_role

    self._warn_zero_temp_rollout(self.kwargs.get("temperature"), self.kwargs.get("rollout_id"))
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

Source code in `dspy/clients/lm.py`

```
def copy(self, **kwargs):
    if kwargs.get("prompt_cache") is not None and not isinstance(kwargs["prompt_cache"], CacheConfig):
        raise TypeError("prompt_cache must be a dspy.lm15.CacheConfig or None")
    return super().copy(**kwargs)
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
        if state["engine"] is not None:
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
    record_history(self, entry)
```

## Choosing an engine

`engine="auto"` prefers native lm15 and may select the optional LiteLLM compatibility engine for unsupported routes, settings or representations before provider I/O. `engine="lm15"` requires native; `engine="litellm"` explicitly selects compatibility. Provider failures never trigger a backend switch. Native imports and calls do not require LiteLLM. Compatibility use requires `pip install "dspy[litellm]"`; without it, an actionable error names that command. `model_type="text"` uses compatibility, not native lm15. All engines retain the canonical `Request`/`Response` contract. See the [migration guide](https://dspy.ai/current/community/normalized-lm-api-migration/index.md) for limitations.

## Application-owned exception: text-only Bedrock Nova

Native Bedrock Nova/Converse is not supported by this DSPy release. The following recipe is a bounded, application-owned exception, **not restored native support or an accepted migration-gate exception**. A custom engine is not the only integration mechanism: optional LiteLLM compatibility also remains available. This recipe does not verify built-in Nova execution through that engine. It uses the public [`complete(Request) -> Response` engine contract](https://dspy.ai/current/tutorials/custom_lm_engines/index.md#what-a-custom-engine-owns) and LiteLLM installed through DSPy’s optional extra, pinned to the version tested here:

```
pip install "dspy[litellm]" "litellm==1.96.0"
```

This example targets **`amazon.nova-lite-v1:0` in `us-east-1`**, using AWS credentials configured for your application. It forwards a string system prompt and all user/assistant text blocks, including demonstrations. Only `max_tokens` and `temperature` (0–1) are supported. Tools, media, developer/tool messages, continuation state, structured outputs, reasoning, provider-side caching and other non-default generation settings are refused before the SDK call.

```
import asyncio
from dataclasses import replace

import dspy
import litellm
from openai import APIError  # LiteLLM's SDK errors inherit this public type.
from dspy import lm15

MODEL = "bedrock/amazon.nova-lite-v1:0"
REGION = "us-east-1"
litellm.num_retries = 0  # Application-owned LiteLLM setting; DSPy owns retries.


class NovaText:
    supported_params = {"max_tokens", "temperature"}

    def complete(self, request: lm15.Request) -> lm15.Response:
        config = request.config
        if request.model != MODEL or request.tools or not isinstance(request.system, (str, type(None))):
            raise lm15.UnsupportedFeatureError("Nova recipe requires its fixed model, no tools, and a string system prompt")
        if replace(config, max_tokens=None, temperature=None) != lm15.Config():
            raise lm15.UnsupportedFeatureError("Nova recipe supports only max_tokens and temperature")
        if config.temperature is not None and config.temperature > 1:
            raise lm15.UnsupportedFeatureError("Nova recipe requires temperature between 0 and 1")
        messages = []
        if request.system is not None:
            messages.append({"role": "system", "content": request.system})
        for message in request.messages:
            if (message.role not in {"user", "assistant"} or message.text is None
                    or message.continuation or any(part.continuation for part in message.parts)):
                raise lm15.UnsupportedFeatureError("Nova recipe accepts only text messages without continuation state")
            messages.append({"role": message.role, "content": [
                {"type": "text", "text": part.text} for part in message.parts
            ]})
        options = {key: getattr(config, key) for key in self.supported_params if getattr(config, key) is not None}
        try:
            raw = litellm.completion(
                model=MODEL, messages=messages, aws_region_name=REGION,
                timeout=30, num_retries=0, caching=False, drop_params=False, **options,
            )
        except APIError as exc:
            status = getattr(exc, "status_code", None)
            error = {400: lm15.InvalidRequestError, 401: lm15.AuthError,
                     403: lm15.AuthError, 429: lm15.RateLimitError}.get(status, lm15.ProviderError)
            if isinstance(exc, litellm.Timeout):
                error = lm15.TimeoutError
            elif isinstance(exc, litellm.APIConnectionError):
                # LiteLLM also uses this class for unclassified HTTP failures.
                error, status = lm15.ProviderError, None
            elif status is not None and status >= 500:
                error = lm15.ServerError
            raise error(str(exc), provider="bedrock", status=status) from exc
        response = lm15.response_from_openai_chat(raw.model_dump(), model=request.model)
        if response.message.text is None:
            raise lm15.UnsupportedFeatureError("Nova recipe received a non-text response")
        return response


class AsyncNovaText(NovaText):
    async def complete(self, request: lm15.Request) -> lm15.Response:
        return await asyncio.to_thread(super().complete, request)
```

Use the pair for ordinary synchronous and asynchronous prediction:

```
lm = dspy.LM(MODEL, engine=NovaText(), async_engine=AsyncNovaText(),
             temperature=0, max_tokens=128, cache=True, num_retries=2)
dspy.configure(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False))
program = dspy.Predict("question -> answer")
print(program(question="What is the capital of France?").answer)
print(asyncio.run(program.acall(question="What is the capital of Germany?")).answer)
```

In an existing event loop, use `await program.acall(...)` instead of `asyncio.run(...)`. The async engine runs the same blocking SDK call in a worker thread; cancelling the await does not stop an in-flight HTTP request. The SDK timeout is 30 seconds. There is no streaming or engine serialization in this recipe; reattach the engine after loading a program saved without its LM.

DSPy owns response caching and retry limits; LiteLLM caching and retries are disabled here. Keep this application’s LiteLLM retry setting at zero and do not add SDK/router fallback or retry policies. Use an isolated DSPy cache for this deployment; changing AWS account, region or endpoint does not itself partition DSPy’s response cache. The public lm15 response reader preserves reported usage, finish reasons and provider data; this recipe makes no cost-accounting promise.

Retry classification is limited to what LiteLLM identifies: rate limits, timeouts and server failures are retryable. Its generic `APIConnectionError` can also represent an unclassified HTTP failure (observed with a local 403), so this recipe reports it as a non-retryable `LMProviderError`, without claiming an HTTP status. The SDK exception remains in the cause chain; connection failures are deliberately not retried automatically either.

**Validation boundary:** the copied recipe was executed through `dspy.Predict` with LiteLLM 1.96.0 against a disposable loopback Converse endpoint, with fake AWS credentials and locally checked SigV4 signatures. Sync/async text output, request model/body, local refusals, SDK failures, DSPy retries and cache hits were checked. This is local protocol validation with canned responses, **not live AWS validation**: AWS model availability, IAM permissions, regional support and real Nova output quality remain unverified. It does not establish acceptance of this exception by the migration gate.

```
 October 28, 2026
```
