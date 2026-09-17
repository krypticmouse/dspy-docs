# dspy.LM

## `dspy.LM(model: str, model_type: Literal['chat', 'text', 'responses'] = 'chat', temperature: float | None = None, max_tokens: int | None = None, cache: bool = True, callbacks: list[BaseCallback] | None = None, num_retries: int = 3, provider: Provider | None = None, finetuning_model: str | None = None, launch_kwargs: dict[str, Any] | None = None, train_kwargs: dict[str, Any] | None = None, use_developer_role: bool = False, **kwargs)`

Bases: `BaseLM`

A language model supporting chat or text completion requests for use with DSPy modules.

Create a new language model instance for use with DSPy modules and programs.

Parameters:

| Name               | Type                                   | Description                                                                                                                                                                                                                                                                                                                                      | Default                                                                                                                             |
| ------------------ | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| `model`            | `str`                                  | The model to use. This should be a string of the form "llm_provider/llm_name" supported by LiteLLM. For example, "openai/gpt-4o".                                                                                                                                                                                                                | *required*                                                                                                                          |
| `model_type`       | `Literal['chat', 'text', 'responses']` | The type of the model, such as "chat", "text", or "responses".                                                                                                                                                                                                                                                                                   | `'chat'`                                                                                                                            |
| `temperature`      | \`float                                | None\`                                                                                                                                                                                                                                                                                                                                           | The sampling temperature to use when generating responses.                                                                          |
| `max_tokens`       | \`int                                  | None\`                                                                                                                                                                                                                                                                                                                                           | The maximum number of tokens to generate per response.                                                                              |
| `cache`            | `bool`                                 | Whether to cache the model responses for reuse to improve performance and reduce costs.                                                                                                                                                                                                                                                          | `True`                                                                                                                              |
| `callbacks`        | \`list[BaseCallback]                   | None\`                                                                                                                                                                                                                                                                                                                                           | A list of callback functions to run before and after each request.                                                                  |
| `num_retries`      | `int`                                  | The number of times to retry a request if it fails transiently due to network error, rate limiting, etc. Requests are retried with exponential backoff.                                                                                                                                                                                          | `3`                                                                                                                                 |
| `provider`         | \`Provider                             | None\`                                                                                                                                                                                                                                                                                                                                           | The provider to use. If not specified, the provider will be inferred from the model.                                                |
| `finetuning_model` | \`str                                  | None\`                                                                                                                                                                                                                                                                                                                                           | The model to finetune. In some providers, the models available for finetuning is different from the models available for inference. |
| `rollout_id`       |                                        | Optional integer used to differentiate cache entries for otherwise identical requests. Different values bypass DSPy’s caches while still caching future calls with the same inputs and rollout ID. Note that rollout_id only affects generation when temperature is non-zero. This argument is stripped before sending requests to the provider. | *required*                                                                                                                          |

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
        provider: The provider to use. If not specified, the provider will be inferred from the model.
        finetuning_model: The model to finetune. In some providers, the models available for finetuning is different
            from the models available for inference.
        rollout_id: Optional integer used to differentiate cache entries for otherwise
            identical requests. Different values bypass DSPy's caches while still caching
            future calls with the same inputs and rollout ID. Note that `rollout_id`
            only affects generation when `temperature` is non-zero. This argument is
            stripped before sending requests to the provider.
    """
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

Call the configured LM asynchronously.

LiteLLM/provider exceptions are wrapped in DSPy’s structured LM error hierarchy before they are re-raised.

Parameters:

| Name       | Type                     | Description                                                 | Default                                                  |
| ---------- | ------------------------ | ----------------------------------------------------------- | -------------------------------------------------------- |
| `prompt`   | \`str                    | None\`                                                      | Optional prompt text. Ignored when messages is provided. |
| `messages` | \`list\[dict[str, Any]\] | None\`                                                      | Optional chat messages to send to the LM.                |
| `**kwargs` |                          | Per-call LM parameters that override defaults from LM(...). | `{}`                                                     |

Raises:

| Type      | Description                                                                                                                                                                                                                                                                     |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LMError` | Base class for wrapped LM configuration, transport, provider, and unsupported-feature failures. Notable subclasses include dspy.ContextWindowExceededError for context-window failures, which adapters use to avoid inappropriate fallback retries when the prompt is too long. |

Source code in `dspy/clients/lm.py`

```
async def aforward(
    self,
    prompt: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    **kwargs,
):
    """Call the configured LM asynchronously.

    LiteLLM/provider exceptions are wrapped in DSPy's structured LM error
    hierarchy before they are re-raised.

    Args:
        prompt: Optional prompt text. Ignored when `messages` is provided.
        messages: Optional chat messages to send to the LM.
        **kwargs: Per-call LM parameters that override defaults from `LM(...)`.

    Raises:
        dspy.LMError: Base class for wrapped LM configuration, transport,
            provider, and unsupported-feature failures. Notable subclasses
            include `dspy.ContextWindowExceededError` for context-window
            failures, which adapters use to avoid inappropriate fallback
            retries when the prompt is too long.
    """
    # Build the request.
    kwargs = dict(kwargs)
    cache = kwargs.pop("cache", self.cache)

    messages = messages or [{"role": "user", "content": prompt}]
    if self.use_developer_role and self.model_type == "responses":
        messages = [{**m, "role": "developer"} if m.get("role") == "system" else m for m in messages]
    kwargs = {**self.kwargs, **kwargs}
    self._warn_zero_temp_rollout(kwargs.get("temperature"), kwargs.get("rollout_id"))
    if kwargs.get("rollout_id") is None:
        kwargs.pop("rollout_id", None)

    if self.model_type == "chat":
        completion = alitellm_completion
    elif self.model_type == "text":
        completion = alitellm_text_completion
    elif self.model_type == "responses":
        completion = alitellm_responses_completion
    completion, litellm_cache_args = self._get_cached_completion_fn(completion, cache)

    try:
        results = await completion(
            request=dict(model=self.model, messages=messages, **kwargs),
            num_retries=self.num_retries,
            cache=litellm_cache_args,
        )
    except Exception as e:
        if isinstance(e, LMError):
            raise
        raise self._wrap_litellm_exception(e) from e

    self._check_truncation(results)

    return results
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

#### `dump_state()`

Return a sanitized reconstruction state for this LM.

Returns:

| Type | Description                                             |
| ---- | ------------------------------------------------------- |
|      | A dictionary that can be passed to BaseLM.load_state to |
|      | reconstruct this LM. The state excludes API keys.       |

Source code in `dspy/clients/lm.py`

```
def dump_state(self):
    """Return a sanitized reconstruction state for this LM.

    Returns:
        A dictionary that can be passed to `BaseLM.load_state` to
        reconstruct this `LM`. The state excludes API keys.
    """
    state = super().dump_state()
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

#### `forward(prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs)`

Call the configured LM synchronously.

LiteLLM/provider exceptions are wrapped in DSPy’s structured LM error hierarchy before they are re-raised.

Parameters:

| Name       | Type                     | Description                                                 | Default                                                  |
| ---------- | ------------------------ | ----------------------------------------------------------- | -------------------------------------------------------- |
| `prompt`   | \`str                    | None\`                                                      | Optional prompt text. Ignored when messages is provided. |
| `messages` | \`list\[dict[str, Any]\] | None\`                                                      | Optional chat messages to send to the LM.                |
| `**kwargs` |                          | Per-call LM parameters that override defaults from LM(...). | `{}`                                                     |

Raises:

| Type      | Description                                                                                                                                                                                                                                                                     |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LMError` | Base class for wrapped LM configuration, transport, provider, and unsupported-feature failures. Notable subclasses include dspy.ContextWindowExceededError for context-window failures, which adapters use to avoid inappropriate fallback retries when the prompt is too long. |

Source code in `dspy/clients/lm.py`

```
def forward(
    self,
    prompt: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    **kwargs
):
    """Call the configured LM synchronously.

    LiteLLM/provider exceptions are wrapped in DSPy's structured LM error
    hierarchy before they are re-raised.

    Args:
        prompt: Optional prompt text. Ignored when `messages` is provided.
        messages: Optional chat messages to send to the LM.
        **kwargs: Per-call LM parameters that override defaults from `LM(...)`.

    Raises:
        dspy.LMError: Base class for wrapped LM configuration, transport,
            provider, and unsupported-feature failures. Notable subclasses
            include `dspy.ContextWindowExceededError` for context-window
            failures, which adapters use to avoid inappropriate fallback
            retries when the prompt is too long.
    """
    # Build the request.
    kwargs = dict(kwargs)
    cache = kwargs.pop("cache", self.cache)

    messages = messages or [{"role": "user", "content": prompt}]
    if self.use_developer_role and self.model_type == "responses":
        messages = [{**m, "role": "developer"} if m.get("role") == "system" else m for m in messages]
    kwargs = {**self.kwargs, **kwargs}
    self._warn_zero_temp_rollout(kwargs.get("temperature"), kwargs.get("rollout_id"))
    if kwargs.get("rollout_id") is None:
        kwargs.pop("rollout_id", None)

    if self.model_type == "chat":
        completion = litellm_completion
    elif self.model_type == "text":
        completion = litellm_text_completion
    elif self.model_type == "responses":
        completion = litellm_responses_completion
    completion, litellm_cache_args = self._get_cached_completion_fn(completion, cache)

    try:
        results = completion(
            request=dict(model=self.model, messages=messages, **kwargs),
            num_retries=self.num_retries,
            cache=litellm_cache_args,
        )
    except Exception as e:
        if isinstance(e, LMError):
            raise
        raise self._wrap_litellm_exception(e) from e

    self._check_truncation(results)

    return results
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

    model = state.get("model")
    if isinstance(model, str) and _is_openai_reasoning_model(model) and "max_completion_tokens" in state:
        if "max_tokens" not in state:
            state["max_tokens"] = state["max_completion_tokens"]
        state.pop("max_completion_tokens")

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
