# dspy.LM

## `dspy.LM(model: str, model_type: Literal['chat', 'text', 'responses'] = 'chat', temperature: float | None = None, max_tokens: int | None = None, cache: bool = True, callbacks: list[BaseCallback] | None = None, num_retries: int = 3, provider: Provider | None = None, finetuning_model: str | None = None, launch_kwargs: dict[str, Any] | None = None, train_kwargs: dict[str, Any] | None = None, use_developer_role: bool = False, **kwargs)`

Bases: `BaseLM`

A language model supporting chat or text completion requests for use with DSPy modules.

Create a new language model instance for use with DSPy modules and programs.

Parameters:

| Name               | Type                                   | Description                                                                                                                                                                                                                                                                                                                                      | Default                                                                                                                             |
| ------------------ | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| `model`            | `str`                                  | The model to use. This should be a string of the form "llm_provider/llm_name" supported by LiteLLM. For example, "openai/gpt-4o".                                                                                                                                                                                                                | *required*                                                                                                                          |
| `model_type`       | `Literal['chat', 'text', 'responses']` | The type of the model, either "chat" or "text".                                                                                                                                                                                                                                                                                                  | `'chat'`                                                                                                                            |
| `temperature`      | \`float                                | None\`                                                                                                                                                                                                                                                                                                                                           | The sampling temperature to use when generating responses.                                                                          |
| `max_tokens`       | \`int                                  | None\`                                                                                                                                                                                                                                                                                                                                           | The maximum number of tokens to generate per response.                                                                              |
| `cache`            | `bool`                                 | Whether to cache the model responses for reuse to improve performance and reduce costs.                                                                                                                                                                                                                                                          | `True`                                                                                                                              |
| `callbacks`        | \`list[BaseCallback]                   | None\`                                                                                                                                                                                                                                                                                                                                           | A list of callback functions to run before and after each request.                                                                  |
| `num_retries`      | `int`                                  | The number of times to retry a request if it fails transiently due to network error, rate limiting, etc. Requests are retried with exponential backoff.                                                                                                                                                                                          | `3`                                                                                                                                 |
| `provider`         | \`Provider                             | None\`                                                                                                                                                                                                                                                                                                                                           | The provider to use. If not specified, the provider will be inferred from the model.                                                |
| `finetuning_model` | \`str                                  | None\`                                                                                                                                                                                                                                                                                                                                           | The model to finetune. In some providers, the models available for finetuning is different from the models available for inference. |
| `rollout_id`       |                                        | Optional integer used to differentiate cache entries for otherwise identical requests. Different values bypass DSPy's caches while still caching future calls with the same inputs and rollout ID. Note that rollout_id only affects generation when temperature is non-zero. This argument is stripped before sending requests to the provider. | *required*                                                                                                                          |

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
    """
    Create a new language model instance for use with DSPy modules and programs.

    Args:
        model: The model to use. This should be a string of the form ``"llm_provider/llm_name"``
               supported by LiteLLM. For example, ``"openai/gpt-4o"``.
        model_type: The type of the model, either ``"chat"`` or ``"text"``.
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
    # Remember to update LM.copy() if you modify the constructor!
    self.model = model
    self.model_type = model_type
    self.cache = cache
    self.provider = provider or self.infer_provider()
    self.callbacks = callbacks or []
    self.history = []
    self.num_retries = num_retries
    self.finetuning_model = finetuning_model
    self.launch_kwargs = launch_kwargs or {}
    self.train_kwargs = train_kwargs or {}
    self.use_developer_role = use_developer_role
    self._warned_zero_temp_rollout = False

    # Handle model-specific configuration for different model families
    model_family = model.split("/")[-1].lower() if "/" in model else model.lower()

    # Recognize OpenAI reasoning models (o1, o3, o4, gpt-5 family)
    # Exclude non-reasoning variants like gpt-5-chat this is in azure ai foundry
    # Allow date suffixes like -2023-01-01 after model name or mini/nano/pro
    # For gpt-5, use negative lookahead to exclude -chat and allow other suffixes
    model_pattern = re.match(
        r"^(?:o[1345](?:-(?:mini|nano|pro))?(?:-\d{4}-\d{2}-\d{2})?|gpt-5(?!-chat)(?:-.*)?)$",
        model_family,
    )

    if model_pattern:
        if (temperature and temperature != 1.0) or (max_tokens and max_tokens < 16000):
            raise ValueError(
                "OpenAI's reasoning models require passing temperature=1.0 or None and max_tokens >= 16000 or None to "
                "`dspy.LM(...)`, e.g., dspy.LM('openai/gpt-5', temperature=1.0, max_tokens=16000)"
            )
        self.kwargs = dict(temperature=temperature, max_completion_tokens=max_tokens, **kwargs)
        if self.kwargs.get("rollout_id") is None:
            self.kwargs.pop("rollout_id", None)
    else:
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
        if self.kwargs.get("rollout_id") is None:
            self.kwargs.pop("rollout_id", None)

    self._warn_zero_temp_rollout(self.kwargs.get("temperature"), self.kwargs.get("rollout_id"))
```

### Functions

#### `__call__(prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs) -> list[dict[str, Any] | str]`

Source code in `dspy/clients/base_lm.py`

```
@with_callbacks
def __call__(
    self,
    prompt: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    **kwargs
) -> list[dict[str, Any] | str]:
    response = self.forward(prompt=prompt, messages=messages, **kwargs)
    outputs = self._process_lm_response(response, prompt, messages, **kwargs)

    return outputs
```

#### `acall(prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs) -> list[dict[str, Any] | str]`

Source code in `dspy/clients/base_lm.py`

```
@with_callbacks
async def acall(
    self,
    prompt: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    **kwargs
) -> list[dict[str, Any] | str]:
    response = await self.aforward(prompt=prompt, messages=messages, **kwargs)
    outputs = self._process_lm_response(response, prompt, messages, **kwargs)
    return outputs
```

#### `aforward(prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs)`

Source code in `dspy/clients/lm.py`

```
async def aforward(
    self,
    prompt: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    **kwargs,
):
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
    except LitellmContextWindowExceededError as e:
        raise ContextWindowExceededError(model=self.model) from e

    self._check_truncation(results)

    if not getattr(results, "cache_hit", False) and dspy.settings.usage_tracker:
        settings.usage_tracker.add_usage(self.model, dict(getattr(results, "usage", {})))
    return results
```

#### `copy(**kwargs)`

Returns a copy of the language model with possibly updated parameters.

Any provided keyword arguments update the corresponding attributes or LM kwargs of the copy. For example, `lm.copy(rollout_id=1, temperature=1.0)` returns an LM whose requests use a different rollout ID at non-zero temperature to bypass cache collisions.

Source code in `dspy/clients/base_lm.py`

```
def copy(self, **kwargs):
    """Returns a copy of the language model with possibly updated parameters.

    Any provided keyword arguments update the corresponding attributes or LM kwargs of
    the copy. For example, ``lm.copy(rollout_id=1, temperature=1.0)`` returns an LM whose
    requests use a different rollout ID at non-zero temperature to bypass cache collisions.
    """

    import copy

    new_instance = copy.deepcopy(self)
    new_instance.history = []

    for key, value in kwargs.items():
        if hasattr(self, key):
            setattr(new_instance, key, value)
        if (key in self.kwargs) or (not hasattr(self, key)):
            if value is None:
                new_instance.kwargs.pop(key, None)
            else:
                new_instance.kwargs[key] = value
    if hasattr(new_instance, "_warned_zero_temp_rollout"):
        new_instance._warned_zero_temp_rollout = False

    return new_instance
```

#### `dump_state()`

Source code in `dspy/clients/lm.py`

```
def dump_state(self):
    state_keys = [
        "model",
        "model_type",
        "cache",
        "num_retries",
        "finetuning_model",
        "launch_kwargs",
        "train_kwargs",
    ]
    # Exclude api_key from kwargs to prevent API keys from being saved in plain text
    filtered_kwargs = {k: v for k, v in self.kwargs.items() if k != "api_key"}
    return {key: getattr(self, key) for key in state_keys} | filtered_kwargs
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
        raise ValueError(
            f"Provider {self.provider} does not support fine-tuning, please specify your provider by explicitly "
            "setting `provider` when creating the `dspy.LM` instance. For example, "
            "`dspy.LM('openai/gpt-4.1-mini-2025-04-14', provider=dspy.OpenAIProvider())`."
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

Source code in `dspy/clients/lm.py`

```
def forward(
    self,
    prompt: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    **kwargs
):
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
    except LitellmContextWindowExceededError as e:
        raise ContextWindowExceededError(model=self.model) from e

    self._check_truncation(results)

    if not getattr(results, "cache_hit", False) and dspy.settings.usage_tracker:
        settings.usage_tracker.add_usage(self.model, dict(getattr(results, "usage", {})))
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

#### `reinforce(train_kwargs) -> ReinforceJob`

Source code in `dspy/clients/lm.py`

```
def reinforce(self, train_kwargs) -> ReinforceJob:
    # TODO(GRPO Team): Should we return an initialized job here?
    from dspy import settings as settings

    err = f"Provider {self.provider} does not implement the reinforcement learning interface."
    assert self.provider.reinforceable, err

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
