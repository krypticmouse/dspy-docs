# dspy.configure

Set the default language model, adapter, and other settings for DSPy.

```
dspy.configure(**kwargs)
```

Call `dspy.configure(...)` once near the top of your script or notebook. Every DSPy module will use these defaults unless you override them with [`dspy.context`](https://dspy.ai/3.3.0/api/utils/context/index.md). The values persist until you call `dspy.configure(...)` again.

Note

Pass a [`dspy.LM`](https://dspy.ai/3.3.0/api/models/LM/index.md) object as `lm`, not a bare model string.

## Settings

| Setting                            | Default | Description                                                                                                                                      |
| ---------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `lm`                               | `None`  | Default language model. Pass a [`dspy.LM`](https://dspy.ai/3.3.0/api/models/LM/index.md) instance.                                               |
| `adapter`                          | `None`  | Formats prompts and parses LM responses. When `None`, modules use [`dspy.ChatAdapter`](https://dspy.ai/3.3.0/api/adapters/ChatAdapter/index.md). |
| `callbacks`                        | `[]`    | Observability and logging hooks. See [Observability](https://dspy.ai/3.3.0/tutorials/observability/index.md).                                    |
| `track_usage`                      | `False` | Record token counts for every LM call.                                                                                                           |
| `async_max_workers`                | `8`     | Maximum concurrent workers for async operations.                                                                                                 |
| `num_threads`                      | `8`     | Thread count for [`dspy.Parallel`](https://dspy.ai/3.3.0/api/modules/Parallel/index.md).                                                         |
| `max_errors`                       | `10`    | Stop parallel execution after this many errors.                                                                                                  |
| `disable_history`                  | `False` | Stop recording LM call history.                                                                                                                  |
| `max_history_size`                 | `10000` | Cap on stored history entries.                                                                                                                   |
| `allow_tool_async_sync_conversion` | `False` | Let async tools run in synchronous code. See [Async](https://dspy.ai/3.3.0/tutorials/async/index.md).                                            |
| `provide_traceback`                | `False` | Include Python tracebacks in error logs.                                                                                                         |
| `warn_on_type_mismatch`            | `True`  | Warn when a module input type does not match the signature.                                                                                      |

## Examples

### Set the default LM

```
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-5-mini"))

qa = dspy.Predict("question -> answer")
result = qa(question="What is the capital of France?")
print(result.answer)
```

### Set the LM and adapter

```
import dspy

dspy.configure(
    lm=dspy.LM("anthropic/claude-sonnet-4-6"),
    adapter=dspy.JSONAdapter(),
)
```

### Enable usage tracking and tune concurrency

```
import dspy

dspy.configure(
    lm=dspy.LM("gemini/gemini-3-flash-preview"),
    track_usage=True,
    async_max_workers=4,
)
```

## When to use `dspy.configure`

Use `dspy.configure(...)` when one set of defaults should apply to most of your program—scripts, notebooks, test setup, or application startup.

If you need different settings for one call or one block, use [`dspy.context`](https://dspy.ai/3.3.0/api/utils/context/index.md) instead.

## Thread safety

Only the thread that first calls `dspy.configure(...)` may call it again. Other threads that try will get a `RuntimeError`. In async code, only the task that first called `dspy.configure(...)` may continue to call it.

For temporary overrides inside worker threads, async tasks, or [`dspy.Parallel`](https://dspy.ai/3.3.0/api/modules/Parallel/index.md) blocks, use [`dspy.context`](https://dspy.ai/3.3.0/api/utils/context/index.md).

## See Also

- [`dspy.context`](https://dspy.ai/3.3.0/api/utils/context/index.md) — temporary overrides that last for one block.
- [`dspy.LM`](https://dspy.ai/3.3.0/api/models/LM/index.md) — create the language model you pass as `lm`.
- [Language Models](https://dspy.ai/3.3.0/learn/programming/language_models/index.md) — overview of LM configuration.
- [Adapters](https://dspy.ai/3.3.0/learn/programming/adapters/index.md) — how adapters format prompts and parse responses.
