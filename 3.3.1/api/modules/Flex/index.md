# dspy.Flex

`Flex` is a DSPy module whose implementation is *optimizable code* rather than a fixed prompt. You construct it from a signature, and it defaults to a thin baseline over that signature. What makes it different is what an optimizer is allowed to do with it: instead of only rewriting instructions, `dspy.GEPA` can rewrite the module’s entire source — splitting the task into multiple predictors, folding deterministic steps into plain Python, and authoring its own helper functions. Being a `Flex` is what tells GEPA that the module’s code is an optimizable parameter.

## When to Use Flex

Reach for `Flex` when you’d rather have the optimizer discover the program’s structure than hand-write it. That’s the case when:

- The best decomposition is **unknown or worth searching** — you have a metric and a dataset to judge candidate structures against.
- Parts of the task are **deterministic** and shouldn’t cost an LM call — arithmetic, parsing, lookups, normalization.
- You want the optimizer to **trade accuracy against cost** — e.g. rewarding programs that answer clear cases in code and reserve the LM for genuinely hard ones.

## Basic Usage

```
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-5"))

# Construct Flex from a signature, like any module.
solve = dspy.Flex("invoice: str -> total_cents: int")

# Runs the baseline (a single dspy.Predict).
result = solve(invoice="2 widgets @ $3.50, shipping $1.00")
print(result.total_cents)
```

Out of the box, `solve` is just a `dspy.Predict` over the signature, wrapped in a module (with `tools`, it starts as a `dspy.RLM` instead — see [Tools](#tools)). The point of `Flex` is what happens when you optimize it (see [Optimizing with GEPA](#optimizing-with-gepa)): GEPA can replace that baseline with, say, a predictor that only extracts quantities and unit prices, and a line of Python that multiplies and sums them.

The generated code always runs in a sandbox (`interpreter_factory` defaults to `dspy.PythonInterpreter`), so the example above needs [Deno](https://deno.land/) installed — see [Sandboxed Execution](#sandboxed-execution).

## How Optimization Works

`dspy.GEPA` discovers `Flex` submodules by type. When GEPA compiles a program containing one or more `Flex` submodules, it treats each one as a **code component**: rather than proposing a new instruction string, its reflection model proposes a new *whole module source*, guided by the signature, any available tools, and your metric’s feedback on failing examples. GEPA binds the candidate source, evaluates it, and keeps it if it advances the Pareto frontier — the same search GEPA runs for prompts, applied to code.

A broken candidate can’t crash the optimization run. If the reflection model emits source that fails to bind, GEPA scores that candidate as a failure and moves on, rather than aborting the optimization.

## Optimizing with GEPA

You optimize a `Flex` the same way you optimize any DSPy program — hand it to `dspy.GEPA` with a metric and a trainset:

```
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-5-mini"))  # runs the program

def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    correct = getattr(pred, "total_cents", None) == gold.total_cents
    fb = "Correct." if correct else (
        f"Wrong total: got {getattr(pred, 'total_cents', None)}, expected {gold.total_cents}. "
        "Have the LM extract line items, then sum them in Python."
    )
    return dspy.Prediction(score=1.0 if correct else 0.0, feedback=fb)

solve = dspy.Flex("invoice: str -> total_cents: int")

optimized = dspy.GEPA(
    metric=metric,
    reflection_lm=dspy.LM("openai/gpt-5", temperature=1.0, max_tokens=8000),
    max_metric_calls=60,
).compile(solve, trainset=trainset, valset=valset)

print(optimized.module_src)  # the discovered program
```

The `metric` returns a `dspy.Prediction(score=..., feedback=...)` — a scalar plus natural-language feedback that GEPA reflects on to revise the module. For how to write an effective feedback metric, see [Implementing Feedback Metrics](https://dspy.ai/3.3.1/api/optimizers/GEPA/overview/#implementing-feedback-metrics) in the GEPA guide and the [dspy.GEPA tutorials](https://dspy.ai/3.3.1/tutorials/gepa_ai_program/index.md).

### Rewarding leaner programs with a trace-aware metric

A common goal with `Flex` is to push work out of the LM and into deterministic code. To optimize for that, your metric needs to see *how* an answer was produced, not just whether it was right. Declare a `program_trace` parameter and GEPA will pass the execution trace to the metric at scoring time, letting you penalize LM calls:

```
LLM_CALL_PENALTY = 0.15

def metric(gold, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None):
    correct = getattr(pred, "total_cents", None) == gold.total_cents
    n_calls = len(program_trace) if program_trace else 0
    score = max(0.0, (1.0 if correct else 0.0) - LLM_CALL_PENALTY * n_calls)
    fb = f"{'Correct' if correct else 'Wrong'} — used {n_calls} LM call(s). Settle clear cases in Python."
    return dspy.Prediction(score=score, feedback=fb)
```

The `program_trace` parameter is opt-in *by declaration*: only metrics that name it receive the trace. Keep the penalty small relative to correctness, so a decomposition has to *hold* accuracy to win.

## Sandboxed Execution

`Flex` always runs its generated code in a sandbox — never in the host Python process. `interpreter_factory` defaults to `dspy.PythonInterpreter` (Deno/Pyodide) and must be a **zero-argument factory** returning a fresh `CodeInterpreter`; a bare instance is not accepted, so parallel evaluations receive isolated sessions. The factory is called once per sandbox session, including separate sessions requested by nested code-executing modules. The code is authored by the reflection model, so isolating it keeps it from running with your host’s full permissions. With the default interpreter, optimizer-authored control flow, string work, arithmetic, and supported imports run inside the sandbox, and only provided-tool calls, predictor construction, and predictor calls bridge back to the host, which makes the real LM calls.

Because the default builds a `PythonInterpreter`, *running* a `Flex` needs [Deno](https://deno.land/) installed; without it, the call raises.

```
solve = dspy.Flex(
    "invoice: str -> total_cents: int",
    interpreter_factory=lambda: dspy.PythonInterpreter(),  # the default; swap in your own CodeInterpreter factory here
)
```

Each call owns and shuts down every interpreter session it creates, so a `Flex` holds no live sessions between calls.

## Tools

Pass `tools` and the baseline starts as a `dspy.RLM` instead of a `dspy.Predict`.

```
def lookup_sku(code: str) -> dict:
    """Look up a product by SKU."""
    return catalog[code]

solve = dspy.Flex("order: str -> total_cents: int", tools=[lookup_sku])
```

The optimizer can then wire your tools into `dspy.RLM(..., tools=[...])` / `dspy.ReAct(..., tools=[...])`, or call them directly from `forward`.

## Saving and Loading

A `Flex` serializes its `module_src`, so saving and loading a program restores the optimized code:

```
optimized.save("solver.json")

restored = dspy.Flex("invoice: str -> total_cents: int")
restored.load("solver.json")  # rebinds the saved module_src
```

The interpreter is a **runtime dependency and is not serialized**. Reconstructing with `dspy.Flex(signature)` restores the default sandbox automatically; if you optimized with a custom `interpreter_factory`, pass the same one when you reconstruct the module before calling `load`.

## Constructor Parameters

| Parameter             | Type                            | Default             | Description                                                                                                                                                                                                                              |
| --------------------- | ------------------------------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `signature`           | `str \| Signature`              | required            | Declares the module’s inputs and outputs (e.g. `"invoice -> total_cents: int"`).                                                                                                                                                         |
| `tools`               | `list[Callable \| dspy.Tool]`   | `None`              | Tools the generated code may call. With tools, the baseline is a `dspy.RLM`; without, a `dspy.Predict`.                                                                                                                                  |
| `interpreter_factory` | `Callable[[], CodeInterpreter]` | `PythonInterpreter` | Zero-arg factory returning a fresh `CodeInterpreter` for each sandbox session; defaults to `dspy.PythonInterpreter` (needs Deno). A bare interpreter instance is not accepted. Supported Python and libraries are interpreter-dependent. |
| `max_predictor_calls` | `int \| None`                   | `100`               | Maximum number of predictor calls the generated code can make in one `forward` — a guard against runaway loops. `None` removes the limit.                                                                                                |

## Notes

Experimental

`Flex` is marked experimental. The API and the optimization behavior may change between releases; pin a version if you depend on it.

Interpreter Requirements

`Flex` always runs generated code in a sandbox (`interpreter_factory` defaults to `dspy.PythonInterpreter`), which requires [Deno](https://deno.land/) for its Pyodide WASM sandbox — see the [RLM page](https://dspy.ai/3.3.1/api/modules/RLM/#deno-installation) for installation notes.

## API Reference

### `dspy.Flex(signature: Any, *, tools: list[Any] | None = None, interpreter_factory: Callable[[], CodeInterpreter] = PythonInterpreter, max_predictor_calls: int | None = 100)`

Bases: `Module`, `Parameter`

A module whose implementation is optimizable code, not just a prompt.

Construct it like any module (`dspy.Flex(MySignature)`). It starts as a baseline that delegates to a single `dspy.Predict` over the signature — or `dspy.RLM` when `tools` are given, so the baseline can call them. `dspy.GEPA` recognizes `Flex` instances by type and rewrites their source — a single `dspy.Module` subclass, exposed as `module_src` — into decomposed predictors plus plain Python instead of only tuning instructions.

The optimizer-authored code runs inside an interpreter. `Flex` never runs it in the host Python process. `interpreter_factory` defaults to `dspy.PythonInterpreter` (Deno/Pyodide) and must be a zero-argument callable returning a new `CodeInterpreter`. Flex may validate or lower source and install its guest shim before execution; a custom interpreter therefore defines the Python and standard-library subset available to that source. The optimizer-authored glue runs isolated; only provided-tool calls, predictor construction, and predictor calls bridge back to the host, which makes the real LM calls.

Parameters:

| Name                  | Type                            | Description                                                                                                                                     | Default                                                                                                          |
| --------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `signature`           | `Any`                           | A dspy.Signature class or string declaring inputs/outputs.                                                                                      | *required*                                                                                                       |
| `tools`               | \`list[Any]                     | None\`                                                                                                                                          | dspy.Tool instances or named callables.                                                                          |
| `interpreter_factory` | `Callable[[], CodeInterpreter]` | Zero-argument callable returning a fresh CodeInterpreter for each sandbox session. Defaults to dspy.PythonInterpreter (sandbox, requires Deno). | `PythonInterpreter`                                                                                              |
| `max_predictor_calls` | \`int                           | None\`                                                                                                                                          | Maximum number of predictor calls the optimizer-authored code can make per forward\`.None\`\` removes the limit. |

Source code in `dspy/predict/flex/flex.py`

```
def __init__(
    self,
    signature: Any,
    *,
    tools: list[Any] | None = None,
    interpreter_factory: Callable[[], CodeInterpreter] = PythonInterpreter,
    max_predictor_calls: int | None = 100,
):
    super().__init__()

    self._signature_cls = ensure_signature(signature)
    self._name = getattr(self._signature_cls, "__name__", None) or "Flex"
    self._flex_ctx = FlexContext(signature_cls=self._signature_cls, tools=list(tools or []))

    self._module_src: str | None = None
    self.lm = None

    _validate_interpreter_factory(interpreter_factory)
    self._interpreter_factory = interpreter_factory
    self._max_predictor_calls = max_predictor_calls

    self._rebuild_bridge()
    self._bind_code(self._baseline_src())
```

#### Attributes

##### `module_src: str | None`

##### `signature: type[Signature]`

#### Methods:

##### `__call__(*args, **kwargs) -> Prediction`

Source code in `dspy/primitives/module.py`

```
@with_callbacks
def __call__(self, *args, **kwargs) -> Prediction:
    from dspy.dsp.utils.settings import thread_local_overrides

    caller_modules = settings.caller_modules or []
    caller_modules = list(caller_modules)
    caller_modules.append(self)

    with settings.context(caller_modules=caller_modules):
        if settings.track_usage and thread_local_overrides.get().get("usage_tracker") is None:
            with track_usage() as usage_tracker:
                output = self.forward(*args, **kwargs)
            tokens = usage_tracker.get_total_tokens()
            self._set_lm_usage(tokens, output)

            return output

        return self.forward(*args, **kwargs)
```

##### `forward(*args: Any, **kwargs: Any) -> Any`

Run the bound `forward` inside the interpreter.

Source code in `dspy/predict/flex/flex.py`

```
def forward(self, *args: Any, **kwargs: Any) -> Any:
    """Run the bound ``forward`` inside the interpreter."""
    if args:
        raise TypeError("dspy.Flex accepts keyword inputs only")
    if self.lm is not None:
        with settings.context(lm=self.lm):
            return self._bridge.forward(kwargs)
    return self._bridge.forward(kwargs)
```

##### `deepcopy()`

Deep copy the module.

This is a tweak to the default python deepcopy that only deep copies `self.parameters()`, and for other attributes, we just do the shallow copy.

Source code in `dspy/primitives/base_module.py`

```
def deepcopy(self):
    """Deep copy the module.

    This is a tweak to the default python deepcopy that only deep copies `self.parameters()`, and for other
    attributes, we just do the shallow copy.
    """
    try:
        # If the instance itself is copyable, we can just deep copy it.
        # Otherwise we will have to create a new instance and copy over the attributes one by one.
        return copy.deepcopy(self)
    except Exception:
        pass

    # Create an empty instance.
    new_instance = self.__class__.__new__(self.__class__)
    # Set attribuetes of the copied instance.
    for attr, value in self.__dict__.items():
        if isinstance(value, BaseModule):
            setattr(new_instance, attr, value.deepcopy())
        else:
            try:
                # Try to deep copy the attribute
                setattr(new_instance, attr, copy.deepcopy(value))
            except Exception:
                logging.warning(
                    f"Failed to deep copy attribute '{attr}' of {self.__class__.__name__}, "
                    "falling back to shallow copy or reference copy."
                )
                try:
                    # Fallback to shallow copy if deep copy fails
                    setattr(new_instance, attr, copy.copy(value))
                except Exception:
                    # If even the shallow copy fails, we just copy over the reference.
                    setattr(new_instance, attr, value)

    return new_instance
```

##### `dump_state(json_mode: bool = True) -> dict[str, Any]`

Source code in `dspy/predict/flex/flex.py`

```
def dump_state(self, json_mode: bool = True) -> dict[str, Any]:
    return {
        "module_src": self._module_src,
        "lm": self.lm.dump_state() if self.lm else None,
    }
```

##### `get_lm()`

Source code in `dspy/predict/flex/flex.py`

```
def get_lm(self):
    return self.lm
```

##### `inspect_history(n: int = 1, file: TextIO | None = None) -> None`

Display the LM call history for this module.

Prints a formatted view of the most recent language model calls made by this module, useful for debugging and understanding the module’s behavior.

Parameters:

| Name   | Type     | Description                                                     | Default                                                                                                                                           |
| ------ | -------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `n`    | `int`    | The number of recent history entries to display. Defaults to 1. | `1`                                                                                                                                               |
| `file` | \`TextIO | None\`                                                          | An optional file-like object to write output to. When provided, ANSI color codes are automatically disabled. Defaults to None (prints to stdout). |

Source code in `dspy/primitives/module.py`

```
def inspect_history(self, n: int = 1, file: "TextIO | None" = None) -> None:
    """Display the LM call history for this module.

    Prints a formatted view of the most recent language model calls
    made by this module, useful for debugging and understanding
    the module's behavior.

    Args:
        n: The number of recent history entries to display.
            Defaults to 1.
        file: An optional file-like object to write output to. When
            provided, ANSI color codes are automatically disabled.
            Defaults to `None` (prints to stdout).
    """
    pretty_print_history(self.history, n, file=file)
```

##### `load(path, allow_pickle=False, allow_unsafe_lm_state=False)`

Load the saved module. You may also want to check out dspy.load, if you want to load an entire program, not just the state for an existing program.

Parameters:

| Name                    | Type   | Description                                                                                                                                                                           | Default    |
| ----------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `path`                  | `str`  | Path to the saved state file, which should be a .json or a .pkl file                                                                                                                  | *required* |
| `allow_pickle`          | `bool` | If True, allow loading .pkl files, which can run arbitrary code. This is dangerous and should only be used if you are sure about the source of the file and in a trusted environment. | `False`    |
| `allow_unsafe_lm_state` | `bool` | If True, preserves unsafe LM endpoint keys (e.g., api_base, base_url, and model_list) from loaded state and allows importing custom LM classes. Enable only for trusted files.        | `False`    |

Source code in `dspy/primitives/base_module.py`

```
def load(self, path, allow_pickle=False, allow_unsafe_lm_state=False):
    """Load the saved module. You may also want to check out dspy.load, if you want to
    load an entire program, not just the state for an existing program.

    Args:
        path (str): Path to the saved state file, which should be a .json or a .pkl file
        allow_pickle (bool): If True, allow loading .pkl files, which can run arbitrary code.
            This is dangerous and should only be used if you are sure about the source of the file and in a trusted environment.
        allow_unsafe_lm_state (bool): If True, preserves unsafe LM endpoint keys (e.g.,
            `api_base`, `base_url`, and `model_list`) from loaded state and allows importing custom LM classes.
            Enable only for trusted files.
    """
    path = Path(path)

    if path.suffix == ".json":
        with open(path, "rb") as f:
            state = orjson.loads(f.read())
    elif path.suffix == ".pkl":
        if not allow_pickle:
            raise ValueError("Loading .pkl files can run arbitrary code, which may be dangerous. Prefer "
                             "saving with .json files if possible. Set `allow_pickle=True` "
                             "if you are sure about the source of the file and in a trusted environment.")
        with open(path, "rb") as f:
            state = cloudpickle.load(f)
    else:
        raise ValueError(f"`path` must end with `.json` or `.pkl`, but received: {path}")

    dependency_versions = get_dependency_versions()
    saved_dependency_versions = state["metadata"]["dependency_versions"]
    for key, saved_version in saved_dependency_versions.items():
        if dependency_versions[key] != saved_version:
            logger.warning(
                f"There is a mismatch of {key} version between saved model and current environment. "
                f"You saved with `{key}=={saved_version}`, but now you have "
                f"`{key}=={dependency_versions[key]}`. This might cause errors or performance downgrade "
                "on the loaded model, please consider loading the model in the same environment as the "
                "saving environment."
            )
    self.load_state(state, allow_unsafe_lm_state=allow_unsafe_lm_state)
```

##### `load_state(state: dict[str, Any], *, allow_unsafe_lm_state: bool = False) -> None`

Source code in `dspy/predict/flex/flex.py`

```
def load_state(self, state: dict[str, Any], *, allow_unsafe_lm_state: bool = False) -> None:
    module_src = state.get("module_src")
    if module_src:
        self._bind_code(module_src)
    lm_state = state.get("lm")
    if lm_state:
        sanitized = _sanitize_lm_state(lm_state, allow_unsafe_lm_state)
        self.lm = (
            BaseLM.load_state(sanitized, allow_custom_lm_class=allow_unsafe_lm_state) if sanitized else None
        )
    else:
        self.lm = None
```

##### `named_parameters()`

Unlike PyTorch, handles (non-recursive) lists of parameters too.

Source code in `dspy/primitives/base_module.py`

```
def named_parameters(self):
    """
    Unlike PyTorch, handles (non-recursive) lists of parameters too.
    """

    import dspy
    from dspy.predict.parameter import Parameter

    visited = set()
    named_parameters = []

    def add_parameter(param_name, param_value):
        if isinstance(param_value, Parameter):
            if id(param_value) not in visited:
                visited.add(id(param_value))
                named_parameters.append((param_name, param_value))

        elif isinstance(param_value, dspy.Module):
            # When a sub-module is pre-compiled, keep it frozen.
            if not getattr(param_value, "_compiled", False):
                for sub_name, param in param_value.named_parameters():
                    add_parameter(f"{param_name}.{sub_name}", param)

    if isinstance(self, Parameter):
        add_parameter("self", self)

    for name, value in self.__dict__.items():
        if isinstance(value, Parameter):
            add_parameter(name, value)

        elif isinstance(value, dspy.Module):
            # When a sub-module is pre-compiled, keep it frozen.
            if not getattr(value, "_compiled", False):
                for sub_name, param in value.named_parameters():
                    add_parameter(f"{name}.{sub_name}", param)

        elif isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                add_parameter(f"{name}[{idx}]", item)

        elif isinstance(value, dict):
            for key, item in value.items():
                add_parameter(f"{name}['{key}']", item)

    return named_parameters
```

##### `named_predictors()`

A Flex’s update unit is only its `module_src`.

Source code in `dspy/predict/flex/flex.py`

```
def named_predictors(self):
    """A Flex's update unit is only its ``module_src``."""
    return []
```

##### `named_sub_modules(type_=None, skip_compiled=False) -> Generator[tuple[str, BaseModule], None, None]`

Find all sub-modules in the module, as well as their names.

Say `self.children[4]['key'].sub_module` is a sub-module. Then the name will be `children[4]['key'].sub_module`. But if the sub-module is accessible at different paths, only one of the paths will be returned.

Source code in `dspy/primitives/base_module.py`

```
def named_sub_modules(self, type_=None, skip_compiled=False) -> Generator[tuple[str, "BaseModule"], None, None]:
    """Find all sub-modules in the module, as well as their names.

    Say `self.children[4]['key'].sub_module` is a sub-module. Then the name will be
    `children[4]['key'].sub_module`. But if the sub-module is accessible at different
    paths, only one of the paths will be returned.
    """
    if type_ is None:
        type_ = BaseModule

    queue = deque([("self", self)])
    seen = {id(self)}

    def add_to_queue(name, item):
        if id(item) not in seen:
            seen.add(id(item))
            queue.append((name, item))

    while queue:
        name, item = queue.popleft()

        if isinstance(item, type_):
            yield name, item

        if isinstance(item, BaseModule):
            if skip_compiled and getattr(item, "_compiled", False):
                continue
            for sub_name, sub_item in item.__dict__.items():
                add_to_queue(f"{name}.{sub_name}", sub_item)

        elif isinstance(item, (list, tuple)):
            for i, sub_item in enumerate(item):
                add_to_queue(f"{name}[{i}]", sub_item)

        elif isinstance(item, dict):
            for key, sub_item in item.items():
                add_to_queue(f"{name}[{key}]", sub_item)
```

##### `parameters()`

Source code in `dspy/primitives/base_module.py`

```
def parameters(self):
    return [param for _, param in self.named_parameters()]
```

##### `predictors()`

Return all Predict modules in this module.

Returns:

| Type | Description                                                      |
| ---- | ---------------------------------------------------------------- |
|      | list\[Predict\]: A list of all Predict instances in this module. |

Examples:

```
>>> import dspy
>>> class MyProgram(dspy.Module):
...     def __init__(self):
...         super().__init__()
...         self.qa = dspy.Predict("question -> answer")
...
>>> program = MyProgram()
>>> len(program.predictors())
1
```

Source code in `dspy/primitives/module.py`

```
def predictors(self):
    """Return all Predict modules in this module.

    Returns:
        list[Predict]: A list of all Predict instances in this module.

    Examples:
        >>> import dspy
        >>> class MyProgram(dspy.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.qa = dspy.Predict("question -> answer")
        ...
        >>> program = MyProgram()
        >>> len(program.predictors())
        1
    """
    return [param for _, param in self.named_predictors()]
```

##### `reset() -> None`

Clear the LM; `module_src` is kept, and predictors are rebuilt from it each forward.

Source code in `dspy/predict/flex/flex.py`

```
def reset(self) -> None:
    """Clear the LM; ``module_src`` is kept, and predictors are rebuilt from it each forward."""
    self.lm = None
```

##### `reset_copy()`

Deep copy the module and reset all parameters.

Source code in `dspy/primitives/base_module.py`

```
def reset_copy(self):
    """Deep copy the module and reset all parameters."""
    new_instance = self.deepcopy()

    for param in new_instance.parameters():
        param.reset()

    return new_instance
```

##### `save(path, save_program=False, modules_to_serialize=None)`

Save the module.

Save the module to a directory or a file. There are two modes:

- `save_program=False`: Save only the state of the module to a json or pickle file, based on the value of the file extension.
- `save_program=True`: Save the whole module to a directory via cloudpickle, which contains both the state and architecture of the model.

If `save_program=True` and `modules_to_serialize` are provided, it will register those modules for serialization with cloudpickle’s `register_pickle_by_value`. This causes cloudpickle to serialize the module by value rather than by reference, ensuring the module is fully preserved along with the saved program. This is useful when you have custom modules that need to be serialized alongside your program. If None, then no modules will be registered for serialization.

We also save the dependency versions, so that the loaded model can check if there is a version mismatch on critical dependencies or DSPy version.

Parameters:

| Name                   | Type   | Description                                                                                                                                | Default    |
| ---------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------ | ---------- |
| `path`                 | `str`  | Path to the saved state file, which should be a .json or .pkl file when save_program=False, and a directory when save_program=True.        | *required* |
| `save_program`         | `bool` | If True, save the whole module to a directory via cloudpickle, otherwise only save the state.                                              | `False`    |
| `modules_to_serialize` | `list` | A list of modules to serialize with cloudpickle’s register_pickle_by_value. If None, then no modules will be registered for serialization. | `None`     |

Source code in `dspy/primitives/base_module.py`

```
def save(self, path, save_program=False, modules_to_serialize=None):
    """Save the module.

    Save the module to a directory or a file. There are two modes:
    - `save_program=False`: Save only the state of the module to a json or pickle file, based on the value of
        the file extension.
    - `save_program=True`: Save the whole module to a directory via cloudpickle, which contains both the state and
        architecture of the model.

    If `save_program=True` and `modules_to_serialize` are provided, it will register those modules for serialization
    with cloudpickle's `register_pickle_by_value`. This causes cloudpickle to serialize the module by value rather
    than by reference, ensuring the module is fully preserved along with the saved program. This is useful
    when you have custom modules that need to be serialized alongside your program. If None, then no modules
    will be registered for serialization.

    We also save the dependency versions, so that the loaded model can check if there is a version mismatch on
    critical dependencies or DSPy version.

    Args:
        path (str): Path to the saved state file, which should be a .json or .pkl file when `save_program=False`,
            and a directory when `save_program=True`.
        save_program (bool): If True, save the whole module to a directory via cloudpickle, otherwise only save
            the state.
        modules_to_serialize (list): A list of modules to serialize with cloudpickle's `register_pickle_by_value`.
            If None, then no modules will be registered for serialization.

    """
    metadata = {}
    metadata["dependency_versions"] = get_dependency_versions()
    path = Path(path)

    if save_program:
        if path.suffix:
            raise ValueError(
                f"`path` must point to a directory without a suffix when `save_program=True`, but received: {path}"
            )
        if path.exists() and not path.is_dir():
            raise NotADirectoryError(f"The path '{path}' exists but is not a directory.")

        if not path.exists():
            # Create the directory (and any parent directories)
            path.mkdir(parents=True)
        logger.warning("Loading untrusted .pkl files can run arbitrary code, which may be dangerous. To avoid "
                      'this, prefer saving using json format using module.save("module.json").')
        try:
            modules_to_serialize = modules_to_serialize or []
            for module in modules_to_serialize:
                cloudpickle.register_pickle_by_value(module)

            with open(path / "program.pkl", "wb") as f:
                cloudpickle.dump(self, f)
        except Exception as e:
            raise RuntimeError(
                f"Saving failed with error: {e}. Please remove the non-picklable attributes from your DSPy program, "
                "or consider using state-only saving by setting `save_program=False`."
            )
        with open(path / "metadata.json", "wb") as f:
            f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE))

        return

    if path.suffix == ".json":
        state = self.dump_state()
        state["metadata"] = metadata
        try:
            with open(path, "wb") as f:
                f.write(orjson.dumps(state, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE))
        except Exception as e:
            raise RuntimeError(
                f"Failed to save state to {path} with error: {e}. Your DSPy program may contain non "
                "json-serializable objects, please consider saving the state in .pkl by using `path` ending "
                "with `.pkl`, or saving the whole program by setting `save_program=True`."
            )
    elif path.suffix == ".pkl":
        logger.warning("Loading untrusted .pkl files can run arbitrary code, which may be dangerous. To avoid "
                      'this, prefer saving using json format using module.save("module.json").')
        state = self.dump_state(json_mode=False)
        state["metadata"] = metadata
        with open(path, "wb") as f:
            cloudpickle.dump(state, f)
    else:
        raise ValueError(f"`path` must end with `.json` or `.pkl` when `save_program=False`, but received: {path}")
```

##### `set_lm(lm) -> None`

Source code in `dspy/predict/flex/flex.py`

```
def set_lm(self, lm) -> None:
    self.lm = lm
```
