# dspy.ReAct

## `dspy.ReAct(signature: type[Signature], tools: list[Callable], max_iters: int = 10)`

Bases: `Module`

ReAct stands for "Reasoning and Acting," a popular paradigm for building tool-using agents. In this approach, the language model is iteratively provided with a list of tools and has to reason about the current situation. The model decides whether to call a tool to gather more information or to finish the task based on its reasoning process. The DSPy version of ReAct is generalized to work over any signature, thanks to signature polymorphism.

Parameters:

| Name        | Type              | Description                                                                          | Default    |
| ----------- | ----------------- | ------------------------------------------------------------------------------------ | ---------- |
| `signature` | `type[Signature]` | The signature of the module, which defines the input and output of the react module. | *required* |
| `tools`     | `list[Callable]`  | A list of functions, callable objects, or dspy.Tool instances.                       | *required* |
| `max_iters` | `Optional[int]`   | The maximum number of iterations to run. Defaults to 10.                             | `10`       |

Example:

```
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."

react = dspy.ReAct(signature="question->answer", tools=[get_weather])
pred = react(question="What is the weather in Tokyo?")
```

Source code in `dspy/predict/react.py`

````
def __init__(self, signature: type["Signature"], tools: list[Callable], max_iters: int = 10):
    """
    ReAct stands for "Reasoning and Acting," a popular paradigm for building tool-using agents.
    In this approach, the language model is iteratively provided with a list of tools and has
    to reason about the current situation. The model decides whether to call a tool to gather more
    information or to finish the task based on its reasoning process. The DSPy version of ReAct is
    generalized to work over any signature, thanks to signature polymorphism.

    Args:
        signature: The signature of the module, which defines the input and output of the react module.
        tools (list[Callable]): A list of functions, callable objects, or `dspy.Tool` instances.
        max_iters (Optional[int]): The maximum number of iterations to run. Defaults to 10.

    Example:

    ```python
    def get_weather(city: str) -> str:
        return f"The weather in {city} is sunny."

    react = dspy.ReAct(signature="question->answer", tools=[get_weather])
    pred = react(question="What is the weather in Tokyo?")
    ```
    """
    super().__init__()
    self.signature = signature = ensure_signature(signature)
    self.max_iters = max_iters

    tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]
    tools = {tool.name: tool for tool in tools}

    inputs = ", ".join([f"`{k}`" for k in signature.input_fields.keys()])
    outputs = ", ".join([f"`{k}`" for k in signature.output_fields.keys()])
    instr = [f"{signature.instructions}\n"] if signature.instructions else []

    instr.extend(
        [
            f"You are an Agent. In each episode, you will be given the fields {inputs} as input. And you can see your past trajectory so far.",
            f"Your goal is to use one or more of the supplied tools to collect any necessary information for producing {outputs}.\n",
            "To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.",
            "After each tool call, you receive a resulting observation, which gets appended to your trajectory.\n",
            "When writing next_thought, you may reason about the current situation and plan for future steps.",
            "When selecting the next_tool_name and its next_tool_args, the tool must be one of:\n",
        ]
    )

    tools["finish"] = Tool(
        func=lambda: "Completed.",
        name="finish",
        desc=f"Marks the task as complete. That is, signals that all information for producing the outputs, i.e. {outputs}, are now available to be extracted.",
        args={},
    )

    for idx, tool in enumerate(tools.values()):
        instr.append(f"({idx + 1}) {tool}")
    instr.append("When providing `next_tool_args`, the value inside the field must be in JSON format")

    react_signature = (
        dspy.Signature({**signature.input_fields}, "\n".join(instr))
        .append("trajectory", dspy.InputField(), type_=str)
        .append("next_thought", dspy.OutputField(), type_=str)
        .append("next_tool_name", dspy.OutputField(), type_=Literal[tuple(tools.keys())])
        .append("next_tool_args", dspy.OutputField(), type_=dict[str, Any])
    )

    fallback_signature = dspy.Signature(
        {**signature.input_fields, **signature.output_fields},
        signature.instructions,
    ).append("trajectory", dspy.InputField(), type_=str)

    self.tools = tools
    self.react = dspy.Predict(react_signature)
    self.extract = dspy.ChainOfThought(fallback_signature)
````

### Functions

#### `__call__(*args, **kwargs) -> Prediction`

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

#### `acall(*args, **kwargs) -> Prediction`

Source code in `dspy/primitives/module.py`

```
@with_callbacks
async def acall(self, *args, **kwargs) -> Prediction:
    from dspy.dsp.utils.settings import thread_local_overrides

    caller_modules = settings.caller_modules or []
    caller_modules = list(caller_modules)
    caller_modules.append(self)

    with settings.context(caller_modules=caller_modules):
        if settings.track_usage and thread_local_overrides.get().get("usage_tracker") is None:
            with track_usage() as usage_tracker:
                output = await self.aforward(*args, **kwargs)
                tokens = usage_tracker.get_total_tokens()
                self._set_lm_usage(tokens, output)

                return output

        return await self.aforward(*args, **kwargs)
```

#### `aforward(**input_args)`

Source code in `dspy/predict/react.py`

```
async def aforward(self, **input_args):
    trajectory = {}
    max_iters = input_args.pop("max_iters", self.max_iters)
    for idx in range(max_iters):
        try:
            pred = await self._async_call_with_potential_trajectory_truncation(self.react, trajectory, **input_args)
        except ValueError as err:
            logger.warning(f"Ending the trajectory: Agent failed to select a valid tool: {_fmt_exc(err)}")
            break

        trajectory[f"thought_{idx}"] = pred.next_thought
        trajectory[f"tool_name_{idx}"] = pred.next_tool_name
        trajectory[f"tool_args_{idx}"] = pred.next_tool_args

        try:
            trajectory[f"observation_{idx}"] = await self.tools[pred.next_tool_name].acall(**pred.next_tool_args)
        except Exception as err:
            trajectory[f"observation_{idx}"] = f"Execution error in {pred.next_tool_name}: {_fmt_exc(err)}"

        if pred.next_tool_name == "finish":
            break

    extract = await self._async_call_with_potential_trajectory_truncation(self.extract, trajectory, **input_args)
    return dspy.Prediction(trajectory=trajectory, **extract)
```

#### `batch(examples: list[Example], num_threads: int | None = None, max_errors: int | None = None, return_failed_examples: bool = False, provide_traceback: bool | None = None, disable_progress_bar: bool = False) -> list[Example] | tuple[list[Example], list[Example], list[Exception]]`

Processes a list of dspy.Example instances in parallel using the Parallel module.

Parameters:

| Name                     | Type            | Description                                       | Default                                                                                                      |
| ------------------------ | --------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `examples`               | `list[Example]` | List of dspy.Example instances to process.        | *required*                                                                                                   |
| `num_threads`            | \`int           | None\`                                            | Number of threads to use for parallel processing.                                                            |
| `max_errors`             | \`int           | None\`                                            | Maximum number of errors allowed before stopping execution. If None, inherits from dspy.settings.max_errors. |
| `return_failed_examples` | `bool`          | Whether to return failed examples and exceptions. | `False`                                                                                                      |
| `provide_traceback`      | \`bool          | None\`                                            | Whether to include traceback information in error logs.                                                      |
| `disable_progress_bar`   | `bool`          | Whether to display the progress bar.              | `False`                                                                                                      |

Returns:

| Type            | Description                                              |
| --------------- | -------------------------------------------------------- |
| \`list[Example] | tuple\[list[Example], list[Example], list[Exception]\]\` |

Source code in `dspy/primitives/module.py`

```
def batch(
    self,
    examples: list[Example],
    num_threads: int | None = None,
    max_errors: int | None = None,
    return_failed_examples: bool = False,
    provide_traceback: bool | None = None,
    disable_progress_bar: bool = False,
) -> list[Example] | tuple[list[Example], list[Example], list[Exception]]:
    """
    Processes a list of dspy.Example instances in parallel using the Parallel module.

    Args:
        examples: List of dspy.Example instances to process.
        num_threads: Number of threads to use for parallel processing.
        max_errors: Maximum number of errors allowed before stopping execution.
            If ``None``, inherits from ``dspy.settings.max_errors``.
        return_failed_examples: Whether to return failed examples and exceptions.
        provide_traceback: Whether to include traceback information in error logs.
        disable_progress_bar: Whether to display the progress bar.

    Returns:
        List of results, and optionally failed examples and exceptions.
    """
    # Create a list of execution pairs (self, example)
    exec_pairs = [(self, example.inputs()) for example in examples]

    # Create an instance of Parallel
    parallel_executor = Parallel(
        num_threads=num_threads,
        max_errors=max_errors,
        return_failed_examples=return_failed_examples,
        provide_traceback=provide_traceback,
        disable_progress_bar=disable_progress_bar,
    )

    # Execute the forward method of Parallel
    if return_failed_examples:
        results, failed_examples, exceptions = parallel_executor.forward(exec_pairs)
        return results, failed_examples, exceptions
    else:
        results = parallel_executor.forward(exec_pairs)
        return results
```

#### `deepcopy()`

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

#### `dump_state(json_mode=True)`

Source code in `dspy/primitives/base_module.py`

```
def dump_state(self, json_mode=True):
    return {name: param.dump_state(json_mode=json_mode) for name, param in self.named_parameters()}
```

#### `forward(**input_args)`

Source code in `dspy/predict/react.py`

```
def forward(self, **input_args):
    trajectory = {}
    max_iters = input_args.pop("max_iters", self.max_iters)
    for idx in range(max_iters):
        try:
            pred = self._call_with_potential_trajectory_truncation(self.react, trajectory, **input_args)
        except ValueError as err:
            logger.warning(f"Ending the trajectory: Agent failed to select a valid tool: {_fmt_exc(err)}")
            break

        trajectory[f"thought_{idx}"] = pred.next_thought
        trajectory[f"tool_name_{idx}"] = pred.next_tool_name
        trajectory[f"tool_args_{idx}"] = pred.next_tool_args

        try:
            trajectory[f"observation_{idx}"] = self.tools[pred.next_tool_name](**pred.next_tool_args)
        except Exception as err:
            trajectory[f"observation_{idx}"] = f"Execution error in {pred.next_tool_name}: {_fmt_exc(err)}"

        if pred.next_tool_name == "finish":
            break

    extract = self._call_with_potential_trajectory_truncation(self.extract, trajectory, **input_args)
    return dspy.Prediction(trajectory=trajectory, **extract)
```

#### `get_lm()`

Source code in `dspy/primitives/module.py`

```
def get_lm(self):
    all_used_lms = [param.lm for _, param in self.named_predictors()]

    if len(set(all_used_lms)) == 1:
        return all_used_lms[0]

    raise ValueError("Multiple LMs are being used in the module. There's no unique LM to return.")
```

#### `inspect_history(n: int = 1)`

Source code in `dspy/primitives/module.py`

```
def inspect_history(self, n: int = 1):
    return pretty_print_history(self.history, n)
```

#### `load(path, allow_pickle=False)`

Load the saved module. You may also want to check out dspy.load, if you want to load an entire program, not just the state for an existing program.

Parameters:

| Name           | Type   | Description                                                                                                                                                                           | Default    |
| -------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `path`         | `str`  | Path to the saved state file, which should be a .json or a .pkl file                                                                                                                  | *required* |
| `allow_pickle` | `bool` | If True, allow loading .pkl files, which can run arbitrary code. This is dangerous and should only be used if you are sure about the source of the file and in a trusted environment. | `False`    |

Source code in `dspy/primitives/base_module.py`

```
def load(self, path, allow_pickle=False):
    """Load the saved module. You may also want to check out dspy.load, if you want to
    load an entire program, not just the state for an existing program.

    Args:
        path (str): Path to the saved state file, which should be a .json or a .pkl file
        allow_pickle (bool): If True, allow loading .pkl files, which can run arbitrary code.
            This is dangerous and should only be used if you are sure about the source of the file and in a trusted environment.
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
    self.load_state(state)
```

#### `load_state(state)`

Source code in `dspy/primitives/base_module.py`

```
def load_state(self, state):
    for name, param in self.named_parameters():
        param.load_state(state[name])
```

#### `map_named_predictors(func)`

Applies a function to all named predictors.

Source code in `dspy/primitives/module.py`

```
def map_named_predictors(self, func):
    """Applies a function to all named predictors."""
    for name, predictor in self.named_predictors():
        set_attribute_by_name(self, name, func(predictor))
    return self
```

#### `named_parameters()`

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

#### `named_predictors()`

Source code in `dspy/primitives/module.py`

```
def named_predictors(self):
    from dspy.predict.predict import Predict

    return [(name, param) for name, param in self.named_parameters() if isinstance(param, Predict)]
```

#### `named_sub_modules(type_=None, skip_compiled=False) -> Generator[tuple[str, BaseModule], None, None]`

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

#### `parameters()`

Source code in `dspy/primitives/base_module.py`

```
def parameters(self):
    return [param for _, param in self.named_parameters()]
```

#### `predictors()`

Source code in `dspy/primitives/module.py`

```
def predictors(self):
    return [param for _, param in self.named_predictors()]
```

#### `reset_copy()`

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

#### `save(path, save_program=False, modules_to_serialize=None)`

Save the module.

Save the module to a directory or a file. There are two modes:

- `save_program=False`: Save only the state of the module to a json or pickle file, based on the value of the file extension.
- `save_program=True`: Save the whole module to a directory via cloudpickle, which contains both the state and architecture of the model.

If `save_program=True` and `modules_to_serialize` are provided, it will register those modules for serialization with cloudpickle's `register_pickle_by_value`. This causes cloudpickle to serialize the module by value rather than by reference, ensuring the module is fully preserved along with the saved program. This is useful when you have custom modules that need to be serialized alongside your program. If None, then no modules will be registered for serialization.

We also save the dependency versions, so that the loaded model can check if there is a version mismatch on critical dependencies or DSPy version.

Parameters:

| Name                   | Type   | Description                                                                                                                                | Default    |
| ---------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------ | ---------- |
| `path`                 | `str`  | Path to the saved state file, which should be a .json or .pkl file when save_program=False, and a directory when save_program=True.        | *required* |
| `save_program`         | `bool` | If True, save the whole module to a directory via cloudpickle, otherwise only save the state.                                              | `False`    |
| `modules_to_serialize` | `list` | A list of modules to serialize with cloudpickle's register_pickle_by_value. If None, then no modules will be registered for serialization. | `None`     |

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

#### `set_lm(lm)`

Source code in `dspy/primitives/module.py`

```
def set_lm(self, lm):
    for _, param in self.named_predictors():
        param.lm = lm
```

#### `truncate_trajectory(trajectory)`

Truncates the trajectory so that it fits in the context window.

Users can override this method to implement their own truncation logic.

Source code in `dspy/predict/react.py`

```
def truncate_trajectory(self, trajectory):
    """Truncates the trajectory so that it fits in the context window.

    Users can override this method to implement their own truncation logic.
    """
    keys = list(trajectory.keys())
    if len(keys) < 4:
        # Every tool call has 4 keys: thought, tool_name, tool_args, and observation.
        raise ValueError(
            "The trajectory is too long so your prompt exceeded the context window, but the trajectory cannot be "
            "truncated because it only has one tool call."
        )

    for key in keys[:4]:
        trajectory.pop(key)

    return trajectory
```

:::
