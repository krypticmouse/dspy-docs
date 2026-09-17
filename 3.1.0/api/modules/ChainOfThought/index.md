# dspy.ChainOfThought

## `dspy.ChainOfThought(signature: str | type[Signature], rationale_field: FieldInfo | None = None, rationale_field_type: type = str, **config: dict[str, Any])`

Bases: `Module`

A module that reasons step by step in order to predict the output of a task.

Parameters:

| Name                   | Type                                      | Description                                | Default    |
| ---------------------- | ----------------------------------------- | ------------------------------------------ | ---------- |
| `signature`            | `Type[Signature]`                         | The signature of the module.               | *required* |
| `rationale_field`      | `Optional[Union[OutputField, FieldInfo]]` | The field that will contain the reasoning. | `None`     |
| `rationale_field_type` | `Type`                                    | The type of the rationale field.           | `str`      |
| `**config`             | `dict[str, Any]`                          | The configuration for the module.          | `{}`       |

Source code in `dspy/predict/chain_of_thought.py`

```
def __init__(
    self,
    signature: str | type[Signature],
    rationale_field: FieldInfo | None = None,
    rationale_field_type: type = str,
    **config: dict[str, Any],
):
    """
    A module that reasons step by step in order to predict the output of a task.

    Args:
        signature (Type[dspy.Signature]): The signature of the module.
        rationale_field (Optional[Union[dspy.OutputField, pydantic.fields.FieldInfo]]): The field that will contain the reasoning.
        rationale_field_type (Type): The type of the rationale field.
        **config: The configuration for the module.
    """
    super().__init__()
    signature = ensure_signature(signature)
    prefix = "Reasoning: Let's think step by step in order to"
    desc = "${reasoning}"
    rationale_field_type = rationale_field.annotation if rationale_field else rationale_field_type
    rationale_field = rationale_field if rationale_field else dspy.OutputField(prefix=prefix, desc=desc)
    extended_signature = signature.prepend(name="reasoning", field=rationale_field, type_=rationale_field_type)
    self.predict = dspy.Predict(extended_signature, **config)
```

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

#### `aforward(**kwargs)`

Source code in `dspy/predict/chain_of_thought.py`

```
async def aforward(self, **kwargs):
    return await self.predict.acall(**kwargs)
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

#### `forward(**kwargs)`

Source code in `dspy/predict/chain_of_thought.py`

```
def forward(self, **kwargs):
    return self.predict(**kwargs)
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

:::
