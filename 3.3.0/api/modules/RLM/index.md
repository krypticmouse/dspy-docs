# dspy.RLM

`RLM` (Recursive Language Model) is a DSPy module that lets LLMs programmatically explore large contexts through a sandboxed Python REPL. Instead of feeding huge contexts directly into the prompt, RLM treats context as external data that the LLM examines via code execution and recursive sub-LLM calls.

This implements the approach described in [“Recursive Language Models” (Zhang, Kraska, Khattab, 2025)](https://arxiv.org/abs/2512.24601).

## When to Use RLM

As contexts grow, LLM performance degrades — a phenomenon known as [context rot](https://research.trychroma.com/context-rot). RLMs address this by separating the *variable space* (information stored in the REPL) from the *token space* (what the LLM actually processes). The LLM dynamically loads only the context it needs, when it needs it.

Use RLM when:

- Your context is **too large** to fit in the LLM’s context window effectively
- The task benefits from **programmatic exploration** (searching, filtering, aggregating, chunking)
- You need the LLM to decide **how to decompose** the problem, not you

## Basic Usage

```
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-5"))

# Create an RLM module
rlm = dspy.RLM("context, query -> answer")

# Call it like any other module
result = rlm(
    context="...very long document or data...",
    query="What is the total revenue mentioned?"
)
print(result.answer)
```

## Deno Installation

RLM relies on [Deno](https://deno.land/) and [Pyodide](https://pyodide.org/) to create a local WASM sandbox for secure Python execution.

You can install Deno with: `brew install deno` on MacOS or `curl -fsSL https://deno.land/install.sh | sh` on MacOS and Linux. See the [Deno Installation Docs](https://docs.deno.com/runtime/getting_started/installation/) for more details. Make sure to accept the prompt when it asks to add it to your shell profile.

After you have installed Deno, **Make sure to restart your shell.**

Deno may get confused by existing `package.json` files it happens to find; use the environment variable `DENO_NO_PACKAGE_JSON=1` to ignore `package.json` entirely and resolve `npm:pyodide` from its own cache, which is what DSPy expects.

Then you can run `dspy.RLM`.

Users have reported issues with the Deno cache not being found by DSPy. We are actively investigating these issues, and your feedback is greatly appreciated.

You can also work with an external sandbox provider. We are still working on creating an example of using external sandbox providers.

## How It Works

RLM operates in an iterative REPL loop:

1. The LLM receives **metadata** about the context (type, length, preview) but not the full context
1. The LLM writes **Python code** to explore the data (print samples, search, filter)
1. Code executes in a **sandboxed interpreter**, and the LLM sees the output
1. The LLM can call `llm_query(prompt)` to run **sub-LLM calls** for semantic analysis on snippets
1. When done, the LLM calls `SUBMIT(output)` to return the final answer

#### What the LLM sees (step-by-step trace):

##### Step 1: Initial Metadata (no direct access to full context)

```
# Step 1: Peek at the data
print(context[:2000])
```

*Output shown to the LLM:*

```
[Preview of the first 2000 characters of the document]
```

##### Step 2: Write Code to Explore Context

```
# Step 2: Search for relevant sections
import re
matches = re.findall(r'revenue.*?\$[\d,]+', context, re.IGNORECASE)
print(matches)
```

*Output shown to the LLM:*

```
['Revenue in Q4: $5,000,000', 'Total revenue: $20,000,000']
```

##### Step 3: Trigger Sub-LLM Calls

```
# Step 3: Use sub-LLM for semantic extraction
result = llm_query(f"Extract the total revenue from: {matches[1]}")
print(result)
```

*Output shown to the LLM:*

```
$20,000,000
```

##### Step 4: Submit Final Answer

```
# Step 4: Return final answer
SUBMIT(result)
```

*Output shown to the user:*

```
$20,000,000
```

## Constructor Parameters

| Parameter             | Type                               | Default             | Description                                                                       |
| --------------------- | ---------------------------------- | ------------------- | --------------------------------------------------------------------------------- |
| `signature`           | `str \| Signature`                 | required            | Defines inputs and outputs (e.g., `"context, query -> answer"`)                   |
| `max_iters`           | `int`                              | `20`                | Maximum REPL interaction loops before fallback extraction                         |
| `max_llm_calls`       | `int`                              | `50`                | Maximum `llm_query`/`llm_query_batched` calls per execution                       |
| `max_output_chars`    | `int`                              | `10_000`            | Maximum characters to include from REPL output                                    |
| `verbose`             | `bool`                             | `False`             | Log detailed execution info                                                       |
| `tools`               | `list[Union[Callable, dspy.Tool]]` | `None`              | Additional tool functions callable from interpreter code                          |
| `sub_lm`              | `dspy.LM`                          | `None`              | LM for sub-queries. Defaults to `dspy.settings.lm`. Use a cheaper model here.     |
| `interpreter_factory` | `Callable[[], CodeInterpreter]`    | `PythonInterpreter` | Creates one interpreter per invocation. RLM shuts down each returned interpreter. |

## Built-in Tools

Inside the REPL, the LLM has access to:

| Tool                         | Description                                                       |
| ---------------------------- | ----------------------------------------------------------------- |
| `llm_query(prompt)`          | Query a sub-LLM for semantic analysis (~500K char capacity)       |
| `llm_query_batched(prompts)` | Query multiple prompts concurrently (faster for batch operations) |
| `print()`                    | Print output (required to see results)                            |
| `SUBMIT(...)`                | Submit final output and end execution                             |
| Standard library             | `re`, `json`, `collections`, `math`, etc.                         |

## Examples

### Long Document Q&A

```
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-5"))

rlm = dspy.RLM("document, question -> answer", max_iters=10)

with open("large_report.txt") as f:
    document = f.read()  # 500K+ characters

result = rlm(
    document=document,
    question="What were the key findings from Q3?"
)
print(result.answer)
```

### Using a Cheaper Sub-LM

```
import dspy

main_lm = dspy.LM("openai/gpt-5")
cheap_lm = dspy.LM("openai/gpt-5-nano")

dspy.configure(lm=main_lm)

# Root LM (gpt-5) decides strategy; sub-LM (gpt-5-nano) handles extraction
rlm = dspy.RLM("data, query -> summary", sub_lm=cheap_lm)
```

### Multiple Typed Outputs

```
rlm = dspy.RLM("logs -> error_count: int, critical_errors: list[str]")

result = rlm(logs=server_logs)
print(f"Found {result.error_count} errors")
print(f"Critical: {result.critical_errors}")
```

### Custom Tools

```
def fetch_metadata(doc_id: str) -> str:
    """Fetch metadata for a document ID."""
    return database.get_metadata(doc_id)

rlm = dspy.RLM(
    "documents, query -> answer",
    tools=[fetch_metadata]
)
```

### Configuring the Interpreter

Pass the interpreter class directly when it needs no arguments. For configured construction, use any zero-argument callable, such as `functools.partial`:

```
from functools import partial

rlm = dspy.RLM(
    "context, query -> answer",
    interpreter_factory=partial(
        dspy.PythonInterpreter,
        enable_network_access=["example.com"],
    ),
)
```

RLM creates and shuts down one interpreter from this factory per invocation. It adds invocation-scoped tools to the returned interpreter’s mutable `tools` dictionary, so remote sandboxes need a `CodeInterpreter` adapter that supports that protocol. To reuse a caller-owned interpreter, pass it as the first positional argument when calling the module: `rlm(interpreter, context=data, query=query)`. RLM updates its tools and output metadata but does not shut down or restore it. Reuse is supported only for sequential calls to the same RLM instance; use the factory path for concurrency.

### Custom Sandbox-Serializable Inputs

For inputs that should be loaded into the sandbox differently from normal Python values, subclass `dspy.SandboxSerializable`. RLM detects these inputs, sends their serialized payload into the interpreter, runs their setup code, and exposes the reconstructed value under the original input name.

```
class DataFrame(dspy.SandboxSerializable):
    def sandbox_setup(self) -> str:
        return "import pandas as pd\nimport base64\nimport io"

    def to_sandbox(self) -> bytes:
        return base64.b64encode(self.data.to_parquet(index=False))

    def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
        return f"{var_name} = pd.read_parquet(io.BytesIO(base64.b64decode({data_expr})))"

    def rlm_preview(self, max_chars: int = 500) -> str:
        return f"DataFrame: {self.data.shape[0]} rows x {self.data.shape[1]} columns"
```

`SandboxSerializable` also defines a Pydantic schema hook so subclasses can be used directly in DSPy signatures, for example `data: DataFrame = dspy.InputField()`. The hook is intentionally pass-through: Pydantic accepts the object as-is and serializes it with `str(value)` for schema/metadata purposes. RLM’s real sandbox transport still comes from `to_sandbox()` and `sandbox_assignment()`.

### Async Execution

```
import asyncio

rlm = dspy.RLM("context, query -> answer")

async def process():
    result = await rlm.acall(context=data, query="Summarize this")
    return result.answer

answer = asyncio.run(process())
```

### Inspecting the Trajectory

```
result = rlm(context=data, query="Find the magic number")

# See what code the LLM executed
for step in result.trajectory:
    print(f"Code:\n{step['code']}")
    print(f"Output:\n{step['output']}\n")
```

## Output

RLM returns a `Prediction` with:

- **Output fields** from your signature (e.g., `result.answer`)
- **`trajectory`**: List of dicts with `reasoning`, `code`, `output` for each step
- **`final_reasoning`**: The LLM’s reasoning on the final step

## Notes

Experimental

RLM is marked as experimental. The API may change in future releases.

Thread Safety

`interpreter_factory` may be called concurrently and must return a fresh interpreter each time. An interpreter passed as the first positional argument to `rlm(...)` or `rlm.acall(...)` is caller-owned and may be reused only for sequential calls to the same RLM instance. `PythonInterpreter` must also stay on the thread where it was first used.

Interpreter Requirements

The default `PythonInterpreter` requires [Deno](https://deno.land/) to be installed for the Pyodide WASM sandbox.

## API Reference

### `dspy.RLM(signature: type[Signature] | str, max_iters: int = 20, max_llm_calls: int = 50, max_output_chars: int = 10000, verbose: bool = False, tools: list[Callable] | None = None, sub_lm: dspy.LM | None = None, interpreter_factory: Callable[[], CodeInterpreter] = PythonInterpreter)`

Bases: `Module`

Recursive Language Model module.

Uses a sandboxed REPL to let the LLM programmatically explore large contexts through code execution. The LLM writes Python code to examine data, call sub-LLMs for semantic analysis, and build up answers iteratively.

The default interpreter is PythonInterpreter (Deno/Pyodide/WASM), but `interpreter_factory` can create another CodeInterpreter implementation, such as an adapter for a remote sandbox. RLM updates the interpreter’s mutable `tools` dictionary with invocation-scoped tools before execution. A caller-owned interpreter may be reused sequentially with the same RLM instance, but must not be shared by overlapping invocations.

Examples:

```
# Basic usage
rlm = dspy.RLM("context, query -> output", max_iters=10)
result = rlm(context="...very long text...", query="What is the magic number?")
print(result.output)
```

Parameters:

| Name                  | Type                            | Description                                                                                                                                                                                                                                            | Default                                                                                                                                    |
| --------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `signature`           | \`type[Signature]               | str\`                                                                                                                                                                                                                                                  | Defines inputs and outputs. String like “context, query -> answer” or a Signature class.                                                   |
| `max_iters`           | `int`                           | Maximum REPL interaction iterations.                                                                                                                                                                                                                   | `20`                                                                                                                                       |
| `max_llm_calls`       | `int`                           | Maximum sub-LLM calls (llm_query/llm_query_batched) per execution.                                                                                                                                                                                     | `50`                                                                                                                                       |
| `max_output_chars`    | `int`                           | Maximum characters to include from REPL output.                                                                                                                                                                                                        | `10000`                                                                                                                                    |
| `verbose`             | `bool`                          | Whether to log detailed execution info.                                                                                                                                                                                                                | `False`                                                                                                                                    |
| `tools`               | \`list[Callable]                | None\`                                                                                                                                                                                                                                                 | List of tool functions or dspy.Tool objects callable from interpreter code. Built-in tools: llm_query(prompt), llm_query_batched(prompts). |
| `sub_lm`              | \`LM                            | None\`                                                                                                                                                                                                                                                 | LM for llm_query/llm_query_batched. Defaults to dspy.settings.lm. Allows using a different (e.g., cheaper) model for sub-queries.          |
| `interpreter_factory` | `Callable[[], CodeInterpreter]` | Zero-argument callable that creates an interpreter for each forward pass. The callable may be invoked concurrently, and DSPy shuts down each interpreter it returns. RLM updates the returned interpreter’s mutable tools dictionary before execution. | `PythonInterpreter`                                                                                                                        |

Source code in `dspy/predict/rlm.py`

```
def __init__(
    self,
    signature: type[Signature] | str,
    max_iters: int = 20,
    max_llm_calls: int = 50,
    max_output_chars: int = 10_000,
    verbose: bool = False,
    tools: list[Callable] | None = None,
    sub_lm: dspy.LM | None = None,
    interpreter_factory: Callable[[], CodeInterpreter] = PythonInterpreter,
):
    """
    Args:
        signature: Defines inputs and outputs. String like "context, query -> answer"
                  or a Signature class.
        max_iters: Maximum REPL interaction iterations.
        max_llm_calls: Maximum sub-LLM calls (llm_query/llm_query_batched) per execution.
        max_output_chars: Maximum characters to include from REPL output.
        verbose: Whether to log detailed execution info.
        tools: List of tool functions or dspy.Tool objects callable from interpreter code.
              Built-in tools: llm_query(prompt), llm_query_batched(prompts).
        sub_lm: LM for llm_query/llm_query_batched. Defaults to dspy.settings.lm.
               Allows using a different (e.g., cheaper) model for sub-queries.
        interpreter_factory: Zero-argument callable that creates an interpreter for each forward pass. The
            callable may be invoked concurrently, and DSPy shuts down each interpreter it returns. RLM updates
            the returned interpreter's mutable ``tools`` dictionary before execution.
    """
    super().__init__()
    _validate_interpreter_factory(interpreter_factory)
    self.signature = ensure_signature(signature)
    self.max_iters = max_iters
    self.max_llm_calls = max_llm_calls
    self.max_output_chars = max_output_chars
    self.verbose = verbose
    self.sub_lm = sub_lm
    self._interpreter_factory = interpreter_factory
    self._user_tools = self._normalize_tools(tools)
    self._validate_namespace(self._user_tools)

    # Build the action and extract signatures
    action_sig, extract_sig = self._build_signatures()
    self.generate_action = dspy.Predict(action_sig)
    self.extract = dspy.Predict(extract_sig)
```

#### Attributes

##### `tools: dict[str, Tool]`

User-provided tools (excludes internal llm_query/llm_query_batched).

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

##### `forward(interpreter: CodeInterpreter | None = None, /, **input_args) -> Prediction`

Execute RLM to produce outputs from the given inputs.

Parameters:

| Name           | Type              | Description                                         | Default                                                                                                                                                                                                        |
| -------------- | ----------------- | --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `interpreter`  | \`CodeInterpreter | None\`                                              | Optional caller-owned interpreter, passed positionally. RLM injects invocation tools and output metadata into it but does not shut it down. Reuse is supported only for sequential calls to this RLM instance. |
| `**input_args` |                   | Input values matching the signature’s input fields. | `{}`                                                                                                                                                                                                           |

Returns:

| Type         | Description                                                                       |
| ------------ | --------------------------------------------------------------------------------- |
| `Prediction` | Prediction with output field(s) from the signature and ‘trajectory’ for debugging |

Raises:

| Type                   | Description                                      |
| ---------------------- | ------------------------------------------------ |
| `ValueError`           | If required input fields are missing             |
| `CodeInterpreterError` | If interpreter setup, process, or protocol fails |

Source code in `dspy/predict/rlm.py`

```
def forward(self, interpreter: CodeInterpreter | None = None, /, **input_args) -> Prediction:
    """Execute RLM to produce outputs from the given inputs.

    Args:
        interpreter: Optional caller-owned interpreter, passed positionally. RLM injects invocation tools and
            output metadata into it but does not shut it down. Reuse is supported only for sequential calls to
            this RLM instance.
        **input_args: Input values matching the signature's input fields.

    Returns:
        Prediction with output field(s) from the signature and 'trajectory' for debugging

    Raises:
        ValueError: If required input fields are missing
        CodeInterpreterError: If interpreter setup, process, or protocol fails
    """
    self._validate_inputs(input_args)

    output_field_names = list(self.signature.output_fields.keys())
    execution_tools = self._prepare_execution_tools()
    variables = self._build_variables(**input_args)

    with self._interpreter_context(execution_tools, interpreter) as repl:
        regular_args = self._prepare_serializable_vars(input_args, repl)
        history: REPLHistory = REPLHistory(max_output_chars=self.max_output_chars)

        for iteration in range(self.max_iters):
            result: Prediction | REPLHistory = self._execute_iteration(
                repl, variables, history, iteration, regular_args, output_field_names
            )
            if isinstance(result, Prediction):
                return result
            history = result

        # Max iterations reached - use extract fallback
        return self._extract_fallback(variables, history, output_field_names)
```

##### `aforward(interpreter: CodeInterpreter | None = None, /, **input_args) -> Prediction`

Async version of forward(). Execute RLM to produce outputs.

Parameters:

| Name           | Type              | Description                                         | Default                                                                                                                                                                                                        |
| -------------- | ----------------- | --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `interpreter`  | \`CodeInterpreter | None\`                                              | Optional caller-owned interpreter, passed positionally. RLM injects invocation tools and output metadata into it but does not shut it down. Reuse is supported only for sequential calls to this RLM instance. |
| `**input_args` |                   | Input values matching the signature’s input fields. | `{}`                                                                                                                                                                                                           |

Returns:

| Type         | Description                                                                       |
| ------------ | --------------------------------------------------------------------------------- |
| `Prediction` | Prediction with output field(s) from the signature and ‘trajectory’ for debugging |

Raises:

| Type                   | Description                                      |
| ---------------------- | ------------------------------------------------ |
| `ValueError`           | If required input fields are missing             |
| `CodeInterpreterError` | If interpreter setup, process, or protocol fails |

Source code in `dspy/predict/rlm.py`

```
async def aforward(self, interpreter: CodeInterpreter | None = None, /, **input_args) -> Prediction:
    """Async version of forward(). Execute RLM to produce outputs.

    Args:
        interpreter: Optional caller-owned interpreter, passed positionally. RLM injects invocation tools and
            output metadata into it but does not shut it down. Reuse is supported only for sequential calls to
            this RLM instance.
        **input_args: Input values matching the signature's input fields.

    Returns:
        Prediction with output field(s) from the signature and 'trajectory' for debugging

    Raises:
        ValueError: If required input fields are missing
        CodeInterpreterError: If interpreter setup, process, or protocol fails
    """
    self._validate_inputs(input_args)

    output_field_names = list(self.signature.output_fields.keys())
    execution_tools = self._prepare_execution_tools()
    variables = self._build_variables(**input_args)

    with self._interpreter_context(execution_tools, interpreter) as repl:
        regular_args = self._prepare_serializable_vars(input_args, repl)
        history = REPLHistory(max_output_chars=self.max_output_chars)

        for iteration in range(self.max_iters):
            result = await self._aexecute_iteration(
                repl, variables, history, iteration, regular_args, output_field_names
            )
            if isinstance(result, Prediction):
                return result
            history = result

        # Max iterations reached - use extract fallback
        return await self._aextract_fallback(variables, history, output_field_names)
```

##### `batch(examples: list[Example], num_threads: int | None = None, max_errors: int | None = None, return_failed_examples: bool = False, provide_traceback: bool | None = None, disable_progress_bar: bool = False, timeout: int = 120, straggler_limit: int = 3) -> list[Example] | tuple[list[Example], list[Example], list[Exception]]`

Processes a list of dspy.Example instances in parallel using the Parallel module.

Parameters:

| Name                     | Type            | Description                                                          | Default                                                                                                      |
| ------------------------ | --------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `examples`               | `list[Example]` | List of dspy.Example instances to process.                           | *required*                                                                                                   |
| `num_threads`            | \`int           | None\`                                                               | Number of threads to use for parallel processing.                                                            |
| `max_errors`             | \`int           | None\`                                                               | Maximum number of errors allowed before stopping execution. If None, inherits from dspy.settings.max_errors. |
| `return_failed_examples` | `bool`          | Whether to return failed examples and exceptions.                    | `False`                                                                                                      |
| `provide_traceback`      | \`bool          | None\`                                                               | Whether to include traceback information in error logs.                                                      |
| `disable_progress_bar`   | `bool`          | Whether to display the progress bar.                                 | `False`                                                                                                      |
| `timeout`                | `int`           | Seconds before a straggler task is resubmitted. Set to 0 to disable. | `120`                                                                                                        |
| `straggler_limit`        | `int`           | Only check for stragglers when this many or fewer tasks remain.      | `3`                                                                                                          |

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
    timeout: int = 120,
    straggler_limit: int = 3,
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
        timeout: Seconds before a straggler task is resubmitted. Set to 0 to disable.
        straggler_limit: Only check for stragglers when this many or fewer tasks remain.

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
        timeout=timeout,
        straggler_limit=straggler_limit,
    )

    # Execute the forward method of Parallel
    if return_failed_examples:
        results, failed_examples, exceptions = parallel_executor.forward(exec_pairs)
        return results, failed_examples, exceptions
    else:
        results = parallel_executor.forward(exec_pairs)
        return results
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

##### `dump_state(json_mode=True)`

Source code in `dspy/primitives/base_module.py`

```
def dump_state(self, json_mode=True):
    return {name: param.dump_state(json_mode=json_mode) for name, param in self.named_parameters()}
```

##### `get_lm()`

Get the language model used by this module’s predictors.

Returns the language model if every module leaf uses the same LM. Raises an error if multiple different LMs are in use.

Returns:

| Type | Description                                                   |
| ---- | ------------------------------------------------------------- |
|      | The language model instance used by this module’s predictors. |

Raises:

| Type         | Description                                                                            |
| ------------ | -------------------------------------------------------------------------------------- |
| `ValueError` | If multiple different language models are being used by the predictors in this module. |

Source code in `dspy/primitives/module.py`

```
def get_lm(self):
    """Get the language model used by this module's predictors.

    Returns the language model if every module leaf uses the same LM. Raises an error if multiple
    different LMs are in use.

    Returns:
        The language model instance used by this module's predictors.

    Raises:
        ValueError: If multiple different language models are being
            used by the predictors in this module.
    """
    all_used_lms = [param.lm for _, param in self.named_parameters() if isinstance(param, Module)]

    if len(set(all_used_lms)) == 1:
        return all_used_lms[0]

    raise ValueError("Multiple LMs are being used in the module. There's no unique LM to return.")
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

##### `load_state(state, *, allow_unsafe_lm_state=False)`

Source code in `dspy/primitives/base_module.py`

```
def load_state(self, state, *, allow_unsafe_lm_state=False):
    from dspy import Module

    def _apply(module):
        for name, param in module.named_parameters():
            if isinstance(param, Module):
                param.load_state(state[name], allow_unsafe_lm_state=allow_unsafe_lm_state)
            else:
                param.load_state(state[name])

    _apply(self.deepcopy())  # trial run raises before self is touched
    _apply(self)
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

Return all named Predict modules in this module.

Iterates through all parameters and returns those that are instances of `dspy.Predict`, along with their names.

Returns:

| Type | Description                                                                                                                             |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------- |
|      | list\[tuple[str, Predict]\]: A list of (name, predictor) tuples where name is the attribute path and predictor is the Predict instance. |

Examples:

```
>>> import dspy
>>> class MyProgram(dspy.Module):
...     def __init__(self):
...         super().__init__()
...         self.qa = dspy.Predict("question -> answer")
...         self.summarize = dspy.Predict("text -> summary")
...
>>> program = MyProgram()
>>> for name, p in program.named_predictors():
...     print(name)
qa
summarize
```

Source code in `dspy/primitives/module.py`

```
def named_predictors(self):
    """Return all named Predict modules in this module.

    Iterates through all parameters and returns those that are instances
    of ``dspy.Predict``, along with their names.

    Returns:
        list[tuple[str, Predict]]: A list of (name, predictor) tuples
            where name is the attribute path and predictor is the
            Predict instance.

    Examples:
        >>> import dspy
        >>> class MyProgram(dspy.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.qa = dspy.Predict("question -> answer")
        ...         self.summarize = dspy.Predict("text -> summary")
        ...
        >>> program = MyProgram()
        >>> for name, p in program.named_predictors():
        ...     print(name)
        qa
        summarize
    """
    from dspy.predict.predict import Predict

    return [(name, param) for name, param in self.named_parameters() if isinstance(param, Predict)]
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

##### `set_lm(lm)`

Set the language model for all predictors in this module.

This method recursively sets the language model on every leaf module in this module.

Parameters:

| Name | Type | Description                                            | Default    |
| ---- | ---- | ------------------------------------------------------ | ---------- |
| `lm` |      | The language model instance to use for all predictors. | *required* |

Examples:

```
>>> import dspy
>>> lm = dspy.LM("openai/gpt-4o-mini")
>>> program = dspy.Predict("question -> answer")
>>> program.set_lm(lm)
```

Source code in `dspy/primitives/module.py`

```
def set_lm(self, lm):
    """Set the language model for all predictors in this module.

    This method recursively sets the language model on every leaf module
    in this module.

    Args:
        lm: The language model instance to use for all predictors.

    Examples:
        >>> import dspy
        >>> lm = dspy.LM("openai/gpt-4o-mini")
        >>> program = dspy.Predict("question -> answer")
        >>> program.set_lm(lm)
    """
    for _, param in self.named_parameters():
        if isinstance(param, Module):
            param.lm = lm
```
