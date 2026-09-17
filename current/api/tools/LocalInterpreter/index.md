# dspy.LocalInterpreter

`LocalInterpreter` runs Python in one persistent local CPython worker. It is useful when generated code needs ordinary Python compatibility plus separation from the DSPy process’s memory, stdout, and lifecycle. State and imports persist until `shutdown()`, host tools cross a JSON protocol, and `execution_timeout` can terminate a stuck worker.

```
import dspy

rlm = dspy.RLM(
    "question: str -> answer: int",
    interpreter_factory=dspy.LocalInterpreter,
)
```

A subprocess is not a security sandbox

Generated code retains the host user’s filesystem, environment, credentials, subprocess, and network authority. Use the default [`PythonInterpreter`](https://dspy.ai/current/api/tools/PythonInterpreter/index.md) or a remote sandbox for untrusted code. The subprocess boundary protects ordinary host memory and stdout from accidental mutation; it does not contain hostile code.

Inputs, host-tool arguments/results, and structured outputs must be JSON-compatible. The worker uses the current Python executable. `execution_timeout` includes time spent in host tools and terminates the worker promptly when the deadline expires. Python cannot forcibly stop a running host callable, so that callable may finish in a detached daemon thread; its result is discarded and the interpreter session remains terminal.

Parallel guest code is supported when all threads finish before the current `execute()` call returns, such as a context-managed `ThreadPoolExecutor`. Leaving a guest thread running makes the session terminal because that thread could otherwise mutate state, write output, or invoke tools during a later execution. Shutdown and terminal failures also terminate ordinary descendant processes with the worker.

`LocalInterpreter.execution_instructions` is stable class metadata. `RLM` adds it to the action prompt without starting a worker.

## `dspy.LocalInterpreter(tools: dict[str, Callable[..., Any]] | None = None, output_fields: list[dict[str, Any]] | None = None, *, execution_timeout: float | None = None, callbacks: list[BaseCallback] | None = None)`

Execute Python in a persistent local subprocess.

This separates generated code from DSPy’s memory, stdout, and lifecycle, but it is not a security sandbox. The worker retains the host user’s files, environment, credentials, subprocess, and network authority.

Parameters:

| Name                | Type                              | Description | Default                                                                                                                                                                                                                                                            |
| ------------------- | --------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `tools`             | \`dict\[str, Callable[..., Any]\] | None\`      | Host functions exposed to executed code by name. Arguments and return values must be JSON-compatible.                                                                                                                                                              |
| `output_fields`     | \`list\[dict[str, Any]\]          | None\`      | Optional field definitions for typed SUBMIT calls.                                                                                                                                                                                                                 |
| `execution_timeout` | \`float                           | None\`      | Maximum seconds for one execution, including host tool calls. A timeout terminates the worker and its session state. Python cannot forcibly stop a running host callable, so a timed-out callable may finish in a detached daemon thread; its result is discarded. |
| `callbacks`         | \`list[BaseCallback]              | None\`      | Optional instance-level callback handlers.                                                                                                                                                                                                                         |

Source code in `dspy/primitives/local_interpreter.py`

```
def __init__(
    self,
    tools: dict[str, Callable[..., Any]] | None = None,
    output_fields: list[dict[str, Any]] | None = None,
    *,
    execution_timeout: float | None = None,
    callbacks: list[BaseCallback] | None = None,
) -> None:
    if execution_timeout is not None and execution_timeout <= 0:
        raise ValueError("execution_timeout must be positive or None")
    self._validate_tool_names(tools or {})
    self.tools = dict(tools or {})
    fields = None if output_fields is None else [dict(field) for field in output_fields]
    self._validate_output_fields(fields)
    self.output_fields = fields
    self.execution_timeout = execution_timeout
    self.callbacks = list(callbacks or [])
    self._process: subprocess.Popen[str] | None = None
    self._responses: queue.Queue[str | BaseException | None] = queue.Queue()
    self._lock = threading.Lock()
    self._send_lock = threading.Lock()
    self._ended = False
```

### Methods:

#### `__call__(code: str, variables: dict[str, Any] | None = None) -> Any`

Source code in `dspy/primitives/local_interpreter.py`

```
def __call__(self, code: str, variables: dict[str, Any] | None = None) -> Any:
    return self.execute(code, variables)
```

#### `execute(code: str, variables: dict[str, Any] | None = None) -> Any`

Execute code in the worker’s persistent namespace.

Source code in `dspy/primitives/local_interpreter.py`

```
@with_callbacks
def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
    """Execute code in the worker's persistent namespace."""
    if not self._lock.acquire(blocking=False):
        raise CodeInterpreterError("LocalInterpreter already has an active execution.")
    try:
        variables = {} if variables is None else variables
        self._validate_tool_names(self.tools)
        self._validate_output_fields(self.output_fields)
        if not isinstance(code, str):
            raise CodeInterpreterError("code must be a string")
        if not isinstance(variables, dict):
            raise CodeInterpreterError("variables must map Python identifiers to JSON-compatible values")
        invalid_variables = [
            name
            for name in variables
            if not isinstance(name, str)
            or not name.isidentifier()
            or keyword.iskeyword(name)
            or name in {"SUBMIT", "__builtins__"}
            or name in self.tools
        ]
        if invalid_variables:
            raise CodeInterpreterError(
                "variable names must be non-reserved Python identifiers distinct from tool names; "
                f"invalid names: {invalid_variables!r}"
            )
        try:
            json.dumps(variables, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise CodeInterpreterError(f"variables must be JSON-compatible: {exc}") from exc

        if self._process is None:
            self.start()
        deadline = None if self.execution_timeout is None else time.monotonic() + self.execution_timeout
        self._send(
            {
                "type": "execute",
                "code": code,
                "variables": variables,
                "tools": list(self.tools),
                "output_fields": self.output_fields,
            }
        )
        timeout_message = (
            "Python worker did not respond."
            if self.execution_timeout is None
            else f"Python worker exceeded execution timeout of {self.execution_timeout:g} seconds."
        )
        process = self._process
        assert process is not None
        tool_calls: list[Future[Any]] = []
        message = self._receive(deadline, timeout_message)
        while message.get("type") == "tool_request":
            tool_calls.append(_run_in_thread(lambda request=message: self._handle_tool(request, process)))
            message = self._receive(deadline, timeout_message)
        for tool_call in tool_calls:
            timeout = None if deadline is None else max(0, deadline - time.monotonic())
            try:
                tool_call.result(timeout)
            except FutureTimeoutError:
                self._raise_terminal_error(timeout_message)
        kind = message.get("type")
        if kind == "terminal_error":
            self._raise_terminal_error(f"Python worker failed: {message.get('error')}")
        if kind == "syntax":
            raise SyntaxError(message.get("error"))
        if kind == "execution_error":
            raise CodeExecutionError(message.get("error"))
        if kind == "final":
            return FinalOutput(message.get("value"))
        if kind == "result":
            value, stdout = message.get("value"), message.get("stdout")
            return value if value is not None else (stdout or None)
        self._raise_terminal_error(f"Python worker returned an unknown message: {message!r}")
    finally:
        self._lock.release()
```

#### `shutdown() -> None`

Terminate the worker and discard its session state.

Source code in `dspy/primitives/local_interpreter.py`

```
@with_callbacks
def shutdown(self) -> None:
    """Terminate the worker and discard its session state."""
    if self._ended:
        return
    self._ended = True
    self._kill()
```

#### `start() -> None`

Start the worker, or return immediately if it is already running.

Source code in `dspy/primitives/local_interpreter.py`

```
@with_callbacks
def start(self) -> None:
    """Start the worker, or return immediately if it is already running."""
    if self._ended:
        raise CodeInterpreterError("LocalInterpreter session has been shut down.")
    if self._process is not None:
        return
    try:
        process = subprocess.Popen(
            [sys.executable, "-I", str(Path(__file__).with_name("local_interpreter_worker.py"))],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            start_new_session=os.name == "posix",
            text=True,
        )
    except OSError as exc:
        raise CodeInterpreterError(f"Unable to start Python worker: {exc}") from exc
    self._process = process
    threading.Thread(target=self._read_responses, args=(process,), daemon=True).start()
    message = self._receive(time.monotonic() + 10, "Python worker did not start within 10 seconds.")
    if message.get("type") != "ready":
        self._raise_terminal_error(f"Python worker returned an invalid startup message: {message!r}")
```
