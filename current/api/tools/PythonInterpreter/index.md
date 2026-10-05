# dspy.PythonInterpreter

## Deno Installation

`PythonInterpreter` uses Deno and Pyodide to run Python in a local WASM sandbox. The recommended installation keeps Deno in the same Python environment as DSPy:

```
pip install "dspy[deno]"
```

The `deno` extra installs the official Deno Python distribution (`>=2.4.5,<3.0.0`). DSPy prefers that managed binary when it is installed, so Python dependency locking also locks the Deno runtime. The extra provides binaries for macOS x86-64/arm64, glibc Linux x86-64/arm64, and Windows x86-64 and adds approximately 40–50 MiB to the environment.

On other platforms, install a compatible Deno 2.x release (`>=2.0.0,<3.0.0`) using the [Deno installation instructions](https://docs.deno.com/runtime/getting_started/installation/). DSPy falls back to the `deno` executable on `PATH`. An explicit `deno_command` passed to `PythonInterpreter` takes precedence over both options.

DSPy disables ambient `deno.json`, lockfile, `package.json`, and local `node_modules` discovery for its default runner. A `package.json` in the current directory or an ancestor therefore cannot redirect the sandbox’s pinned Pyodide dependency. No `DENO_NO_PACKAGE_JSON` environment variable is required.

## Execution Instructions

`PythonInterpreter.execution_instructions` describes stable constraints of its Pyodide execution environment. It is class metadata, so code-generating modules can inspect it without starting Deno or allocating an interpreter. `RLM` includes these instructions in its action prompt, which adapters render in the model’s system prompt.

Custom interpreter factories may expose their own `execution_instructions` string. This metadata is optional; a factory without it remains valid and uses RLM’s generic action prompt.

## `dspy.PythonInterpreter(deno_command: list[str] | None = None, enable_read_paths: list[PathLike | str] | None = None, enable_write_paths: list[PathLike | str] | None = None, enable_env_vars: list[str] | None = None, enable_network_access: list[str] | None = None, sync_files: bool = True, tools: dict[str, Callable[..., str]] | None = None, output_fields: list[dict] | None = None, callbacks: list[BaseCallback] | None = None)`

Local interpreter for secure Python execution using Deno and Pyodide.

Implements the Interpreter protocol for secure code execution in a WASM-based sandbox. Code runs in an isolated Pyodide environment with no access to the host filesystem, network, or environment by default.

Prerequisites

Install the managed Deno runtime with `pip install "dspy[deno]"`, or install a compatible Deno 2.x release system-wide.

Examples:

```
# Basic execution
with PythonInterpreter() as interp:
    result = interp("print(1 + 2)")  # Returns "3"

# With host-side tools
def my_tool(question: str) -> str:
    return "answer"

with PythonInterpreter(tools={"my_tool": my_tool}) as interp:
    result = interp("print(my_tool(question='test'))")
```

Parameters:

| Name                    | Type                              | Description                                                                      | Default                                                                                                                                                                        |
| ----------------------- | --------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `deno_command`          | \`list[str]                       | None\`                                                                           | command list to launch Deno.                                                                                                                                                   |
| `enable_read_paths`     | \`list\[PathLike                  | str\]                                                                            | None\`                                                                                                                                                                         |
| `enable_write_paths`    | \`list\[PathLike                  | str\]                                                                            | None\`                                                                                                                                                                         |
| `enable_env_vars`       | \`list[str]                       | None\`                                                                           | Environment variable names to allow in the sandbox.                                                                                                                            |
| `enable_network_access` | \`list[str]                       | None\`                                                                           | Domains or IPs to allow network access in the sandbox.                                                                                                                         |
| `sync_files`            | `bool`                            | If set, syncs changes within the sandbox back to original files after execution. | `True`                                                                                                                                                                         |
| `tools`                 | \`dict\[str, Callable[..., str]\] | None\`                                                                           | Dictionary mapping tool names to callable functions. Each function should accept keyword arguments and return a string. Tools are callable directly from sandbox code by name. |
| `output_fields`         | \`list[dict]                      | None\`                                                                           | List of output field definitions for typed SUBMIT signature. Each dict should have ‘name’ and optionally ‘type’ keys.                                                          |
| `callbacks`             | \`list[BaseCallback]              | None\`                                                                           | Optional instance-level callback handlers.                                                                                                                                     |

Source code in `dspy/primitives/python_interpreter.py`

```
def __init__(
    self,
    deno_command: list[str] | None = None,
    enable_read_paths: list[PathLike | str] | None = None,
    enable_write_paths: list[PathLike | str] | None = None,
    enable_env_vars: list[str] | None = None,
    enable_network_access: list[str] | None = None,
    sync_files: bool = True,
    tools: dict[str, Callable[..., str]] | None = None,
    output_fields: list[dict] | None = None,
    callbacks: list[BaseCallback] | None = None,
) -> None:
    """
    Args:
        deno_command: command list to launch Deno.
        enable_read_paths: Files or directories to allow reading from in the sandbox.
        enable_write_paths: Files or directories to allow writing to in the sandbox.
            All write paths will also be able to be read from for mounting.
        enable_env_vars: Environment variable names to allow in the sandbox.
        enable_network_access: Domains or IPs to allow network access in the sandbox.
        sync_files: If set, syncs changes within the sandbox back to original files after execution.
        tools: Dictionary mapping tool names to callable functions.
               Each function should accept keyword arguments and return a string.
               Tools are callable directly from sandbox code by name.
        output_fields: List of output field definitions for typed SUBMIT signature.
               Each dict should have 'name' and optionally 'type' keys.
        callbacks: Optional instance-level callback handlers.
    """
    if isinstance(deno_command, dict):
        raise TypeError("deno_command must be a list of strings, not a dict")

    self.enable_read_paths = enable_read_paths or []
    self.enable_write_paths = enable_write_paths or []
    mounts = {}
    for path in [*self.enable_read_paths, *self.enable_write_paths]:
        if path:
            virtual_path = os.path.basename(os.fspath(path))
            host_path = _canonicalize_path(path)
            if virtual_path in mounts and mounts[virtual_path] != host_path:
                raise CodeInterpreterError("Mounted files must have unique basenames inside the sandbox.")
            mounts[virtual_path] = host_path
    self.enable_env_vars = enable_env_vars or []
    self.enable_network_access = enable_network_access or []
    self.sync_files = sync_files
    self.tools = dict(tools) if tools else {}
    self.output_fields = output_fields
    self.callbacks = list(callbacks or [])
    self._tools_registered = False
    # TODO later on add enable_run (--allow-run) by proxying subprocess.run through Deno.run() to fix 'emscripten does not support processes' error

    self._uses_default_deno_command = not deno_command
    if deno_command:
        self.deno_command = list(deno_command)
    else:
        deno_executable = _find_deno_executable()
        args = [
            deno_executable,
            "run",
            "--no-config",
            "--no-lock",
            "--node-modules-dir=false",
        ]

        # Also allow reading Deno's cache directory so Pyodide can load its files
        deno_dir = self._get_deno_dir(deno_executable)
        protected = [_canonicalize_path(self._get_runner_path()), *([_canonicalize_path(deno_dir)] if deno_dir else [])]
        if any(_paths_overlap(_canonicalize_path(path), item) for path in self.enable_write_paths for item in protected):
            raise CodeInterpreterError("Write paths cannot overlap PythonInterpreter runtime files.")
        raw_read_paths = [
            self._get_runner_path(),
            *([deno_dir] if deno_dir else []),
            *self.enable_read_paths,
            *self.enable_write_paths,
        ]
        allowed_read_paths = [_canonicalize_path(p) for p in raw_read_paths]
        args.append(f"--allow-read={','.join(allowed_read_paths)}")

        self._env_arg = ""
        if self.enable_env_vars:
            user_vars = [str(v).strip() for v in self.enable_env_vars]
            args.append("--allow-env=" + ",".join(user_vars))
            self._env_arg = ",".join(user_vars)
        if self.enable_network_access:
            args.append(f"--allow-net={','.join(str(x) for x in self.enable_network_access)}")
        if self.enable_write_paths:
            args.append(f"--allow-write={','.join(_canonicalize_path(x) for x in self.enable_write_paths)}")

        args.append(_canonicalize_path(self._get_runner_path()))

        # For runner.js to load in env vars and revoke cache access after startup
        args.append(self._env_arg)
        if deno_dir:
            args.append(f"--dspy-deno-dir={_canonicalize_path(deno_dir)}")
        self.deno_command = args

    self.deno_process = None
    self._mounted_files = False
    self._last_diagnostic: str | None = None
    self._owner_thread: int | None = None
    self._handling_tool_call = False
    self._pending_large_vars = {}
    self._session_ended = False
```

### Methods:

#### `__call__(code: str, variables: dict[str, Any] | None = None) -> Any`

Source code in `dspy/primitives/python_interpreter.py`

```
def __call__(
    self,
    code: str,
    variables: dict[str, Any] | None = None,
) -> Any:
    return self.execute(code, variables)
```

#### `execute(code: str, variables: dict[str, Any] | None = None) -> Any`

Source code in `dspy/primitives/python_interpreter.py`

```
@with_callbacks
def execute(
    self,
    code: str,
    variables: dict[str, Any] | None = None,
) -> Any:
    if self._handling_tool_call:
        raise CodeInterpreterError("PythonInterpreter cannot execute recursively from one of its tools.")
    self._check_session_active()
    self._check_thread_ownership()
    variables = variables or {}
    code = self._inject_variables(code, variables)
    self._ensure_deno_process()
    self._mount_files()
    self._register_tools()

    for name, value in self._pending_large_vars.items():
        self._inject_large_var(name, value)

    # Send the code as JSON-RPC request
    execute_request_id = self._next_request_id()
    input_data = _jsonrpc_request("execute", {"code": code}, execute_request_id)
    self._write_message(input_data, "during execution")

    # Read and handle messages until we get the final output.
    # Loop is needed because tool calls require back-and-forth communication.
    skipped = 0
    while skipped <= self._MAX_SKIP_LINES:
        output_line = self._read_response_line("during execution")
        msg = self._parse_response_line(output_line, "during execution")
        if msg is None:
            skipped += 1
            continue

        # Handle incoming requests (tool calls from sandbox)
        if "method" in msg:
            if msg["method"] == "tool_call":
                self._handle_tool_call(msg)
                continue

        if self._handle_out_of_band_message(msg, "during execution"):
            skipped += 1
            continue

        # Handle success response
        if "result" in msg:
            if msg.get("id") != execute_request_id:
                self._raise_terminal_error(
                    f"Response ID mismatch: expected {execute_request_id}, got {msg.get('id')}"
                )
            result = msg["result"]
            if not isinstance(result, dict):
                self._raise_terminal_error(f"Malformed execution result: {msg}")
            self._sync_files()
            # Check for SUBMIT (encoded as success with "final" field)
            if "final" in result:
                return FinalOutput(result["final"])
            return result.get("output", None)

        # Handle error response
        if "error" in msg:
            if msg.get("id") != execute_request_id:
                self._raise_terminal_error(
                    f"Response ID mismatch: expected {execute_request_id}, got {msg.get('id')}"
                )
            error = msg["error"]
            if not isinstance(error, dict):
                self._raise_terminal_error(f"Malformed execution error: {msg}")
            error_code = error.get("code", JSONRPC_APP_ERRORS["Unknown"])
            error_message = error.get("message", "Unknown error")
            error_data = error.get("data", {})
            if not isinstance(error_data, dict):
                self._raise_terminal_error(f"Malformed execution error data: {msg}")
            error_type = error_data.get("type", "Error")

            if error_code == JSONRPC_APP_ERRORS["SyntaxError"]:
                raise SyntaxError(f"Invalid Python syntax. message: {error_message}")
            if error_code in JSONRPC_APP_ERRORS.values():
                raise CodeExecutionError(f"{error_type}: {error_data.get('args') or error_message}")
            self._raise_terminal_error(f"{error_type}: {error_data.get('args') or error_message}")

        # Unexpected message format - neither a recognized method nor a response
        self._raise_terminal_error(f"Unexpected message format from sandbox: {msg}")

    self._raise_terminal_error(f"Too many skipped lines ({skipped}) during execution")
```

#### `shutdown() -> None`

Source code in `dspy/primitives/python_interpreter.py`

```
@with_callbacks
def shutdown(self) -> None:
    session_was_active = not self._session_ended
    self._session_ended = True
    if self.deno_process and self.deno_process.poll() is None:
        if session_was_active:
            self.deno_process.stdin.write(_jsonrpc_notification("shutdown") + "\n")
            self.deno_process.stdin.flush()
            self.deno_process.stdin.close()
        else:
            self.deno_process.terminate()
        self.deno_process.wait()
    self.deno_process = None
    self._owner_thread = None
```

#### `start() -> None`

Initialize the Deno/Pyodide sandbox.

This pre-warms the sandbox by starting the Deno subprocess. Can be called explicitly for pooling, or will be called lazily on first execute().

Idempotent while the session is active. A stopped or shut-down session cannot be restarted because its Python state cannot be reconstructed.

Source code in `dspy/primitives/python_interpreter.py`

```
@with_callbacks
def start(self) -> None:
    """Initialize the Deno/Pyodide sandbox.

    This pre-warms the sandbox by starting the Deno subprocess.
    Can be called explicitly for pooling, or will be called lazily
    on first execute().

    Idempotent while the session is active. A stopped or shut-down session
    cannot be restarted because its Python state cannot be reconstructed.
    """
    if self.deno_process is None:
        self._check_session_active()
        self._spawn_process()
    else:
        self._ensure_deno_process()
```
