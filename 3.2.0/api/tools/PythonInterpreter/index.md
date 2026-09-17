# dspy.PythonInterpreter

## `dspy.PythonInterpreter(deno_command: list[str] | None = None, enable_read_paths: list[PathLike | str] | None = None, enable_write_paths: list[PathLike | str] | None = None, enable_env_vars: list[str] | None = None, enable_network_access: list[str] | None = None, sync_files: bool = True, tools: dict[str, Callable[..., str]] | None = None, output_fields: list[dict] | None = None)`

Local interpreter for secure Python execution using Deno and Pyodide.

Implements the Interpreter protocol for secure code execution in a WASM-based sandbox. Code runs in an isolated Pyodide environment with no access to the host filesystem, network, or environment by default.

Prerequisites

Deno must be installed: https://docs.deno.com/runtime/getting_started/installation/

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
| `output_fields`         | \`list[dict]                      | None\`                                                                           | List of output field definitions for typed SUBMIT signature. Each dict should have 'name' and optionally 'type' keys.                                                          |

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
    """
    if isinstance(deno_command, dict):
        raise TypeError("deno_command must be a list of strings, not a dict")

    self.enable_read_paths = enable_read_paths or []
    self.enable_write_paths = enable_write_paths or []
    self.enable_env_vars = enable_env_vars or []
    self.enable_network_access = enable_network_access or []
    self.sync_files = sync_files
    self.tools = dict(tools) if tools else {}
    self.output_fields = output_fields
    self._tools_registered = False
    # TODO later on add enable_run (--allow-run) by proxying subprocess.run through Deno.run() to fix 'emscripten does not support processes' error

    if deno_command:
        self.deno_command = list(deno_command)
    else:
        args = ["deno", "run"]

        # Allow reading runner.js and explicitly enabled paths
        allowed_read_paths = [self._get_runner_path()]

        # Also allow reading Deno's cache directory so Pyodide can load its files
        deno_dir = self._get_deno_dir()
        if deno_dir:
            allowed_read_paths.append(deno_dir)

        if self.enable_read_paths:
            allowed_read_paths.extend(str(p) for p in self.enable_read_paths)
        if self.enable_write_paths:
            allowed_read_paths.extend(str(p) for p in self.enable_write_paths)
        args.append(f"--allow-read={','.join(allowed_read_paths)}")

        self._env_arg = ""
        if self.enable_env_vars:
            user_vars = [str(v).strip() for v in self.enable_env_vars]
            args.append("--allow-env=" + ",".join(user_vars))
            self._env_arg = ",".join(user_vars)
        if self.enable_network_access:
            args.append(f"--allow-net={','.join(str(x) for x in self.enable_network_access)}")
        if self.enable_write_paths:
            args.append(f"--allow-write={','.join(str(x) for x in self.enable_write_paths)}")

        args.append(self._get_runner_path())

        # For runner.js to load in env vars
        if self._env_arg:
            args.append(self._env_arg)
        self.deno_command = args

    self.deno_process = None
    self._mounted_files = False
    self._request_id = 0
    self._owner_thread: int | None = None
    self._pending_large_vars = {}
```

### Functions

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
def execute(
    self,
    code: str,
    variables: dict[str, Any] | None = None,
) -> Any:
    self._check_thread_ownership()
    variables = variables or {}
    code = self._inject_variables(code, variables)
    self._ensure_deno_process()
    self._mount_files()
    self._register_tools()

    for name, value in self._pending_large_vars.items():
        self._inject_large_var(name, value)

    # Send the code as JSON-RPC request
    self._request_id += 1
    execute_request_id = self._request_id
    input_data = _jsonrpc_request("execute", {"code": code}, execute_request_id)
    try:
        self.deno_process.stdin.write(input_data + "\n")
        self.deno_process.stdin.flush()
    except BrokenPipeError:
        # If the process died, restart and try again once
        self._ensure_deno_process()
        self._mount_files()
        self._register_tools()
        for name, value in self._pending_large_vars.items():
            self._inject_large_var(name, value)
        self.deno_process.stdin.write(input_data + "\n")
        self.deno_process.stdin.flush()

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

        # Handle success response
        if "result" in msg:
            if msg.get("id") != execute_request_id:
                raise CodeInterpreterError(f"Response ID mismatch: expected {execute_request_id}, got {msg.get('id')}")
            result = msg["result"]
            self._sync_files()
            # Check for SUBMIT (encoded as success with "final" field)
            if "final" in result:
                return FinalOutput(result["final"])
            return result.get("output", None)

        # Handle error response
        if "error" in msg:
            # Errors with id=null are unsolicited errors (e.g., unhandled async rejections)
            # Treat them as errors for the current request
            if msg.get("id") is not None and msg.get("id") != execute_request_id:
                raise CodeInterpreterError(f"Response ID mismatch: expected {execute_request_id}, got {msg.get('id')}")
            error = msg["error"]
            error_code = error.get("code", JSONRPC_APP_ERRORS["Unknown"])
            error_message = error.get("message", "Unknown error")
            error_data = error.get("data", {})
            error_type = error_data.get("type", "Error")

            if error_code == JSONRPC_APP_ERRORS["SyntaxError"]:
                raise SyntaxError(f"Invalid Python syntax. message: {error_message}")
            else:
                raise CodeInterpreterError(f"{error_type}: {error_data.get('args') or error_message}")

        # Unexpected message format - neither a recognized method nor a response
        raise CodeInterpreterError(f"Unexpected message format from sandbox: {msg}")

    raise CodeInterpreterError(f"Too many non-JSON lines ({skipped}) during execution")
```

#### `shutdown() -> None`

Source code in `dspy/primitives/python_interpreter.py`

```
def shutdown(self) -> None:
    if self.deno_process and self.deno_process.poll() is None:
        self.deno_process.stdin.write(_jsonrpc_notification("shutdown") + "\n")
        self.deno_process.stdin.flush()
        self.deno_process.stdin.close()
        self.deno_process.wait()
    self.deno_process = None
    self._owner_thread = None
```

#### `start() -> None`

Initialize the Deno/Pyodide sandbox.

This pre-warms the sandbox by starting the Deno subprocess. Can be called explicitly for pooling, or will be called lazily on first execute().

Idempotent: safe to call multiple times.

Source code in `dspy/primitives/python_interpreter.py`

```
def start(self) -> None:
    """Initialize the Deno/Pyodide sandbox.

    This pre-warms the sandbox by starting the Deno subprocess.
    Can be called explicitly for pooling, or will be called lazily
    on first execute().

    Idempotent: safe to call multiple times.
    """
    self._ensure_deno_process()
```
