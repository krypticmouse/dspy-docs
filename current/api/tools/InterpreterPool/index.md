# dspy.InterpreterPool

`InterpreterPool` is an `interpreter_factory` that keeps interpreters started ahead of need. Each call still returns a fresh interpreter that has run no code, but usually one that has already started, so a `dspy.Flex` forward or a `dspy.RLM` call doesn’t wait for a sandbox to boot. With the default `PythonInterpreter`, a pooled interpreter has also loaded dspy, which is the first thing a `Flex` forward imports.

```
import dspy

pool = dspy.InterpreterPool(size=2)
program = dspy.Flex("question -> answer", interpreter_factory=pool)
```

The pool starts `size` interpreters in the background and starts a replacement whenever it hands one out. When none is ready, a call makes a new interpreter on the spot, as the plain factory would.

Pick `size` by how calls arrive. One interpreter is enough when each call spends longer in LM calls than a sandbox takes to start (well under a second). Calls that run back to back with little LM time, or in parallel threads such as `dspy.Evaluate(num_threads=...)`, need one for each call in flight.

Idle interpreters hold memory

Every pooled interpreter is a running sandbox process. A `PythonInterpreter` with dspy loaded holds about 400 MB, so size the pool to your machine. Idle interpreters are shut down by `close()`, when the pool is used as a context manager, when the pool is garbage-collected, and at exit.

`factory` must return interchangeable interpreters, because the pool calls it before the interpreter is needed. A factory whose interpreters depend on when it is called, for example on the current input, should not be pooled.

## `dspy.InterpreterPool(factory: Callable[[], CodeInterpreter] = PythonInterpreter, size: int = 1)`

An `interpreter_factory` that keeps interpreters started ahead of need.

Each call returns a fresh interpreter from `factory` that has run no code, as `factory()` would, except that it is usually already started: the pool starts `size` interpreters in the background and starts a replacement whenever it hands one out, or after a start failed. When none is ready, a call returns a new, unstarted interpreter instead. A pooled `PythonInterpreter` has dspy installed, and usually already imported, which is what `dspy.Flex` imports first.

Pass it wherever an `interpreter_factory` goes:

```
pool = dspy.InterpreterPool(size=2)
program = dspy.Flex(MySignature, interpreter_factory=pool)
```

An idle interpreter is a running sandbox process: a `PythonInterpreter` with dspy loaded holds about 400 MB, so size the pool to your memory. One interpreter hides the start-up cost when each call spends longer in LM calls than a sandbox takes to start; calls that run back to back, or in parallel threads, need one per call in flight. `factory` must return interchangeable interpreters, since the pool calls it ahead of time. Idle interpreters are shut down by `close()`, when the pool is garbage-collected, and at exit.

Source code in `dspy/primitives/interpreter_pool.py`

```
def __init__(self, factory: Callable[[], CodeInterpreter] = PythonInterpreter, size: int = 1) -> None:
    if size < 1:
        raise ValueError("size must be at least 1")
    self._factory = factory
    self._size = size
    self._starting = 0
    self._ready: queue.Queue[CodeInterpreter] = queue.Queue()
    self._lock = threading.Lock()
    self._closed = False
    # Shuts the idle interpreters down when the pool is closed, garbage-collected, or the process exits.
    self._finalizer = weakref.finalize(self, _shut_down_idle, self._ready)
    self._refill()
```

### Methods:

#### `__call__() -> CodeInterpreter`

Source code in `dspy/primitives/interpreter_pool.py`

```
def __call__(self) -> CodeInterpreter:
    try:
        interpreter = self._ready.get_nowait()
    except queue.Empty:
        interpreter = None
    self._refill()
    return interpreter if interpreter is not None else self._factory()
```

#### `close() -> None`

Shut down the idle interpreters and stop starting new ones.

Source code in `dspy/primitives/interpreter_pool.py`

```
def close(self) -> None:
    """Shut down the idle interpreters and stop starting new ones."""
    with self._lock:
        self._closed = True
    self._finalizer()
```

```
  November 5, 2026
```
