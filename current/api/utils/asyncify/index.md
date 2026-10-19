# dspy.asyncify

## `dspy.asyncify(program: Callable[P, T]) -> Callable[P, Coroutine[Any, Any, T]]`

Wraps a DSPy program so that it can be called asynchronously. This is useful for running a program in parallel with another task (e.g., another DSPy program).

This implementation propagates the current thread’s configuration context to the worker thread.

Parameters:

| Name      | Type             | Description                                                | Default    |
| --------- | ---------------- | ---------------------------------------------------------- | ---------- |
| `program` | `Callable[P, T]` | The DSPy program to be wrapped for asynchronous execution. | *required* |

Returns:

| Type                                  | Description                                                                                                                                                          |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Callable[P, Coroutine[Any, Any, T]]` | An async function: An async function that, when awaited, runs the program in a worker thread. The current thread’s configuration context is inherited for each call. |

Source code in `dspy/utils/asyncify.py`

```
def asyncify(program: Callable[P, T]) -> Callable[P, Coroutine[Any, Any, T]]:
    """
    Wraps a DSPy program so that it can be called asynchronously. This is useful for running a
    program in parallel with another task (e.g., another DSPy program).

    This implementation propagates the current thread's configuration context to the worker thread.

    Args:
        program: The DSPy program to be wrapped for asynchronous execution.

    Returns:
        An async function: An async function that, when awaited, runs the program in a worker thread.
            The current thread's configuration context is inherited for each call.
    """

    async def async_program(*args: P.args, **kwargs: P.kwargs) -> T:
        # Capture the current overrides at call-time.
        from dspy.dsp.utils.settings import thread_local_overrides

        parent_overrides = thread_local_overrides.get().copy()

        def wrapped_program(*a: P.args, **kw: P.kwargs) -> T:
            from dspy.dsp.utils.settings import thread_local_overrides

            original_overrides = thread_local_overrides.get()
            token = thread_local_overrides.set({**original_overrides, **parent_overrides.copy()})
            try:
                return program(*a, **kw)
            finally:
                thread_local_overrides.reset(token)

        partial_f = functools.partial(wrapped_program, *args, **kwargs)
        return await anyio.to_thread.run_sync(partial_f, abandon_on_cancel=True, limiter=get_limiter())

    return async_program
```
