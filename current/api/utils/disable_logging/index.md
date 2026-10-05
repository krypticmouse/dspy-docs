# dspy.disable_logging

## `dspy.disable_logging()`

Disables the `DSPyLoggingStream` used by event logging APIs throughout DSPy (`eprint()`, `logger.info()`, etc), silencing all subsequent event logs.

Source code in `dspy/utils/logging_utils.py`

```
def disable_logging():
    """
    Disables the `DSPyLoggingStream` used by event logging APIs throughout DSPy
    (`eprint()`, `logger.info()`, etc), silencing all subsequent event logs.
    """
    DSPY_LOGGING_STREAM.enabled = False
```
