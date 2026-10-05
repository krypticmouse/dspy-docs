# dspy.enable_logging

## `dspy.enable_logging()`

Enables the `DSPyLoggingStream` used by event logging APIs throughout DSPy (`eprint()`, `logger.info()`, etc), emitting all subsequent event logs. This reverses the effects of `disable_logging()`.

Source code in `dspy/utils/logging_utils.py`

```
def enable_logging():
    """
    Enables the `DSPyLoggingStream` used by event logging APIs throughout DSPy
    (`eprint()`, `logger.info()`, etc), emitting all subsequent event logs. This
    reverses the effects of `disable_logging()`.
    """
    DSPY_LOGGING_STREAM.enabled = True
```
