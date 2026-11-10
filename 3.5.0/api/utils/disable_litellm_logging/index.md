# dspy.disable_litellm_logging

Disable logging for the optional LiteLLM compatibility integration installed with `pip install "dspy[litellm]"`. This is separate from DSPy’s own logging and does not disable the compatibility engine.

## `dspy.disable_litellm_logging()`

Source code in `dspy/clients/__init__.py`

```
def disable_litellm_logging():
    litellm = get_litellm(feature="LiteLLM logging")
    litellm.suppress_debug_info = True
    litellm._dspy_logging_configured = True
    configure_litellm_logging("ERROR")
```

```
  October 28, 2026
```
