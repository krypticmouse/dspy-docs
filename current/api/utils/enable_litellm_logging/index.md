# dspy.enable_litellm_logging

Enable logging for the optional LiteLLM compatibility integration installed with `pip install "dspy[litellm]"`. This is separate from DSPy’s own logging and does not enable LiteLLM routing for native-only calls.

## `dspy.enable_litellm_logging()`

Source code in `dspy/clients/__init__.py`

```
def enable_litellm_logging():
    litellm = get_litellm(feature="LiteLLM logging")
    litellm.suppress_debug_info = False
    litellm._dspy_logging_configured = True
    configure_litellm_logging("DEBUG")
```

```
  October 28, 2026
```
