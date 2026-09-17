# dspy.enable_litellm_logging

## `dspy.enable_litellm_logging()`

Source code in `dspy/clients/__init__.py`

```
def enable_litellm_logging():
    litellm = get_litellm(feature="LiteLLM logging")
    litellm.suppress_debug_info = False
    litellm._dspy_logging_configured = True
    configure_litellm_logging("DEBUG")
```
