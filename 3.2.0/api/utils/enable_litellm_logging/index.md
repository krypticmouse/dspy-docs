# dspy.enable_litellm_logging

## `dspy.enable_litellm_logging()`

Source code in `dspy/clients/__init__.py`

```
def enable_litellm_logging():
    litellm.suppress_debug_info = False
    configure_litellm_logging("DEBUG")
```
