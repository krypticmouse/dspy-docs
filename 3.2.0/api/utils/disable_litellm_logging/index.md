# dspy.disable_litellm_logging

## `dspy.disable_litellm_logging()`

Source code in `dspy/clients/__init__.py`

```
def disable_litellm_logging():
    litellm.suppress_debug_info = True
    configure_litellm_logging("ERROR")
```
