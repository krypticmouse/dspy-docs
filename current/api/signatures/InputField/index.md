# dspy.InputField

## `dspy.InputField(**kwargs)`

Source code in `dspy/signatures/field.py`

```
def InputField(**kwargs): # noqa: N802
    _warn_deprecated_field_args(**kwargs)
    return pydantic.Field(**move_kwargs(**kwargs, __dspy_field_type="input"))
```
