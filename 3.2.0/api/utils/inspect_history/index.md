# dspy.inspect_history

## `dspy.inspect_history(n: int = 1, file: TextIO | None = None) -> None`

The global history shared across all LMs.

Parameters:

| Name   | Type     | Description                                         | Default                                                                                                                                           |
| ------ | -------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `n`    | `int`    | Number of recent entries to display. Defaults to 1. | `1`                                                                                                                                               |
| `file` | \`TextIO | None\`                                              | An optional file-like object to write output to. When provided, ANSI color codes are automatically disabled. Defaults to None (prints to stdout). |

Source code in `dspy/clients/base_lm.py`

```
def inspect_history(n: int = 1, file: "TextIO | None" = None) -> None:
    """The global history shared across all LMs.

    Args:
        n: Number of recent entries to display. Defaults to 1.
        file: An optional file-like object to write output to. When
            provided, ANSI color codes are automatically disabled.
            Defaults to `None` (prints to stdout).
    """
    pretty_print_history(GLOBAL_HISTORY, n, file=file)
```
