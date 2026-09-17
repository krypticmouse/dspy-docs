# dspy.ColBERTv2

## `dspy.ColBERTv2(url: str = 'http://0.0.0.0', port: str | int | None = None, post_requests: bool = False)`

Wrapper for the ColBERTv2 Retrieval.

Source code in `dspy/dsp/colbertv2.py`

```
def __init__(
    self,
    url: str = "http://0.0.0.0",
    port: str | int | None = None,
    post_requests: bool = False,
):
    self.post_requests = post_requests
    self.url = f"{url}:{port}" if port else url
```

### Functions

#### `__call__(query: str, k: int = 10, simplify: bool = False) -> list[str] | list[dotdict]`

Source code in `dspy/dsp/colbertv2.py`

```
def __call__(
    self,
    query: str,
    k: int = 10,
    simplify: bool = False,
) -> list[str] | list[dotdict]:
    if self.post_requests:
        topk: list[dict[str, Any]] = colbertv2_post_request(self.url, query, k)
    else:
        topk: list[dict[str, Any]] = colbertv2_get_request(self.url, query, k)

    if simplify:
        return [psg["long_text"] for psg in topk]

    return [dotdict(psg) for psg in topk]
```
