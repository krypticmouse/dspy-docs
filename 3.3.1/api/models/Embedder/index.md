# dspy.Embedder

Requires numpy

`dspy.Embedder` requires numpy. Install it with `pip install dspy[numpy]`.

## `dspy.Embedder(model: str | Callable, batch_size: int = 200, caching: bool = True, **kwargs: dict[str, Any])`

DSPy embedding class.

The class for computing embeddings for text inputs. This class provides a unified interface for both:

1. Hosted embedding models (e.g. OpenAI’s text-embedding-3-small) via litellm integration
1. Custom embedding functions that you provide

For hosted models, simply pass the model name as a string (e.g., “openai/text-embedding-3-small”). The class will use litellm to handle the API calls and caching.

For custom embedding models, pass a callable function that:

- Takes a list of strings as input.
- Returns embeddings as either:
- A 2D numpy array of float32 values
- A 2D list of float32 values
- Each row should represent one embedding vector

Parameters:

| Name         | Type             | Description                                                                          | Default                                                                                                                                                                                                                |
| ------------ | ---------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`      | \`str            | Callable\`                                                                           | The embedding model to use. This can be either a string (representing the name of the hosted embedding model, must be an embedding model supported by litellm) or a callable that represents a custom embedding model. |
| `batch_size` | `int`            | The default batch size for processing inputs in batches. Defaults to 200.            | `200`                                                                                                                                                                                                                  |
| `caching`    | `bool`           | Whether to cache the embedding response when using a hosted model. Defaults to True. | `True`                                                                                                                                                                                                                 |
| `**kwargs`   | `dict[str, Any]` | Additional default keyword arguments to pass to the embedding model.                 | `{}`                                                                                                                                                                                                                   |

Examples:

Example 1: Using a hosted model.

```
import dspy

embedder = dspy.Embedder("openai/text-embedding-3-small", batch_size=100)
embeddings = embedder(["hello", "world"])

assert embeddings.shape == (2, 1536)
```

Example 2: Using any local embedding model, e.g. from https://huggingface.co/models?library=sentence-transformers.

```
# pip install sentence_transformers
import dspy
from sentence_transformers import SentenceTransformer

# Load an extremely efficient local model for retrieval
model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device="cpu")

embedder = dspy.Embedder(model.encode)
embeddings = embedder(["hello", "world"], batch_size=1)

assert embeddings.shape == (2, 1024)
```

Example 3: Using a custom function.

```
import dspy
import numpy as np

def my_embedder(texts):
    return np.random.rand(len(texts), 10)

embedder = dspy.Embedder(my_embedder)
embeddings = embedder(["hello", "world"], batch_size=1)

assert embeddings.shape == (2, 10)
```

Source code in `dspy/clients/embedding.py`

```
def __init__(self, model: str | Callable, batch_size: int = 200, caching: bool = True, **kwargs: dict[str, Any]):
    self.model = model
    self.batch_size = batch_size
    self.caching = caching
    self.default_kwargs = kwargs
```

### Methods:

#### `__call__(inputs: str | list[str], batch_size: int | None = None, caching: bool | None = None, **kwargs: dict[str, Any]) -> np.ndarray`

Compute embeddings for the given inputs.

Parameters:

| Name         | Type             | Description                                                                                                                         | Default                                                                            |
| ------------ | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `inputs`     | \`str            | list[str]\`                                                                                                                         | The inputs to compute embeddings for, can be a single string or a list of strings. |
| `batch_size` | `int`            | The batch size for processing inputs. If None, defaults to the batch_size set during initialization.                                | `None`                                                                             |
| `caching`    | `bool`           | Whether to cache the embedding response when using a hosted model. If None, defaults to the caching setting from initialization.    | `None`                                                                             |
| `kwargs`     | `dict[str, Any]` | Additional keyword arguments to pass to the embedding model. These will override the default kwargs provided during initialization. | `{}`                                                                               |

Returns:

| Type      | Description                                                                                          |
| --------- | ---------------------------------------------------------------------------------------------------- |
| `ndarray` | numpy.ndarray: If the input is a single string, returns a 1D numpy array representing the embedding. |
| `ndarray` | If the input is a list of strings, returns a 2D numpy array of embeddings, one embedding per row.    |

Source code in `dspy/clients/embedding.py`

```
def __call__(self, inputs: str | list[str], batch_size: int | None = None, caching: bool | None = None, **kwargs: dict[str, Any]) -> np.ndarray:
    """Compute embeddings for the given inputs.

    Args:
        inputs: The inputs to compute embeddings for, can be a single string or a list of strings.
        batch_size (int, optional): The batch size for processing inputs. If None, defaults to the batch_size set
            during initialization.
        caching (bool, optional): Whether to cache the embedding response when using a hosted model. If None,
            defaults to the caching setting from initialization.
        kwargs: Additional keyword arguments to pass to the embedding model. These will override the default
            kwargs provided during initialization.

    Returns:
        numpy.ndarray: If the input is a single string, returns a 1D numpy array representing the embedding.
        If the input is a list of strings, returns a 2D numpy array of embeddings, one embedding per row.
    """
    input_batches, caching, kwargs, is_single_input = self._preprocess(inputs, batch_size, caching, **kwargs)

    compute_embeddings = _cached_compute_embeddings if caching else _compute_embeddings

    embeddings_list = []

    for batch in input_batches:
        embeddings_list.extend(compute_embeddings(self.model, batch, caching=caching, **kwargs))
    return self._postprocess(embeddings_list, is_single_input)
```

#### `acall(inputs, batch_size=None, caching=None, **kwargs)`

Source code in `dspy/clients/embedding.py`

```
async def acall(self, inputs, batch_size=None, caching=None, **kwargs):
    input_batches, caching, kwargs, is_single_input = self._preprocess(inputs, batch_size, caching, **kwargs)

    embeddings_list = []
    acompute_embeddings = _cached_acompute_embeddings if caching else _acompute_embeddings

    for batch in input_batches:
        embeddings_list.extend(await acompute_embeddings(self.model, batch, caching=caching, **kwargs))
    return self._postprocess(embeddings_list, is_single_input)
```
