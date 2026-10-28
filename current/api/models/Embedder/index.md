# dspy.Embedder

Requires numpy

`dspy.Embedder` requires numpy. Install it with `pip install dspy[numpy]`.

## Hosted endpoints and custom embedding functions

A string model uses the OpenAI SDK against OpenAI, Azure OpenAI, or an OpenAI-compatible embeddings endpoint. For example:

```
import dspy

embedder = dspy.Embedder("openai/text-embedding-3-small", dimensions=512)
vectors = embedder(["hello", "world"])  # uses OPENAI_API_KEY

compatible = dspy.Embedder(
    "openai/your-org/your-embedding-model",
    api_base="https://your-provider.example/v1",
    api_key="YOUR_PROVIDER_KEY",
)
```

Bare OpenAI model names are also accepted. For compatible endpoints, only the leading `openai/` is stripped, preserving an organization/model id. `base_url` is an alias for `api_base`; the default endpoint can also come from `OPENAI_BASE_URL` or `OPENAI_API_BASE`, with credentials in `OPENAI_API_KEY`.

Azure embeddings use `azure/<deployment>` and the Azure SDK client:

```
azure = dspy.Embedder(
    "azure/your-embedding-deployment",
    api_base="https://your-resource.openai.azure.com",  # resource root, not /embeddings
    api_version="YOUR_SUPPORTED_API_VERSION",
    api_key="YOUR_AZURE_KEY",
)
```

For Azure, `azure_endpoint` can replace `api_base`; use `base_url` only for an explicit SDK-style base path. Environment defaults are `AZURE_OPENAI_ENDPOINT` / `AZURE_API_BASE`, `OPENAI_API_VERSION` / `AZURE_API_VERSION`, and `AZURE_OPENAI_API_KEY` / `AZURE_API_KEY`. This embeddings configuration is separate from native `dspy.LM`’s Azure v1 route.

This is not a multi-provider routing layer. Other LiteLLM embedding prefixes and their provider-specific kwargs are removed. Installing `dspy[litellm]` enables LM compatibility, not embedding fallback. For a provider with another protocol or authentication flow, pass a callable that accepts a list of strings and returns one embedding vector per input, for example `dspy.Embedder(model.encode)` for a local embedding model. LM provider registration does not register an embeddings backend. See the [migration guide](https://dspy.ai/current/community/normalized-lm-api-migration/index.md).

## Retaining a Cohere retrieval index

Cohere’s `search_document` and `search_query` modes are not interchangeable. Its OpenAI-compatible endpoint does not support `input_type`, so changing only the model prefix does not preserve this workflow. Keep the provider call in an application-owned callable instead. For example, install LiteLLM separately with `pip install litellm` and set `COHERE_API_KEY`:

```
import dspy
import litellm
from dspy.retrievers import Embeddings


def cohere_vectors(texts, input_type):
    response = litellm.embedding(
        model="cohere/embed-english-v3.0",
        input=texts,
        input_type=input_type,
    )
    return [item["embedding"] for item in response.data]


embedder = dspy.Embedder(cohere_vectors, input_type="search_document")
retrieve = Embeddings(
    ["Paris is the capital of France", "Berlin is the capital of Germany"],
    embedder,
    k=1,
)
# Construction above embeds the corpus. Subsequent calls embed queries.
embedder.default_kwargs["input_type"] = "search_query"
result = retrieve("What is the capital of France?")
```

Keep a separate document-mode embedder if you also index new documents after this switch; do not toggle shared defaults while serving concurrent requests. Preserve the original model, dimensions, normalization and input modes when using an existing index. The callable can instead use Cohere’s SDK; DSPy does not install or automatically select either dependency. DSPy still handles batching and embedding caching, including `input_type` in callable kwargs.

## `dspy.Embedder(model: str | Callable, batch_size: int = 200, caching: bool = True, **kwargs: dict[str, Any])`

DSPy embedding class.

The class for computing embeddings for text inputs. This class provides a unified interface for both:

1. Hosted embedding models via the OpenAI SDK (including OpenAI-compatible endpoints and Azure OpenAI)
1. Custom embedding functions that you provide

For OpenAI models, pass a bare model name or an “openai/”-prefixed name, such as “openai/text-embedding-3-small”. Use api_base (or base_url) for OpenAI-compatible endpoints. For Azure OpenAI, use “azure/” with api_base (or azure_endpoint) and api_version. Other provider prefixes are not supported; use a custom callable for other APIs. Hosted requests use DSPy’s cache and close their SDK clients after each uncached batch.

For custom embedding models, pass a callable function that: - Takes a list of strings as input. - Returns embeddings as either: - A 2D numpy array of float32 values - A 2D list of float32 values - Each row should represent one embedding vector

Parameters:

| Name         | Type             | Description                                                                          | Default                                                                                                            |
| ------------ | ---------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| `model`      | \`str            | Callable\`                                                                           | A bare OpenAI model name, an “openai/” or “azure/” string, or a callable that represents a custom embedding model. |
| `batch_size` | `int`            | The default batch size for processing inputs in batches. Defaults to 200.            | `200`                                                                                                              |
| `caching`    | `bool`           | Whether to cache the embedding response when using a hosted model. Defaults to True. | `True`                                                                                                             |
| `**kwargs`   | `dict[str, Any]` | Additional default keyword arguments to pass to the embedding model.                 | `{}`                                                                                                               |

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
def __call__(
    self,
    inputs: str | list[str],
    batch_size: int | None = None,
    caching: bool | None = None,
    **kwargs: dict[str, Any],
) -> np.ndarray:
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

    compute_embeddings = _cached_compute_embeddings if caching else _cached_compute_embeddings.__wrapped__

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
    acompute_embeddings = _cached_acompute_embeddings if caching else _cached_acompute_embeddings.__wrapped__

    for batch in input_batches:
        embeddings_list.extend(await acompute_embeddings(self.model, batch, caching=caching, **kwargs))
    return self._postprocess(embeddings_list, is_single_input)
```

```
  October 28, 2026
```
