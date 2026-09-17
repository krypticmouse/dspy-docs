# dspy.retrievers.Embeddings

## `dspy.Embeddings(corpus: list[str], embedder, k: int = 5, callbacks: list[Any] | None = None, cache: bool = False, brute_force_threshold: int = 20000, normalize: bool = True)`

DSPy Embeddings retriever.

This class retrieves the top-k most similar passages from a corpus using embedding-based similarity search. For large corpora, a FAISS index is built for fast approximate candidate retrieval, followed by exact re-ranking. For small corpora, brute-force search is used.

Source code in `dspy/retrievers/embeddings.py`

```
def __init__(
    self,
    corpus: list[str],
    embedder,
    k: int = 5,
    callbacks: list[Any] | None = None,
    cache: bool = False,
    brute_force_threshold: int = 20_000,
    normalize: bool = True,
):
    assert cache is False, "Caching is not supported for embeddings-based retrievers"

    self.embedder = embedder
    self.k = k
    self.corpus = corpus
    self.normalize = normalize

    self.corpus_embeddings = self.embedder(self.corpus)
    self.corpus_embeddings = self._normalize(self.corpus_embeddings) if self.normalize else self.corpus_embeddings

    self.index = self._build_faiss() if len(corpus) >= brute_force_threshold else None
    self.search_fn = Unbatchify(self._batch_forward)
```

### Functions

#### `__call__(query: str)`

Source code in `dspy/retrievers/embeddings.py`

```
def __call__(self, query: str):
    return self.forward(query)
```

#### `forward(query: str)`

Search for the top-k passages most similar to the query.

Parameters:

| Name    | Type  | Description             | Default    |
| ------- | ----- | ----------------------- | ---------- |
| `query` | `str` | The search query string | *required* |

Returns:

| Type | Description                                                                 |
| ---- | --------------------------------------------------------------------------- |
|      | dspy.Prediction: A prediction containing passages and their corpus indices. |

Source code in `dspy/retrievers/embeddings.py`

```
def forward(self, query: str):
    """Search for the top-k passages most similar to the query.

    Args:
        query (str): The search query string

    Returns:
        dspy.Prediction: A prediction containing passages and their corpus indices.
    """
    import dspy

    passages, indices, _scores = self.search_fn(query)
    return dspy.Prediction(passages=passages, indices=indices)
```

#### `from_saved(path: str, embedder)`

Create an Embeddings instance from a saved index.

This is the recommended way to load saved embeddings as it creates a new instance without unnecessarily computing embeddings.

Parameters:

| Name       | Type  | Description                                    | Default    |
| ---------- | ----- | ---------------------------------------------- | ---------- |
| `path`     | `str` | Directory path where the embeddings were saved | *required* |
| `embedder` |       | The embedder function to use for new queries   | *required* |

Returns:

| Type | Description                          |
| ---- | ------------------------------------ |
|      | Embeddings instance loaded from disk |

Examples:

```
# Save embeddings
embeddings = Embeddings(corpus, embedder)
embeddings.save("./saved_embeddings")

# Load embeddings later
loaded_embeddings = Embeddings.from_saved("./saved_embeddings", embedder)
```

Source code in `dspy/retrievers/embeddings.py`

````
@classmethod
def from_saved(cls, path: str, embedder):
    """
    Create an Embeddings instance from a saved index.

    This is the recommended way to load saved embeddings as it creates a new
    instance without unnecessarily computing embeddings.

    Args:
        path: Directory path where the embeddings were saved
        embedder: The embedder function to use for new queries

    Returns:
        Embeddings instance loaded from disk

    Examples:
        ```python
        # Save embeddings
        embeddings = Embeddings(corpus, embedder)
        embeddings.save("./saved_embeddings")

        # Load embeddings later
        loaded_embeddings = Embeddings.from_saved("./saved_embeddings", embedder)
        ```
    """
    # Create a minimal instance without triggering embedding computation
    instance = cls.__new__(cls)
    # Initialize the search function (required since we bypassed __init__)
    instance.search_fn = Unbatchify(instance._batch_forward)
    instance.load(path, embedder)
    return instance
````

#### `load(path: str, embedder)`

Load the embeddings index from disk into the current instance.

Parameters:

| Name       | Type  | Description                                    | Default    |
| ---------- | ----- | ---------------------------------------------- | ---------- |
| `path`     | `str` | Directory path where the embeddings were saved | *required* |
| `embedder` |       | The embedder function to use for new queries   | *required* |

Returns:

| Name   | Type | Description                      |
| ------ | ---- | -------------------------------- |
| `self` |      | Returns self for method chaining |

Raises:

| Type                | Description                                         |
| ------------------- | --------------------------------------------------- |
| `FileNotFoundError` | If the save directory or required files don't exist |
| `ValueError`        | If the saved config is invalid or incompatible      |

Source code in `dspy/retrievers/embeddings.py`

```
def load(self, path: str, embedder):
    """
    Load the embeddings index from disk into the current instance.

    Args:
        path: Directory path where the embeddings were saved
        embedder: The embedder function to use for new queries

    Returns:
        self: Returns self for method chaining

    Raises:
        FileNotFoundError: If the save directory or required files don't exist
        ValueError: If the saved config is invalid or incompatible
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Save directory not found: {path}")

    config_path = os.path.join(path, "config.json")
    embeddings_path = os.path.join(path, "corpus_embeddings.npy")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    # Load configuration and corpus
    with open(config_path) as f:
        config = json.load(f)

    # Validate required config fields
    required_fields = ["k", "normalize", "corpus", "has_faiss_index"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Invalid config: missing required field '{field}'")

    # Restore configuration
    self.k = config["k"]
    self.normalize = config["normalize"]
    self.corpus = config["corpus"]
    self.embedder = embedder

    # Load embeddings
    self.corpus_embeddings = np.load(embeddings_path)

    # Load FAISS index if it was saved and FAISS is available
    faiss_index_path = os.path.join(path, "faiss_index.bin")
    if config["has_faiss_index"] and os.path.exists(faiss_index_path):
        try:
            import faiss
            self.index = faiss.read_index(faiss_index_path)
        except ImportError:
            # If FAISS is not available, fall back to brute force
            self.index = None
    else:
        self.index = None

    return self
```

#### `save(path: str)`

Save the embeddings index to disk.

This saves the corpus, embeddings, FAISS index (if present), and configuration to allow for fast loading without recomputing embeddings.

Parameters:

| Name   | Type  | Description                                       | Default    |
| ------ | ----- | ------------------------------------------------- | ---------- |
| `path` | `str` | Directory path where the embeddings will be saved | *required* |

Source code in `dspy/retrievers/embeddings.py`

```
def save(self, path: str):
    """
    Save the embeddings index to disk.

    This saves the corpus, embeddings, FAISS index (if present), and configuration
    to allow for fast loading without recomputing embeddings.

    Args:
        path: Directory path where the embeddings will be saved
    """
    os.makedirs(path, exist_ok=True)

    # Save configuration and corpus
    config = {
        "k": self.k,
        "normalize": self.normalize,
        "corpus": self.corpus,
        "has_faiss_index": self.index is not None,
    }

    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save embeddings
    np.save(os.path.join(path, "corpus_embeddings.npy"), self.corpus_embeddings)

    # Save FAISS index if it exists
    if self.index is not None:
        try:
            import faiss
            faiss.write_index(self.index, os.path.join(path, "faiss_index.bin"))
        except ImportError:
            # If FAISS is not available, we can't save the index
            # but we can still save the embeddings for brute force search
            pass
```
