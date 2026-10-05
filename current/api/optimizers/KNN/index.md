# dspy.KNN

Requires numpy

`dspy.KNN` requires numpy. Install it with `pip install dspy[numpy]`.

## `dspy.KNN(k: int, trainset: list[Example], vectorizer: Embedder)`

A k-nearest neighbors retriever that finds similar examples from a training set.

Parameters:

| Name         | Type            | Description                                 | Default    |
| ------------ | --------------- | ------------------------------------------- | ---------- |
| `k`          | `int`           | Number of nearest neighbors to retrieve     | *required* |
| `trainset`   | `list[Example]` | List of training examples to search through | *required* |
| `vectorizer` | `Embedder`      | The Embedder to use for vectorization       | *required* |

Examples:

```
import dspy
from sentence_transformers import SentenceTransformer

# Create a training dataset with examples
trainset = [
    dspy.Example(input="hello", output="world"),
    # ... more examples ...
]

# Initialize KNN with a sentence transformer model
knn = KNN(
    k=3,
    trainset=trainset,
    vectorizer=dspy.Embedder(SentenceTransformer("all-MiniLM-L6-v2").encode)
)

# Find similar examples
similar_examples = knn(input="hello")
```

Source code in `dspy/predict/knn.py`

````
def __init__(self, k: int, trainset: list[Example], vectorizer: Embedder):
    """
    A k-nearest neighbors retriever that finds similar examples from a training set.

    Args:
        k: Number of nearest neighbors to retrieve
        trainset: List of training examples to search through
        vectorizer: The `Embedder` to use for vectorization

    Examples:
        ```python
        import dspy
        from sentence_transformers import SentenceTransformer

        # Create a training dataset with examples
        trainset = [
            dspy.Example(input="hello", output="world"),
            # ... more examples ...
        ]

        # Initialize KNN with a sentence transformer model
        knn = KNN(
            k=3,
            trainset=trainset,
            vectorizer=dspy.Embedder(SentenceTransformer("all-MiniLM-L6-v2").encode)
        )

        # Find similar examples
        similar_examples = knn(input="hello")
        ```
    """
    self.k = k
    self.trainset = trainset
    self.embedding = vectorizer
    trainset_casted_to_vectorize = [
        " | ".join([f"{key}: {value}" for key, value in example.items() if key in example._input_keys])
        for example in self.trainset
    ]
    self.trainset_vectors = self.embedding(trainset_casted_to_vectorize).astype(np.float32)
````

### Methods:

#### `__call__(**kwargs) -> list`

Source code in `dspy/predict/knn.py`

```
def __call__(self, **kwargs) -> list:
    input_example_vector = self.embedding([" | ".join([f"{key}: {val}" for key, val in kwargs.items()])])
    scores = np.dot(self.trainset_vectors, input_example_vector.T).squeeze()
    nearest_samples_idxs = scores.argsort()[-self.k :][::-1]
    return [self.trainset[cur_idx] for cur_idx in nearest_samples_idxs]
```
