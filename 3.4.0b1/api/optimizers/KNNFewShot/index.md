# dspy.KNNFewShot

Requires numpy

`dspy.KNNFewShot` requires numpy. Install it with `pip install dspy[numpy]`.

## `dspy.KNNFewShot(k: int, trainset: list[Example], vectorizer: Embedder, **few_shot_bootstrap_args: dict[str, Any])`

Bases: `Teleprompter`

KNNFewShot is an optimizer that uses an in-memory KNN retriever to find the k nearest neighbors in a trainset at test time. For each input example in a forward call, it identifies the k most similar examples from the trainset and attaches them as demonstrations to the student module.

Parameters:

| Name                        | Type             | Description                                                     | Default    |
| --------------------------- | ---------------- | --------------------------------------------------------------- | ---------- |
| `k`                         | `int`            | The number of nearest neighbors to attach to the student model. | *required* |
| `trainset`                  | `list[Example]`  | The training set to use for few-shot prompting.                 | *required* |
| `vectorizer`                | `Embedder`       | The Embedder to use for vectorization                           | *required* |
| `**few_shot_bootstrap_args` | `dict[str, Any]` | Additional arguments for the BootstrapFewShot optimizer.        | `{}`       |

Examples:

```
import dspy
from sentence_transformers import SentenceTransformer

# Define a QA module with chain of thought
qa = dspy.ChainOfThought("question -> answer")

# Create a training dataset with examples
trainset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    # ... more examples ...
]

# Initialize KNNFewShot with a sentence transformer model
knn_few_shot = KNNFewShot(
    k=3,
    trainset=trainset,
    vectorizer=dspy.Embedder(SentenceTransformer("all-MiniLM-L6-v2").encode)
)

# Compile the QA module with few-shot learning
compiled_qa = knn_few_shot.compile(qa)

# Use the compiled module
result = compiled_qa("What is the capital of Belgium?")
```

Source code in `dspy/teleprompt/knn_fewshot.py`

````
def __init__(self, k: int, trainset: list[Example], vectorizer: Embedder, **few_shot_bootstrap_args: dict[str, Any]):
    """
    KNNFewShot is an optimizer that uses an in-memory KNN retriever to find the k nearest neighbors
    in a trainset at test time. For each input example in a forward call, it identifies the k most
    similar examples from the trainset and attaches them as demonstrations to the student module.

    Args:
        k: The number of nearest neighbors to attach to the student model.
        trainset: The training set to use for few-shot prompting.
        vectorizer: The `Embedder` to use for vectorization
        **few_shot_bootstrap_args: Additional arguments for the `BootstrapFewShot` optimizer.

    Examples:
        ```python
        import dspy
        from sentence_transformers import SentenceTransformer

        # Define a QA module with chain of thought
        qa = dspy.ChainOfThought("question -> answer")

        # Create a training dataset with examples
        trainset = [
            dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
            # ... more examples ...
        ]

        # Initialize KNNFewShot with a sentence transformer model
        knn_few_shot = KNNFewShot(
            k=3,
            trainset=trainset,
            vectorizer=dspy.Embedder(SentenceTransformer("all-MiniLM-L6-v2").encode)
        )

        # Compile the QA module with few-shot learning
        compiled_qa = knn_few_shot.compile(qa)

        # Use the compiled module
        result = compiled_qa("What is the capital of Belgium?")
        ```
    """
    self.KNN = KNN(k, trainset, vectorizer=vectorizer)
    self.few_shot_bootstrap_args = few_shot_bootstrap_args
````

### Methods:

#### `compile(student, *, teacher=None)`

Source code in `dspy/teleprompt/knn_fewshot.py`

```
def compile(self, student, *, teacher=None):
    student_copy = student.reset_copy()

    def forward_pass(_, **kwargs):
        knn_trainset = self.KNN(**kwargs)
        few_shot_bootstrap = BootstrapFewShot(**self.few_shot_bootstrap_args)
        compiled_program = few_shot_bootstrap.compile(
            student,
            teacher=teacher,
            trainset=knn_trainset,
        )
        return compiled_program(**kwargs)

    student_copy.forward = types.MethodType(forward_pass, student_copy)
    return student_copy
```

#### `get_params() -> dict[str, Any]`

Get the parameters of the teleprompter.

Returns:

| Type             | Description                         |
| ---------------- | ----------------------------------- |
| `dict[str, Any]` | The parameters of the teleprompter. |

Source code in `dspy/teleprompt/teleprompt.py`

```
def get_params(self) -> dict[str, Any]:
    """
    Get the parameters of the teleprompter.

    Returns:
        The parameters of the teleprompter.
    """
    return self.__dict__
```
