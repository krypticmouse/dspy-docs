# dspy.experimental.ReAnchor

Experimental API

`ReAnchor` is experimental and may change without warning.

`ReAnchor` fits the numeric decision settings in a program to your metric. These settings are the `threshold`, `cuts`, and `weights` entries in each predictor’s `fields`. They are described in [Decision types and System One models](https://dspy.ai/current/api/experimental/DecisionTypes/index.md). ReAnchor does not change any instructions, descriptions, or demos.

You give ReAnchor a metric, a program, and training examples. The program can be a single `dspy.Predict` or any module that holds predictors with decision outputs. A decision output is a `Noul`, `Score`, or `Choice` output, or a native `bool` or `Literal` output. `RLM` does not support decision outputs, so ReAnchor does not support RLM programs either.

```
import dspy
from dspy.experimental import ReAnchor, TypeSafe


class Match(dspy.Signature):
    """Decide whether two product listings describe the same item."""

    pair: str = dspy.InputField(desc="Two listings.")
    match: bool = dspy.OutputField(desc="Are they the same item?")


def metric(example, prediction, trace=None):
    return float(prediction.match == example.match)


dspy.configure(lm=TypeSafe("jev-latest"))
optimizer = ReAnchor(metric)
tuned = optimizer.compile(dspy.Predict(Match), trainset=trainset, valset=valset)
print(optimizer.report)
```

## What ReAnchor fits

ReAnchor searches each output’s numeric settings against the whole-program metric. A new setting must improve the training score and pass a fold check.

For every compatible output, ReAnchor saves the original configuration and score, enables probability-based execution, and fits the numeric settings. It then compares the fitted behavior against the original using a fold check. If rejected, it restores the exact original configuration, including removing an entry that was originally absent. The adapter handles the backend used on each call; calibration does not inspect it.

Before fitting each output, ReAnchor runs the program and records that output’s probabilities on every call. A threshold anywhere between two neighboring P(True) values makes the same decisions, so ReAnchor tries the midpoint of each gap between them. For example, when the model only returns P(True) of 0.98 and 1.0, ReAnchor tries 0.5 and 0.99.

- For a Boolean output, ReAnchor tries each midpoint between the observed P(True) values.
- For a Score output, the level depends on the mean level index. ReAnchor tries each `cut` at the midpoints between the observed mean indexes, and it keeps the cuts in order. The cuts choose `.level` and do not change `.value`.
- For a Choice output, ReAnchor moves one option’s multiplier at a time. It tries the multipliers that fall between the points where that option’s pick would flip on some call.

The current setting stays unless another setting scores strictly better and passes a fold check. The check splits the training examples into up to five parts. For each part, ReAnchor picks a setting using the remaining parts and scores that pick on the held-out part. A new setting is kept only when the combined held-out score beats the current setting. The check discourages gains confined to small portions of the dataset, but can accept them when they recur across folds. Among improving candidates with equal scores, ReAnchor prefers the widest gap, which leaves the most room on either side. Each search step tries at most 40 gap midpoints, thinning large lists to settings spaced evenly through the observed values. For Boolean outputs, it also tries threshold zero when P(True)=0 is observed, since that boundary is the only way to classify those answers as True. When two observed probabilities are adjacent floats, it tries the upper value as a threshold: `p >= threshold` separates them without a midpoint.

## Requests and the cache

The numeric settings are not part of the request. Repeated identical requests reuse cached answers. In a composed program, changing an upstream decision can change downstream inputs or which predictors run, producing new requests and additional backend calls. Native-output promotion also changes the request.

`compile` requires caching by default to avoid repeating identical backend calls; it does not guarantee a fixed request count. Pass `require_cache=False` to run without this check. The cache check inspects statically bound or globally configured clients. For programs selecting clients inside `forward()`, disable the check and manage caching on those clients.

## Native outputs on a generative LM

A generative LM answers a native `bool` or `Literal` output with a single value. It does not report probabilities, so ReAnchor has nothing to fit. When an output has an entry in the predictor’s `fields`, `Predict` asks the LM for probabilities instead. `Predict` then applies the threshold or weights to pick the value, and the output still returns a `bool` or a `Literal` member.

The first pass with probabilities sends new requests, because the request changes. If the fitted behavior is rejected, restoring the original configuration returns an unconfigured native field to direct generation.

```
dspy.configure(lm=dspy.LM("your-provider/your-model"))  # Use your model's identifier.
tuned = ReAnchor(metric).compile(dspy.Predict(Match), trainset=trainset)
```

A System One model such as Jev always returns probabilities. Enabling probability execution leaves its request unchanged; restoring a field restores its original decoding settings, not direct generation.

## Errors

ReAnchor stops at the first error from the program or the metric. On a generative LM, a malformed answer counts as an error. For example, `Predict` raises when a `Score` answer leaves out a probability for any level. Use a model that follows JSON schemas reliably, and set a client `timeout` that fits your backend’s load.

## Results

`compile` returns a copy of the program and leaves your program unchanged. After `compile`, `report` holds:

- `train_score_before` and `train_score`, the mean metric score on the training examples before and after calibration.
- `val_score_before` and `val_score`, the same scores on `valset` when you pass one. ReAnchor never fits settings on `valset`.
- `fitted`, one row per output with the fitted value, or the reason ReAnchor skipped it. Each fitted row has an `observed` entry with the number of calls and settings tried; threshold and cut reports also summarize observed values. Its `fold_check` entry counts the search steps whose better training score passed or failed the fold check. Per-field `train_score_original` is the score before enabling probabilities, `train_score_at_start` is the probability-based baseline, and `train_score` describes the retained behavior. Rejected configurations have a `skipped` reason instead of a fitted `value`.

Your metric may return a number or a `dspy.Prediction` with a `score`.

When you set `log_dir`, ReAnchor writes `report.json` to that folder.

The fitted settings are part of each predictor’s `fields`, so the tuned program saves and loads like any other `Predict` program.

### `dspy.experimental.ReAnchor(metric, *, num_threads=None, log_dir=None, require_cache=True)`

Bases: `Teleprompter`

Calibrate a program’s decisions: fit every threshold, Score cut, and Choice weight against the metric.

The program can be a single `Predict` or any `dspy.Module` holding predictors with decision outputs, and the metric is the only supervision. Each parameter changes how `Predict` reads the backend’s probabilities without adding request parameters. Identical requests reuse cached answers, but changed upstream decisions can produce new downstream requests.

ReAnchor enables probability-based execution for every compatible output through the predictor’s `fields`. It keeps each fitted configuration only when it beats the original behavior and passes a fold check; otherwise it restores the original configuration, including absent field entries.

Parameters:

| Name            | Type | Description                                                                                                                                              | Default    |
| --------------- | ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `metric`        |      | Per-example metric to maximize, as in dspy.Evaluate. It may return a number, or a dspy.Prediction with a score.                                          | *required* |
| `num_threads`   |      | Evaluation concurrency, as in dspy.Evaluate.                                                                                                             | `None`     |
| `log_dir`       |      | When set, report.json is written here.                                                                                                                   | `None`     |
| `require_cache` |      | When True, check caching on statically bound or globally configured clients. Set it to False for runtime client selection or to allow uncached requests. | `True`     |

After `compile`, `report` holds the fitted parameters and the metric’s mean before and after calibration, on the training set and on the validation set when one is given.

Source code in `dspy/teleprompt/reanchor/reanchor.py`

```
def __init__(self, metric, *, num_threads=None, log_dir=None, require_cache=True):
    super().__init__()
    self.metric = metric
    self.num_threads = num_threads
    self.log_dir = Path(log_dir) if log_dir else None
    self.require_cache = require_cache
    self.report: dict[str, Any] = {}
```

#### Methods:

##### `compile(student, *, trainset, valset=None)`

Return a calibrated copy; leave the student unchanged.

Parameters:

| Name       | Type | Description                                                                             | Default    |
| ---------- | ---- | --------------------------------------------------------------------------------------- | ---------- |
| `student`  |      | A Predict, or a module holding predictors with decision outputs.                        | *required* |
| `trainset` |      | Examples the parameters are fitted on.                                                  | *required* |
| `valset`   |      | Examples scored before and after calibration for the report. Nothing is fitted on them. | `None`     |

Source code in `dspy/teleprompt/reanchor/reanchor.py`

```
def compile(self, student, *, trainset, valset=None):
    """Return a calibrated copy; leave the student unchanged.

    Args:
        student: A `Predict`, or a module holding predictors with decision outputs.
        trainset: Examples the parameters are fitted on.
        valset: Examples scored before and after calibration for the report. Nothing is fitted on them.
    """
    if not trainset:
        raise ValueError("trainset must contain at least one example.")
    program = student.deepcopy()
    found = predictors(program)
    if not found:
        raise ValueError("The student must contain at least one Predict with a decision output.")
    uncached = [name for name, predict in found if not caches(predict)] if self.require_cache else []
    if self.require_cache and uncached:
        raise ValueError(
            f"Predictors {uncached} do not cache responses, so every candidate setting would send new requests. "
            "Enable the client's cache, or pass require_cache=False to calibrate anyway."
        )

    logger.info("answering %d training examples", len(trainset))
    before = self._score(program, trainset, progress=True)
    val_before = self._score(program, valset) if valset else None
    logger.info("fitting thresholds, cuts, and weights")
    fitted = calibrate(program, trainset, self.metric, num_threads=self.num_threads)
    self.report = {"train_score_before": before, "train_score": self._score(program, trainset), "fitted": fitted}
    logger.info("calibrated: train %s -> %s", before, self.report["train_score"])
    if valset:
        self.report.update(val_score_before=val_before, val_score=self._score(program, valset))
        logger.info("validation: %s -> %s", val_before, self.report["val_score"])

    if self.log_dir:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "report.json").write_text(json.dumps(self.report, indent=2, default=str), encoding="utf-8")
    program._compiled = True
    return program
```
