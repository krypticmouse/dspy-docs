# dspy.GEPA: Reflective Prompt Optimizer

**GEPA** (Genetic-Pareto) is a reflective optimizer proposed in “GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning” (Agrawal et al., 2025, [arxiv:2507.19457](https://arxiv.org/abs/2507.19457)), that adaptively evolves *textual components* (such as prompts) of arbitrary systems. In addition to scalar scores returned by metrics, users can also provide GEPA with a text feedback to guide the optimization process. Such textual feedback provides GEPA more visibility into why the system got the score that it did, and then GEPA can introspect to identify how to improve the score. This allows GEPA to propose high performing prompts in very few rollouts.

## `dspy.GEPA(*args, **kwargs)`

Bases: `Teleprompter`

Stand-in for GEPA when the gepa package is not installed.

DSPy installs gepa only on Python < 3.15, because gepa declares requires-python \<3.15. The stand-in keeps GEPA’s Teleprompter interface, so code that inspects or wraps GEPA.**init** and GEPA.compile still imports.

Source code in `dspy/teleprompt/__init__.py`

```
def __init__(self, *args, **kwargs):
    raise ImportError(
        "gepa is required to use dspy.GEPA. It is not installed by default on Python 3.15 and later "
        "because gepa declares requires-python <3.15."
    )
```

### Methods:

#### `compile(student: Module, *, trainset: list[Example], teacher: Module | None = None, valset: list[Example] | None = None, **kwargs) -> Module`

Optimize the student program.

Parameters:

| Name       | Type            | Description                               | Default                                      |
| ---------- | --------------- | ----------------------------------------- | -------------------------------------------- |
| `student`  | `Module`        | The student program to optimize.          | *required*                                   |
| `trainset` | `list[Example]` | The training set to use for optimization. | *required*                                   |
| `teacher`  | \`Module        | None\`                                    | The teacher program to use for optimization. |
| `valset`   | \`list[Example] | None\`                                    | The validation set to use for optimization.  |

Returns:

| Type     | Description                    |
| -------- | ------------------------------ |
| `Module` | The optimized student program. |

Source code in `dspy/teleprompt/teleprompt.py`

```
def compile(self, student: Module, *, trainset: list[Example], teacher: Module | None = None, valset: list[Example] | None = None, **kwargs) -> Module:
    """
    Optimize the student program.

    Args:
        student: The student program to optimize.
        trainset: The training set to use for optimization.
        teacher: The teacher program to use for optimization.
        valset: The validation set to use for optimization.

    Returns:
        The optimized student program.
    """
    raise NotImplementedError
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

One of the key insights behind GEPA is its ability to leverage domain-specific textual feedback. Users should provide a feedback function as the GEPA metric, which has the following call signature:

## `dspy.teleprompt.gepa.gepa.GEPAFeedbackMetric`

Bases: `Protocol`

### Methods:

#### `__call__(gold: Example, pred: Prediction, trace: Optional[DSPyTrace], pred_name: str | None, pred_trace: Optional[DSPyTrace], program_trace: Optional[DSPyTrace] = None) -> Union[float, ScoreWithFeedback]`

This function is called with the following arguments: - gold: The gold example. - pred: The predicted output. - trace: Optional. The trace of the program’s execution. - pred_name: Optional. The name of the target predictor currently being optimized by GEPA, for which the feedback is being requested. - pred_trace: Optional. The trace of the target predictor’s execution GEPA is seeking feedback for. - program_trace: Optional. The full execution trace of the program, supplied at scoring time when a `dspy.Flex` submodule is being optimized. Declare this parameter to score against how an answer was produced (e.g. `len(program_trace)` as an LM-call count), rather than only whether it was correct. Unlike `trace`, it is populated during candidate *scoring*.

Note the `pred_name` and `pred_trace` arguments. During optimization, GEPA will call the metric to obtain feedback for individual predictors being optimized. GEPA provides the name of the predictor in `pred_name` and the sub-trace (of the trace) corresponding to the predictor in `pred_trace`. If available at the predictor level, the metric should return dspy.Prediction(score: float, feedback: str) corresponding to the predictor. If not available at the predictor level, the metric can also return a text feedback at the program level (using just the gold, pred and trace). If no feedback is returned, GEPA will use a simple text feedback consisting of just the score: f”This trajectory got a score of {score}.”

Source code in `dspy/teleprompt/gepa/gepa.py`

```
def __call__(
    self,
    gold: Example,
    pred: Prediction,
    trace: Optional["DSPyTrace"],
    pred_name: str | None,
    pred_trace: Optional["DSPyTrace"],
    program_trace: Optional["DSPyTrace"] = None,
) -> Union[float, "ScoreWithFeedback"]:
    """
    This function is called with the following arguments:
    - gold: The gold example.
    - pred: The predicted output.
    - trace: Optional. The trace of the program's execution.
    - pred_name: Optional. The name of the target predictor currently being optimized by GEPA, for which
        the feedback is being requested.
    - pred_trace: Optional. The trace of the target predictor's execution GEPA is seeking feedback for.
    - program_trace: Optional. The full execution trace of the program, supplied at scoring time when a
        `dspy.Flex` submodule is being optimized. Declare this parameter to score against how an answer was
        produced (e.g. `len(program_trace)` as an LM-call count), rather than only whether it was correct.
        Unlike `trace`, it is populated during candidate *scoring*.

    Note the `pred_name` and `pred_trace` arguments. During optimization, GEPA will call the metric to obtain
    feedback for individual predictors being optimized. GEPA provides the name of the predictor in `pred_name`
    and the sub-trace (of the trace) corresponding to the predictor in `pred_trace`.
    If available at the predictor level, the metric should return dspy.Prediction(score: float, feedback: str)
    corresponding to the predictor.
    If not available at the predictor level, the metric can also return a text feedback at the program level
    (using just the gold, pred and trace).
    If no feedback is returned, GEPA will use a simple text feedback consisting of just the score:
    f"This trajectory got a score of {score}."
    """
    ...
```

When `track_stats=True`, GEPA returns detailed results about all of the proposed candidates, and metadata about the optimization run. The results are available in the `detailed_results` attribute of the optimized program returned by GEPA, and has the following type:

## `dspy.teleprompt.gepa.gepa.DspyGEPAResult(candidates: list[Module], parents: list[list[int | None]], val_aggregate_scores: list[float], val_subscores: list[dict[Any, float]], per_val_instance_best_candidates: dict[Any, set[int]], discovery_eval_counts: list[int], best_outputs_valset: dict[Any, list[tuple[int, Prediction]]] | None = None, total_metric_calls: int | None = None, num_full_val_evals: int | None = None, log_dir: str | None = None, seed: int | None = None, val_aggregate_subscores: list[dict[str, float]] | None = None, per_objective_best_candidates: dict[str, set[int]] | None = None, objective_pareto_front: dict[str, float] | None = None)`

Additional data related to the GEPA run.

Fields: - candidates: list of proposed candidates (compiled DSPy modules) - parents: lineage info; for each candidate i, parents[i] is a list of parent indices or None - val_aggregate_scores: per-candidate aggregate score on the validation set (higher is better) - val_subscores: per-candidate scores keyed by validation instance id - per_val_instance_best_candidates: for each val instance id, a set of candidate indices achieving the best score - discovery_eval_counts: Budget (number of metric calls / rollouts) consumed up to the discovery of each candidate - val_aggregate_subscores: per-candidate objective scores aggregated over the validation set - per_objective_best_candidates: candidate indices achieving the best score for each objective - objective_pareto_front: highest score achieved for each objective

- total_metric_calls: total number of metric calls made across the run
- num_full_val_evals: number of full validation evaluations performed
- log_dir: where artifacts were written (if any)
- seed: RNG seed for reproducibility (if known)
- best_idx: candidate index with the highest val_aggregate_scores
- best_candidate: the compiled DSPy module for best_idx

### Attributes

#### `candidates: list[Module]`

#### `parents: list[list[int | None]]`

#### `val_aggregate_scores: list[float]`

#### `val_subscores: list[dict[Any, float]]`

#### `per_val_instance_best_candidates: dict[Any, set[int]]`

#### `discovery_eval_counts: list[int]`

#### `best_outputs_valset: dict[Any, list[tuple[int, Prediction]]] | None = None`

#### `total_metric_calls: int | None = None`

#### `num_full_val_evals: int | None = None`

#### `log_dir: str | None = None`

#### `seed: int | None = None`

#### `val_aggregate_subscores: list[dict[str, float]] | None = None`

#### `per_objective_best_candidates: dict[str, set[int]] | None = None`

#### `objective_pareto_front: dict[str, float] | None = None`

#### `best_idx: int`

#### `best_candidate: Module`

#### `highest_score_achieved_per_val_task: dict[Any, float]`

### Methods:

#### `to_dict() -> dict[str, Any]`

Source code in `dspy/teleprompt/gepa/gepa.py`

```
def to_dict(self) -> dict[str, Any]:
    cands = [self._candidate_components(cand) for cand in self.candidates]

    return dict(
        candidates=cands,
        parents=self.parents,
        val_aggregate_scores=self.val_aggregate_scores,
        best_outputs_valset=self.best_outputs_valset,
        val_subscores=self.val_subscores,
        per_val_instance_best_candidates={
            val_id: list(s) for val_id, s in self.per_val_instance_best_candidates.items()
        },
        val_aggregate_subscores=self.val_aggregate_subscores,
        per_objective_best_candidates=(
            {objective: list(s) for objective, s in self.per_objective_best_candidates.items()}
            if self.per_objective_best_candidates is not None
            else None
        ),
        objective_pareto_front=self.objective_pareto_front,
        discovery_eval_counts=self.discovery_eval_counts,
        total_metric_calls=self.total_metric_calls,
        num_full_val_evals=self.num_full_val_evals,
        log_dir=self.log_dir,
        seed=self.seed,
        best_idx=self.best_idx,
    )
```

#### `from_gepa_result(gepa_result: GEPAResult, adapter: DspyAdapter) -> DspyGEPAResult`

Source code in `dspy/teleprompt/gepa/gepa.py`

```
@staticmethod
def from_gepa_result(gepa_result: "GEPAResult", adapter: "DspyAdapter") -> "DspyGEPAResult":
    return DspyGEPAResult(
        candidates=[adapter.build_program(c) for c in gepa_result.candidates],
        parents=gepa_result.parents,
        val_aggregate_scores=gepa_result.val_aggregate_scores,
        best_outputs_valset=gepa_result.best_outputs_valset,
        val_subscores=gepa_result.val_subscores,
        per_val_instance_best_candidates=gepa_result.per_val_instance_best_candidates,
        discovery_eval_counts=gepa_result.discovery_eval_counts,
        val_aggregate_subscores=gepa_result.val_aggregate_subscores,
        per_objective_best_candidates=gepa_result.per_objective_best_candidates,
        objective_pareto_front=gepa_result.objective_pareto_front,
        total_metric_calls=gepa_result.total_metric_calls,
        num_full_val_evals=gepa_result.num_full_val_evals,
        log_dir=gepa_result.run_dir,
        seed=gepa_result.seed,
    )
```

## Usage Examples

See GEPA usage tutorials in [GEPA Tutorials](https://dspy.ai/current/tutorials/gepa_ai_program/index.md).

### Inference-Time Search

GEPA can act as a test-time/inference search mechanism. By setting your `valset` to your *evaluation batch* and using `track_best_outputs=True`, GEPA produces for each batch element the highest-scoring outputs found during the evolutionary search.

```
gepa = dspy.GEPA(metric=metric, track_stats=True, ...)
new_prog = gepa.compile(student, trainset=my_tasks, valset=my_tasks)
highest_score_achieved_per_task = new_prog.detailed_results.highest_score_achieved_per_val_task
best_outputs = new_prog.detailed_results.best_outputs_valset
```

### Objective-Aware Frontier Tracking

Metrics can return named `objective_scores` alongside the scalar score and feedback:

```
def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    quality = measure_quality(gold, pred)
    privacy = measure_privacy(gold, pred)
    return dspy.Prediction(
        score=(quality + privacy) / 2,
        objective_scores={"quality": quality, "privacy": privacy},
        feedback="...",
    )

gepa = dspy.GEPA(metric=metric, gepa_kwargs={"frontier_type": "objective"}, track_stats=True)
```

`frontier_type` can be `"instance"` (the default), `"objective"`, `"hybrid"` (both instance and objective), or `"cartesian"` (each validation-instance/objective pair). The scalar score still gates acceptance and determines the final candidate; objectives only affect parent and merge selection. Each objective is maximized and averaged over examples that report it. Predictor-level objective scores are ignored. `objective_pareto_front` contains the independent maximum for each objective, not nondominated objective vectors.

## How Does GEPA Work?

### 1. **Reflective Prompt Mutation**

GEPA uses LLMs to *reflect* on structured execution traces (inputs, outputs, failures, feedback), targeting a chosen module and proposing a new instruction/program text tailored to real observed failures and rich textual/environmental feedback.

### 2. **Rich Textual Feedback as Optimization Signal**

GEPA can leverage *any* textual feedback available—not just scalar rewards. This includes evaluation logs, code traces, failed parses, constraint violations, error message strings, or even isolated submodule-specific feedback. This allows actionable, domain-aware optimization.

### 3. **Pareto-based Candidate Selection**

Rather than evolving just the *best* global candidate (which leads to local optima or stagnation), GEPA maintains a Pareto frontier: the set of candidates which achieve the highest score on at least one evaluation instance. In each iteration, the next candidate to mutate is sampled (with probability proportional to coverage) from this frontier, guaranteeing both exploration and robust retention of complementary strategies.

### Algorithm Summary

1. **Initialize** the candidate pool with the unoptimized program.
1. **Iterate**:
1. **Sample a candidate** (from Pareto frontier).
1. **Sample a minibatch** from the train set.
1. **Collect execution traces + feedbacks** for module rollout on minibatch.
1. **Select a module** of the candidate for targeted improvement.
1. **LLM Reflection:** Propose a new instruction/prompt for the targeted module using reflective meta-prompting and the gathered feedback.
1. **Roll out the new candidate** on the minibatch; **if improved, evaluate on Pareto validation set**.
1. **Update the candidate pool/Pareto frontier.**
1. **[Optionally] System-aware merge/crossover**: Combine best-performing modules from distinct lineages.
1. **Continue** until rollout or metric budget is exhausted.
1. **Return** candidate with best aggregate performance on validation.

## Implementing Feedback Metrics

A well-designed metric is central to GEPA’s sample efficiency and learning signal richness. GEPA expects the metric to returns a `dspy.Prediction(score=..., feedback=...)`. GEPA leverages natural language traces from LLM-based workflows for optimization, preserving intermediate trajectories and errors in plain text rather than reducing them to numerical rewards. This mirrors human diagnostic processes, enabling clearer identification of system behaviors and bottlenecks.

Practical Recipe for GEPA-Friendly Feedback:

- **Leverage Existing Artifacts**: Use logs, unit tests, evaluation scripts, and profiler outputs; surfacing these often suffices.
- **Decompose Outcomes**: Break scores into per-objective components (e.g., correctness, latency, cost, safety) and attribute errors to steps.
- **Expose Trajectories**: Label pipeline stages, reporting pass/fail with salient errors (e.g., in code generation pipelines).
- **Ground in Checks**: Employ automatic validators (unit tests, schemas, simulators) or LLM-as-a-judge for non-verifiable tasks (as in PUPA).
- **Prioritize Clarity**: Focus on error coverage and decision points over technical complexity.

### Examples

- **Document Retrieval** (e.g., HotpotQA): List correctly retrieved, incorrect, or missed documents, beyond mere Recall/F1 scores.
- **Multi-Objective Tasks** (e.g., PUPA): Decompose aggregate scores to reveal contributions from each objective, highlighting tradeoffs (e.g., quality vs. privacy).
- **Stacked Pipelines** (e.g., code generation: parse → compile → run → profile → evaluate): Expose stage-specific failures; natural-language traces often suffice for LLM self-correction.

## Custom Instruction Proposal

For advanced customization of GEPA’s instruction proposal mechanism, including custom instruction proposers and component selectors, see [Advanced Features](https://dspy.ai/current/api/optimizers/GEPA/GEPA_Advanced/index.md).

## Further Reading

- [GEPA Paper: arxiv:2507.19457](https://arxiv.org/abs/2507.19457)

- [GEPA Github](https://github.com/gepa-ai/gepa) - This repository provides the core GEPA evolution pipeline used by `dspy.GEPA` optimizer.

- [DSPy Tutorials](https://dspy.ai/current/tutorials/gepa_ai_program/index.md)

  August 21, 2026
