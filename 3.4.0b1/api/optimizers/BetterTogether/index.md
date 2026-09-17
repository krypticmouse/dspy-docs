# dspy.BetterTogether

**BetterTogether** is a meta-optimizer proposed in the paper [Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together](https://arxiv.org/abs/2407.10930) by Dilara Soylu, Christopher Potts, and Omar Khattab. It combines prompt optimization and weight optimization (fine-tuning) by applying them in a configurable sequence, allowing a student program to iteratively improve both its prompts and model parameters. The core insight is that prompt and weight optimization can complement each other: prompt optimization can potentially discover effective task decompositions and reasoning strategies, while weight optimization can specialize the model to execute these patterns more efficiently. Using these approaches together in sequences (e.g., prompt optimization then weight optimization) may allow each to build on the improvements made by the other.

## `dspy.BetterTogether(metric: Callable, **optimizers: Teleprompter)`

Bases: `Teleprompter`

A meta-optimizer that combines prompt and weight optimization in configurable sequences.

BetterTogether is a meta-optimizer proposed in the paper [Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together](https://arxiv.org/abs/2407.10930). It combines prompt optimization and weight optimization (fine-tuning) by applying them in a configurable sequence, allowing a student program to iteratively improve both its prompts and model parameters.

The core insight is that prompt and weight optimization can complement each other: prompt optimization can potentially discover effective task decompositions and reasoning strategies, while weight optimization can specialize the model to execute these patterns more efficiently. Using these approaches together in sequences (e.g., prompt optimization then weight optimization) may allow each to build on the improvements made by the other. Empirically, this approach often outperforms either strategy alone, even with state-of-the-art optimizers. For example, a [Databricks case study](https://www.databricks.com/blog/building-state-art-enterprise-agents-90x-cheaper-automated-prompt-optimization) shows that combining BetterTogether with GEPA and fine-tuning outperforms either approach alone.

The optimizer is initialized with a metric and custom optimizers. For example, you can combine GEPA for prompt optimization with BootstrapFinetune for weight optimization: `BetterTogether(metric=metric, p=GEPA(...), w=BootstrapFinetune(...))`. The `compile()` method takes a student program, trainset, and strategy string where strategy keys correspond to the optimizer names from initialization. It executes each optimizer in the specified sequence. When a validation set is provided, the best performing program is returned; otherwise, the latest program is returned. Note: Weight optimizers like BootstrapFinetune require student programs to have LMs explicitly set (not relying on global dspy.settings.lm), and BetterTogether mirrors this requirement for simplicity. Therefore we call `set_lm` before compiling.

```
>>> from dspy.teleprompt import GEPA, BootstrapFinetune
>>>
>>> # Combine GEPA for prompt optimization with BootstrapFinetune for weight optimization
>>> optimizer = BetterTogether(
...     metric=metric,
...     p=GEPA(metric=metric, auto="medium"),
...     w=BootstrapFinetune(metric=metric)
... )
>>>
>>> student.set_lm(lm)
>>> compiled = optimizer.compile(
...     student,
...     trainset=trainset,
...     valset=valset,
...     strategy="p -> w"
... )
```

You can pass optimizer-specific arguments to each optimizer’s `compile()` method using `optimizer_compile_args`. This allows you to customize each optimizer’s behavior:

```
>>> from dspy.teleprompt import MIPROv2
>>>
>>> # Use MIPROv2 for prompt optimization with custom parameters
>>> optimizer = BetterTogether(
...     metric=metric,
...     p=MIPROv2(metric=metric),
...     w=BootstrapFinetune(metric=metric)
... )
>>>
>>> student.set_lm(lm)
>>> compiled = optimizer.compile(
...     student,
...     trainset=trainset,
...     valset=valset,
...     strategy="p -> w",
...     optimizer_compile_args={
...         "p": {"num_trials": 10, "max_bootstrapped_demos": 8},  # Configure MIPROv2's compile arguments
...     }
... )
```

Since BetterTogether is a meta-optimizer that can run arbitrary optimizers in sequence, any sequence of optimizers can be combined together. The optimizer names used in the strategy string correspond to the keyword arguments specified in the constructor. For example, different prompt optimizers can be alternated multiple times (though note this is just an illustration of BetterTogether’s flexibility, not a recommended configuration):

```
>>> from dspy.teleprompt import MIPROv2, GEPA
>>>
>>> # Chain two optimizers three times: MIPROv2 -> GEPA -> MIPROv2
>>> optimizer = BetterTogether(
...     metric=metric,
...     mipro=MIPROv2(metric=metric, auto="light"),
...     gepa=GEPA(metric=metric, auto="light")
... )
>>>
>>> student.set_lm(lm)
>>> compiled = optimizer.compile(
...     student,
...     trainset=trainset,
...     valset=valset,
...     strategy="mipro -> gepa -> mipro"
... )
```

Note

Output Attributes: The returned program includes two additional attributes: `candidate_programs` and `flag_compilation_error_occurred`. The `candidate_programs` attribute is a list of dicts, each containing ‘program’, ‘score’, and ‘strategy’ (e.g., ‘’, ‘p’, ‘p -> w’, ‘p -> w -> p’), sorted by descending score (similar to `dspy.MIPROv2.candidate_programs`). If any optimizer step fails, `flag_compilation_error_occurred` is set to True and the best program found so far is returned.

Model Lifecycle Management: BetterTogether automatically manages language model lifecycle (launching, killing, and relaunching after fine-tuning), which are no-ops for most API-based LMs. This is particularly important when using weight optimizers like BootstrapFinetune with local providers (e.g., dspy.LocalProvider), as it handles model initialization and cleanup between optimization steps.

Initialize BetterTogether with a metric and custom optimizers.

Parameters:

| Name           | Type           | Description                                                                                                                                                                                                                                                                                                                                                                              | Default    |
| -------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `metric`       | `Callable`     | Evaluation metric function for scoring programs. Should accept (example, prediction, trace=None) and return a numeric score (higher is better). This metric is used to evaluate candidate programs during optimization and is passed to the default optimizers if no custom optimizers are provided.                                                                                     | *required* |
| `**optimizers` | `Teleprompter` | Custom optimizers as keyword arguments, where keys become the optimizer names used in the strategy string. For example, p=GEPA(...), w=BootstrapFinetune(...) makes ‘p’ and ‘w’ available for use in strategies like "p -> w". If not provided, defaults to p=BootstrapFewShotWithRandomSearch(metric=metric) and w=BootstrapFinetune(metric=metric). Any DSPy Teleprompter can be used. | `{}`       |

Examples:

```
>>> # Use custom optimizers
>>> from dspy.teleprompt import GEPA, BootstrapFinetune
>>> optimizer = BetterTogether(
...     metric=metric,
...     p=GEPA(metric=metric, auto="medium"),
...     w=BootstrapFinetune(metric=metric)
... )
>>>
>>> # Use default optimizers
>>> optimizer = BetterTogether(metric=metric)
```

Source code in `dspy/teleprompt/bettertogether.py`

```
def __init__(
    self,
    metric: Callable,
    **optimizers: Teleprompter,
):
    """Initialize BetterTogether with a metric and custom optimizers.

    Args:
        metric: Evaluation metric function for scoring programs. Should accept
            ``(example, prediction, trace=None)`` and return a numeric score (higher is better).
            This metric is used to evaluate candidate programs during optimization and is passed
            to the default optimizers if no custom optimizers are provided.
        **optimizers: Custom optimizers as keyword arguments, where keys become the optimizer
            names used in the strategy string. For example, ``p=GEPA(...), w=BootstrapFinetune(...)``
            makes 'p' and 'w' available for use in strategies like ``"p -> w"``. If not provided,
            defaults to ``p=BootstrapFewShotWithRandomSearch(metric=metric)`` and
            ``w=BootstrapFinetune(metric=metric)``. Any DSPy Teleprompter can be used.

    Examples:
        >>> # Use custom optimizers
        >>> from dspy.teleprompt import GEPA, BootstrapFinetune
        >>> optimizer = BetterTogether(
        ...     metric=metric,
        ...     p=GEPA(metric=metric, auto="medium"),
        ...     w=BootstrapFinetune(metric=metric)
        ... )
        >>>
        >>> # Use default optimizers
        >>> optimizer = BetterTogether(metric=metric)
    """
    self.metric = metric

    if not optimizers:
        logger.info(
            "No optimizers provided. Using defaults: "
            "BootstrapFewShotWithRandomSearch (p) and BootstrapFinetune (w). "
            "You can use the letters p and w to specify the compile strategy. "
            "For example, to run weight optimization after prompt optimization, use strategy='p -> w'."
        )
        optimizers = {
            "p": BootstrapFewShotWithRandomSearch(metric=metric),
            "w": BootstrapFinetune(metric=metric),
        }
    for key, optimizer in optimizers.items():
        if not isinstance(optimizer, Teleprompter):
            raise TypeError(
                f"Optimizer '{key}' must be a Teleprompter, "
                f"got {type(optimizer).__name__}"
            )
    self.optimizers: dict[str, Teleprompter] = optimizers
```

### Methods:

#### `compile(student: Module, *, trainset: list[Example], teacher: Module | list[Module] | None = None, valset: list[Example] | None = None, num_threads: int | None = None, max_errors: int | None = None, provide_traceback: bool | None = None, seed: int | None = None, valset_ratio: float = 0.1, shuffle_trainset_between_steps: bool = True, strategy: str = 'p -> w -> p', optimizer_compile_args: dict[str, dict[str, Any]] | None = None) -> Module`

Compile and optimize a student program using a sequence of optimization strategies.

Executes the optimizers specified in the strategy string sequentially, evaluating each intermediate result and returning the best performing program.

Parameters:

| Name                             | Type                          | Description                                                                                                                                                                                                                                                            | Default                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| -------------------------------- | ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `student`                        | `Module`                      | DSPy program to optimize. All predictors must have language models assigned. program.set_lm(lm) can be used to assign a language model to all modules of a program.                                                                                                    | *required*                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `trainset`                       | `list[Example]`               | Training examples for optimization. Each optimizer receives the full trainset (or a shuffled version if shuffle_trainset_between_steps=True).                                                                                                                          | *required*                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `teacher`                        | \`Module                      | list[Module]                                                                                                                                                                                                                                                           | None\`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `valset`                         | \`list[Example]               | None\`                                                                                                                                                                                                                                                                 | Validation set for evaluating optimization steps. If not provided, a portion of trainset is held out (controlled by valset_ratio). If both valset and valset_ratio are None/0, no validation occurs and the latest program is returned.                                                                                                                                                                                                                                                                                                   |
| `num_threads`                    | \`int                         | None\`                                                                                                                                                                                                                                                                 | Number of parallel evaluation threads. Default is None, which means sequential evaluation.                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `max_errors`                     | \`int                         | None\`                                                                                                                                                                                                                                                                 | Maximum errors to tolerate during evaluation. Defaults to dspy.settings.max_errors.                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `provide_traceback`              | \`bool                        | None\`                                                                                                                                                                                                                                                                 | Whether to show detailed tracebacks for evaluation errors.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `seed`                           | \`int                         | None\`                                                                                                                                                                                                                                                                 | Random seed for reproducibility. Controls trainset shuffling and evaluation sampling.                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `valset_ratio`                   | `float`                       | Fraction of trainset to hold out as validation (range \[0, 1)). For example, 0.1 holds out 10%. Set to 0 to skip validation. Default is 0.1.                                                                                                                           | `0.1`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `shuffle_trainset_between_steps` | `bool`                        | Whether to shuffle trainset before each optimization step. Helps prevent overfitting to example ordering. Default is True.                                                                                                                                             | `True`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `strategy`                       | `str`                         | Sequence of optimizers to apply, separated by " -> ". Each element must be a key from the optimizers provided in __init__. For example, "p -> w -> p" applies prompt optimization, then weight optimization, then prompt optimization again. Default is "p -> w -> p". | `'p -> w -> p'`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `optimizer_compile_args`         | \`dict\[str, dict[str, Any]\] | None\`                                                                                                                                                                                                                                                                 | Optional dict mapping optimizer keys to their compile() arguments. If trainset, valset, or teacher are provided in the dict for a specific optimizer, they override the defaults from BetterTogether’s compile method. For example: {"p": {"num_trials": 10}, "w": {"trainset": custom_trainset}}. This is useful to override the default compile arguments for specific optimizers. The student argument cannot be included in optimizer_compile_args; BetterTogether’s compile method manages the student reference for all optimizers. |

Returns:

| Type     | Description                                                                                                                                                           |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Module` | Optimized student program with two additional attributes:                                                                                                             |
| `Module` | candidate_programs: List of dicts with ‘program’, ‘score’, and ‘strategy’ keys, sorted by score (best first). Contains all evaluated programs including the baseline. |
| `Module` | flag_compilation_error_occurred: Boolean indicating if any optimization step failed.                                                                                  |

Raises:

| Type         | Description                                                                                                                                                    |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If trainset is empty, valset_ratio not in \[0, 1), strategy is empty or contains invalid optimizer keys, or optimizer_compile_args contains invalid arguments. |
| `TypeError`  | If optimizer_compile_args contains a ‘student’ key (not allowed).                                                                                              |

Examples:

```
>>> optimizer = BetterTogether(
...     metric=metric,
...     p=GEPA(metric=metric),
...     w=BootstrapFinetune(metric=metric)
... )
>>> student.set_lm(lm)
>>> compiled = optimizer.compile(
...     student,
...     trainset=trainset,
...     valset=valset,
...     strategy="p -> w"
... )
>>> print(f"Best score: {compiled.candidate_programs[0]['score']}")
```

Source code in `dspy/teleprompt/bettertogether.py`

```
def compile(
    self,
    student: Module,
    *,
    trainset: list[Example],
    teacher: Module | list[Module] | None = None,
    valset: list[Example] | None = None,
    # often specified in init in other optimizers
    num_threads: int | None = None,
    max_errors: int | None = None,
    provide_traceback: bool | None = None,
    seed: int | None = None,
    # specific to BetterTogether
    valset_ratio: float = 0.1,
    shuffle_trainset_between_steps: bool = True,
    strategy: str = "p -> w -> p",
    optimizer_compile_args: dict[str, dict[str, Any]] | None = None,
) -> Module:
    """Compile and optimize a student program using a sequence of optimization strategies.

    Executes the optimizers specified in the strategy string sequentially, evaluating each
    intermediate result and returning the best performing program.

    Args:
        student: DSPy program to optimize. All predictors must have language models assigned.
            program.set_lm(lm) can be used to assign a language model to all modules of a 
            program.
        trainset: Training examples for optimization. Each optimizer receives the full trainset
            (or a shuffled version if ``shuffle_trainset_between_steps=True``).
        teacher: Optional teacher module(s) for bootstrapping. Can be a single module or list.
            Passed to optimizers.
        valset: Validation set for evaluating optimization steps. If not provided, a portion of
            trainset is held out (controlled by ``valset_ratio``). If both ``valset`` and
            ``valset_ratio`` are None/0, no validation occurs and the latest program is returned.
        num_threads: Number of parallel evaluation threads. Default is None, which means sequential evaluation.
        max_errors: Maximum errors to tolerate during evaluation. Defaults to
            ``dspy.settings.max_errors``.
        provide_traceback: Whether to show detailed tracebacks for evaluation errors.
        seed: Random seed for reproducibility. Controls trainset shuffling and evaluation sampling.
        valset_ratio: Fraction of trainset to hold out as validation (range [0, 1)). For example,
            0.1 holds out 10%. Set to 0 to skip validation. Default is 0.1.
        shuffle_trainset_between_steps: Whether to shuffle trainset before each optimization step.
            Helps prevent overfitting to example ordering. Default is True.
        strategy: Sequence of optimizers to apply, separated by ``" -> "``. Each element must be
            a key from the optimizers provided in ``__init__``. For example, ``"p -> w -> p"``
            applies prompt optimization, then weight optimization, then prompt optimization again.
            Default is ``"p -> w -> p"``.
        optimizer_compile_args: Optional dict mapping optimizer keys to their ``compile()``
            arguments. If trainset, valset, or teacher are provided in the dict for a specific
            optimizer, they override the defaults from BetterTogether's compile method. For example:
            ``{"p": {"num_trials": 10}, "w": {"trainset": custom_trainset}}``. This is useful to
            override the default compile arguments for specific optimizers. The ``student`` argument
            cannot be included in optimizer_compile_args; BetterTogether's compile method manages
            the student reference for all optimizers.

    Returns:
        Optimized student program with two additional attributes:

        - ``candidate_programs``: List of dicts with 'program', 'score', and 'strategy' keys,
          sorted by score (best first). Contains all evaluated programs including the baseline.
        - ``flag_compilation_error_occurred``: Boolean indicating if any optimization step failed.

    Raises:
        ValueError: If trainset is empty, valset_ratio not in [0, 1), strategy is empty or
            contains invalid optimizer keys, or optimizer_compile_args contains invalid arguments.
        TypeError: If optimizer_compile_args contains a 'student' key (not allowed).

    Examples:
        >>> optimizer = BetterTogether(
        ...     metric=metric,
        ...     p=GEPA(metric=metric),
        ...     w=BootstrapFinetune(metric=metric)
        ... )
        >>> student.set_lm(lm)
        >>> compiled = optimizer.compile(
        ...     student,
        ...     trainset=trainset,
        ...     valset=valset,
        ...     strategy="p -> w"
        ... )
        >>> print(f"Best score: {compiled.candidate_programs[0]['score']}")
    """
    logger.info(f"\n{BOLD}==> BETTERTOGETHER COMPILATION STARTED <=={ENDC}")
    logger.info(f"{BLUE}Strategy:{ENDC} {strategy}")
    logger.info(f"{BLUE}Trainset size:{ENDC} {len(trainset)}")
    logger.info(f"{BLUE}Validation ratio:{ENDC} {valset_ratio if valset is None else 'using provided valset'}")

    student, teacher = self._prepare_student_and_teacher(student, teacher)
    trainset, valset = self._prepare_trainset_and_valset(trainset, valset, valset_ratio)
    effective_max_errors = max_errors if max_errors is not None else dspy.settings.max_errors
    parsed_strategy = self._prepare_strategy(strategy)
    optimizer_compile_args = self._prepare_optimizer_compile_args(optimizer_compile_args, teacher)

    student = self._run_strategies(
        student,
        trainset,
        teacher,
        valset,
        num_threads,
        effective_max_errors,
        provide_traceback,
        seed,
        parsed_strategy,
        shuffle_trainset_between_steps,
        optimizer_compile_args,
    )

    logger.info(f"\n{BOLD}{GREEN}==> BETTERTOGETHER COMPILATION COMPLETE <=={ENDC}")
    logger.info(f"{GREEN}Best score achieved:{ENDC} {student.candidate_programs[0]['score']}")
    logger.info(f"{GREEN}Best strategy:{ENDC} {student.candidate_programs[0]['strategy'] or 'original (no optimization)'}")

    student._compiled = True
    return student
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

## How BetterTogether Works

BetterTogether executes optimizers in a configurable sequence, evaluating each intermediate result and returning the best performing program. Here’s how it works:

### 1. **Initialization with Custom Optimizers**

When initialized, BetterTogether accepts any DSPy optimizers (Teleprompters) as keyword arguments. The keys become the optimizer names used in the strategy string:

```
optimizer = BetterTogether(
    metric=metric,
    p=GEPA(...),           # 'p' can be used in strategy
    w=BootstrapFinetune(...) # 'w' can be used in strategy
)
```

If no optimizers are provided, BetterTogether defaults to `BootstrapFewShotWithRandomSearch` (key: ‘p’) and `BootstrapFinetune` (key: ‘w’).

### 2. **Strategy Execution**

The strategy string defines the sequence of optimizers to apply. For example:

```
compiled = optimizer.compile(
    student,
    trainset=trainset,
    valset=valset,
    strategy="p -> w -> p"
)
```

This strategy `"p -> w -> p"` means:

1. Run prompt optimizer (‘p’)
1. Run weight optimizer (‘w’) on the result
1. Run prompt optimizer (‘p’) again on the result

At each step:

- The trainset is shuffled (if `shuffle_trainset_between_steps=True`)
- The optimizer is run on the current student program
- The result is evaluated on the validation set
- The candidate program and score are recorded

Since BetterTogether is a meta-optimizer, any sequence of optimizers can be combined. The optimizer names in the strategy string correspond to the keyword arguments from initialization. For example, you can sequence different prompt optimizers (note: this illustrates BetterTogether’s flexibility, not necessarily a recommended configuration):

```
optimizer = BetterTogether(
    metric=metric,
    mipro=MIPROv2(metric=metric, auto="light"),
    gepa=GEPA(metric=metric, auto="light")
)

compiled = optimizer.compile(
    student,
    trainset=trainset,
    valset=valset,
    strategy="mipro -> gepa -> mipro"
)
```

### 3. **Validation and Program Selection**

BetterTogether can use a validation set in three ways:

- **Explicit valset**: If `valset` is provided, it’s used for evaluation
- **Auto-split**: If `valset_ratio > 0`, a portion of trainset is held out for validation
- **No validation**: If both `valset` and `valset_ratio` are None/0, no validation occurs

After all optimization steps complete, the best program is selected based on validation set availability:

- **With validation**: The program with the **best score** is returned (with earlier programs winning ties)
- **Without validation**: The **latest program** is returned

If an optimization step fails:

- The error is logged with full traceback
- Optimization stops early
- The best program found so far is returned
- `flag_compilation_error_occurred` is set to `True`

The returned program includes two additional attributes:

- `candidate_programs`: List of all evaluated programs with their scores and strategies, sorted by score (best first). When an error occurs, this contains all successfully evaluated programs up to the point of failure.
- `flag_compilation_error_occurred`: Boolean indicating if any optimization step failed during compilation.

### 4. **Further Details**

**Model Lifecycle Management**: For local models (like LocalLM), BetterTogether automatically launches models before first use, kills them after optimization completes, and relaunches them after fine-tuning when model names change. These operations are no-ops for API-based LMs but needed for local model serving.

**Custom Compile Arguments**: You can pass custom compile arguments to specific optimizers using the `optimizer_compile_args` parameter:

- **Override default arguments**: Pass custom trainset/valset/teacher to specific optimizers
- **Customize per optimizer**: Each optimizer can have different compile arguments (e.g., `num_trials`, `max_bootstrapped_demos`)

Note: The `student` argument cannot be included in `optimizer_compile_args` - BetterTogether manages the student program for all optimizers. See the `compile()` method docstring for detailed argument documentation.

## Best Practices

### When to Use BetterTogether

BetterTogether is the right optimizer when:

- **You want to squeeze every bit of performance**: Prompt optimization is often the best bang for buck, quickly discovering high-level strategies. When opportunity allows, adding weight optimization on top compounds these gains, yielding benefits that exceed either approach alone.
- **You have fine-tuning capabilities**: Weight optimizers like `BootstrapFinetune` require LMs with a fine-tuning interface. Currently supported: `LocalProvider`, `DatabricksProvider`, and `OpenAIProvider`. You can extend the `Provider` class for custom use cases, or use BetterTogether to combine prompt optimizers only.

The [Databricks case study](https://www.databricks.com/blog/building-state-art-enterprise-agents-90x-cheaper-automated-prompt-optimization) demonstrates this effectiveness. They evaluated on IE Bench—a comprehensive suite spanning enterprise domains (finance, legal, commerce, healthcare) with complex challenges: 100+ page documents, 70+ extraction fields, and hierarchical schemas. Using GPT-4.1:

- **SFT alone**: +1.9 points over baseline
- **GEPA alone**: +2.1 points over baseline (slightly exceeding SFT)
- **GEPA + SFT (BetterTogether)**: +4.8 points over baseline

This demonstrates that prompt optimization can match or surpass supervised fine-tuning, and combining these techniques yields strong compounding benefits.

### Common Strategies and Optimizers

Common strategies:

- `"p -> w"`: Optimize prompts first, then fine-tune (simple and often effective)
- `"p -> w -> p"`: Optimize prompts, fine-tune, then optimize prompts again (can build on fine-tuning improvements)
- `"w -> p"`: Fine-tune first, then optimize prompts

Example optimizer combinations:

- **GEPA + BootstrapFinetune**: Prompt optimization with fine-tuning
- **MIPROv2 + BootstrapFinetune**: Prompt optimization with fine-tuning
- **Multiple prompt optimizers**: Alternate between different prompt optimization approaches (experimental)

## Further Reading

- [BetterTogether Paper: arxiv:2407.10930](https://arxiv.org/abs/2407.10930)
- [Databricks Case Study](https://www.databricks.com/blog/building-state-art-enterprise-agents-90x-cheaper-automated-prompt-optimization) - Real-world application combining BetterTogether with GEPA
- [Optimizers: choosing one](https://dspy.ai/3.4.0b1/diving-deeper/choosing-an-optimizer/index.md)
