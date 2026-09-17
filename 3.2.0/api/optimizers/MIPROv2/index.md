# dspy.MIPROv2

`MIPROv2` (Multiprompt Instruction PRoposal Optimizer Version 2) is an prompt optimizer capable of optimizing both instructions and few-shot examples jointly. It does this by bootstrapping few-shot example candidates, proposing instructions grounded in different dynamics of the task, and finding an optimized combination of these options using Bayesian Optimization. It can be used for optimizing few-shot examples & instructions jointly, or just instructions for 0-shot optimization.

## `dspy.MIPROv2(metric: Callable, prompt_model: Any | None = None, task_model: Any | None = None, teacher_settings: dict | None = None, max_bootstrapped_demos: int = 4, max_labeled_demos: int = 4, auto: Literal['light', 'medium', 'heavy'] | None = 'light', num_candidates: int | None = None, num_threads: int | None = None, max_errors: int | None = None, seed: int = 9, init_temperature: float = 1.0, verbose: bool = False, track_stats: bool = True, log_dir: str | None = None, metric_threshold: float | None = None)`

Bases: `Teleprompter`

Source code in `dspy/teleprompt/mipro_optimizer_v2.py`

```
def __init__(
    self,
    metric: Callable,
    prompt_model: Any | None = None,
    task_model: Any | None = None,
    teacher_settings: dict | None = None,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 4,
    auto: Literal["light", "medium", "heavy"] | None = "light",
    num_candidates: int | None = None,
    num_threads: int | None = None,
    max_errors: int | None = None,
    seed: int = 9,
    init_temperature: float = 1.0,
    verbose: bool = False,
    track_stats: bool = True,
    log_dir: str | None = None,
    metric_threshold: float | None = None,
):
    # Validate 'auto' parameter
    allowed_modes = {None, "light", "medium", "heavy"}
    if auto not in allowed_modes:
        raise ValueError(f"Invalid value for auto: {auto}. Must be one of {allowed_modes}.")
    self.auto = auto
    self.num_fewshot_candidates = num_candidates
    self.num_instruct_candidates = num_candidates
    self.num_candidates = num_candidates
    self.metric = metric
    self.init_temperature = init_temperature
    self.task_model = task_model if task_model else dspy.settings.lm
    self.prompt_model = prompt_model if prompt_model else dspy.settings.lm
    self.max_bootstrapped_demos = max_bootstrapped_demos
    self.max_labeled_demos = max_labeled_demos
    self.verbose = verbose
    self.track_stats = track_stats
    self.log_dir = log_dir
    self.teacher_settings = teacher_settings or {}
    self.prompt_model_total_calls = 0
    self.total_calls = 0
    self.num_threads = num_threads
    self.max_errors = max_errors
    self.metric_threshold = metric_threshold
    self.seed = seed
    self.rng = None

    if not self.prompt_model or not self.task_model:
        raise ValueError("Either provide both prompt_model and task_model or set a default LM through dspy.configure(lm=...)")
```

### Functions

#### `compile(student: Any, *, trainset: list, teacher: Any = None, valset: list | None = None, num_trials: int | None = None, max_bootstrapped_demos: int | None = None, max_labeled_demos: int | None = None, seed: int | None = None, minibatch: bool = True, minibatch_size: int = 35, minibatch_full_eval_steps: int = 5, program_aware_proposer: bool = True, data_aware_proposer: bool = True, view_data_batch_size: int = 10, tip_aware_proposer: bool = True, fewshot_aware_proposer: bool = True, requires_permission_to_run: bool | None = None, provide_traceback: bool | None = None) -> Any`

Source code in `dspy/teleprompt/mipro_optimizer_v2.py`

```
def compile(
    self,
    student: Any,
    *,
    trainset: list,
    teacher: Any = None,
    valset: list | None = None,
    num_trials: int | None = None,
    max_bootstrapped_demos: int | None = None,
    max_labeled_demos: int | None = None,
    seed: int | None = None,
    minibatch: bool = True,
    minibatch_size: int = 35,
    minibatch_full_eval_steps: int = 5,
    program_aware_proposer: bool = True,
    data_aware_proposer: bool = True,
    view_data_batch_size: int = 10,
    tip_aware_proposer: bool = True,
    fewshot_aware_proposer: bool = True,
    requires_permission_to_run: bool | None = None, # deprecated
    provide_traceback: bool | None = None,
) -> Any:
    if requires_permission_to_run == False:
        logger.warning(
            "'requires_permission_to_run' is deprecated and will be removed in a future version."
        )
    elif requires_permission_to_run == True:
        raise ValueError("User confirmation is removed from MIPROv2. Please remove the 'requires_permission_to_run' argument.")

    effective_max_errors = (
        self.max_errors
        if self.max_errors is not None
        else dspy.settings.max_errors
    )

    effective_max_bootstrapped_demos = (
        max_bootstrapped_demos if max_bootstrapped_demos is not None else self.max_bootstrapped_demos
    )
    effective_max_labeled_demos = (
        max_labeled_demos if max_labeled_demos is not None else self.max_labeled_demos
    )

    zeroshot_opt = (effective_max_bootstrapped_demos == 0) and (effective_max_labeled_demos == 0)

    # If auto is None, and num_trials is not provided (but num_candidates is), raise an error that suggests a good num_trials value
    if self.auto is None and (self.num_candidates is not None and num_trials is None):
        raise ValueError(
            f"If auto is None, num_trials must also be provided. Given num_candidates={self.num_candidates}, we'd recommend setting num_trials to ~{self._set_num_trials_from_num_candidates(student, zeroshot_opt, self.num_candidates)}."
        )

    # If auto is None, and num_candidates or num_trials is None, raise an error
    if self.auto is None and (self.num_candidates is None or num_trials is None):
        raise ValueError("If auto is None, num_candidates must also be provided.")

    # If auto is provided, and either num_candidates or num_trials is not None, raise an error
    if self.auto is not None and (self.num_candidates is not None or num_trials is not None):
        raise ValueError(
            "If auto is not None, num_candidates and num_trials cannot be set, since they would be overridden by the auto settings. Please either set auto to None, or do not specify num_candidates and num_trials."
        )

    # Set random seeds
    seed = seed or self.seed
    self._set_random_seeds(seed)


    # Set training & validation sets
    trainset, valset = self._set_and_validate_datasets(trainset, valset)

    num_instruct_candidates = (
        self.num_instruct_candidates
        if self.num_instruct_candidates is not None
        else self.num_candidates
    )
    num_fewshot_candidates = (
        self.num_fewshot_candidates
        if self.num_fewshot_candidates is not None
        else self.num_candidates
    )

    # Set hyperparameters based on run mode (if set)
    (
        num_trials,
        valset,
        minibatch,
        num_instruct_candidates,
        num_fewshot_candidates,
    ) = self._set_hyperparams_from_run_mode(
        student,
        num_trials,
        minibatch,
        zeroshot_opt,
        valset,
        num_instruct_candidates,
        num_fewshot_candidates,
    )

    if self.auto:
        self._print_auto_run_settings(
            num_trials,
            minibatch,
            valset,
            num_fewshot_candidates,
            num_instruct_candidates,
        )

    if minibatch and minibatch_size > len(valset):
        raise ValueError(f"Minibatch size cannot exceed the size of the valset. Valset size: {len(valset)}.")

    # Initialize program and evaluator
    program = student.deepcopy()
    evaluate = Evaluate(
        devset=valset,
        metric=self.metric,
        num_threads=self.num_threads,
        max_errors=effective_max_errors,
        display_table=False,
        display_progress=True,
        provide_traceback=provide_traceback,
    )

    with dspy.context(lm=self.task_model):
        # Step 1: Bootstrap few-shot examples
        demo_candidates = self._bootstrap_fewshot_examples(
            program,
            trainset,
            seed,
            teacher,
            num_fewshot_candidates=num_fewshot_candidates,
            max_bootstrapped_demos=effective_max_bootstrapped_demos,
            max_labeled_demos=effective_max_labeled_demos,
            max_errors=effective_max_errors,
            metric_threshold=self.metric_threshold,
        )

    # Step 2: Propose instruction candidates
    instruction_candidates = self._propose_instructions(
        program,
        trainset,
        demo_candidates,
        view_data_batch_size,
        program_aware_proposer,
        data_aware_proposer,
        tip_aware_proposer,
        fewshot_aware_proposer,
        num_instruct_candidates=num_instruct_candidates,
    )

    # If zero-shot, discard demos
    if zeroshot_opt:
        demo_candidates = None

    with dspy.context(lm=self.task_model):
        # Step 3: Find optimal prompt parameters
        best_program = self._optimize_prompt_parameters(
            program,
            instruction_candidates,
            demo_candidates,
            evaluate,
            valset,
            num_trials,
            minibatch,
            minibatch_size,
            minibatch_full_eval_steps,
            seed,
        )

    return best_program
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

## Example Usage

The program below shows optimizing a math program with MIPROv2

```
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

# Import the optimizer
from dspy.teleprompt import MIPROv2

# Initialize the LM
lm = dspy.LM('openai/gpt-4o-mini', api_key='YOUR_OPENAI_API_KEY')
dspy.configure(lm=lm)

# Initialize optimizer
teleprompter = MIPROv2(
    metric=gsm8k_metric,
    auto="medium", # Can choose between light, medium, and heavy optimization runs
)

# Optimize program
print(f"Optimizing program with MIPROv2...")
gsm8k = GSM8K()
optimized_program = teleprompter.compile(
    dspy.ChainOfThought("question -> answer"),
    trainset=gsm8k.train,
)

# Save optimize program for future use
optimized_program.save(f"optimized.json")
```

## How `MIPROv2` works

At a high level, `MIPROv2` works by creating both few-shot examples and new instructions for each predictor in your LM program, and then searching over these using Bayesian Optimization to find the best combination of these variables for your program. If you want a visual explanation check out this [twitter thread](https://x.com/michaelryan207/status/1804189184988713065).

These steps are broken down in more detail below:

1. **Bootstrap Few-Shot Examples**: Randomly samples examples from your training set, and run them through your LM program. If the output from the program is correct for this example, it is kept as a valid few-shot example candidate. Otherwise, we try another example until we've curated the specified amount of few-shot example candidates. This step creates `num_candidates` sets of `max_bootstrapped_demos` bootstrapped examples and `max_labeled_demos` basic examples sampled from the training set.

1. **Propose Instruction Candidates**. The instruction proposer includes (1) a generated summary of properties of the training dataset, (2) a generated summary of your LM program's code and the specific predictor that an instruction is being generated for, (3) the previously bootstrapped few-shot examples to show reference inputs / outputs for a given predictor and (4) a randomly sampled tip for generation (i.e. "be creative", "be concise", etc.) to help explore the feature space of potential instructions. This context is provided to a `prompt_model` which writes high quality instruction candidates.

1. **Find an Optimized Combination of Few-Shot Examples & Instructions**. Finally, we use Bayesian Optimization to choose which combinations of instructions and demonstrations work best for each predictor in our program. This works by running a series of `num_trials` trials, where a new set of prompts are evaluated over our validation set at each trial. The new set of prompts are only evaluated on a minibatch of size `minibatch_size` at each trial (when `minibatch`=`True`). The best averaging set of prompts is then evaluated on the full validation set every `minibatch_full_eval_steps`. At the end of the optimization process, the LM program with the set of prompts that performed best on the full validation set is returned.

For those interested in more details, more information on `MIPROv2` along with a study on `MIPROv2` compared with other DSPy optimizers can be found in [this paper](https://arxiv.org/abs/2406.11695).
