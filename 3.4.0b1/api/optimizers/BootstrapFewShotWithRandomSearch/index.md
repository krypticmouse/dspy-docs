# dspy.BootstrapFewShotWithRandomSearch

## `dspy.BootstrapFewShotWithRandomSearch(metric, teacher_settings=None, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1, num_candidate_programs=16, num_threads=None, max_errors=None, stop_at_score=None, metric_threshold=None)`

Bases: `Teleprompter`

Source code in `dspy/teleprompt/random_search.py`

```
def __init__(
    self,
    metric,
    teacher_settings=None,
    max_bootstrapped_demos=4,
    max_labeled_demos=16,
    max_rounds=1,
    num_candidate_programs=16,
    num_threads=None,
    max_errors=None,
    stop_at_score=None,
    metric_threshold=None,
):
    self.metric = metric
    self.teacher_settings = teacher_settings or {}
    self.max_rounds = max_rounds

    self.num_threads = num_threads
    self.stop_at_score = stop_at_score
    self.metric_threshold = metric_threshold
    self.min_num_samples = 1
    self.max_num_samples = max_bootstrapped_demos
    self.max_errors = max_errors
    self.num_candidate_sets = num_candidate_programs
    self.max_labeled_demos = max_labeled_demos

    print(f"Going to sample between {self.min_num_samples} and {self.max_num_samples} traces per predictor.")
    print(f"Will attempt to bootstrap {self.num_candidate_sets} candidate sets.")
```

### Methods:

#### `compile(student, *, teacher=None, trainset, valset=None, restrict=None, labeled_sample=True)`

Source code in `dspy/teleprompt/random_search.py`

```
def compile(self, student, *, teacher=None, trainset, valset=None, restrict=None, labeled_sample=True):
    self.trainset = trainset
    self.valset = valset or trainset  # TODO: FIXME: Note this choice.

    if restrict is not None:
        restrict = set(restrict)
        if not restrict.intersection(range(-3, self.num_candidate_sets)):
            raise ValueError(
                f"`restrict` {restrict!r} does not match any candidate seed in "
                f"range(-3, {self.num_candidate_sets}); no candidate programs would be evaluated."
            )

    effective_max_errors = self.max_errors if self.max_errors is not None else dspy.settings.max_errors

    scores = []
    all_subscores = []
    score_data = []

    for seed in range(-3, self.num_candidate_sets):
        if (restrict is not None) and (seed not in restrict):
            continue

        trainset_copy = list(self.trainset)

        if seed == -3:
            # zero-shot
            program = student.reset_copy()

        elif seed == -2:
            # labels only
            teleprompter = LabeledFewShot(k=self.max_labeled_demos)
            program = teleprompter.compile(student, trainset=trainset_copy, sample=labeled_sample)

        elif seed == -1:
            # unshuffled few-shot
            optimizer = BootstrapFewShot(
                metric=self.metric,
                metric_threshold=self.metric_threshold,
                max_bootstrapped_demos=self.max_num_samples,
                max_labeled_demos=self.max_labeled_demos,
                teacher_settings=self.teacher_settings,
                max_rounds=self.max_rounds,
                max_errors=effective_max_errors,
            )
            program = optimizer.compile(student, teacher=teacher, trainset=trainset_copy)

        else:
            assert seed >= 0, seed

            random.Random(seed).shuffle(trainset_copy)
            size = random.Random(seed).randint(self.min_num_samples, self.max_num_samples)

            optimizer = BootstrapFewShot(
                metric=self.metric,
                metric_threshold=self.metric_threshold,
                max_bootstrapped_demos=size,
                max_labeled_demos=self.max_labeled_demos,
                teacher_settings=self.teacher_settings,
                max_rounds=self.max_rounds,
                max_errors=effective_max_errors,
            )

            program = optimizer.compile(student, teacher=teacher, trainset=trainset_copy)

        evaluate = Evaluate(
            devset=self.valset,
            metric=self.metric,
            num_threads=self.num_threads,
            max_errors=effective_max_errors,
            display_table=False,
            display_progress=True,
        )

        result = evaluate(program)

        score, subscores = result.score, [output[2] for output in result.results]

        all_subscores.append(subscores)

        if len(scores) == 0 or score > max(scores):
            print("New best score:", score, "for seed", seed)
            best_program = program

        scores.append(score)
        print(f"Scores so far: {scores}")
        print(f"Best score so far: {max(scores)}")

        score_data.append({"score": score, "subscores": subscores, "seed": seed, "program": program})

        if self.stop_at_score is not None and score >= self.stop_at_score:
            print(f"Stopping early because score {score} is >= stop_at_score {self.stop_at_score}")
            break

    # To best program, attach all program candidates in decreasing average score
    best_program.candidate_programs = score_data
    best_program.candidate_programs = sorted(
        best_program.candidate_programs, key=lambda x: x["score"], reverse=True
    )

    print(f"{len(best_program.candidate_programs)} candidate programs found.")

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
