# dspy.InferRules

## `dspy.InferRules(num_candidates=10, num_rules=10, num_threads=None, teacher_settings=None, **kwargs)`

Bases: `BootstrapFewShot`

Source code in `dspy/teleprompt/infer_rules.py`

```
def __init__(self, num_candidates=10, num_rules=10, num_threads=None, teacher_settings=None, **kwargs):
    super().__init__(teacher_settings=teacher_settings, **kwargs)

    self.num_candidates = num_candidates
    self.num_rules = num_rules
    self.num_threads = num_threads
    self.rules_induction_program = RulesInductionProgram(num_rules, teacher_settings=teacher_settings)
    self.metric = kwargs.get("metric")
    self.max_errors = kwargs.get("max_errors")
```

### Methods:

#### `compile(student, *, teacher=None, trainset, valset=None)`

Source code in `dspy/teleprompt/infer_rules.py`

```
def compile(self, student, *, teacher=None, trainset, valset=None):
    if valset is None:
        train_size = int(0.5 * len(trainset))
        trainset, valset = trainset[:train_size], trainset[train_size:]

    super().compile(student, teacher=teacher, trainset=trainset)

    original_program = self.student.deepcopy()
    all_predictors = [p for p in original_program.predictors() if hasattr(p, "signature")]
    instructions_list = [p.signature.instructions for p in all_predictors]

    best_score = -math.inf
    best_program = None

    for candidate_idx in range(self.num_candidates):
        candidate_program = original_program.deepcopy()
        candidate_predictors = [p for p in candidate_program.predictors() if hasattr(p, "signature")]

        for i, predictor in enumerate(candidate_predictors):
            predictor.signature.instructions = instructions_list[i]

        for i, predictor in enumerate(candidate_predictors):
            rules = self.induce_natural_language_rules(predictor, trainset)
            predictor.signature.instructions = instructions_list[i]
            self.update_program_instructions(predictor, rules)

        score = self.evaluate_program(candidate_program, valset)

        if score > best_score:
            best_score = score
            best_program = candidate_program

        logger.info(f"Evaluated Candidate {candidate_idx + 1} with score {score}. Current best score: {best_score}")

    logger.info(f"Final best score: {best_score}")

    return best_program
```

#### `evaluate_program(program, dataset)`

Source code in `dspy/teleprompt/infer_rules.py`

```
def evaluate_program(self, program, dataset):
    effective_max_errors = (
        self.max_errors if self.max_errors is not None else dspy.settings.max_errors
    )
    evaluate = Evaluate(
        devset=dataset,
        metric=self.metric,
        num_threads=self.num_threads,
        max_errors=effective_max_errors,
        display_table=False,
        display_progress=True,
    )
    score = evaluate(program, metric=self.metric).score
    return score
```

#### `format_examples(demos, signature)`

Source code in `dspy/teleprompt/infer_rules.py`

```
def format_examples(self, demos, signature):
    examples_text = ""
    for demo in demos:
        input_fields = {k: v for k, v in demo.items() if k in signature.input_fields}
        output_fields = {k: v for k, v in demo.items() if k in signature.output_fields}
        input_text = "\n".join(f"{k}: {v}" for k, v in input_fields.items())
        output_text = "\n".join(f"{k}: {v}" for k, v in output_fields.items())
        examples_text += f"Input Fields:\n{input_text}\n\n=========\nOutput Fields:\n{output_text}\n\n"
    return examples_text
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

#### `get_predictor_demos(trainset, predictor)`

Source code in `dspy/teleprompt/infer_rules.py`

```
def get_predictor_demos(self, trainset, predictor):
    # TODO: Consider how this handled "incomplete" demos.
    signature = predictor.signature
    return [
        {
            key: value
            for key, value in example.items()
            if key in signature.input_fields or key in signature.output_fields
        }
        for example in trainset
    ]
```

#### `induce_natural_language_rules(predictor, trainset)`

Source code in `dspy/teleprompt/infer_rules.py`

```
def induce_natural_language_rules(self, predictor, trainset):
    demos = self.get_predictor_demos(trainset, predictor)
    signature = predictor.signature
    while True:
        examples_text = self.format_examples(demos, signature)
        try:
            return self.rules_induction_program(examples_text)
        except Exception as e:
            assert (
                isinstance(e, ValueError)
                or e.__class__.__name__ == "BadRequestError"
                or "ContextWindowExceededError" in str(e)
            )
            if len(demos) > 1:
                demos = demos[:-1]
            else:
                raise RuntimeError(
                    "Failed to generate natural language rules since a single example couldn't fit in the model's "
                    "context window."
                ) from e
```

#### `update_program_instructions(predictor, natural_language_rules)`

Source code in `dspy/teleprompt/infer_rules.py`

```
def update_program_instructions(self, predictor, natural_language_rules):
    predictor.signature.instructions = (
        f"{predictor.signature.instructions}\n\n"
        f"Please adhere to the following rules when making your prediction:\n{natural_language_rules}"
    )
```
