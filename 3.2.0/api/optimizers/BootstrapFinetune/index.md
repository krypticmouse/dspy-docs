# dspy.BootstrapFinetune

## `dspy.BootstrapFinetune(metric: Callable | None = None, multitask: bool = True, train_kwargs: dict[str, Any] | dict[LM, dict[str, Any]] | None = None, adapter: Adapter | dict[LM, Adapter] | None = None, exclude_demos: bool = False, num_threads: int | None = None)`

Bases: `FinetuneTeleprompter`

Source code in `dspy/teleprompt/bootstrap_finetune.py`

```
def __init__(
    self,
    metric: Callable | None = None,
    multitask: bool = True,
    train_kwargs: dict[str, Any] | dict[LM, dict[str, Any]] | None = None,
    adapter: Adapter | dict[LM, Adapter] | None = None,
    exclude_demos: bool = False,
    num_threads: int | None = None,
):
    # TODO(feature): Inputs train_kwargs (a dict with string keys) and
    # adapter (Adapter) can depend on the LM they are used with. We are
    # takingthese as parameters for the time being. However, they can be
    # attached to LMs themselves -- an LM could know which adapter it should
    # be used with along with the train_kwargs. This will lead the only
    # required argument for LM.finetune() to be the train dataset.

    super().__init__(train_kwargs=train_kwargs)
    self.metric = metric
    self.multitask = multitask
    self.adapter: dict[LM, Adapter] = self.convert_to_lm_dict(adapter)
    self.exclude_demos = exclude_demos
    self.num_threads = num_threads
```

### Functions

#### `compile(student: Module, trainset: list[Example], teacher: Module | list[Module] | None = None) -> Module`

Source code in `dspy/teleprompt/bootstrap_finetune.py`

```
def compile(
    self, student: Module, trainset: list[Example], teacher: Module | list[Module] | None = None
) -> Module:
    # TODO: Print statements can be converted to logger.info if we ensure
    # that the default DSPy logger logs info level messages in notebook
    # environments.
    logger.info("Preparing the student and teacher programs...")
    all_predictors_have_lms(student)

    logger.info("Bootstrapping data...")
    trace_data = []

    teachers = teacher if isinstance(teacher, list) else [teacher]
    teachers = [prepare_teacher(student, t) for t in teachers]
    num_threads = self.num_threads or dspy.settings.num_threads
    for t in teachers:
        trace_data += bootstrap_trace_data(program=t, dataset=trainset, metric=self.metric, num_threads=num_threads)

    logger.info("Preparing the train data...")
    key_to_data = {}
    for pred_ind, pred in enumerate(student.predictors()):
        data_pred_ind = None if self.multitask else pred_ind
        if pred.lm is None:
            raise ValueError(
                f"Predictor {pred_ind} does not have an LM assigned. "
                f"Please ensure the module's predictors have their LM set before fine-tuning. "
                f"You can set it using: your_module.set_lm(your_lm)"
            )
        training_key = (pred.lm, data_pred_ind)

        if training_key not in key_to_data:
            train_data, data_format = self._prepare_finetune_data(
                trace_data=trace_data, lm=pred.lm, pred_ind=data_pred_ind
            )
            logger.info(f"Using {len(train_data)} data points for fine-tuning the model: {pred.lm.model}")
            finetune_kwargs = {
                "lm": pred.lm,
                "train_data": train_data,
                "train_data_format": data_format,
                "train_kwargs": self.train_kwargs[pred.lm],
            }
            key_to_data[training_key] = finetune_kwargs

    logger.info("Starting LM fine-tuning...")
    # TODO(feature): We could run batches of fine-tuning jobs in sequence
    # to avoid exceeding the number of threads.
    if len(key_to_data) > num_threads:
        raise ValueError(
            "BootstrapFinetune requires `num_threads` to be bigger than or equal to the number of fine-tuning "
            f"jobs. There are {len(key_to_data)} fine-tuning jobs to start, but the number of threads is: "
            f"{num_threads}! If the `multitask` flag is set to False, the number of fine-tuning jobs will "
            "be equal to the number of predictors in the student program. If the `multitask` flag is set to True, "
            "the number of fine-tuning jobs will be equal to: 1 if there is only a context LM, or the number of "
            "unique LMs attached to the predictors in the student program. In any case, the number of fine-tuning "
            "jobs will be less than or equal to the number of predictors."
        )
    logger.info(f"{len(key_to_data)} fine-tuning job(s) to start")
    key_to_lm = self.finetune_lms(key_to_data)

    logger.info("Updating the student program with the fine-tuned LMs...")
    for pred_ind, pred in enumerate(student.predictors()):
        data_pred_ind = None if self.multitask else pred_ind
        training_key = (pred.lm, data_pred_ind)
        finetuned_lm = key_to_lm[training_key]
        if isinstance(finetuned_lm, Exception):
            raise RuntimeError(f"Finetuned LM for predictor {pred_ind} failed.") from finetuned_lm
        pred.lm = finetuned_lm
        # TODO: What should the correct behavior be here? Should
        # BootstrapFinetune modify the prompt demos according to the
        # train data?
        pred.demos = [] if self.exclude_demos else pred.demos

    logger.info("BootstrapFinetune has finished compiling the student program")
    student._compiled = True
    return student
```

#### `convert_to_lm_dict(arg) -> dict[LM, Any]`

Source code in `dspy/teleprompt/bootstrap_finetune.py`

```
@staticmethod
def convert_to_lm_dict(arg) -> dict[LM, Any]:
    non_empty_dict = arg and isinstance(arg, dict)
    if non_empty_dict and all(isinstance(k, LM) for k in arg.keys()):
        return arg
    # Default to using the same value for all LMs
    return defaultdict(lambda: arg)
```

#### `finetune_lms(finetune_dict) -> dict[Any, LM]`

Source code in `dspy/teleprompt/bootstrap_finetune.py`

```
@staticmethod
def finetune_lms(finetune_dict) -> dict[Any, LM]:
    num_jobs = len(finetune_dict)
    logger.info(f"Starting {num_jobs} fine-tuning job(s)...")
    # TODO(nit) Pass an identifier to the job so that we can tell the logs
    # coming from different fine-tune threads.

    key_to_job = {}
    for key, finetune_kwargs in finetune_dict.items():
        lm: LM = finetune_kwargs.pop("lm")
        # TODO: The following line is a hack. We should re-think how to free
        # up resources for fine-tuning. This might mean introducing a new
        # provider method (e.g. prepare_for_finetune) that can be called
        # before fine-tuning is started.
        logger.info(
            "Calling lm.kill() on the LM to be fine-tuned to free up resources. This won't have any effect if the "
            "LM is not running."
        )
        lm.kill()
        key_to_job[key] = lm.finetune(**finetune_kwargs)

    key_to_lm = {}
    for ind, (key, job) in enumerate(key_to_job.items()):
        result = job.result()
        if isinstance(result, Exception):
            raise result
        key_to_lm[key] = result
        job.thread.join()
        logger.info(f"Job {ind + 1}/{num_jobs} is done")

    return key_to_lm
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
