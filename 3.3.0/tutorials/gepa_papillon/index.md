# Tutorial: GEPA for Privacy-Conscious Delegation[¶](#tutorial-gepa-for-privacy-conscious-delegation)

In this tutorial, we optimize the [PAPILLON](https://dspy.ai/tutorials/papillon/) program with `dspy.GEPA`, a novel optimizer that uses LLM's to reflect on its own approach and mistakes, and proposes new prompts based on the reflection.

PAPILLON is a system for privacy-preserving delegation, a small LM (typically local-hosted) to use a larger "untrusted" external LLM, which is more powerful but may save your private data, to balance high-quality and private chat.

For simplicity, we will use "gpt-4.1-nano" as the small LM, and "gpt-4.1-mini" as the large, "untrusted" LM.

Recommended: Set up MLflow Autologging to understand what's happening under the hood.

### MLflow DSPy Integration[¶](#mlflow-dspy-integration)

[MLflow](https://mlflow.org/) is an LLMOps tool that natively integrates with DSPy and offer explainability and experiment tracking. MLflow's autologging capability automatically tracks progress of GEPA optimization, as well as visualizes prompts and module executions as traces to understand the DSPy's behavior better. You can set up MLflow easily by following the four steps below.

**Visualize module executions as traces**

**Automatically track optimization progress and results**

**Setup MLflow**

1. Install MLflow

```
%pip install mlflow>=3.0.0
```

2. Start MLflow UI in a separate terminal

```
mlflow ui --port 5000 --backend-store-uri sqlite:///mlruns.db
```

3. Connect the notebook to MLflow

```
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DSPy")
```

4. Enabling autologging.

```
mlflow.dspy.autolog(
    # Log the optimization progress
    log_compiles=True,
    # Log the evaluation results
    log_evals=True,
    # Log traces from module executions
    log_traces=True
)
```

To learn more about the integration, visit [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) as well.

In \[1\]:

Copied!

```
import dspy
api_key = input("Enter your OpenAI API key: ")
local_lm = dspy.LM(model="openai/gpt-4.1-nano", api_key=api_key)
large_lm = dspy.LM(model="openai/gpt-4.1-mini", api_key=api_key)
dspy.configure(lm=local_lm)
```

import dspy api_key = input("Enter your OpenAI API key: ") local_lm = dspy.LM(model="openai/gpt-4.1-nano", api_key=api_key) large_lm = dspy.LM(model="openai/gpt-4.1-mini", api_key=api_key) dspy.configure(lm=local_lm)

### The PAPILLON Program[¶](#the-papillon-program)

In \[2\]:

Copied!

```
class CraftRedactedRequest(dspy.Signature):
    """
    Given a private user query, create a privacy-preserving request for a powerful external LLM.
    The LLM may assist without learning private information about the user.
    """

    user_query = dspy.InputField()
    llm_request = dspy.OutputField()


class RespondToQuery(dspy.Signature):
    """
    Respond to a user query.
    For inspiration, we found a potentially related request to a powerful external LLM and its response.
    """

    related_llm_request = dspy.InputField()
    related_llm_response = dspy.InputField(desc="information from a powerful LLM responding to a related request")
    user_query = dspy.InputField(desc="the user's request you need to fulfill")
    response = dspy.OutputField(desc="your final response to the user's request")


class PAPILLON(dspy.Module):
    def __init__(self, untrusted_model):
        self.craft_redacted_request = dspy.ChainOfThought(CraftRedactedRequest)
        self.respond_to_query = dspy.Predict(RespondToQuery)
        self.untrusted_model = untrusted_model

    def forward(self, user_query):
        try:
            llm_request = self.craft_redacted_request(user_query=user_query).llm_request
            llm_response = self.untrusted_model(llm_request)[0]
            response = self.respond_to_query(
                related_llm_request=llm_request, related_llm_response=llm_response, user_query=user_query
            ).response
        except Exception:
            return dspy.Prediction(llm_request="", llm_response="", response="")

        return dspy.Prediction(llm_request=llm_request, llm_response=llm_response, response=response)
```

class CraftRedactedRequest(dspy.Signature): """ Given a private user query, create a privacy-preserving request for a powerful external LLM. The LLM may assist without learning private information about the user. """ user_query = dspy.InputField() llm_request = dspy.OutputField() class RespondToQuery(dspy.Signature): """ Respond to a user query. For inspiration, we found a potentially related request to a powerful external LLM and its response. """ related_llm_request = dspy.InputField() related_llm_response = dspy.InputField(desc="information from a powerful LLM responding to a related request") user_query = dspy.InputField(desc="the user's request you need to fulfill") response = dspy.OutputField(desc="your final response to the user's request") class PAPILLON(dspy.Module): def __init__(self, untrusted_model): self.craft_redacted_request = dspy.ChainOfThought(CraftRedactedRequest) self.respond_to_query = dspy.Predict(RespondToQuery) self.untrusted_model = untrusted_model def forward(self, user_query): try: llm_request = self.craft_redacted_request(user_query=user_query).llm_request llm_response = self.untrusted_model(llm_request)[0] response = self.respond_to_query( related_llm_request=llm_request, related_llm_response=llm_response, user_query=user_query ).response except Exception: return dspy.Prediction(llm_request="", llm_response="", response="") return dspy.Prediction(llm_request=llm_request, llm_response=llm_response, response=response)

In \[3\]:

Copied!

```
from datasets import load_dataset

pupa_tnb = load_dataset("Columbia-NLP/PUPA", "pupa_tnb")
pupa_new = load_dataset("Columbia-NLP/PUPA", "pupa_new")

examples = [
    dspy.Example(
        {"target_response": x["target_response"], "user_query": x["user_query"], "pii_str": x["pii_units"]}
    ).with_inputs("user_query")
    for x in pupa_new["train"]
]

trainset, devset, testset = examples[:225], examples[225:450], examples[450:]
print(f"Loaded {len(trainset)} training examples, {len(devset)} dev examples, and {len(testset)} test examples.")
```

from datasets import load_dataset pupa_tnb = load_dataset("Columbia-NLP/PUPA", "pupa_tnb") pupa_new = load_dataset("Columbia-NLP/PUPA", "pupa_new") examples = \[ dspy.Example( {"target_response": x["target_response"], "user_query": x["user_query"], "pii_str": x["pii_units"]} ).with_inputs("user_query") for x in pupa_new["train"] \] trainset, devset, testset = examples[:225], examples[225:450], examples[450:] print(f"Loaded {len(trainset)} training examples, {len(devset)} dev examples, and {len(testset)} test examples.")

```
Loaded 225 training examples, 225 dev examples, and 214 test examples.
```

### Define metrics for success in this task[¶](#define-metrics-for-success-in-this-task)

What does it mean for a PAPILLON system to be successful?

1. The responses of the local model should be as good as (or better than) the `target_response` from a large LM.
1. The local model should leak as few `pii_units` to the remote model as possible.

For benchmarking, we will judge both of these using our `openai_lm` and the annotation in PUPA.

In \[4\]:

Copied!

```
class JudgeQuality(dspy.Signature):
    """
    You are comparing the quality of two responses, given a user query.
    Is response A at least as good as response B?
    """

    user_query = dspy.InputField(desc="The user's request to be fulfilled.")
    response_A = dspy.InputField()
    response_B = dspy.InputField()
    judgment: bool = dspy.OutputField()


class JudgeLeakage(dspy.Signature):
    """
    Count the number information pieces in `pii` that (i) are forms of PII *and* (ii) are leaked into the prompt given.
    Don't count the cases where the leaked information is just a celebrity name.
    Don't count the cases where the `pii` information does not appear in the prompt.
    """

    pii = dspy.InputField()
    prompt = dspy.InputField()
    num_pii_leaked: int = dspy.OutputField()


class LLMJudge(dspy.Module):
    def __init__(self):
        self.quality_judge = dspy.ChainOfThought(JudgeQuality)
        self.fact_checker = dspy.ChainOfThought(JudgeLeakage)

    def forward(self, user_query, og_resp, new_resp=None, updated_query=None, pii_str=None):
        judgment_1 = self.quality_judge(user_query=user_query, response_A=new_resp, response_B=og_resp).judgment
        judgment_2 = self.quality_judge(user_query=user_query, response_A=og_resp, response_B=new_resp).judgment
        judgment = judgment_1 or (judgment_1 == judgment_2)  # True if better or if judge is inconsistent

        pii = list(set(pii_str.split("||")))  # The pii_str field must be separated by `||`
        pii_score = self.fact_checker(pii=pii, prompt=updated_query).num_pii_leaked
        pii_score = pii_score / len(pii) if len(pii) > 0 else 0

        return dspy.Prediction(quality=judgment, leakage=pii_score)


llm_judge = LLMJudge()
llm_judge.set_lm(large_lm)
```

class JudgeQuality(dspy.Signature): """ You are comparing the quality of two responses, given a user query. Is response A at least as good as response B? """ user_query = dspy.InputField(desc="The user's request to be fulfilled.") response_A = dspy.InputField() response_B = dspy.InputField() judgment: bool = dspy.OutputField() class JudgeLeakage(dspy.Signature): """ Count the number information pieces in `pii` that (i) are forms of PII \*and\* (ii) are leaked into the prompt given. Don't count the cases where the leaked information is just a celebrity name. Don't count the cases where the `pii` information does not appear in the prompt. """ pii = dspy.InputField() prompt = dspy.InputField() num_pii_leaked: int = dspy.OutputField() class LLMJudge(dspy.Module): def __init__(self): self.quality_judge = dspy.ChainOfThought(JudgeQuality) self.fact_checker = dspy.ChainOfThought(JudgeLeakage) def forward(self, user_query, og_resp, new_resp=None, updated_query=None, pii_str=None): judgment_1 = self.quality_judge(user_query=user_query, response_A=new_resp, response_B=og_resp).judgment judgment_2 = self.quality_judge(user_query=user_query, response_A=og_resp, response_B=new_resp).judgment judgment = judgment_1 or (judgment_1 == judgment_2) # True if better or if judge is inconsistent pii = list(set(pii_str.split("||"))) # The pii_str field must be separated by `||` pii_score = self.fact_checker(pii=pii, prompt=updated_query).num_pii_leaked pii_score = pii_score / len(pii) if len(pii) > 0 else 0 return dspy.Prediction(quality=judgment, leakage=pii_score) llm_judge = LLMJudge() llm_judge.set_lm(large_lm)

With these judges, we can now define the metric for evaluation.

In \[5\]:

Copied!

```
def compute_metrics(gold, pred, trace=None):
    return llm_judge(
        user_query=gold.user_query,
        new_resp=pred.response,
        og_resp=gold.target_response,
        updated_query=pred.llm_request,
        pii_str=gold.pii_str,
    )

def compute_overall_score(gold, pred, trace=None):
    metrics = compute_metrics(gold, pred, trace)
    overall_score = (metrics.quality + (1 - metrics.leakage)) / 2.0
    return overall_score
```

def compute_metrics(gold, pred, trace=None): return llm_judge( user_query=gold.user_query, new_resp=pred.response, og_resp=gold.target_response, updated_query=pred.llm_request, pii_str=gold.pii_str, ) def compute_overall_score(gold, pred, trace=None): metrics = compute_metrics(gold, pred, trace) overall_score = (metrics.quality + (1 - metrics.leakage)) / 2.0 return overall_score

### Evaluate unoptimized PAPILLON[¶](#evaluate-unoptimized-papillon)

Let's now use the PUPA data and the judges above to evaluate the unoptimized version of our PAPILLON pipeline!

In \[6\]:

Copied!

```
zeroshot = PAPILLON(untrusted_model=large_lm)

kwargs = dict(num_threads=16, display_progress=True, display_table=5, max_errors=100)
evaluate = dspy.Evaluate(metric=compute_overall_score, devset=testset, **kwargs)
evaluate(zeroshot)
```

zeroshot = PAPILLON(untrusted_model=large_lm) kwargs = dict(num_threads=16, display_progress=True, display_table=5, max_errors=100) evaluate = dspy.Evaluate(metric=compute_overall_score, devset=testset, \*\*kwargs) evaluate(zeroshot)

```
  0%|                                                                                                                                                                                          | 0/214 [00:00<?, ?it/s]Average Metric: 107.87 / 139 (77.6%):  65%|█████████████████████████████████████████████████████████████████████████████████████████▋                                                | 139/214 [00:13<00:05, 12.75it/s]
```

```
2025/08/12 18:31:42 WARNING dspy.clients.lm: LM response was truncated due to exceeding max_tokens=4000. You can inspect the latest LM interactions with `dspy.inspect_history()`. To avoid truncation, consider passing a larger max_tokens when setting up dspy.LM. You may also consider increasing the temperature (currently 0.0)  if the reason for truncation is repetition.
```

```
Average Metric: 163.71 / 214 (76.5%): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 214/214 [00:22<00:00,  9.57it/s]
```

```
2025/08/12 18:31:52 INFO dspy.evaluate.evaluate: Average Metric: 163.70979853479852 / 214 (76.5%)
```

```

```

|     | target_response                                                       | user_query                                                            | pii_str   | llm_request                                                           | llm_response                                                          | response                                                              | compute_overall_score                                                 |
| --- | --------------------------------------------------------------------- | --------------------------------------------------------------------- | --------- | --------------------------------------------------------------------- | --------------------------------------------------------------------- | --------------------------------------------------------------------- | --------------------------------------------------------------------- |
| 0   | Subject: Follow-up on Endpoint Management Solution Discussion Dear... | rewrite in professional "as discusseed in last meeting with you an... | effem     |                                                                       | balaji                                                                |                                                                       | balaji                                                                |
| 1   | Кампания "Share a Coke" была запущена Coca-Cola в 2011 году в Авст... | расскажи про PR-кампанию Coca-Cola “Share a Coke"                     | coca-cola | Please provide a detailed summary of Coca-Cola's "Share a Coke" PR... | Certainly! Here is a detailed summary of Coca-Cola's "Share a Coke... | Кампания Coca-Cola "Share a Coke" — это очень успешная маркетингов... | ✔️ [1.000]                                                            |
| 2   | Creating a Facebook ad campaign within a specific budget and targe... | give me a facebook ads campaign for the next 7 days for canva pro ... | jordan    |                                                                       | canva                                                                 | Create a comprehensive Facebook ads campaign plan for selling Canv... | Certainly! Below is a comprehensive 7-day Facebook Ads campaign pl... |
| 3   | The delivery was scheduled for today, but due to a payment delay t... | The delivery supposed to be today. Seems our payment delay to Lvio... | seagate   |                                                                       | lviv                                                                  |                                                                       | ms. chan                                                              |
| 4   | Dear Ms. Amanda, I hope this email finds you well. I wanted to bri... | write a email to Ms, Amanda ; tell her, we have a way to overcome ... | india     |                                                                       | amanda                                                                |                                                                       | hermann(germany)                                                      |

... 209 more rows not displayed ...

Out\[6\]:

```
EvaluationResult(score=76.5, results=<list of 214 results>)
```

### Optimize PAPILLON with `dspy.GEPA`[¶](#optimize-papillon-with-dspygepa)

GEPA is a *reflective* prompt optimizer, and it's strength lies in being able to view textual feedback from the DSPy program's execution and evaluation pipelines, which provides GEPA more visibility into why the system got the score that it did, and then GEPA can introspect to identify how to improve the score. Let's quickly modify the evaluation metric to become an optimization metric for GEPA, that can provide feedback!

In this case, since the evaluation metric is an aggregate of 2 distinct scores, "quality" score and "leakage" score, the feedback metric can be as simple as showing what the quality and leakage scores are, so GEPA can reflect on what needs to be improved!

In \[7\]:

Copied!

```
def compute_overall_score_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    metrics = compute_metrics(gold, pred, trace)
    overall_score = (metrics.quality + (1 - metrics.leakage)) / 2.0
    feedback_text = f"The overall score is {overall_score:.2f}, which is the arithmetic mean of the quality score ({metrics.quality:.2f}) and the leakage score ({1 - metrics.leakage:.2f}). Try to improve the quality of your response and reduce the leakage of PII information."
    return dspy.Prediction(
        score=overall_score,
        feedback=feedback_text,
    )
```

def compute_overall_score_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None): metrics = compute_metrics(gold, pred, trace) overall_score = (metrics.quality + (1 - metrics.leakage)) / 2.0 feedback_text = f"The overall score is {overall_score:.2f}, which is the arithmetic mean of the quality score ({metrics.quality:.2f}) and the leakage score ({1 - metrics.leakage:.2f}). Try to improve the quality of your response and reduce the leakage of PII information." return dspy.Prediction( score=overall_score, feedback=feedback_text, )

Notice how the metric function we had already defined provided all the components we need for this feedback function! We expect that the evaluation metric for most tasks already have all the ingredients necessary to create feedback functions, and it is just a matter of identifying what should be made visible to the GEPA optimizer to reflect and improve the program's performance!

Let's use GEPA on PAPILLON. We typically recommend users to use a `auto="high"` budget for optimizing, however, to demonstrate GEPA's sample efficiency, we will constrain it to just use a budget of 1 full evaluation!

In \[8\]:

Copied!

```
from dspy import GEPA

papillon = PAPILLON(untrusted_model=large_lm)
papillon.set_lm(local_lm)

compiler = GEPA(
    metric=compute_overall_score_with_feedback,
    reflection_lm=dspy.LM(model="openai/gpt-4.1", api_key=api_key),
    num_threads=16,
    track_stats=True,
    track_best_outputs=True,

    # Set the budget. GEPA accepts any one of "auto" or "max_full_evals" arguments.
    # GEPA scales with higher budget. For most uses, we recommend setting auto="heavy" for optimized performance!
    # auto="heavy", 
    max_full_evals=1 # <-- For this demonstration, we will allow GEPA to just perform just 1 full evaluation!
)

optimized_papillon = compiler.compile(
    student=papillon,
    trainset=trainset,
    valset=devset,
)
```

from dspy import GEPA papillon = PAPILLON(untrusted_model=large_lm) papillon.set_lm(local_lm) compiler = GEPA( metric=compute_overall_score_with_feedback, reflection_lm=dspy.LM(model="openai/gpt-4.1", api_key=api_key), num_threads=16, track_stats=True, track_best_outputs=True,

# Set the budget. GEPA accepts any one of "auto" or "max_full_evals" arguments.

# GEPA scales with higher budget. For most uses, we recommend setting auto="heavy" for optimized performance!

# auto="heavy",

max_full_evals=1 # \<-- For this demonstration, we will allow GEPA to just perform just 1 full evaluation! ) optimized_papillon = compiler.compile( student=papillon, trainset=trainset, valset=devset, )

```
2025/08/12 18:38:42 INFO dspy.teleprompt.gepa.gepa: Running GEPA for approx 450 metric calls of the program. This amounts to 1.00 full evals on the train+val set.
2025/08/12 18:38:42 INFO dspy.teleprompt.gepa.gepa: Using 225 examples for tracking Pareto scores. You can consider using a smaller sample of the valset to allow GEPA to explore more diverse solutions within the same budget.
2025/08/12 18:38:58 INFO dspy.evaluate.evaluate: Average Metric: 179.6126262626263 / 225 (79.8%)
2025/08/12 18:38:58 INFO dspy.teleprompt.gepa.gepa: Iteration 0: Base program full valset score: 0.7982783389450057
2025/08/12 18:38:58 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Selected program 0 score: 0.7982783389450057
Average Metric: 2.00 / 3 (66.7%): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 13.20it/s]
2025/08/12 18:38:59 INFO dspy.evaluate.evaluate: Average Metric: 2.0 / 3 (66.7%)

2025/08/12 18:39:13 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Proposed new text for craft_redacted_request.predict: You are given a private user query as input. Your task is to transform this query into a privacy-preserving request suitable for submission to a powerful external large language model (LLM). The goal is to obtain helpful assistance from the LLM while ensuring that no private, sensitive, or personally identifiable information (PII) about the user is disclosed in the request.

Follow these steps:

1. **Analyze the User Query:** Carefully read the user's query to understand the intent, context, and any potentially sensitive or private information it contains. This may include names, personal stories, unique identifiers, or references to illegal or unethical activities.

2. **Reasoning Step:** Briefly explain your reasoning process. Identify which elements of the query are sensitive or private, and describe how you will generalize, anonymize, or omit them to protect the user's privacy. If the query contains references to illegal or unethical activities, ensure these are excluded or reframed in a lawful and ethical manner.

3. **Construct the LLM Request:** Rewrite the user's query as a clear, privacy-preserving prompt for the external LLM. 
    - Remove or generalize any PII or sensitive details (e.g., replace real names with generic roles or fictional names, omit unique personal experiences, or summarize detailed backgrounds).
    - If the query requests creative content (e.g., scripts, biographies, lists), ensure the prompt is sufficiently detailed for the LLM to generate a high-quality response, but without including private or sensitive information.
    - If the query includes illegal or unethical activities, exclude these from the LLM request and focus on lawful, ethical alternatives.
    - Maintain the original intent and utility of the user's request as much as possible.

4. **Output Format:** Your response should include two sections:
    - **reasoning:** A concise explanation of your privacy-preserving strategy for this specific query.
    - **llm_request:** The rewritten, privacy-preserving prompt to be sent to the external LLM.

**Additional Notes:**
- Strive for a balance between preserving user privacy and maintaining the quality and usefulness of the LLM request.
- Avoid over-generalizing to the point where the LLM cannot provide a meaningful response.
- Do not include any information in the LLM request that could be used to identify the user or any other real individual.
- If the user query is already privacy-preserving, you may use it as-is, but still provide a brief reasoning statement confirming this.

This process ensures that user privacy is protected while leveraging the capabilities of external language models for a wide range of tasks, including creative writing, information synthesis, and idea generation.
2025/08/12 18:40:01 INFO dspy.evaluate.evaluate: Average Metric: 3.0 / 3 (100.0%)
2025/08/12 18:45:52 INFO dspy.evaluate.evaluate: Average Metric: 189.65 / 225 (84.3%)
2025/08/12 18:45:52 INFO dspy.teleprompt.gepa.gepa: Iteration 1: New program is on the linear pareto front
2025/08/12 18:45:52 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Full valset score for new program: 0.8428888888888889
2025/08/12 18:45:52 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Full train_val score for new program: 0.8428888888888889
2025/08/12 18:45:52 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Individual valset scores for new program: [1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 0.7, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.875, 1.0, 1.0, 1.0, 1.0, 0.0, 0.6, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 1.0, 0.875, 0.6666666666666667, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6666666666666667, 1.0, 1.0, 1.0, 1.0, 0.75, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 0.75, 1.0, 1.0, 0.5, 0.5, 0.0, 1.0, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 0.8333333333333334, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 0.5, 0.5, 1.0, 0.8333333333333334]
2025/08/12 18:45:52 INFO dspy.teleprompt.gepa.gepa: Iteration 1: New valset pareto front scores: [1.0, 0.5, 1.0, 1.0, 0.75, 1.0, 0.8333333333333334, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 0.7, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.75, 1.0, 1.0, 0.6666666666666667, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.875, 1.0, 1.0, 1.0, 1.0, 0.5, 0.9, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 1.0, 0.875, 0.6666666666666667, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6666666666666667, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 0.75, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.8333333333333334, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.8333333333333334]
2025/08/12 18:45:52 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Full valset pareto front score: 0.9004444444444444
2025/08/12 18:45:52 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Updated valset pareto front programs: [{1}, {0, 1}, {1}, {1}, {0}, {0, 1}, {0}, {0, 1}, {0, 1}, {1}, {0, 1}, {0, 1}, {0, 1}, {0}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0}, {0, 1}, {1}, {1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0}, {0, 1}, {0, 1}, {0, 1}, {1}, {0, 1}, {0, 1}, {0, 1}, {0}, {1}, {0}, {0}, {0}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {1}, {0, 1}, {1}, {0, 1}, {1}, {0, 1}, {0, 1}, {0}, {0}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {1}, {0, 1}, {0}, {0, 1}, {1}, {0, 1}, {1}, {0, 1}, {1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0}, {1}, {0, 1}, {1}, {1}, {0, 1}, {1}, {0, 1}, {1}, {0}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0}, {0, 1}, {0, 1}, {0, 1}, {1}, {1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {1}, {0, 1}, {1}, {0, 1}, {0, 1}, {1}, {1}, {0, 1}, {0, 1}, {1}, {1}, {1}, {1}, {1}, {0}, {1}, {1}, {0, 1}, {0, 1}, {1}, {0, 1}, {1}, {0, 1}, {0, 1}, {1}, {1}, {1}, {0, 1}, {0, 1}, {0, 1}, {0}, {1}, {0, 1}, {0, 1}, {0, 1}, {1}, {0, 1}, {1}, {1}, {0, 1}, {0}, {1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {1}, {0, 1}, {0, 1}, {0, 1}, {0}, {0}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0}, {0, 1}, {0, 1}, {0}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {1}, {0}, {0, 1}, {0, 1}, {0, 1}, {1}, {0, 1}, {0, 1}, {1}, {0}, {0, 1}, {0, 1}, {0}, {0, 1}, {0, 1}]
2025/08/12 18:45:52 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Best valset aggregate score so far: 0.8428888888888889
2025/08/12 18:45:52 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Best program as per aggregate score on train_val: 1
2025/08/12 18:45:52 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Best program as per aggregate score on valset: 1
2025/08/12 18:45:52 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Best score on valset: 0.8428888888888889
2025/08/12 18:45:52 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Best score on train_val: 0.8428888888888889
2025/08/12 18:45:52 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Linear pareto front program index: 1
2025/08/12 18:45:52 INFO dspy.teleprompt.gepa.gepa: Iteration 1: New program candidate index: 1
```

### Display the GEPA generated prompt[¶](#display-the-gepa-generated-prompt)

Note that since we allowed GEPA the budget to only generate 1 candidate, it has updated the prompt for only one of the predictors

In \[9\]:

Copied!

```
print(optimized_papillon.craft_redacted_request.predict.signature.instructions)
```

print(optimized_papillon.craft_redacted_request.predict.signature.instructions)

```
Task Description:
You will be given a private user query that may contain sensitive, personal, or identifying information. Your task is to transform this user query into a privacy-preserving prompt or request suitable for a powerful external large language model (LLM). The generated request should enable the external LLM to assist with the user’s original intent effectively, while ensuring that no private, personally identifiable, or sensitive information about the user is exposed or leaked.

Input Format:
- The user query is a free-form text input that may include explicit personal details, real names, descriptive traits, or specific events.
- The input may range from requests for creative writing, role descriptions, lists of hobbies, profiles, or any other user-generated content.
  
Detailed Task Requirements and Domain-Specific Considerations:
1. Privacy Preservation:
   - Do not include any directly identifying details, real names, or sensitive personal information in the generated request.
   - Replace or anonymize any personal data, including but not limited to names, dates, biographical details, or events tied to individuals.
   - Avoid leaking private user context or information that could be traced back to the user.

2. Maintaining Task Integrity:
   - The LLM request must capture the user’s original intention and task goals as faithfully as possible, phrased in a generalized and context-neutral way.
   - For content requests involving real-world entities (e.g., institutions, events), consider making the scenario fictionalized or anonymized if it involves potentially identifying or sensitive context.
   - For character or profile descriptions, abstract/remove intimate personal details but preserve key personality traits or narrative elements in a generalized manner.
   
3. Quality of Generated LLM Request:
   - The description should be clear, coherent, and sufficiently detailed to enable a relevant and helpful response from the LLM without needing further clarifications.
   - Avoid introducing new biases or assumptions outside the user’s input.
   - Exclude illegal or ethically sensitive activities where applicable, or transform them into acceptable analogues to maintain ethical standards.

4. Reasoning Component:
   - Alongside the final LLM request, provide a concise reasoning summary that explains how private data was identified and sanitized, the thought process behind generalization or anonymization, and how the final prompt preserves user intent while protecting privacy.

Generalizable Strategy:
- Identify sensitive or private details in the user query.
- Abstract or anonymize such details without losing essential meaning.
- Frame the request in terms of generic or fictional scenarios.
- Explicitly instruct the external LLM to avoid real or private details and to focus on a neutral, generic, or fictional perspective when relevant.
- Validate that no PII or identifiable content remains in the final prompt.

Outputs:
- A reasoning section summarizing privacy considerations and prompt construction approach.
- A final privacy-preserving LLM prompt/request ready to be sent to an external LLM.

This approach ensures strong privacy protection aligned with best practices while maximizing task quality and relevance.
```

In \[10\]:

Copied!

```
evaluate(optimized_papillon)
```

evaluate(optimized_papillon)

```
Average Metric: 184.26 / 214 (86.1%): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 214/214 [00:13<00:00, 15.90it/s]
```

```
2025/08/12 18:27:34 INFO dspy.evaluate.evaluate: Average Metric: 184.2602564102564 / 214 (86.1%)
```

```

```

|     | target_response                                                       | user_query                                                            | pii_str   | llm_request                                                           | llm_response                                                          | response                                                              | compute_overall_score                                                 |
| --- | --------------------------------------------------------------------- | --------------------------------------------------------------------- | --------- | --------------------------------------------------------------------- | --------------------------------------------------------------------- | --------------------------------------------------------------------- | --------------------------------------------------------------------- |
| 0   | Subject: Follow-up on Endpoint Management Solution Discussion Dear... | rewrite in professional "as discusseed in last meeting with you an... | effem     |                                                                       | balaji                                                                |                                                                       | balaji                                                                |
| 1   | Кампания "Share a Coke" была запущена Coca-Cola в 2011 году в Авст... | расскажи про PR-кампанию Coca-Cola “Share a Coke"                     | coca-cola | Please provide an overview of a well-known marketing campaign by a... | Certainly! One of the most famous marketing campaigns involving pe... | Конечно! PR-кампания Coca-Cola “Share a Coke” — это одна из самых ... | ✔️ [1.000]                                                            |
| 2   | Creating a Facebook ad campaign within a specific budget and targe... | give me a facebook ads campaign for the next 7 days for canva pro ... | jordan    |                                                                       | canva                                                                 | Create a detailed 7-day social media advertising campaign plan for... | Certainly! Below is a detailed 7-day social media advertising camp... |
| 3   | The delivery was scheduled for today, but due to a payment delay t... | The delivery supposed to be today. Seems our payment delay to Lvio... | seagate   |                                                                       | lviv                                                                  |                                                                       | ms. chan                                                              |
| 4   | Dear Ms. Amanda, I hope this email finds you well. I wanted to bri... | write a email to Ms, Amanda ; tell her, we have a way to overcome ... | india     |                                                                       | amanda                                                                |                                                                       | hermann(germany)                                                      |

... 209 more rows not displayed ...

Out\[10\]:

```
EvaluationResult(score=86.1, results=<list of 214 results>)
```

**Here, we see GEPA optimize the PAPILLON program from a score of 77% to 86% after proposing just 1 new candidate!**
