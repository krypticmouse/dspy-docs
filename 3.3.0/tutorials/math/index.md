# Tutorial: Math Reasoning[¶](#tutorial-math-reasoning)

Let's walk through a quick example of setting up a `dspy.ChainOfThought` module and optimizing it for answering algebra questions.

Install the latest DSPy via `pip install -U dspy` and follow along. You also need to run `pip install datasets`.

Recommended: Set up MLflow Tracing to understand what's happening under the hood.

### MLflow DSPy Integration[¶](#mlflow-dspy-integration)

[MLflow](https://mlflow.org/) is an LLMOps tool that natively integrates with DSPy and offer explainability and experiment tracking. In this tutorial, you can use MLflow to visualize prompts and optimization progress as traces to understand the DSPy's behavior better. You can set up MLflow easily by following the four steps below.

1. Install MLflow

```
%pip install mlflow>=2.20
```

2. Start MLflow UI in a separate terminal

```
mlflow ui --port 5000
```

3. Connect the notebook to MLflow

```
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DSPy")
```

4. Enabling tracing.

```
mlflow.dspy.autolog()
```

Once you have completed the steps above, you can see traces for each program execution on the notebook. They provide great visibility into the model's behavior and helps you understand the DSPy's concepts better throughout the tutorial.

To learn more about the integration, visit [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) as well.

Let's tell DSPy that we will use OpenAI's `gpt-4o-mini` in our modules. To authenticate, DSPy will look into your `OPENAI_API_KEY`. You can easily swap this out for [other providers or local models](https://github.com/stanfordnlp/dspy/blob/main/examples/migration.ipynb).

In \[1\]:

Copied!

```
import dspy

gpt4o_mini = dspy.LM('openai/gpt-4o-mini', max_tokens=2000)
gpt4o = dspy.LM('openai/gpt-4o', max_tokens=2000)
dspy.configure(lm=gpt4o_mini)  # we'll use gpt-4o-mini as the default LM, unless otherwise specified
```

import dspy gpt4o_mini = dspy.LM('openai/gpt-4o-mini', max_tokens=2000) gpt4o = dspy.LM('openai/gpt-4o', max_tokens=2000) dspy.configure(lm=gpt4o_mini) # we'll use gpt-4o-mini as the default LM, unless otherwise specified

Next, let's load some data examples from the [MATH](https://arxiv.org/abs/2103.03874) benchmark. We'll use a training split for optimization and evaluate it on a held-out dev set.

Please note that the following step will require:

```
%pip install git+https://github.com/hendrycks/math.git
```

In \[2\]:

Copied!

```
from dspy.datasets import MATH

dataset = MATH(subset='algebra')
print(len(dataset.train), len(dataset.dev))
```

from dspy.datasets import MATH dataset = MATH(subset='algebra') print(len(dataset.train), len(dataset.dev))

```
350 350
```

Let's inspect one example from the training set.

In \[3\]:

Copied!

```
example = dataset.train[0]
print("Question:", example.question)
print("Answer:", example.answer)
```

example = dataset.train[0] print("Question:", example.question) print("Answer:", example.answer)

```
Question: The doctor has told Cal O'Ree that during his ten weeks of working out at the gym, he can expect each week's weight loss to be $1\%$ of his weight at the end of the previous week. His weight at the beginning of the workouts is $244$ pounds. How many pounds does he expect to weigh at the end of the ten weeks? Express your answer to the nearest whole number.
Answer: 221
```

Now let's define our module. It's extremely simple: just a chain-of-thought step that takes a `question` and produces an `answer`.

In \[4\]:

Copied!

```
module = dspy.ChainOfThought("question -> answer")
module(question=example.question)
```

module = dspy.ChainOfThought("question -> answer") module(question=example.question)

Out\[4\]:

```
Prediction(
    reasoning="Cal O'Ree's weight loss each week is $1\\%$ of his weight at the end of the previous week. This means that at the end of each week, he retains $99\\%$ of his weight from the previous week. \n\nIf we denote his weight at the beginning as \\( W_0 = 244 \\) pounds, then his weight at the end of week \\( n \\) can be expressed as:\n\\[\nW_n = W_{n-1} \\times 0.99\n\\]\nThis can be simplified to:\n\\[\nW_n = W_0 \\times (0.99)^n\n\\]\nAfter 10 weeks, his weight will be:\n\\[\nW_{10} = 244 \\times (0.99)^{10}\n\\]\n\nNow, we calculate \\( (0.99)^{10} \\):\n\\[\n(0.99)^{10} \\approx 0.904382\n\\]\n\nNow, we can calculate his expected weight after 10 weeks:\n\\[\nW_{10} \\approx 244 \\times 0.904382 \\approx 220.5\n\\]\n\nRounding to the nearest whole number, Cal O'Ree can expect to weigh approximately \\( 221 \\) pounds at the end of the ten weeks.",
    answer='221'
)
```

Next, let's set up an evaluator for the zero-shot module above, before prompt optimization.

In \[5\]:

Copied!

```
THREADS = 24
kwargs = dict(num_threads=THREADS, display_progress=True, display_table=5)
evaluate = dspy.Evaluate(devset=dataset.dev, metric=dataset.metric, **kwargs)

evaluate(module)
```

THREADS = 24 kwargs = dict(num_threads=THREADS, display_progress=True, display_table=5) evaluate = dspy.Evaluate(devset=dataset.dev, metric=dataset.metric, \*\*kwargs) evaluate(module)

```
Average Metric: 259.00 / 350 (74.0%): 100%|██████████| 350/350 [01:30<00:00,  3.85it/s]
```

```
2024/11/28 18:41:55 INFO dspy.evaluate.evaluate: Average Metric: 259 / 350 (74.0%)
```

```

```

|     | question                                                              | example_reasoning                                                         | example_answer | pred_reasoning                                                           | pred_answer | method                                     |
| --- | --------------------------------------------------------------------- | ------------------------------------------------------------------------- | -------------- | ------------------------------------------------------------------------ | ----------- | ------------------------------------------ |
| 0   | What is the smallest integer value of $c$ such that the function $... | The given function has a domain of all real numbers if and only if...     | 1              | To determine the smallest integer value of ( c ) such that the f...      | 1           | ✔️ [True]                                  |
| 1   | What is the least value of $x$ that is a solution of $                | {-x+3}                                                                    | =7$?           | In order to have $                                                       | {-x+3}      | = 7$, we must have $-x + 3 = 7$ or $-x ... |
| 2   | Evaluate $\\left\\lceil -\\frac{7}{4}\\right\\rceil$.                 | $-\\frac{7}{4}$ is between $-1$ and $-2$, so $\\left\\lceil -\\frac{7}... | -1             | To evaluate (\\left\\lceil -\\frac{7}{4}\\right\\rceil), we first nee... | -1          | ✔️ [True]                                  |
| 3   | A triangle has vertices at coordinates $(11,1)$, $(2,3)$ and $(3,7... | We must find the distance between each pair of points by using the...     | 10             | To find the length of the longest side of the triangle with vertic...    | 10          | ✔️ [True]                                  |
| 4   | Let $f(x) = x + 2$ and $g(x) = 1/f(x)$. What is $g(f(-3))$?           | First, we find that $f(-3) = (-3) + 2 = -1$. Then, $$g(f(-3)) = g(...     | 1              | To find ( g(f(-3)) ), we first need to evaluate ( f(-3) ). The...        | 1           | ✔️ [True]                                  |

... 345 more rows not displayed ...

Out\[5\]:

```
74.0
```

Tracking Evaluation Results in MLflow Experiment

To track and visualize the evaluation results over time, you can record the results in MLflow Experiment.

```
import mlflow

# Start an MLflow Run to record the evaluation
with mlflow.start_run(run_name="math_evaluation"):
    kwargs = dict(num_threads=THREADS, display_progress=True)
    evaluate = dspy.Evaluate(devset=dataset.dev, metric=dataset.metric, **kwargs)

    # Evaluate the program as usual
    result = evaluate(module)

    # Log the aggregated score
    mlflow.log_metric("correctness", result.score)
    # Log the detailed evaluation results as a table
    mlflow.log_table(
        {
            "Question": [example.question for example in dataset.dev],
            "Gold Answer": [example.answer for example in dataset.dev],
            "Predicted Answer": [output[1] for output in result.results],
            "Correctness": [output[2] for output in result.results],
        },
        artifact_file="eval_results.json",
    )
```

To learn more about the integration, visit [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) as well.

And lastly let's optimize our module. Since we want strong reasoning, we'll use the large GPT-4o as the teacher model (used to bootstrap reasoning for the small LM at optimization time) but not as the prompt model (used to craft instructions) or the task model (trained).

GPT-4o will be invoked only a small number of times. The model involved directly in optimization and in the resulting (optimized) program will be GPT-4o-mini.

We will also specify `max_bootstrapped_demos=4` which means we want at most four bootstrapped examples in the prompt and `max_labeled_demos=4` which means that, in total between bootstrapped and pre-labeled examples, we want at most four.

In \[ \]:

Copied!

```
kwargs = dict(num_threads=THREADS, teacher_settings=dict(lm=gpt4o), prompt_model=gpt4o_mini)
optimizer = dspy.MIPROv2(metric=dataset.metric, auto="medium", **kwargs)

kwargs = dict(max_bootstrapped_demos=4, max_labeled_demos=4)
optimized_module = optimizer.compile(module, trainset=dataset.train, **kwargs)
```

kwargs = dict(num_threads=THREADS, teacher_settings=dict(lm=gpt4o), prompt_model=gpt4o_mini) optimizer = dspy.MIPROv2(metric=dataset.metric, auto="medium", \*\*kwargs) kwargs = dict(max_bootstrapped_demos=4, max_labeled_demos=4) optimized_module = optimizer.compile(module, trainset=dataset.train, \*\*kwargs)

In \[7\]:

Copied!

```
evaluate(optimized_module)
```

evaluate(optimized_module)

```
Average Metric: 310.00 / 350 (88.6%): 100%|██████████| 350/350 [01:31<00:00,  3.84it/s]
```

```
2024/11/28 18:59:19 INFO dspy.evaluate.evaluate: Average Metric: 310 / 350 (88.6%)
```

```

```

|     | question                                                              | example_reasoning                                                         | example_answer | pred_reasoning                                                           | pred_answer | method                                     |
| --- | --------------------------------------------------------------------- | ------------------------------------------------------------------------- | -------------- | ------------------------------------------------------------------------ | ----------- | ------------------------------------------ |
| 0   | What is the smallest integer value of $c$ such that the function $... | The given function has a domain of all real numbers if and only if...     | 1              | The function ( f(x) = \\frac{x^2 + 1}{x^2 - x + c} ) will have a ...     | 1           | ✔️ [True]                                  |
| 1   | What is the least value of $x$ that is a solution of $                | {-x+3}                                                                    | =7$?           | In order to have $                                                       | {-x+3}      | = 7$, we must have $-x + 3 = 7$ or $-x ... |
| 2   | Evaluate $\\left\\lceil -\\frac{7}{4}\\right\\rceil$.                 | $-\\frac{7}{4}$ is between $-1$ and $-2$, so $\\left\\lceil -\\frac{7}... | -1             | To evaluate (\\left\\lceil -\\frac{7}{4}\\right\\rceil), we first nee... | -1          | ✔️ [True]                                  |
| 3   | A triangle has vertices at coordinates $(11,1)$, $(2,3)$ and $(3,7... | We must find the distance between each pair of points by using the...     | 10             | To find the length of the sides of the triangle formed by the vert...    | 10          | ✔️ [True]                                  |
| 4   | Let $f(x) = x + 2$ and $g(x) = 1/f(x)$. What is $g(f(-3))$?           | First, we find that $f(-3) = (-3) + 2 = -1$. Then, $$g(f(-3)) = g(...     | 1              | To find ( g(f(-3)) ), we first need to evaluate ( f(-3) ). Usi...        | 1           | ✔️ [True]                                  |

... 345 more rows not displayed ...

Out\[7\]:

```
88.57
```

Neat. It was pretty straightforward to improve quality from 74% to over 88% on a held-out set here.

That said, for reasoning tasks like this, you will often want to consider more advanced strategies, like:

- A `dspy.ReAct` module with access to a calculator function or `dspy.LocalSandbox`
- Ensembling multiple optimized prompts with a majority vote (or an Aggregator module) on top

Just to understand what changed, let's view the prompt after optimization. Alternatively, if you enabled MLflow tracing following above instructions, you can compare the prompts before and after optimization in the rich trace UI.

In \[8\]:

Copied!

```
dspy.inspect_history()
```

dspy.inspect_history()

```
[2024-11-28T18:59:19.176586]

System message:

Your input fields are:
1. `question` (str)

Your output fields are:
1. `reasoning` (str)
2. `answer` (str)

All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## question ## ]]
{question}

[[ ## reasoning ## ]]
{reasoning}

[[ ## answer ## ]]
{answer}

[[ ## completed ## ]]

In adhering to this structure, your objective is: 
        Analyze the `question` provided, and systematically apply mathematical reasoning to derive the `answer`. Ensure to articulate each step of your thought process in the `reasoning` field, detailing how you identify relationships and formulate equations to arrive at the solution.


User message:

[[ ## question ## ]]
If $|4x+2|=10$ and $x<0$, what is the value of $x$?

Respond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]`, then `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.


Assistant message:

[[ ## reasoning ## ]]
The equation $|4x+2|=10$ implies two possible equations: $4x+2=10$ or $4x+2=-10$. We solve each equation separately.

1. For $4x+2=10$:
   \[
   4x + 2 = 10 \\
   4x = 8 \\
   x = 2
   \]
   However, $x=2$ does not satisfy the condition $x<0$.

2. For $4x+2=-10$:
   \[
   4x + 2 = -10 \\
   4x = -12 \\
   x = -3
   \]
   The solution $x = -3$ satisfies the condition $x<0$.

Therefore, the value of $x$ is $\boxed{-3}$.

[[ ## answer ## ]]
-3

[[ ## completed ## ]]


User message:

[[ ## question ## ]]
Given that $-4$ is a solution to $x^2 + bx -36 = 0$, what is the value of $b$?

Respond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]`, then `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.


Assistant message:

[[ ## reasoning ## ]]
The product of the roots of this quadratic is $-36/1=-36$, so the other solution must be $-36/-4=9$. That means that the sum of the solutions is $-4+9=5$. The sum of the solutions is also $-b/1=-b$. Thus, $-b=5$ and $b=\boxed{-5}$.

[[ ## answer ## ]]
-5

[[ ## completed ## ]]


User message:

[[ ## question ## ]]
Richard is building a rectangular playground from 200 feet of fencing. The fencing must entirely enclose the playground. What is the maximum area of this playground?

Respond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]`, then `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.


Assistant message:

[[ ## reasoning ## ]]
Let the length of the playground be $l$ and the width be $w$. We have the equation $2l+2w=200 \Rightarrow l + w = 100$. We want to maximize the area of this rectangular playground, which is given by $lw$. From our equation, we know that $l=100-w$. Substituting this in to our expression for area, we have \[(100-w)(w)=100w-w^2\]We will now complete the square to find the maximum value of this expression. Factoring a $-1$ out, we have \[-(w^2-100w)\]In order for the expression inside the parenthesis to be a perfect square, we need to add and subtract $(100/2)^2=2500$ inside the parenthesis. Doing this, we get \[-(w^2-100w+2500-2500) \Rightarrow -(w-50)^2+2500\]Since the maximum value of $-(w-50)^2$ is 0 (perfect squares are always nonnegative), the maximum value of the entire expression is 2500, which is achieved when $w=50$ and $l=100-w=50$ (the playground is a square). Thus, the maximum area of the playground is $\boxed{2500}$ square feet.

[[ ## answer ## ]]
2500

[[ ## completed ## ]]


User message:

[[ ## question ## ]]
Alice and Bob are playing a game. Alice starts first. On Alice's turn, she flips a coin. If she gets a heads, she wins. If not, it becomes Bob's turn. On Bob's turn, he flips a coin. If he gets a tails, he wins. If not, it becomes Alice's turn. What is the probability that Alice wins the game?

Respond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]`, then `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.


Assistant message:

[[ ## reasoning ## ]]
Alice has a $1/2$ chance of winning the game on her first turn. If she doesn't, then the probability that she wins the game on her second turn is $1/8,$ since she must not win on her first flip ($1/2$ chance), Bob must not win on his first flip ($1/2$ chance), and then Alice must win on her second flip ($1/2$ chance). The probability that she wins the game on her third turn is $1/32,$ and in general, the probability that she wins the game on her $k^\text{th}$ turn is $(1/2)^{2k-1}.$ Thus, the probability that Alice wins is an infinite geometric series with first term $1/2$ and common ratio $1/4.$ So, the probability that Alice wins the game is $$\frac{\frac12}{1-\frac14} = \boxed{\frac{2}{3}}.$$OR

Note that the only difference between the odds of Alice or Bob winning is who goes first. Because Bob goes second, the odds of him winning on his $k^\text{th}$ flip is half of the odds that Alice wins on her $k^\text{th}$ flip, since Alice must first get a tails before Bob gets a chance to win. Thus, if $a$ is Alice's chance of winning, and $b$ is Bob's chance of winning, then $a = 2b.$ Also, since someone must win, $a + b = 1.$ It follows that $a = 2/3$ and $b = 1/3,$ so Alice has a $\boxed{\frac{2}{3}}$ chance of winning the game.

[[ ## answer ## ]]
\frac{2}{3}

[[ ## completed ## ]]


User message:

[[ ## question ## ]]
Find the smallest possible value of $$\frac{(y-x)^2}{(y-z)(z-x)} + \frac{(z-y)^2}{(z-x)(x-y)} + \frac{(x-z)^2}{(x-y)(y-z)},$$ where $x,y,$ and $z$ are distinct real numbers.

Respond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]`, then `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.


Response:

[[ ## reasoning ## ]]
Let \( S = \frac{(y-x)^2}{(y-z)(z-x)} + \frac{(z-y)^2}{(z-x)(x-y)} + \frac{(x-z)^2}{(x-y)(y-z)} \).

To find the minimum value of \( S \), we can use the method of symmetry and the Cauchy-Schwarz inequality. 

By applying the Cauchy-Schwarz inequality, we have:
\[
\left( \frac{(y-x)^2}{(y-z)(z-x)} + \frac{(z-y)^2}{(z-x)(x-y)} + \frac{(x-z)^2}{(x-y)(y-z)} \right) \left( (y-z)(z-x) + (z-x)(x-y) + (x-y)(y-z) \right) \geq \left( (y-x) + (z-y) + (x-z) \right)^2.
\]
The right-hand side simplifies to:
\[
(y-x) + (z-y) + (x-z) = 0,
\]
which means the right-hand side is \( 0 \). Therefore, we have:
\[
S \cdot \left( (y-z)(z-x) + (z-x)(x-y) + (x-y)(y-z) \right) \geq 0.
\]
This implies that \( S \) is non-negative.

Next, we can test specific values for \( x, y, z \) to find the minimum. Let's set \( x = 0, y = 1, z = 2 \):
\[
S = \frac{(1-0)^2}{(1-2)(2-0)} + \frac{(2-1)^2}{(2-0)(0-1)} + \frac{(0-2)^2}{(0-1)(1-2)}.
\]
Calculating each term:
1. The first term:
   \[
   \frac{1^2}{(-1)(2)} = \frac{1}{-2} = -\frac{1}{2}.
   \]
2. The second term:
   \[
   \frac{1^2}{(2)(-1)} = \frac{1}{-2} = -\frac{1}{2}.
   \]
3. The third term:
   \[
   \frac{(-2)^2}{(-1)(-1)} = \frac{4}{1} = 4.
   \]

Thus,
\[
S = -\frac{1}{2} - \frac{1}{2} + 4 = 3.
\]

To check if this is indeed the minimum, we can also consider the case when \( x, y, z \) are the roots of a quadratic polynomial. By symmetry and the nature of the expression, we can conclude that the minimum value of \( S \) is indeed \( 3 \).

Therefore, the smallest possible value of \( S \) is \( \boxed{3} \).

[[ ## answer ## ]]
3

[[ ## completed ## ]]
```
