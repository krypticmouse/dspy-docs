# Tutorial: Using Audio in DSPy Programs

This tutorial walks through building pipelines for audio-based applications using DSPy.

### Install Dependencies

Ensure you’re using the latest DSPy version:

```
pip install -U dspy
```

To handle audio data, install the following dependencies:

```
pip install datasets soundfile torch==2.0.1+cu118 torchaudio==2.0.2+cu118
```

### Load the Spoken-SQuAD Dataset

We’ll use the Spoken-SQuAD dataset ([Official](https://github.com/Chia-Hsuan-Lee/Spoken-SQuAD) & [HuggingFace version](https://huggingface.co/datasets/AudioLLMs/spoken_squad_test) for tutorial demonstration), which contains spoken audio passages used for question-answering:

```
import random
import dspy
from dspy.datasets import DataLoader

kwargs = dict(fields=("context", "instruction", "answer"), input_keys=("context", "instruction"))
spoken_squad = DataLoader().from_huggingface(dataset_name="AudioLLMs/spoken_squad_test", split="train", trust_remote_code=True, **kwargs)

random.Random(42).shuffle(spoken_squad)
spoken_squad = spoken_squad[:100]

split_idx = len(spoken_squad) // 2
trainset_raw, testset_raw = spoken_squad[:split_idx], spoken_squad[split_idx:]
```

### Preprocess Audio Data

The audio clips in the dataset require some preprocessing into byte arrays with their corresponding sampling rates.

```
def preprocess(x):
    audio = dspy.Audio.from_array(x.context["array"], x.context["sampling_rate"])
    return dspy.Example(
        passage_audio=audio,
        question=x.instruction,
        answer=x.answer
    ).with_inputs("passage_audio", "question")

trainset = [preprocess(x) for x in trainset_raw]
testset = [preprocess(x) for x in testset_raw]

len(trainset), len(testset)
```

## DSPy program for spoken question answering

Let’s define a simple DSPy program that uses audio inputs to answer questions directly. This is very similar to the [BasicQA](https://dspy.ai/cheatsheet/?h=basicqa#dspysignature) task, with the only difference being that the passage context is provided as an audio file for the model to listen to and answer the question:

```
class SpokenQASignature(dspy.Signature):
    """Answer the question based on the audio clip."""
    passage_audio: dspy.Audio = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc = 'factoid answer between 1 and 5 words')

spoken_qa = dspy.ChainOfThought(SpokenQASignature)
```

Now let’s configure our LLM which can process input audio.

```
dspy.configure(lm=dspy.LM(model='gpt-4o-mini-audio-preview-2024-12-17'))
```

Note: Using `dspy.Audio` in signatures allows passing in audio directly to the model.

### Define Evaluation Metric

We’ll use the Exact Match metric (`dspy.evaluate.answer_exact_match`) to measure answer accuracy compared to the provided reference answers:

```
evaluate_program = dspy.Evaluate(devset=testset, metric=dspy.evaluate.answer_exact_match,display_progress=True, num_threads = 10, display_table=True)

evaluate_program(spoken_qa)
```

### Optimize with DSPy

You can optimize this audio-based program as you would for any DSPy program using any DSPy optimizer.

Note: Audio tokens can be costly so it is recommended to configure optimizers like `dspy.BootstrapFewShotWithRandomSearch` or `dspy.MIPROv2` conservatively with 0-2 few shot examples and less candidates / trials than the optimizer default parameters.

```
optimizer = dspy.BootstrapFewShotWithRandomSearch(metric = dspy.evaluate.answer_exact_match, max_bootstrapped_demos=2, max_labeled_demos=2, num_candidate_programs=5)

optimized_program = optimizer.compile(spoken_qa, trainset = trainset)

evaluate_program(optimized_program)
```

```
prompt_lm = dspy.LM(model='gpt-4o-mini') #NOTE - this is the LLM guiding the MIPROv2 instruction candidate proposal
optimizer = dspy.MIPROv2(metric=dspy.evaluate.answer_exact_match, auto="light", prompt_model = prompt_lm)

#NOTE - MIPROv2's dataset summarizer cannot process the audio files in the dataset, so we turn off the data_aware_proposer 
optimized_program = optimizer.compile(spoken_qa, trainset=trainset, max_bootstrapped_demos=2, max_labeled_demos=2, data_aware_proposer=False)

evaluate_program(optimized_program)
```

With this small subset, MIPROv2 led to a ~10% improvement over baseline performance.

______________________________________________________________________

Now that we’ve seen how to use an audio-input-capable LLM in DSPy, let’s flip the setup.

In this next task, we’ll use a standard text-based LLM to generate prompts for a text-to-speech model and then evaluate the quality of the produced speech for some downstream task. This approach is generally more cost-effective than asking an LLM like `gpt-4o-mini-audio-preview-2024-12-17` to generate audio directly, while still enabling a pipeline that can be optimized for higher-quality speech output.

### Load the CREMA-D Dataset

We’ll use the CREMA-D dataset ([Official](https://github.com/CheyneyComputerScience/CREMA-D) & [HuggingFace version](https://huggingface.co/datasets/myleslinder/crema-d) for tutorial demonstration), which includes audio clips of chosen participants speaking the same line with one of six target emotions: neutral, happy, sad, anger, fear, and disgust.

```
from collections import defaultdict

label_map = ['neutral', 'happy', 'sad', 'anger', 'fear', 'disgust']

kwargs = dict(fields=("sentence", "label", "audio"), input_keys=("sentence", "label"))
crema_d = DataLoader().from_huggingface(dataset_name="myleslinder/crema-d", split="train", trust_remote_code=True, **kwargs)

def preprocess(x):
    return dspy.Example(
        raw_line=x.sentence,
        target_style=label_map[x.label],
        reference_audio=dspy.Audio.from_array(x.audio["array"], x.audio["sampling_rate"])
    ).with_inputs("raw_line", "target_style")

random.Random(42).shuffle(crema_d)
crema_d = crema_d[:100]

random.seed(42)
label_to_indices = defaultdict(list)
for idx, x in enumerate(crema_d):
    label_to_indices[x.label].append(idx)

per_label = 100 // len(label_map)
train_indices, test_indices = [], []
for indices in label_to_indices.values():
    selected = random.sample(indices, min(per_label, len(indices)))
    split = len(selected) // 2
    train_indices.extend(selected[:split])
    test_indices.extend(selected[split:])

trainset = [preprocess(crema_d[idx]) for idx in train_indices]
testset = [preprocess(crema_d[idx]) for idx in test_indices]
```

## DSPy pipeline for generating TTS instructions for speaking with a target emotion

We’ll now build a pipeline that generates emotionally expressive speech by prompting a TTS model with both a line of text and an instruction on how to say it. The goal of this task will be to use DSPy to generate prompts that guide the TTS output to match the emotion and style of reference audio from the dataset.

First let’s set up the TTS generator to produce generate spoken audio with a specified emotion or style. We utilize `gpt-4o-mini-tts` as it supports prompting the model with raw input and speaking and produces an audio response as a `.wav` file processed with `dspy.Audio`. We also set up a cache for the TTS outputs.

```
import os
import base64
import hashlib
from openai import OpenAI

CACHE_DIR = ".audio_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def hash_key(raw_line: str, prompt: str) -> str:
    return hashlib.sha256(f"{raw_line}|||{prompt}".encode("utf-8")).hexdigest()

def generate_dspy_audio(raw_line: str, prompt: str) -> dspy.Audio:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    key = hash_key(raw_line, prompt)
    wav_path = os.path.join(CACHE_DIR, f"{key}.wav")
    if not os.path.exists(wav_path):
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="coral", #NOTE - this can be configured to any of the 11 offered OpenAI TTS voices - https://platform.openai.com/docs/guides/text-to-speech#voice-options. 
            input=raw_line,
            instructions=prompt,
            response_format="wav"
        )
        with open(wav_path, "wb") as f:
            f.write(response.content)
    with open(wav_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return dspy.Audio(data=encoded, audio_format="wav")
```

Now let’s define the DSPy program for generating TTS instructions. For this program, we can use standard text-based LLMs again since we’re just generating instructions.

```
class EmotionStylePromptSignature(dspy.Signature):
    """Generate an OpenAI TTS instruction that makes the TTS model speak the given line with the target emotion or style."""
    raw_line: str = dspy.InputField()
    target_style: str = dspy.InputField()
    openai_instruction: str = dspy.OutputField()

class EmotionStylePrompter(dspy.Module):
    def __init__(self):
        self.prompter = dspy.ChainOfThought(EmotionStylePromptSignature)

    def forward(self, raw_line, target_style):
        out = self.prompter(raw_line=raw_line, target_style=target_style)
        audio = generate_dspy_audio(raw_line, out.openai_instruction)
        return dspy.Prediction(audio=audio)

dspy.configure(lm=dspy.LM(model='gpt-4o-mini'))
```

### Define Evaluation Metric

Audio reference comparisons is generally a non-trivial task due to subjective variations of evaluating speech, especially with emotional expression. For the purposes of this tutorial, we use an embedding-based similarity metric for objective evaluation, leveraging Wav2Vec 2.0 to convert audio into embeddings and computing cosine similarity between the reference and generated audio. To evaluate audio quality more accurately, human feedback or perceptual metrics would be more suitable.

```
import torch
import torchaudio
import soundfile as sf
import io

bundle = torchaudio.pipelines.WAV2VEC2_BASE
model = bundle.get_model().eval()

def decode_dspy_audio(dspy_audio):
    audio_bytes = base64.b64decode(dspy_audio.data)
    array, _ = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    return torch.tensor(array).unsqueeze(0)

def extract_embedding(audio_tensor):
    with torch.inference_mode():
        return model(audio_tensor)[0].mean(dim=1)

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a, b).item()

def audio_similarity_metric(example, pred, trace=None):
    ref_audio = decode_dspy_audio(example.reference_audio)
    gen_audio = decode_dspy_audio(pred.audio)

    ref_embed = extract_embedding(ref_audio)
    gen_embed = extract_embedding(gen_audio)

    score = cosine_similarity(ref_embed, gen_embed)

    if trace is not None:
        return score > 0.8 
    return score

evaluate_program = dspy.Evaluate(devset=testset, metric=audio_similarity_metric, display_progress=True, num_threads = 10, display_table=True)

evaluate_program(EmotionStylePrompter())
```

We can look at an example to see what instructions the DSPy program generated and the corresponding score:

```
program = EmotionStylePrompter()

pred = program(raw_line=testset[1].raw_line, target_style=testset[1].target_style)

print(audio_similarity_metric(testset[1], pred)) #0.5725605487823486

dspy.inspect_history(n=1)
```

```
[34m[2025-05-15T22:01:22.667596][0m

[31mSystem message:[0m

Your input fields are:
1. `raw_line` (str)
2. `target_style` (str)
Your output fields are:
1. `reasoning` (str)
2. `openai_instruction` (str)
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## raw_line ## ]]
{raw_line}

[[ ## target_style ## ]]
{target_style}

[[ ## reasoning ## ]]
{reasoning}

[[ ## openai_instruction ## ]]
{openai_instruction}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        Generate an OpenAI TTS instruction that makes the TTS model speak the given line with the target emotion or style.


[31mUser message:[0m

[[ ## raw_line ## ]]
It's eleven o'clock

[[ ## target_style ## ]]
disgust

Respond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]`, then `[[ ## openai_instruction ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.


[31mResponse:[0m

[32m[[ ## reasoning ## ]]
To generate the OpenAI TTS instruction, we need to specify the target emotion or style, which in this case is 'disgust'. We will use the OpenAI TTS instruction format, which includes the text to be spoken and the desired emotion or style.

[[ ## openai_instruction ## ]]
"Speak the following line with a tone of disgust: It's eleven o'clock"

[[ ## completed ## ]][0m
```

TTS Instruction:

```
Speak the following line with a tone of disgust: It's eleven o'clock
```

```
from IPython.display import Audio

audio_bytes = base64.b64decode(pred.audio.data)
array, rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
Audio(array, rate=rate)
```

Your browser does not support the audio element.

The instruction specifies the target emotion, but is not too informative beyond that. We can also see that the audio score for this sample is not too high. Let’s see if we can do better by optimizing this pipeline.

### Optimize with DSPy

We can leverage `dspy.MIPROv2` to refine the downstream task objective and produce higher quality TTS instructions, leading to more accurate and expressive audio generations:

```
prompt_lm = dspy.LM(model='gpt-4o-mini')

teleprompter = dspy.MIPROv2(metric=audio_similarity_metric, auto="light", prompt_model = prompt_lm)

optimized_program = teleprompter.compile(EmotionStylePrompter(),trainset=trainset)

evaluate_program(optimized_program)
```

Let’s take a look at how the optimized program performs:

```
pred = optimized_program(raw_line=testset[1].raw_line, target_style=testset[1].target_style)

print(audio_similarity_metric(testset[1], pred)) #0.6691027879714966

dspy.inspect_history(n=1)
```

```
[34m[2025-05-15T22:09:40.088592][0m

[31mSystem message:[0m

Your input fields are:
1. `raw_line` (str)
2. `target_style` (str)
Your output fields are:
1. `reasoning` (str)
2. `openai_instruction` (str)
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## raw_line ## ]]
{raw_line}

[[ ## target_style ## ]]
{target_style}

[[ ## reasoning ## ]]
{reasoning}

[[ ## openai_instruction ## ]]
{openai_instruction}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        Generate an OpenAI TTS instruction that makes the TTS model speak the given line with the target emotion or style, as if the speaker is a [insert persona relevant to the task, e.g. "irate customer", "angry boss", etc.]. The instruction should specify the tone, pitch, and other characteristics of the speaker's voice to convey the target emotion.


[31mUser message:[0m

[[ ## raw_line ## ]]
It's eleven o'clock

[[ ## target_style ## ]]
disgust

Respond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]`, then `[[ ## openai_instruction ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.


[31mResponse:[0m

[32m[[ ## reasoning ## ]]
To convey disgust, the speaker's voice should be characterized by a high-pitched tone, a slightly nasal quality, and a sense of revulsion. The speaker's words should be delivered with a sense of distaste and aversion, as if the speaker is trying to convey their strong negative emotions.

[[ ## openai_instruction ## ]]
Generate a text-to-speech synthesis of the input text "It's eleven o'clock" with the following characteristics: 
- Tone: Disgusted
- Pitch: High-pitched, slightly nasal
- Emphasis: Emphasize the words to convey a sense of distaste and aversion
- Volume: Moderate to loud, with a sense of rising inflection at the end to convey the speaker's strong negative emotions
- Speaker: A person who is visibly and audibly disgusted, such as a character who has just been served a spoiled meal.

[[ ## completed ## ]][0m
```

MIPROv2 Optimized Program Instruction:

```
Generate an OpenAI TTS instruction that makes the TTS model speak the given line with the target emotion or style, as if the speaker is a [insert persona relevant to the task, e.g. "irate customer", "angry boss", etc.]. The instruction should specify the tone, pitch, and other characteristics of the speaker's voice to convey the target emotion.
```

TTS Instruction:

```
Generate a text-to-speech synthesis of the input text "It's eleven o'clock" with the following characteristics: 
- Tone: Disgusted
- Pitch: High-pitched, slightly nasal
- Emphasis: Emphasize the words to convey a sense of distaste and aversion
- Volume: Moderate to loud, with a sense of rising inflection at the end to convey the speaker's strong negative emotions
- Speaker: A person who is visibly and audibly disgusted, such as a character who has just been served a spoiled meal.
```

```
from IPython.display import Audio

audio_bytes = base64.b64decode(pred.audio.data)
array, rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
Audio(array, rate=rate)
```

Your browser does not support the audio element.

MIPROv2’s instruction tuning added more flavor to the overall task objective, giving more criteria to how the TTS instruction should be defined, and in turn, the generated instruction is much more specific to the various factors of speech prosody and produces a higher similarity score.
