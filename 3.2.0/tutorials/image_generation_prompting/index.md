# Image Generation Prompt iteration[¶](#image-generation-prompt-iteration)

This is based off of a tweet from [@ThorondorLLC](https://x.com/ThorondorLLC)

Tweet is [here](https://x.com/ThorondorLLC/status/1880048546382221313)

This will take an initial desired prompt, and iteratively refine it until the image generated matches the desired prompt.

This is not DSPy prompt optimization as it is normally used, but it is a good example of how to use multimodal DSPy.

A future upgrade would be to create a dataset of initial, final prompts to optimize the prompt generation.

You can install DSPy via:

```
pip install -U dspy
```

For this example, we'll use Flux Pro from FAL. You can get an API key [here](https://fal.com/flux-pro)

We will also need to install Pillow and dotenv.

```
pip install fal-client pillow dotenv
```

Now, let's import the necessary libraries and set up the environment:

In \[ \]:

Copied!

```
# Optional
#os.environ["FAL_API_KEY"] = "your_fal_api_key"
#os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
```

# Optional

#os.environ["FAL_API_KEY"] = "your_fal_api_key" #os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

In \[1\]:

Copied!

```
import dspy

from PIL import Image
from io import BytesIO
import requests
import fal_client

from dotenv import load_dotenv
load_dotenv()

# import display
from IPython.display import display

lm = dspy.LM(model="gpt-4o-mini", temperature=0.5)
dspy.configure(lm=lm)
```

import dspy from PIL import Image from io import BytesIO import requests import fal_client from dotenv import load_dotenv load_dotenv()

# import display

from IPython.display import display lm = dspy.LM(model="gpt-4o-mini", temperature=0.5) dspy.configure(lm=lm)

```
/Users/isaac/sd_optimizer/.venv/lib/python3.12/site-packages/pydantic/_internal/_config.py:345: UserWarning: Valid config keys have changed in V2:
* 'fields' has been removed
  warnings.warn(message, UserWarning)
/Users/isaac/sd_optimizer/.venv/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
```

In \[9\]:

Copied!

```
def generate_image(prompt):

    request_id = fal_client.submit(
        "fal-ai/flux-pro/v1.1-ultra",
        arguments={
            "prompt": prompt
        },
    ).request_id

    result = fal_client.result("fal-ai/flux-pro/v1.1-ultra", request_id)
    url = result["images"][0]["url"]

    return dspy.Image.from_url(url)

def display_image(image):
    url = image.url
    # download the image
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))

    # display at 25% of original size
    display(image.resize((image.width // 4, image.height // 4)))
```

def generate_image(prompt): request_id = fal_client.submit( "fal-ai/flux-pro/v1.1-ultra", arguments={ "prompt": prompt }, ).request_id result = fal_client.result("fal-ai/flux-pro/v1.1-ultra", request_id) url = result["images"][0]["url"] return dspy.Image.from_url(url) def display_image(image): url = image.url

# download the image

response = requests.get(url) image = Image.open(BytesIO(response.content))

# display at 25% of original size

display(image.resize((image.width // 4, image.height // 4)))

In \[18\]:

Copied!

```
check_and_revise_prompt = dspy.Predict("desired_prompt: str, current_image: dspy.Image, current_prompt:str -> feedback:str, image_strictly_matches_desired_prompt: bool, revised_prompt: str")

initial_prompt = "A scene that's both peaceful and tense"
current_prompt = initial_prompt

max_iter = 5
for i in range(max_iter):
    print(f"Iteration {i+1} of {max_iter}")
    current_image = generate_image(current_prompt)
    result = check_and_revise_prompt(desired_prompt=initial_prompt, current_image=current_image, current_prompt=current_prompt)
    display_image(current_image)
    if result.image_strictly_matches_desired_prompt:
        break
    else:
        current_prompt = result.revised_prompt
        print(f"Feedback: {result.feedback}")
        print(f"Revised prompt: {result.revised_prompt}")

print(f"Final prompt: {current_prompt}")
```

check_and_revise_prompt = dspy.Predict("desired_prompt: str, current_image: dspy.Image, current_prompt:str -> feedback:str, image_strictly_matches_desired_prompt: bool, revised_prompt: str") initial_prompt = "A scene that's both peaceful and tense" current_prompt = initial_prompt max_iter = 5 for i in range(max_iter): print(f"Iteration {i+1} of {max_iter}") current_image = generate_image(current_prompt) result = check_and_revise_prompt(desired_prompt=initial_prompt, current_image=current_image, current_prompt=current_prompt) display_image(current_image) if result.image_strictly_matches_desired_prompt: break else: current_prompt = result.revised_prompt print(f"Feedback: {result.feedback}") print(f"Revised prompt: {result.revised_prompt}") print(f"Final prompt: {current_prompt}")

```
Iteration 1 of 5
```

```
Feedback: The image depicts a peaceful autumn scene with people walking among colorful leaves, which aligns with the peaceful aspect of the prompt. However, it lacks any elements that convey tension, making it not fully representative of the desired prompt.
Iteration 2 of 5
```

```
Feedback: The image depicts a serene autumn scene with vibrant foliage and a calm river, which aligns well with the idea of peace. However, it lacks explicit elements that suggest underlying tension, making it less effective in conveying both aspects of the desired prompt.
Iteration 3 of 5
```

```
Feedback: The image depicts a serene autumn scene with warm colors and soft lighting, which aligns with the peaceful aspect of the desired prompt. However, it lacks elements that evoke tension or unease, making it not fully meet the requirement for a scene that is both peaceful and tense.
Iteration 4 of 5
```

```
Final prompt: A serene autumn scene with fog and shadows, capturing both peace and tension.
```

In \[19\]:

Copied!

```
dspy.inspect_history(5)
```

dspy.inspect_history(5)

```
[2025-01-17T11:36:55.947579]

System message:

Your input fields are:
1. `desired_prompt` (str)
2. `current_image` (Image)
3. `current_prompt` (str)

Your output fields are:
1. `feedback` (str)
2. `revised_prompt` (str)
3. `image_strictly_matches_desired_prompt` (bool)

All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## desired_prompt ## ]]
{desired_prompt}

[[ ## current_image ## ]]
{current_image}

[[ ## current_prompt ## ]]
{current_prompt}

[[ ## feedback ## ]]
{feedback}

[[ ## revised_prompt ## ]]
{revised_prompt}

[[ ## image_strictly_matches_desired_prompt ## ]]
{image_strictly_matches_desired_prompt}        # note: the value you produce must be True or False

[[ ## completed ## ]]

In adhering to this structure, your objective is: 
        Given the fields `desired_prompt`, `current_image`, `current_prompt`, produce the fields `feedback`, `revised_prompt`, `image_strictly_matches_desired_prompt`.


User message:

[[ ## desired_prompt ## ]]
A scene that's both peaceful and tense

[[ ## current_image ## ]]
<image_url: https://fal.media/files/panda/HLKaiKFc_sN_fwBBN1C_2_1d1a72732e294452afb056a26d6b6c96.jpg>

[[ ## current_prompt ## ]]
A scene that's both peaceful and tense

Respond with the corresponding output fields, starting with the field `[[ ## feedback ## ]]`, then `[[ ## revised_prompt ## ]]`, then `[[ ## image_strictly_matches_desired_prompt ## ]]` (must be formatted as a valid Python bool), and then ending with the marker for `[[ ## completed ## ]]`.


Response:

[[ ## feedback ## ]]
The image depicts a calm café scene with two individuals engaged in conversation, which conveys a sense of peace. However, the woman's posture and expression suggest an underlying tension, as she appears to be contemplative or distressed. This duality aligns well with the desired prompt of being both peaceful and tense.

[[ ## revised_prompt ## ]]
A scene that captures the duality of peace and tension in a café setting.

[[ ## image_strictly_matches_desired_prompt ## ]]
True

[[ ## completed ## ]]





[2025-01-17T11:37:55.458433]

System message:

Your input fields are:
1. `desired_prompt` (str)
2. `current_image` (Image)
3. `current_prompt` (str)

Your output fields are:
1. `feedback` (str)
2. `image_strictly_matches_desired_prompt` (bool)
3. `revised_prompt` (str)

All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## desired_prompt ## ]]
{desired_prompt}

[[ ## current_image ## ]]
{current_image}

[[ ## current_prompt ## ]]
{current_prompt}

[[ ## feedback ## ]]
{feedback}

[[ ## image_strictly_matches_desired_prompt ## ]]
{image_strictly_matches_desired_prompt}        # note: the value you produce must be True or False

[[ ## revised_prompt ## ]]
{revised_prompt}

[[ ## completed ## ]]

In adhering to this structure, your objective is: 
        Given the fields `desired_prompt`, `current_image`, `current_prompt`, produce the fields `feedback`, `image_strictly_matches_desired_prompt`, `revised_prompt`.


User message:

[[ ## desired_prompt ## ]]
A scene that's both peaceful and tense

[[ ## current_image ## ]]
<image_url: https://fal.media/files/elephant/p3M00ca786Ov5ITX8nXUs_7b815b2087e54c1fbd798b9b9e0c5f5e.jpg>

[[ ## current_prompt ## ]]
A scene that's both peaceful and tense

Respond with the corresponding output fields, starting with the field `[[ ## feedback ## ]]`, then `[[ ## image_strictly_matches_desired_prompt ## ]]` (must be formatted as a valid Python bool), then `[[ ## revised_prompt ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.


Response:

[[ ## feedback ## ]]
The image depicts a peaceful autumn scene with people walking among colorful leaves, which aligns with the peaceful aspect of the prompt. However, it lacks any elements that convey tension, making it not fully representative of the desired prompt.

[[ ## image_strictly_matches_desired_prompt ## ]]
False

[[ ## revised_prompt ## ]]
A serene autumn scene with elements that suggest underlying tension

[[ ## completed ## ]]





[2025-01-17T11:38:24.032318]

System message:

Your input fields are:
1. `desired_prompt` (str)
2. `current_image` (Image)
3. `current_prompt` (str)

Your output fields are:
1. `feedback` (str)
2. `image_strictly_matches_desired_prompt` (bool)
3. `revised_prompt` (str)

All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## desired_prompt ## ]]
{desired_prompt}

[[ ## current_image ## ]]
{current_image}

[[ ## current_prompt ## ]]
{current_prompt}

[[ ## feedback ## ]]
{feedback}

[[ ## image_strictly_matches_desired_prompt ## ]]
{image_strictly_matches_desired_prompt}        # note: the value you produce must be True or False

[[ ## revised_prompt ## ]]
{revised_prompt}

[[ ## completed ## ]]

In adhering to this structure, your objective is: 
        Given the fields `desired_prompt`, `current_image`, `current_prompt`, produce the fields `feedback`, `image_strictly_matches_desired_prompt`, `revised_prompt`.


User message:

[[ ## desired_prompt ## ]]
A scene that's both peaceful and tense

[[ ## current_image ## ]]
<image_url: https://fal.media/files/monkey/QhPlYYa5tob38190Gtn0t_17e963cacc1542d5977031aff954c718.jpg>

[[ ## current_prompt ## ]]
A serene autumn scene with elements that suggest underlying tension

Respond with the corresponding output fields, starting with the field `[[ ## feedback ## ]]`, then `[[ ## image_strictly_matches_desired_prompt ## ]]` (must be formatted as a valid Python bool), then `[[ ## revised_prompt ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.


Response:

[[ ## feedback ## ]]
The image depicts a serene autumn scene with vibrant foliage and a calm river, which aligns well with the idea of peace. However, it lacks explicit elements that suggest underlying tension, making it less effective in conveying both aspects of the desired prompt.

[[ ## image_strictly_matches_desired_prompt ## ]]
False

[[ ## revised_prompt ## ]]
A serene autumn scene with elements that evoke a sense of unease or foreboding

[[ ## completed ## ]]





[2025-01-17T11:38:46.424883]

System message:

Your input fields are:
1. `desired_prompt` (str)
2. `current_image` (Image)
3. `current_prompt` (str)

Your output fields are:
1. `feedback` (str)
2. `image_strictly_matches_desired_prompt` (bool)
3. `revised_prompt` (str)

All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## desired_prompt ## ]]
{desired_prompt}

[[ ## current_image ## ]]
{current_image}

[[ ## current_prompt ## ]]
{current_prompt}

[[ ## feedback ## ]]
{feedback}

[[ ## image_strictly_matches_desired_prompt ## ]]
{image_strictly_matches_desired_prompt}        # note: the value you produce must be True or False

[[ ## revised_prompt ## ]]
{revised_prompt}

[[ ## completed ## ]]

In adhering to this structure, your objective is: 
        Given the fields `desired_prompt`, `current_image`, `current_prompt`, produce the fields `feedback`, `image_strictly_matches_desired_prompt`, `revised_prompt`.


User message:

[[ ## desired_prompt ## ]]
A scene that's both peaceful and tense

[[ ## current_image ## ]]
<image_url: https://fal.media/files/panda/kNBH7FF1uLk06nXonN0Zr_256e1bf2f71c4dcc8eb94027f09f3452.jpg>

[[ ## current_prompt ## ]]
A serene autumn scene with elements that evoke a sense of unease or foreboding

Respond with the corresponding output fields, starting with the field `[[ ## feedback ## ]]`, then `[[ ## image_strictly_matches_desired_prompt ## ]]` (must be formatted as a valid Python bool), then `[[ ## revised_prompt ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.


Response:

[[ ## feedback ## ]]
The image depicts a serene autumn scene with warm colors and soft lighting, which aligns with the peaceful aspect of the desired prompt. However, it lacks elements that evoke tension or unease, making it not fully meet the requirement for a scene that is both peaceful and tense.

[[ ## image_strictly_matches_desired_prompt ## ]]
False

[[ ## revised_prompt ## ]]
A serene autumn scene that includes subtle elements of tension or foreboding, such as dark shadows or an unsettling atmosphere.

[[ ## completed ## ]]





[2025-01-17T11:39:03.586839]

System message:

Your input fields are:
1. `desired_prompt` (str)
2. `current_image` (Image)
3. `current_prompt` (str)

Your output fields are:
1. `feedback` (str)
2. `image_strictly_matches_desired_prompt` (bool)
3. `revised_prompt` (str)

All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## desired_prompt ## ]]
{desired_prompt}

[[ ## current_image ## ]]
{current_image}

[[ ## current_prompt ## ]]
{current_prompt}

[[ ## feedback ## ]]
{feedback}

[[ ## image_strictly_matches_desired_prompt ## ]]
{image_strictly_matches_desired_prompt}        # note: the value you produce must be True or False

[[ ## revised_prompt ## ]]
{revised_prompt}

[[ ## completed ## ]]

In adhering to this structure, your objective is: 
        Given the fields `desired_prompt`, `current_image`, `current_prompt`, produce the fields `feedback`, `image_strictly_matches_desired_prompt`, `revised_prompt`.


User message:

[[ ## desired_prompt ## ]]
A scene that's both peaceful and tense

[[ ## current_image ## ]]
<image_url: https://fal.media/files/penguin/AXRlgHfIxbZ26Ipgd64Rn_d012474791904e70bd419945f7179980.jpg>

[[ ## current_prompt ## ]]
A serene autumn scene that includes subtle elements of tension or foreboding, such as dark shadows or an unsettling atmosphere.

Respond with the corresponding output fields, starting with the field `[[ ## feedback ## ]]`, then `[[ ## image_strictly_matches_desired_prompt ## ]]` (must be formatted as a valid Python bool), then `[[ ## revised_prompt ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.


Response:

[[ ## feedback ## ]]
The image depicts a foggy forest scene with autumn leaves, creating a peaceful atmosphere. However, the overall mood is somewhat tense due to the fog and shadows, aligning well with the desired prompt of being both peaceful and tense. 

[[ ## image_strictly_matches_desired_prompt ## ]]
True

[[ ## revised_prompt ## ]]
A serene autumn scene with fog and shadows, capturing both peace and tension.

[[ ## completed ## ]]
```
