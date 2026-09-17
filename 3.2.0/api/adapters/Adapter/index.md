# dspy.Adapter

## `dspy.Adapter(callbacks: list[BaseCallback] | None = None, use_native_function_calling: bool = False, native_response_types: list[type[Type]] | None = None)`

Base Adapter class.

The Adapter serves as the interface layer between DSPy module/signature and Language Models (LMs). It handles the complete transformation pipeline from DSPy inputs to LM calls and back to structured outputs.

Key responsibilities

- Transform user inputs and signatures into properly formatted LM prompts, which also instructs the LM to format the response in a specific format.
- Parse LM outputs into dictionaries matching the signature's output fields.
- Enable/disable native LM features (function calling, citations, etc.) based on configuration.
- Handle conversation history, few-shot examples, and custom type processing.

The adapter pattern allows DSPy to work with different LM interfaces while maintaining a consistent programming model for users.

Parameters:

| Name                          | Type                 | Description                                                                                                                                                                                                                         | Default                                                                                                                                                                                                                                    |
| ----------------------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `callbacks`                   | \`list[BaseCallback] | None\`                                                                                                                                                                                                                              | List of callback functions to execute during format() and parse() methods. Callbacks can be used for logging, monitoring, or custom processing. Defaults to None (empty list).                                                             |
| `use_native_function_calling` | `bool`               | Whether to enable native function calling capabilities when the LM supports it. If True, the adapter will automatically configure function calling when input fields contain dspy.Tool or list[dspy.Tool] types. Defaults to False. | `False`                                                                                                                                                                                                                                    |
| `native_response_types`       | \`list\[type[Type]\] | None\`                                                                                                                                                                                                                              | List of output field types that should be handled by native LM features rather than adapter parsing. For example, dspy.Citations can be populated directly by citation APIs (e.g., Anthropic's citation feature). Defaults to [Citations]. |

Source code in `dspy/adapters/base.py`

```
def __init__(
    self,
    callbacks: list[BaseCallback] | None = None,
    use_native_function_calling: bool = False,
    native_response_types: list[type[Type]] | None = None,
):
    """
    Args:
        callbacks: List of callback functions to execute during `format()` and `parse()` methods. Callbacks can be
            used for logging, monitoring, or custom processing. Defaults to None (empty list).
        use_native_function_calling: Whether to enable native function calling capabilities when the LM supports it.
            If True, the adapter will automatically configure function calling when input fields contain `dspy.Tool`
            or `list[dspy.Tool]` types. Defaults to False.
        native_response_types: List of output field types that should be handled by native LM features rather than
            adapter parsing. For example, `dspy.Citations` can be populated directly by citation APIs
            (e.g., Anthropic's citation feature). Defaults to `[Citations]`.
    """
    self.callbacks = callbacks or []
    self.use_native_function_calling = use_native_function_calling
    self.native_response_types = native_response_types or _DEFAULT_NATIVE_RESPONSE_TYPES
```

### Functions

#### `__call__(lm: BaseLM, lm_kwargs: dict[str, Any], signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]`

Execute the adapter pipeline: format inputs, call LM, and parse outputs.

Parameters:

| Name        | Type                   | Description                                                                                                                                                                                            | Default    |
| ----------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- |
| `lm`        | `BaseLM`               | The Language Model instance to use for generation. Must be an instance of dspy.BaseLM.                                                                                                                 | *required* |
| `lm_kwargs` | `dict[str, Any]`       | Additional keyword arguments to pass to the LM call (e.g., temperature, max_tokens). These are passed directly to the LM.                                                                              | *required* |
| `signature` | `type[Signature]`      | The DSPy signature associated with this LM call.                                                                                                                                                       | *required* |
| `demos`     | `list[dict[str, Any]]` | List of few-shot examples to include in the prompt. Each dictionary should contain keys matching the signature's input and output field names. Examples are formatted as user/assistant message pairs. | *required* |
| `inputs`    | `dict[str, Any]`       | The current input values for this call. Keys must match the signature's input field names.                                                                                                             | *required* |

Returns:

| Type                   | Description                                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------------------- |
| `list[dict[str, Any]]` | List of dictionaries representing parsed LM responses. Each dictionary contains keys matching the |
| `list[dict[str, Any]]` | signature's output field names. For multiple generations (n > 1), returns multiple dictionaries.  |

Source code in `dspy/adapters/base.py`

```
def __call__(
    self,
    lm: BaseLM,
    lm_kwargs: dict[str, Any],
    signature: type[Signature],
    demos: list[dict[str, Any]],
    inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Execute the adapter pipeline: format inputs, call LM, and parse outputs.

    Args:
        lm: The Language Model instance to use for generation. Must be an instance of `dspy.BaseLM`.
        lm_kwargs: Additional keyword arguments to pass to the LM call (e.g., temperature, max_tokens). These are
            passed directly to the LM.
        signature: The DSPy signature associated with this LM call.
        demos: List of few-shot examples to include in the prompt. Each dictionary should contain keys matching the
            signature's input and output field names. Examples are formatted as user/assistant message pairs.
        inputs: The current input values for this call. Keys must match the signature's input field names.

    Returns:
        List of dictionaries representing parsed LM responses. Each dictionary contains keys matching the
        signature's output field names. For multiple generations (n > 1), returns multiple dictionaries.
    """
    processed_signature = self._call_preprocess(lm, lm_kwargs, signature, inputs)
    inputs = self.format(processed_signature, demos, inputs)

    outputs = lm(messages=inputs, **lm_kwargs)
    return self._call_postprocess(processed_signature, signature, outputs, lm, lm_kwargs)
```

#### `acall(lm: BaseLM, lm_kwargs: dict[str, Any], signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]`

Source code in `dspy/adapters/base.py`

```
async def acall(
    self,
    lm: BaseLM,
    lm_kwargs: dict[str, Any],
    signature: type[Signature],
    demos: list[dict[str, Any]],
    inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    processed_signature = self._call_preprocess(lm, lm_kwargs, signature, inputs)
    inputs = self.format(processed_signature, demos, inputs)

    outputs = await lm.acall(messages=inputs, **lm_kwargs)
    return self._call_postprocess(processed_signature, signature, outputs, lm, lm_kwargs)
```

#### `format(signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]`

Format the input messages for the LM call.

This method converts the DSPy structured input along with few-shot examples and conversation history into multiturn messages as expected by the LM. For custom adapters, this method can be overridden to customize the formatting of the input messages.

In general we recommend the messages to have the following structure:

```
[
    {"role": "system", "content": system_message},
    # Begin few-shot examples
    {"role": "user", "content": few_shot_example_1_input},
    {"role": "assistant", "content": few_shot_example_1_output},
    {"role": "user", "content": few_shot_example_2_input},
    {"role": "assistant", "content": few_shot_example_2_output},
    ...
    # End few-shot examples
    # Begin conversation history
    {"role": "user", "content": conversation_history_1_input},
    {"role": "assistant", "content": conversation_history_1_output},
    {"role": "user", "content": conversation_history_2_input},
    {"role": "assistant", "content": conversation_history_2_output},
    ...
    # End conversation history
    {"role": "user", "content": current_input},
]

And system message should contain the field description, field structure, and task description.
```

Parameters:

| Name        | Type                   | Description                                                | Default    |
| ----------- | ---------------------- | ---------------------------------------------------------- | ---------- |
| `signature` | `type[Signature]`      | The DSPy signature for which to format the input messages. | *required* |
| `demos`     | `list[dict[str, Any]]` | A list of few-shot examples.                               | *required* |
| `inputs`    | `dict[str, Any]`       | The input arguments to the DSPy module.                    | *required* |

Returns:

| Type                   | Description                                         |
| ---------------------- | --------------------------------------------------- |
| `list[dict[str, Any]]` | A list of multiturn messages as expected by the LM. |

Source code in `dspy/adapters/base.py`

````
def format(
    self,
    signature: type[Signature],
    demos: list[dict[str, Any]],
    inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Format the input messages for the LM call.

    This method converts the DSPy structured input along with few-shot examples and conversation history into
    multiturn messages as expected by the LM. For custom adapters, this method can be overridden to customize
    the formatting of the input messages.

    In general we recommend the messages to have the following structure:
    ```
    [
        {"role": "system", "content": system_message},
        # Begin few-shot examples
        {"role": "user", "content": few_shot_example_1_input},
        {"role": "assistant", "content": few_shot_example_1_output},
        {"role": "user", "content": few_shot_example_2_input},
        {"role": "assistant", "content": few_shot_example_2_output},
        ...
        # End few-shot examples
        # Begin conversation history
        {"role": "user", "content": conversation_history_1_input},
        {"role": "assistant", "content": conversation_history_1_output},
        {"role": "user", "content": conversation_history_2_input},
        {"role": "assistant", "content": conversation_history_2_output},
        ...
        # End conversation history
        {"role": "user", "content": current_input},
    ]

    And system message should contain the field description, field structure, and task description.
    ```


    Args:
        signature: The DSPy signature for which to format the input messages.
        demos: A list of few-shot examples.
        inputs: The input arguments to the DSPy module.

    Returns:
        A list of multiturn messages as expected by the LM.
    """
    inputs_copy = dict(inputs)

    # If the signature and inputs have conversation history, we need to format the conversation history and
    # remove the history field from the signature.
    history_field_name = self._get_history_field_name(signature)
    if history_field_name:
        # In order to format the conversation history, we need to remove the history field from the signature.
        signature_without_history = signature.delete(history_field_name)
        conversation_history = self.format_conversation_history(
            signature_without_history,
            history_field_name,
            inputs_copy,
        )

    messages = []
    system_message = self.format_system_message(signature)
    messages.append({"role": "system", "content": system_message})
    messages.extend(self.format_demos(signature, demos))
    if history_field_name:
        # Conversation history and current input
        content = self.format_user_message_content(signature_without_history, inputs_copy, main_request=True)
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": content})
    else:
        # Only current input
        content = self.format_user_message_content(signature, inputs_copy, main_request=True)
        messages.append({"role": "user", "content": content})

    messages = split_message_content_for_custom_types(messages)
    return messages
````

#### `format_assistant_message_content(signature: type[Signature], outputs: dict[str, Any], missing_field_message: str | None = None) -> str`

Format the assistant message content.

This method formats the assistant message content, which can be used in formatting few-shot examples, conversation history.

Parameters:

| Name                    | Type              | Description                                                           | Default                                       |
| ----------------------- | ----------------- | --------------------------------------------------------------------- | --------------------------------------------- |
| `signature`             | `type[Signature]` | The DSPy signature for which to format the assistant message content. | *required*                                    |
| `outputs`               | `dict[str, Any]`  | The output fields to be formatted.                                    | *required*                                    |
| `missing_field_message` | \`str             | None\`                                                                | A message to be used when a field is missing. |

Returns:

| Type  | Description                                           |
| ----- | ----------------------------------------------------- |
| `str` | A string that contains the assistant message content. |

Source code in `dspy/adapters/base.py`

```
def format_assistant_message_content(
    self,
    signature: type[Signature],
    outputs: dict[str, Any],
    missing_field_message: str | None = None,
) -> str:
    """Format the assistant message content.

    This method formats the assistant message content, which can be used in formatting few-shot examples,
    conversation history.

    Args:
        signature: The DSPy signature for which to format the assistant message content.
        outputs: The output fields to be formatted.
        missing_field_message: A message to be used when a field is missing.

    Returns:
        A string that contains the assistant message content.
    """
    raise NotImplementedError
```

#### `format_conversation_history(signature: type[Signature], history_field_name: str, inputs: dict[str, Any]) -> list[dict[str, Any]]`

Format the conversation history.

This method formats the conversation history and the current input as multiturn messages.

Parameters:

| Name                 | Type              | Description                                                      | Default    |
| -------------------- | ----------------- | ---------------------------------------------------------------- | ---------- |
| `signature`          | `type[Signature]` | The DSPy signature for which to format the conversation history. | *required* |
| `history_field_name` | `str`             | The name of the history field in the signature.                  | *required* |
| `inputs`             | `dict[str, Any]`  | The input arguments to the DSPy module.                          | *required* |

Returns:

| Type                   | Description                   |
| ---------------------- | ----------------------------- |
| `list[dict[str, Any]]` | A list of multiturn messages. |

Source code in `dspy/adapters/base.py`

```
def format_conversation_history(
    self,
    signature: type[Signature],
    history_field_name: str,
    inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Format the conversation history.

    This method formats the conversation history and the current input as multiturn messages.

    Args:
        signature: The DSPy signature for which to format the conversation history.
        history_field_name: The name of the history field in the signature.
        inputs: The input arguments to the DSPy module.

    Returns:
        A list of multiturn messages.
    """
    conversation_history = inputs[history_field_name].messages if history_field_name in inputs else None

    if conversation_history is None:
        return []

    messages = []
    for message in conversation_history:
        messages.append(
            {
                "role": "user",
                "content": self.format_user_message_content(signature, message),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": self.format_assistant_message_content(signature, message),
            }
        )

    # Remove the history field from the inputs
    del inputs[history_field_name]

    return messages
```

#### `format_demos(signature: type[Signature], demos: list[dict[str, Any]]) -> list[dict[str, Any]]`

Format the few-shot examples.

This method formats the few-shot examples as multiturn messages.

Parameters:

| Name        | Type                   | Description                                                                                                          | Default    |
| ----------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------- | ---------- |
| `signature` | `type[Signature]`      | The DSPy signature for which to format the few-shot examples.                                                        | *required* |
| `demos`     | `list[dict[str, Any]]` | A list of few-shot examples, each element is a dictionary with keys of the input and output fields of the signature. | *required* |

Returns:

| Type                   | Description                   |
| ---------------------- | ----------------------------- |
| `list[dict[str, Any]]` | A list of multiturn messages. |

Source code in `dspy/adapters/base.py`

```
def format_demos(self, signature: type[Signature], demos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format the few-shot examples.

    This method formats the few-shot examples as multiturn messages.

    Args:
        signature: The DSPy signature for which to format the few-shot examples.
        demos: A list of few-shot examples, each element is a dictionary with keys of the input and output fields of
            the signature.

    Returns:
        A list of multiturn messages.
    """
    complete_demos = []
    incomplete_demos = []

    for demo in demos:
        # Check if all fields are present and not None
        is_complete = all(k in demo and demo[k] is not None for k in signature.fields)

        # Check if demo has at least one input and one output field
        has_input = any(k in demo for k in signature.input_fields)
        has_output = any(k in demo for k in signature.output_fields)

        if is_complete:
            complete_demos.append(demo)
        elif has_input and has_output:
            # We only keep incomplete demos that have at least one input and one output field
            incomplete_demos.append(demo)

    messages = []

    incomplete_demo_prefix = "This is an example of the task, though some input or output fields are not supplied."
    for demo in incomplete_demos:
        messages.append(
            {
                "role": "user",
                "content": self.format_user_message_content(signature, demo, prefix=incomplete_demo_prefix),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": self.format_assistant_message_content(
                    signature, demo, missing_field_message="Not supplied for this particular example. "
                ),
            }
        )

    for demo in complete_demos:
        messages.append({"role": "user", "content": self.format_user_message_content(signature, demo)})
        messages.append(
            {
                "role": "assistant",
                "content": self.format_assistant_message_content(
                    signature, demo, missing_field_message="Not supplied for this conversation history message. "
                ),
            }
        )

    return messages
```

#### `format_field_description(signature: type[Signature]) -> str`

Format the field description for the system message.

This method formats the field description for the system message. It should return a string that contains the field description for the input fields and the output fields.

Parameters:

| Name        | Type              | Description                                                   | Default    |
| ----------- | ----------------- | ------------------------------------------------------------- | ---------- |
| `signature` | `type[Signature]` | The DSPy signature for which to format the field description. | *required* |

Returns:

| Type  | Description                                                                              |
| ----- | ---------------------------------------------------------------------------------------- |
| `str` | A string that contains the field description for the input fields and the output fields. |

Source code in `dspy/adapters/base.py`

```
def format_field_description(self, signature: type[Signature]) -> str:
    """Format the field description for the system message.

    This method formats the field description for the system message. It should return a string that contains
    the field description for the input fields and the output fields.

    Args:
        signature: The DSPy signature for which to format the field description.

    Returns:
        A string that contains the field description for the input fields and the output fields.
    """
    raise NotImplementedError
```

#### `format_field_structure(signature: type[Signature]) -> str`

Format the field structure for the system message.

This method formats the field structure for the system message. It should return a string that dictates the format the input fields should be provided to the LM, and the format the output fields will be in the response. Refer to the ChatAdapter and JsonAdapter for an example.

Parameters:

| Name        | Type              | Description                                                 | Default    |
| ----------- | ----------------- | ----------------------------------------------------------- | ---------- |
| `signature` | `type[Signature]` | The DSPy signature for which to format the field structure. | *required* |

Source code in `dspy/adapters/base.py`

```
def format_field_structure(self, signature: type[Signature]) -> str:
    """Format the field structure for the system message.

    This method formats the field structure for the system message. It should return a string that dictates the
    format the input fields should be provided to the LM, and the format the output fields will be in the response.
    Refer to the ChatAdapter and JsonAdapter for an example.

    Args:
        signature: The DSPy signature for which to format the field structure.
    """
    raise NotImplementedError
```

#### `format_system_message(signature: type[Signature]) -> str`

Format the system message for the LM call.

Parameters:

| Name        | Type              | Description                                                | Default    |
| ----------- | ----------------- | ---------------------------------------------------------- | ---------- |
| `signature` | `type[Signature]` | The DSPy signature for which to format the system message. | *required* |

Source code in `dspy/adapters/base.py`

```
def format_system_message(self, signature: type[Signature]) -> str:
    """Format the system message for the LM call.


    Args:
        signature: The DSPy signature for which to format the system message.
    """
    return (
        f"{self.format_field_description(signature)}\n"
        f"{self.format_field_structure(signature)}\n"
        f"{self.format_task_description(signature)}"
    )
```

#### `format_task_description(signature: type[Signature]) -> str`

Format the task description for the system message.

This method formats the task description for the system message. In most cases this is just a thin wrapper over `signature.instructions`.

Parameters:

| Name        | Type              | Description                            | Default    |
| ----------- | ----------------- | -------------------------------------- | ---------- |
| `signature` | `type[Signature]` | The DSPy signature of the DSpy module. | *required* |

Returns:

| Type  | Description                       |
| ----- | --------------------------------- |
| `str` | A string that describes the task. |

Source code in `dspy/adapters/base.py`

```
def format_task_description(self, signature: type[Signature]) -> str:
    """Format the task description for the system message.

    This method formats the task description for the system message. In most cases this is just a thin wrapper
    over `signature.instructions`.

    Args:
        signature: The DSPy signature of the DSpy module.

    Returns:
        A string that describes the task.
    """
    raise NotImplementedError
```

#### `format_user_message_content(signature: type[Signature], inputs: dict[str, Any], prefix: str = '', suffix: str = '', main_request: bool = False) -> str`

Format the user message content.

This method formats the user message content, which can be used in formatting few-shot examples, conversation history, and the current input.

Parameters:

| Name        | Type              | Description                                                      | Default    |
| ----------- | ----------------- | ---------------------------------------------------------------- | ---------- |
| `signature` | `type[Signature]` | The DSPy signature for which to format the user message content. | *required* |
| `inputs`    | `dict[str, Any]`  | The input arguments to the DSPy module.                          | *required* |
| `prefix`    | `str`             | A prefix to the user message content.                            | `''`       |
| `suffix`    | `str`             | A suffix to the user message content.                            | `''`       |

Returns:

| Type  | Description                                      |
| ----- | ------------------------------------------------ |
| `str` | A string that contains the user message content. |

Source code in `dspy/adapters/base.py`

```
def format_user_message_content(
    self,
    signature: type[Signature],
    inputs: dict[str, Any],
    prefix: str = "",
    suffix: str = "",
    main_request: bool = False,
) -> str:
    """Format the user message content.

    This method formats the user message content, which can be used in formatting few-shot examples, conversation
    history, and the current input.

    Args:
        signature: The DSPy signature for which to format the user message content.
        inputs: The input arguments to the DSPy module.
        prefix: A prefix to the user message content.
        suffix: A suffix to the user message content.

    Returns:
        A string that contains the user message content.
    """
    raise NotImplementedError
```

#### `parse(signature: type[Signature], completion: str) -> dict[str, Any]`

Parse the LM output into a dictionary of the output fields.

This method parses the LM output into a dictionary of the output fields.

Parameters:

| Name         | Type              | Description                                          | Default    |
| ------------ | ----------------- | ---------------------------------------------------- | ---------- |
| `signature`  | `type[Signature]` | The DSPy signature for which to parse the LM output. | *required* |
| `completion` | `str`             | The LM output to be parsed.                          | *required* |

Returns:

| Type             | Description                        |
| ---------------- | ---------------------------------- |
| `dict[str, Any]` | A dictionary of the output fields. |

Source code in `dspy/adapters/base.py`

```
def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
    """Parse the LM output into a dictionary of the output fields.

    This method parses the LM output into a dictionary of the output fields.

    Args:
        signature: The DSPy signature for which to parse the LM output.
        completion: The LM output to be parsed.

    Returns:
        A dictionary of the output fields.
    """
    raise NotImplementedError
```
