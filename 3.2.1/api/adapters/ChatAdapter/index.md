# dspy.ChatAdapter

## `dspy.ChatAdapter(callbacks: list[BaseCallback] | None = None, use_native_function_calling: bool = False, native_response_types: list[type[type]] | None = None, use_json_adapter_fallback: bool = True)`

Bases: `Adapter`

Default Adapter for most language models.

The ChatAdapter formats DSPy signatures into a format compatible with most language models. It uses delimiter patterns like `[[ ## field_name ## ]]` to clearly separate input and output fields in the message content.

Key features

- Structures inputs and outputs using field header markers for clear field delineation.
- Provides automatic fallback to JSONAdapter if the chat format fails.

Parameters:

| Name                          | Type                 | Description                                                                                                                                                                                               | Default                                                       |
| ----------------------------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `callbacks`                   | \`list[BaseCallback] | None\`                                                                                                                                                                                                    | List of callback functions to execute during adapter methods. |
| `use_native_function_calling` | `bool`               | Whether to enable native function calling capabilities.                                                                                                                                                   | `False`                                                       |
| `native_response_types`       | \`list\[type[type]\] | None\`                                                                                                                                                                                                    | List of output field types handled by native LM features.     |
| `use_json_adapter_fallback`   | `bool`               | Whether to automatically fallback to JSONAdapter if the ChatAdapter fails. If True, when an error occurs (except ContextWindowExceededError), the adapter will retry using JSONAdapter. Defaults to True. | `True`                                                        |

Source code in `dspy/adapters/chat_adapter.py`

```
def __init__(
    self,
    callbacks: list[BaseCallback] | None = None,
    use_native_function_calling: bool = False,
    native_response_types: list[type[type]] | None = None,
    use_json_adapter_fallback: bool = True,
):
    """
    Args:
        callbacks: List of callback functions to execute during adapter methods.
        use_native_function_calling: Whether to enable native function calling capabilities.
        native_response_types: List of output field types handled by native LM features.
        use_json_adapter_fallback: Whether to automatically fallback to JSONAdapter if the ChatAdapter fails.
            If True, when an error occurs (except ContextWindowExceededError), the adapter will retry using
            JSONAdapter. Defaults to True.
    """
    super().__init__(
        callbacks=callbacks,
        use_native_function_calling=use_native_function_calling,
        native_response_types=native_response_types,
    )
    self.use_json_adapter_fallback = use_json_adapter_fallback
```

### Functions

#### `__call__(lm: BaseLM, lm_kwargs: dict[str, Any], signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]`

Source code in `dspy/adapters/chat_adapter.py`

```
def __call__(
    self,
    lm: BaseLM,
    lm_kwargs: dict[str, Any],
    signature: type[Signature],
    demos: list[dict[str, Any]],
    inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    try:
        return super().__call__(lm, lm_kwargs, signature, demos, inputs)
    except Exception as e:
        # fallback to JSONAdapter
        from dspy.adapters.json_adapter import JSONAdapter

        if (
            isinstance(e, ContextWindowExceededError)
            or isinstance(self, JSONAdapter)
            or not self.use_json_adapter_fallback
        ):
            # On context window exceeded error, already using JSONAdapter, or use_json_adapter_fallback is False
            # we don't want to retry with a different adapter. Raise the original error instead of the fallback error.
            raise e
        return JSONAdapter()(lm, lm_kwargs, signature, demos, inputs)
```

#### `acall(lm: BaseLM, lm_kwargs: dict[str, Any], signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]`

Source code in `dspy/adapters/chat_adapter.py`

```
async def acall(
    self,
    lm: BaseLM,
    lm_kwargs: dict[str, Any],
    signature: type[Signature],
    demos: list[dict[str, Any]],
    inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    try:
        return await super().acall(lm, lm_kwargs, signature, demos, inputs)
    except Exception as e:
        # fallback to JSONAdapter
        from dspy.adapters.json_adapter import JSONAdapter

        if (
            isinstance(e, ContextWindowExceededError)
            or isinstance(self, JSONAdapter)
            or not self.use_json_adapter_fallback
        ):
            # On context window exceeded error, already using JSONAdapter, or use_json_adapter_fallback is False
            # we don't want to retry with a different adapter. Raise the original error instead of the fallback error.
            raise e
        return await JSONAdapter().acall(lm, lm_kwargs, signature, demos, inputs)
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

#### `format_assistant_message_content(signature: type[Signature], outputs: dict[str, Any], missing_field_message=None) -> str`

Source code in `dspy/adapters/chat_adapter.py`

```
def format_assistant_message_content(
    self,
    signature: type[Signature],
    outputs: dict[str, Any],
    missing_field_message=None,
) -> str:
    assistant_message_content = self.format_field_with_value(
        {
            FieldInfoWithName(name=k, info=v): outputs.get(k, missing_field_message)
            for k, v in signature.output_fields.items()
        },
    )
    assistant_message_content += "\n\n[[ ## completed ## ]]\n"
    return assistant_message_content
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

Source code in `dspy/adapters/chat_adapter.py`

```
def format_field_description(self, signature: type[Signature]) -> str:
    return (
        f"Your input fields are:\n{get_field_description_string(signature.input_fields)}\n"
        f"Your output fields are:\n{get_field_description_string(signature.output_fields)}"
    )
```

#### `format_field_structure(signature: type[Signature]) -> str`

`ChatAdapter` requires input and output fields to be in their own sections, with section header using markers `[[ ## field_name ## ]]`. An arbitrary field `completed` (\[[ ## completed ## ]\]) is added to the end of the output fields section to indicate the end of the output fields.

Source code in `dspy/adapters/chat_adapter.py`

```
def format_field_structure(self, signature: type[Signature]) -> str:
    """
    `ChatAdapter` requires input and output fields to be in their own sections, with section header using markers
    `[[ ## field_name ## ]]`. An arbitrary field `completed` ([[ ## completed ## ]]) is added to the end of the
    output fields section to indicate the end of the output fields.
    """
    parts = []
    parts.append("All interactions will be structured in the following way, with the appropriate values filled in.")

    def format_signature_fields_for_instructions(fields: dict[str, FieldInfo]):
        return self.format_field_with_value(
            fields_with_values={
                FieldInfoWithName(name=field_name, info=field_info): translate_field_type(field_name, field_info)
                for field_name, field_info in fields.items()
            },
        )

    parts.append(format_signature_fields_for_instructions(signature.input_fields))
    parts.append(format_signature_fields_for_instructions(signature.output_fields))
    parts.append("[[ ## completed ## ]]\n")
    return "\n\n".join(parts).strip()
```

#### `format_field_with_value(fields_with_values: dict[FieldInfoWithName, Any]) -> str`

Formats the values of the specified fields according to the field's DSPy type (input or output), annotation (e.g. str, int, etc.), and the type of the value itself. Joins the formatted values into a single string, which is a multiline string if there are multiple fields.

Parameters:

| Name                 | Type                           | Description                                                                | Default    |
| -------------------- | ------------------------------ | -------------------------------------------------------------------------- | ---------- |
| `fields_with_values` | `dict[FieldInfoWithName, Any]` | A dictionary mapping information about a field to its corresponding value. | *required* |

Returns:

| Type  | Description                                                        |
| ----- | ------------------------------------------------------------------ |
| `str` | The joined formatted values of the fields, represented as a string |

Source code in `dspy/adapters/chat_adapter.py`

```
def format_field_with_value(self, fields_with_values: dict[FieldInfoWithName, Any]) -> str:
    """
    Formats the values of the specified fields according to the field's DSPy type (input or output),
    annotation (e.g. str, int, etc.), and the type of the value itself. Joins the formatted values
    into a single string, which is a multiline string if there are multiple fields.

    Args:
        fields_with_values: A dictionary mapping information about a field to its corresponding
            value.

    Returns:
        The joined formatted values of the fields, represented as a string
    """
    output = []
    for field, field_value in fields_with_values.items():
        formatted_field_value = format_field_value(field_info=field.info, value=field_value)
        output.append(f"[[ ## {field.name} ## ]]\n{formatted_field_value}")

    return "\n\n".join(output).strip()
```

#### `format_finetune_data(signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, list[Any]]`

Format the call data into finetuning data according to the OpenAI API specifications.

For the chat adapter, this means formatting the data as a list of messages, where each message is a dictionary with a "role" and "content" key. The role can be "system", "user", or "assistant". Then, the messages are wrapped in a dictionary with a "messages" key.

Source code in `dspy/adapters/chat_adapter.py`

```
def format_finetune_data(
    self,
    signature: type[Signature],
    demos: list[dict[str, Any]],
    inputs: dict[str, Any],
    outputs: dict[str, Any],
) -> dict[str, list[Any]]:
    """
    Format the call data into finetuning data according to the OpenAI API specifications.

    For the chat adapter, this means formatting the data as a list of messages, where each message is a dictionary
    with a "role" and "content" key. The role can be "system", "user", or "assistant". Then, the messages are
    wrapped in a dictionary with a "messages" key.
    """
    system_user_messages = self.format(  # returns a list of dicts with the keys "role" and "content"
        signature=signature, demos=demos, inputs=inputs
    )
    assistant_message_content = self.format_assistant_message_content(  # returns a string, without the role
        signature=signature, outputs=outputs
    )
    assistant_message = {"role": "assistant", "content": assistant_message_content}
    messages = system_user_messages + [assistant_message]
    return {"messages": messages}
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

Source code in `dspy/adapters/chat_adapter.py`

```
def format_task_description(self, signature: type[Signature]) -> str:
    instructions = textwrap.dedent(signature.instructions)
    objective = ("\n" + " " * 8).join([""] + instructions.splitlines())
    return f"In adhering to this structure, your objective is: {objective}"
```

#### `format_user_message_content(signature: type[Signature], inputs: dict[str, Any], prefix: str = '', suffix: str = '', main_request: bool = False) -> str`

Source code in `dspy/adapters/chat_adapter.py`

```
def format_user_message_content(
    self,
    signature: type[Signature],
    inputs: dict[str, Any],
    prefix: str = "",
    suffix: str = "",
    main_request: bool = False,
) -> str:
    messages = [prefix]
    for k, v in signature.input_fields.items():
        if k in inputs:
            value = inputs.get(k)
            formatted_field_value = format_field_value(field_info=v, value=value)
            messages.append(f"[[ ## {k} ## ]]\n{formatted_field_value}")

    if main_request:
        output_requirements = self.user_message_output_requirements(signature)
        if output_requirements is not None:
            messages.append(output_requirements)

    messages.append(suffix)
    return "\n\n".join(messages).strip()
```

#### `parse(signature: type[Signature], completion: str) -> dict[str, Any]`

Source code in `dspy/adapters/chat_adapter.py`

```
def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
    sections = [(None, [])]

    for line in completion.splitlines():
        match = field_header_pattern.match(line.strip())
        if match:
            # If the header pattern is found, split the rest of the line as content
            header = match.group(1)
            remaining_content = line[match.end() :].strip()
            sections.append((header, [remaining_content] if remaining_content else []))
        else:
            sections[-1][1].append(line)

    sections = [(k, "\n".join(v).strip()) for k, v in sections]

    fields = {}
    for k, v in sections:
        if (k not in fields) and (k in signature.output_fields):
            try:
                fields[k] = parse_value(v, signature.output_fields[k].annotation)
            except Exception as e:
                raise AdapterParseError(
                    adapter_name="ChatAdapter",
                    signature=signature,
                    lm_response=completion,
                    message=f"Failed to parse field {k} with value {v} from the LM response. Error message: {e}",
                )
    if fields.keys() != signature.output_fields.keys():
        raise AdapterParseError(
            adapter_name="ChatAdapter",
            signature=signature,
            lm_response=completion,
            parsed_result=fields,
        )

    return fields
```

#### `user_message_output_requirements(signature: type[Signature]) -> str`

Returns a simplified format reminder for the language model.

In chat-based interactions, language models may lose track of the required output format as the conversation context grows longer. This method generates a concise reminder of the expected output structure that can be included in user messages.

Parameters:

| Name        | Type              | Description                                                   | Default    |
| ----------- | ----------------- | ------------------------------------------------------------- | ---------- |
| `signature` | `Type[Signature]` | The DSPy signature defining the expected input/output fields. | *required* |

Returns:

| Name  | Type  | Description                                             |
| ----- | ----- | ------------------------------------------------------- |
| `str` | `str` | A simplified description of the required output format. |

Note

This is a more lightweight version of `format_field_structure` specifically designed for inline reminders within chat messages.

Source code in `dspy/adapters/chat_adapter.py`

```
def user_message_output_requirements(self, signature: type[Signature]) -> str:
    """Returns a simplified format reminder for the language model.

    In chat-based interactions, language models may lose track of the required output format
    as the conversation context grows longer. This method generates a concise reminder of
    the expected output structure that can be included in user messages.

    Args:
        signature (Type[Signature]): The DSPy signature defining the expected input/output fields.

    Returns:
        str: A simplified description of the required output format.

    Note:
        This is a more lightweight version of `format_field_structure` specifically designed
        for inline reminders within chat messages.
    """

    def type_info(v):
        if v.annotation is not str:
            return f" (must be formatted as a valid Python {get_annotation_name(v.annotation)})"
        else:
            return ""

    message = "Respond with the corresponding output fields, starting with the field "
    message += ", then ".join(f"`[[ ## {f} ## ]]`{type_info(v)}" for f, v in signature.output_fields.items())
    message += ", and then ending with the marker for `[[ ## completed ## ]]`."
    return message
```
