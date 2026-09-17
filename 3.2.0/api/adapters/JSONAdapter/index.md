# dspy.JSONAdapter

## `dspy.JSONAdapter(callbacks: list[BaseCallback] | None = None, use_native_function_calling: bool = True)`

Bases: `ChatAdapter`

Source code in `dspy/adapters/json_adapter.py`

```
def __init__(self, callbacks: list[BaseCallback] | None = None, use_native_function_calling: bool = True):
    # JSONAdapter uses native function calling by default.
    super().__init__(callbacks=callbacks, use_native_function_calling=use_native_function_calling)
```

### Functions

#### `__call__(lm: BaseLM, lm_kwargs: dict[str, Any], signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]`

Source code in `dspy/adapters/json_adapter.py`

```
def __call__(
    self,
    lm: BaseLM,
    lm_kwargs: dict[str, Any],
    signature: type[Signature],
    demos: list[dict[str, Any]],
    inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    result = self._json_adapter_call_common(lm, lm_kwargs, signature, demos, inputs, super().__call__)
    if result:
        return result

    try:
        structured_output_model = _get_structured_outputs_response_format(
            signature, self.use_native_function_calling
        )
        lm_kwargs["response_format"] = structured_output_model
        return super().__call__(lm, lm_kwargs, signature, demos, inputs)
    except Exception:
        logger.warning("Failed to use structured output format, falling back to JSON mode.")
        lm_kwargs["response_format"] = {"type": "json_object"}
        return super().__call__(lm, lm_kwargs, signature, demos, inputs)
```

#### `acall(lm: BaseLM, lm_kwargs: dict[str, Any], signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]`

Source code in `dspy/adapters/json_adapter.py`

```
async def acall(
    self,
    lm: BaseLM,
    lm_kwargs: dict[str, Any],
    signature: type[Signature],
    demos: list[dict[str, Any]],
    inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    result = self._json_adapter_call_common(lm, lm_kwargs, signature, demos, inputs, super().acall)
    if result:
        return await result

    try:
        structured_output_model = _get_structured_outputs_response_format(
            signature, self.use_native_function_calling
        )
        lm_kwargs["response_format"] = structured_output_model
        return await super().acall(lm, lm_kwargs, signature, demos, inputs)
    except Exception:
        logger.warning("Failed to use structured output format, falling back to JSON mode.")
        lm_kwargs["response_format"] = {"type": "json_object"}
        return await super().acall(lm, lm_kwargs, signature, demos, inputs)
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

Source code in `dspy/adapters/json_adapter.py`

```
def format_assistant_message_content(
    self,
    signature: type[Signature],
    outputs: dict[str, Any],
    missing_field_message=None,
) -> str:
    fields_with_values = {
        FieldInfoWithName(name=k, info=v): outputs.get(k, missing_field_message)
        for k, v in signature.output_fields.items()
    }
    return self.format_field_with_value(fields_with_values, role="assistant")
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

Source code in `dspy/adapters/json_adapter.py`

```
def format_field_structure(self, signature: type[Signature]) -> str:
    parts = []
    parts.append("All interactions will be structured in the following way, with the appropriate values filled in.")

    def format_signature_fields_for_instructions(fields: dict[str, FieldInfo], role: str):
        return self.format_field_with_value(
            fields_with_values={
                FieldInfoWithName(name=field_name, info=field_info): translate_field_type(field_name, field_info)
                for field_name, field_info in fields.items()
            },
            role=role,
        )

    parts.append("Inputs will have the following structure:")
    parts.append(format_signature_fields_for_instructions(signature.input_fields, role="user"))
    parts.append("Outputs will be a JSON object with the following fields.")
    parts.append(format_signature_fields_for_instructions(signature.output_fields, role="assistant"))
    return "\n\n".join(parts).strip()
```

#### `format_field_with_value(fields_with_values: dict[FieldInfoWithName, Any], role: str = 'user') -> str`

Formats the values of the specified fields according to the field's DSPy type (input or output), annotation (e.g. str, int, etc.), and the type of the value itself. Joins the formatted values into a single string, which is a multiline string if there are multiple fields.

Parameters:

| Name                 | Type                           | Description                                                                | Default    |
| -------------------- | ------------------------------ | -------------------------------------------------------------------------- | ---------- |
| `fields_with_values` | `dict[FieldInfoWithName, Any]` | A dictionary mapping information about a field to its corresponding value. | *required* |

Returns: The joined formatted values of the fields, represented as a string.

Source code in `dspy/adapters/json_adapter.py`

```
def format_field_with_value(self, fields_with_values: dict[FieldInfoWithName, Any], role: str = "user") -> str:
    """
    Formats the values of the specified fields according to the field's DSPy type (input or output),
    annotation (e.g. str, int, etc.), and the type of the value itself. Joins the formatted values
    into a single string, which is a multiline string if there are multiple fields.

    Args:
        fields_with_values: A dictionary mapping information about a field to its corresponding value.
    Returns:
        The joined formatted values of the fields, represented as a string.
    """
    if role == "user":
        output = []
        for field, field_value in fields_with_values.items():
            formatted_field_value = format_field_value(field_info=field.info, value=field_value)
            output.append(f"[[ ## {field.name} ## ]]\n{formatted_field_value}")
        return "\n\n".join(output).strip()
    else:
        d = fields_with_values.items()
        d = {k.name: v for k, v in d}
        return json.dumps(serialize_for_json(d), indent=2, ensure_ascii=False)
```

#### `format_finetune_data(signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, list[Any]]`

Source code in `dspy/adapters/json_adapter.py`

```
def format_finetune_data(
    self, signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any], outputs: dict[str, Any]
) -> dict[str, list[Any]]:
    # TODO: implement format_finetune_data method in JSONAdapter
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

Source code in `dspy/adapters/json_adapter.py`

```
def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
    fields = json_repair.loads(completion)

    if not isinstance(fields, dict):
        pattern = r"\{(?:[^{}]|(?R))*\}"
        match = regex.search(pattern, completion, regex.DOTALL)
        if match:
            completion = match.group(0)
            fields = json_repair.loads(completion)

    if not isinstance(fields, dict):
        raise AdapterParseError(
            adapter_name="JSONAdapter",
            signature=signature,
            lm_response=completion,
            message="LM response cannot be serialized to a JSON object.",
        )

    fields = {k: v for k, v in fields.items() if k in signature.output_fields}

    # Attempt to cast each value to type signature.output_fields[k].annotation.
    for k, v in fields.items():
        if k in signature.output_fields:
            fields[k] = parse_value(v, signature.output_fields[k].annotation)

    if fields.keys() != signature.output_fields.keys():
        raise AdapterParseError(
            adapter_name="JSONAdapter",
            signature=signature,
            lm_response=completion,
            parsed_result=fields,
        )

    return fields
```

#### `user_message_output_requirements(signature: type[Signature]) -> str`

Source code in `dspy/adapters/json_adapter.py`

```
def user_message_output_requirements(self, signature: type[Signature]) -> str:
    def type_info(v):
        return (
            f" (must be formatted as a valid Python {get_annotation_name(v.annotation)})"
            if v.annotation is not str
            else ""
        )

    message = "Respond with a JSON object in the following order of fields: "
    message += ", then ".join(f"`{f}`{type_info(v)}" for f, v in signature.output_fields.items())
    message += "."
    return message
```
