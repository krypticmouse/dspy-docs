# dspy.XMLAdapter

## `dspy.XMLAdapter(callbacks: list[BaseCallback] | None = None, use_native_function_calling: bool = False, native_response_types: list[type[type]] | None = None, use_json_adapter_fallback: bool = True, parallel_tool_calls: bool | None = None)`

Bases: `ChatAdapter`

Parameters:

| Name                          | Type                 | Description                                                                                                                                                                                               | Default                                                                                                                                                       |
| ----------------------------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `callbacks`                   | \`list[BaseCallback] | None\`                                                                                                                                                                                                    | List of callback functions to execute during adapter methods.                                                                                                 |
| `use_native_function_calling` | `bool`               | Whether to enable native function calling capabilities.                                                                                                                                                   | `False`                                                                                                                                                       |
| `native_response_types`       | \`list\[type[type]\] | None\`                                                                                                                                                                                                    | List of output field types handled by native LM features.                                                                                                     |
| `use_json_adapter_fallback`   | `bool`               | Whether to automatically fallback to JSONAdapter if the ChatAdapter fails. If True, when an error occurs (except ContextWindowExceededError), the adapter will retry using JSONAdapter. Defaults to True. | `True`                                                                                                                                                        |
| `parallel_tool_calls`         | \`bool               | None\`                                                                                                                                                                                                    | Whether to request provider-side parallel tool-call generation when native function calling is active. If None, the adapter does not set the provider option. |

Source code in `dspy/adapters/chat_adapter.py`

```
def __init__(
    self,
    callbacks: list[BaseCallback] | None = None,
    use_native_function_calling: bool = False,
    native_response_types: list[type[type]] | None = None,
    use_json_adapter_fallback: bool = True,
    parallel_tool_calls: bool | None = None,
):
    """
    Args:
        callbacks: List of callback functions to execute during adapter methods.
        use_native_function_calling: Whether to enable native function calling capabilities.
        native_response_types: List of output field types handled by native LM features.
        use_json_adapter_fallback: Whether to automatically fallback to JSONAdapter if the ChatAdapter fails.
            If True, when an error occurs (except ContextWindowExceededError), the adapter will retry using
            JSONAdapter. Defaults to True.
        parallel_tool_calls: Whether to request provider-side parallel tool-call generation when native function
            calling is active. If None, the adapter does not set the provider option.
    """
    super().__init__(
        callbacks=callbacks,
        use_native_function_calling=use_native_function_calling,
        parallel_tool_calls=parallel_tool_calls,
        native_response_types=native_response_types,
    )
    self.use_json_adapter_fallback = use_json_adapter_fallback
```

### Methods:

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

        if isinstance(e, LMError) or isinstance(self, JSONAdapter) or not self.use_json_adapter_fallback:
            # On LM errors, already using JSONAdapter, or use_json_adapter_fallback is False, we don't want to
            # retry with a different adapter. Raise the original error instead of the fallback error.
            raise
        return self._make_json_adapter_fallback()(lm, lm_kwargs, signature, demos, inputs)
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

        if isinstance(e, LMError) or isinstance(self, JSONAdapter) or not self.use_json_adapter_fallback:
            # On LM errors, already using JSONAdapter, or use_json_adapter_fallback is False, we don't want to
            # retry with a different adapter. Raise the original error instead of the fallback error.
            raise
        return await self._make_json_adapter_fallback().acall(lm, lm_kwargs, signature, demos, inputs)
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
        if content:
            messages.append({"role": "user", "content": content})
    else:
        # Only current input
        content = self.format_user_message_content(signature, inputs_copy, main_request=True)
        if content:
            messages.append({"role": "user", "content": content})

    return [_expand_legacy_custom_type_markers_in_chat_message(message) for message in messages]
````

#### `format_assistant_message_content(signature: type[Signature], outputs: dict[str, Any], missing_field_message=None) -> str`

Source code in `dspy/adapters/xml_adapter.py`

```
def format_assistant_message_content(
    self, signature: type[Signature], outputs: dict[str, Any], missing_field_message=None
) -> str:
    fields = {
        FieldInfoWithName(name=k, info=v): outputs.get(k, missing_field_message)
        for k, v in signature.output_fields.items()
    }
    return self.format_field_with_value(fields)
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

| Type                   | Description                                         |
| ---------------------- | --------------------------------------------------- |
| `list[dict[str, Any]]` | A list of multiturn messages as expected by the LM. |

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
        A list of multiturn messages as expected by the LM.
    """
    conversation_history = inputs[history_field_name].messages if history_field_name in inputs else None

    if conversation_history is None:
        return []

    messages = []
    for message in conversation_history:
        tool_call_field_name, tool_calls = _tool_calls_from_message(message)
        tool_call_results = (
            ToolCallResults.model_validate(tool_calls.tool_call_results)
            if tool_calls is not None and tool_calls.tool_call_results is not None
            else None
        )

        user_content = self.format_user_message_content(signature, message)
        if user_content:
            messages.append({"role": "user", "content": user_content})

        if self.use_native_function_calling and tool_calls is not None:
            content_signature = signature
            for name, field in signature.output_fields.items():
                if field.annotation == ToolCalls or message.get(name) is None:
                    content_signature = content_signature.delete(name)

            content = (
                self.format_assistant_message_content(content_signature, message)
                if content_signature.output_fields
                else ""
            )

            if tool_call_results is not None:
                tool_call_ids = [tool_call.id for tool_call in tool_calls.tool_calls]
                result_ids = [result.call_id for result in tool_call_results.tool_call_results]
                if tool_call_ids != result_ids or not all(tool_call_ids):
                    tool_call_results = None

            if content or tool_call_results is not None:
                assistant_message: dict[str, Any] = {"role": "assistant", "content": content or None}
                if tool_call_results is not None:
                    assistant_message["tool_calls"] = [
                        _tool_call_as_openai_message_tool_call(tool_call) for tool_call in tool_calls.tool_calls
                    ]
                messages.append(assistant_message)

            if tool_call_results is not None:
                for result in tool_call_results.tool_call_results:
                    content = _tool_result_content(result.value)
                    messages.append(
                        {"role": "tool", "tool_call_id": result.call_id, "name": result.name, "content": content}
                    )
            continue

        assistant_values = message
        if tool_call_field_name is not None and tool_call_results is not None:
            assistant_values = dict(message)
            assistant_values[tool_call_field_name] = tool_calls.model_copy(update={"tool_call_results": None})

        assistant_content = self.format_assistant_message_content(signature, assistant_values)
        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})
        if tool_call_results is not None:
            result_input = {"tool_call_results": tool_call_results}
            content = self.format_user_message_content(_TOOL_CALL_RESULTS_SIGNATURE, result_input)
            messages.append({"role": "user", "content": content})

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

Source code in `dspy/adapters/xml_adapter.py`

```
def format_field_structure(self, signature: type[Signature]) -> str:
    def format_field(name, field):
        if (field.json_schema_extra or {}).get("__dspy_field_type") == "output" and self._uses_nested_xml(
            field.annotation
        ):
            return self._xml_schema(name, field.annotation)
        return self.format_field_with_value(
            {FieldInfoWithName(name=name, info=field): translate_field_type(name, field)}
        )

    fields = (
        "\n\n".join(format_field(name, field) for name, field in group.items())
        for group in (signature.input_fields, signature.output_fields)
    )
    return "\n\n".join(
        (
            "All interactions will be structured in the following way, with the appropriate values filled in.",
            *fields,
        )
    )
```

#### `format_field_with_value(fields_with_values: dict[FieldInfoWithName, Any]) -> str`

Source code in `dspy/adapters/xml_adapter.py`

```
def format_field_with_value(self, fields_with_values: dict[FieldInfoWithName, Any]) -> str:
    output = []
    for field, value in fields_with_values.items():
        serialized = serialize_for_json(value)
        is_output = (field.info.json_schema_extra or {}).get("__dspy_field_type") == "output"
        if is_output and self._uses_nested_xml(field.info.annotation) and isinstance(serialized, (dict, list)):
            output.append(self._value_to_xml(serialized, field.name))
            continue
        formatted = format_field_value(field_info=field.info, value=value)
        if is_output and field.info.annotation is str:
            formatted = formatted.replace("&", "&amp;").replace("<", "&lt;")
        output.append(f"<{field.name}>\n{formatted}\n</{field.name}>")
    return "\n\n".join(output).strip()
```

#### `format_finetune_data(signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, list[Any]]`

Format the call data into finetuning data according to the OpenAI API specifications.

For the chat adapter, this means formatting the data as a list of messages, where each message is a dictionary with a “role” and “content” key. The role can be “system”, “user”, or “assistant”. Then, the messages are wrapped in a dictionary with a “messages” key.

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

Source code in `dspy/adapters/xml_adapter.py`

```
def format_user_message_content(
    self,
    signature: type[Signature],
    inputs: dict[str, Any],
    prefix: str = "",
    suffix: str = "",
    main_request: bool = False,
) -> str:
    fields = {
        FieldInfoWithName(name=k, info=v): inputs[k] for k, v in signature.input_fields.items() if k in inputs
    }
    messages = [prefix, self.format_field_with_value(fields)]
    if main_request:
        messages.append(self.user_message_output_requirements(signature))
    return "\n\n".join((*messages, suffix)).strip()
```

#### `parse(signature: type[Signature], completion: str) -> dict[str, Any]`

Source code in `dspy/adapters/xml_adapter.py`

```
def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
    try:
        root = ET.fromstring(f"<dspy_root>{completion}</dspy_root>")
    except ET.ParseError as e:
        raise AdapterParseError(
            adapter_name="XMLAdapter",
            signature=signature,
            lm_response=completion,
            message=f"Failed to parse XML: {e}",
        ) from e
    elements = self._group_children(root)
    fields = {}
    for name, field in signature.output_fields.items():
        if name not in elements:
            continue
        adapter = TypeAdapter(field.annotation)
        schema = adapter.json_schema(by_alias=False)
        value = None
        for candidate in [schema, *schema.get("anyOf", [])]:
            try:
                value = self._elements_to_value(elements[name], candidate, schema.get("$defs", {}))
                try:
                    fields[name] = parse_value(value, field.annotation)
                except pydantic.ValidationError:
                    fields[name] = adapter.validate_python(value, by_name=True)
                break
            except Exception as e:
                error = e
        else:
            raise AdapterParseError(
                adapter_name="XMLAdapter",
                signature=signature,
                lm_response=completion,
                message=f"Failed to parse field {field} with value {value}: {error}",
            ) from error
    fields = apply_output_field_defaults(signature, fields)
    if fields.keys() != signature.output_fields.keys():
        raise AdapterParseError(
            adapter_name="XMLAdapter", signature=signature, lm_response=completion, parsed_result=fields
        )
    return fields
```

#### `user_message_output_requirements(signature: type[Signature]) -> str`

Source code in `dspy/adapters/xml_adapter.py`

```
def user_message_output_requirements(self, signature: type[Signature]) -> str:
    fields = ", then ".join(f"`<{name}>`" for name in signature.output_fields)
    schemas = [
        self._xml_schema(name, field.annotation)
        for name, field in signature.output_fields.items()
        if self._uses_nested_xml(field.annotation)
    ]
    return f"Respond with the corresponding output fields wrapped in XML tags {fields}." + (
        f" Use this nested XML structure: {' '.join(schemas)}" if schemas else ""
    )
```
