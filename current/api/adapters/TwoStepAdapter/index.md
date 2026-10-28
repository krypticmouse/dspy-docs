# dspy.TwoStepAdapter

## `dspy.TwoStepAdapter(extraction_model: BaseLM, **kwargs)`

Bases: `Adapter`

A two-stage adapter that

1. Uses a simpler, more natural prompt for the main LM
1. Uses a smaller LM with chat adapter to extract structured data from the response of main LM

This adapter uses a common **call** logic defined in base Adapter class. This class is particularly useful when interacting with reasoning models as the main LM since reasoning models are known to struggle with structured outputs.

Examples:

```
import dspy
lm = dspy.LM(model="openai/o3-mini", max_tokens=16000, temperature = 1.0)
adapter = dspy.TwoStepAdapter(dspy.LM("openai/gpt-4o-mini"))
dspy.configure(lm=lm, adapter=adapter)
program = dspy.ChainOfThought("question->answer")
result = program("What is the capital of France?")
print(result)
```

Source code in `dspy/adapters/two_step_adapter.py`

```
def __init__(self, extraction_model: BaseLM, **kwargs):
    super().__init__(**kwargs)
    if not isinstance(extraction_model, BaseLM):
        raise ValueError("extraction_model must be an instance of dspy.BaseLM")
    self.extraction_model = extraction_model
```

### Methods:

#### `__call__(lm: BaseLM, lm_kwargs: dict[str, Any], signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]`

Execute the adapter pipeline: format inputs, call LM, and parse outputs.

Parameters:

| Name        | Type                   | Description                                                                                                                                                                                            | Default    |
| ----------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- |
| `lm`        | `BaseLM`               | The Language Model instance to use for generation. Must be an instance of dspy.BaseLM.                                                                                                                 | *required* |
| `lm_kwargs` | `dict[str, Any]`       | Additional keyword arguments to pass to the LM call (e.g., temperature, max_tokens). These are passed directly to the LM.                                                                              | *required* |
| `signature` | `type[Signature]`      | The DSPy signature associated with this LM call.                                                                                                                                                       | *required* |
| `demos`     | `list[dict[str, Any]]` | List of few-shot examples to include in the prompt. Each dictionary should contain keys matching the signature’s input and output field names. Examples are formatted as user/assistant message pairs. | *required* |
| `inputs`    | `dict[str, Any]`       | The current input values for this call. Keys must match the signature’s input field names.                                                                                                             | *required* |

Returns:

| Type                   | Description                                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------------------- |
| `list[dict[str, Any]]` | List of dictionaries representing parsed LM responses. Each dictionary contains keys matching the |
| `list[dict[str, Any]]` | signature’s output field names. For multiple generations (n > 1), returns multiple dictionaries.  |

Source code in `dspy/adapters/base.py`

```
@with_capability_planning
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
    prompt = self.format(processed_signature, demos, inputs)
    request = build_request(lm, prompt, lm_kwargs)
    if settings.send_stream and settings.stream_listeners:
        from dspy.streaming.messages import _StreamSignature, sync_send_to_stream

        sync_send_to_stream(settings.send_stream, _StreamSignature(id(settings.caller_predict), processed_signature))
    responses = [lm(candidate, **options) for candidate, options in _candidate_calls(lm, lm_kwargs, request)]
    return self._call_postprocess(processed_signature, signature, responses)
```

#### `acall(lm: BaseLM, lm_kwargs: dict[str, Any], signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]`

Source code in `dspy/adapters/two_step_adapter.py`

```
async def acall(
    self,
    lm: BaseLM,
    lm_kwargs: dict[str, Any],
    signature: type[Signature],
    demos: list[dict[str, Any]],
    inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    from dspy.adapters.base import _candidate_calls

    request = build_request(lm, self.format(signature, demos, inputs), lm_kwargs)
    responses = [await lm.acall(candidate, **options) for candidate, options in _candidate_calls(lm, lm_kwargs, request)]
    # The signature is supposed to be "text -> {original output fields}"
    extractor_signature = self._create_extractor_signature(signature)

    values = []

    tool_call_output_field_name = self._get_tool_call_output_field_name(signature)
    for response in responses:
        text = response_text(response) or ""
        try:
            # Call the smaller LM to extract structured data from the raw completion text with ChatAdapter
            value = await ChatAdapter().acall(
                lm=self.extraction_model,
                lm_kwargs={},
                signature=extractor_signature,
                demos=[],
                inputs={"text": text},
            )
            value = value[0]

        except AdapterParseError as e:
            raise AdapterParseError(
                adapter_name="TwoStepAdapter",
                signature=signature,
                lm_response=text,
                message=f"Failed to parse response from the original completion: {e}",
            ) from e

        if response.tool_calls and tool_call_output_field_name:
            value[tool_call_output_field_name] = ToolCalls.from_dict_list(
                [{"id": call.id, "name": call.name, "args": call.input} for call in response.tool_calls]
            )

        if response.logprobs is not None:
            value["logprobs"] = response.logprobs

        values.append(value)
    return values
```

#### `format(signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]) -> Prompt`

Format a prompt for the first stage with the main LM. This no specific structure is required for the main LM, we customize the format method instead of format_field_description or format_field_structure.

Parameters:

| Name        | Type                   | Description                        | Default    |
| ----------- | ---------------------- | ---------------------------------- | ---------- |
| `signature` | `type[Signature]`      | The signature of the original task | *required* |
| `demos`     | `list[dict[str, Any]]` | A list of demo examples            | *required* |
| `inputs`    | `dict[str, Any]`       | The current input                  | *required* |

Returns:

| Type     | Description                 |
| -------- | --------------------------- |
| `Prompt` | The prompt for the main LM. |

Source code in `dspy/adapters/two_step_adapter.py`

```
def format(self, signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]) -> Prompt:
    """
    Format a prompt for the first stage with the main LM.
    This no specific structure is required for the main LM, we customize the format method
    instead of format_field_description or format_field_structure.

    Args:
        signature: The signature of the original task
        demos: A list of demo examples
        inputs: The current input

    Returns:
        The prompt for the main LM.
    """
    messages = list(self.format_demos(signature, demos))
    messages.append(user_message(self.format_user_message_content(signature, inputs)))
    return Prompt(system=self.format_task_description(signature), messages=tuple(messages))
```

#### `format_assistant_message_content(signature: type[Signature], outputs: dict[str, Any], missing_field_message: str | None = None) -> str`

Source code in `dspy/adapters/two_step_adapter.py`

```
def format_assistant_message_content(
    self,
    signature: type[Signature],
    outputs: dict[str, Any],
    missing_field_message: str | None = None,
) -> str:
    parts = []

    for name in signature.output_fields.keys():
        if name in outputs:
            parts.append(f"{name}: {outputs.get(name, missing_field_message)}")

    return "\n\n".join(parts).strip()
```

#### `format_conversation_history(signature: type[Signature], history_field_name: str, inputs: dict[str, Any]) -> list[Message]`

Format the conversation history.

This method formats the conversation history and the current input as multiturn messages.

Parameters:

| Name                 | Type              | Description                                                      | Default    |
| -------------------- | ----------------- | ---------------------------------------------------------------- | ---------- |
| `signature`          | `type[Signature]` | The DSPy signature for which to format the conversation history. | *required* |
| `history_field_name` | `str`             | The name of the history field in the signature.                  | *required* |
| `inputs`             | `dict[str, Any]`  | The input arguments to the DSPy module.                          | *required* |

Returns:

| Type            | Description                        |
| --------------- | ---------------------------------- |
| `list[Message]` | A list of multiturn lm15 messages. |

Source code in `dspy/adapters/base.py`

```
def format_conversation_history(
    self,
    signature: type[Signature],
    history_field_name: str,
    inputs: dict[str, Any],
) -> list[Message]:
    """Format the conversation history.

    This method formats the conversation history and the current input as multiturn messages.

    Args:
        signature: The DSPy signature for which to format the conversation history.
        history_field_name: The name of the history field in the signature.
        inputs: The input arguments to the DSPy module.

    Returns:
        A list of multiturn lm15 messages.
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

        # Preserve the format reminder from the original request for prompt-cache reuse.
        # Output-only events must not acquire a synthetic user message containing just the reminder.
        user_content = self.format_user_message_content(
            signature, message, main_request=any(name in message for name in signature.input_fields)
        )
        if user_content:
            messages.append(user_message(user_content))

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
                parts = [TextPart(content)] if content else []
                if tool_call_results is not None:
                    parts.extend(
                        ToolCallPart(id=tool_call.id, name=tool_call.name, input=serialize_for_json(tool_call.args))
                        for tool_call in tool_calls.tool_calls
                    )
                messages.append(Message.assistant(parts))

            if tool_call_results is not None:
                messages.append(Message.tool([
                    tool_result(result.call_id, _tool_result_content(result.value), name=result.name)
                    for result in tool_call_results.tool_call_results
                ]))
            continue

        assistant_values = message
        if tool_call_field_name is not None and tool_call_results is not None:
            assistant_values = dict(message)
            assistant_values[tool_call_field_name] = tool_calls.model_copy(update={"tool_call_results": None})

        assistant_content = self.format_assistant_message_content(signature, assistant_values)
        if assistant_content:
            messages.append(Message.assistant(assistant_content))
        if tool_call_results is not None:
            result_input = {"tool_call_results": tool_call_results}
            content = self.format_user_message_content(_TOOL_CALL_RESULTS_SIGNATURE, result_input)
            messages.append(user_message(content))

    # Remove the history field from the inputs
    del inputs[history_field_name]

    return messages
```

#### `format_demos(signature: type[Signature], demos: list[dict[str, Any]]) -> list[Message]`

Format the few-shot examples.

This method formats the few-shot examples as multiturn messages.

Parameters:

| Name        | Type                   | Description                                                                                                          | Default    |
| ----------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------- | ---------- |
| `signature` | `type[Signature]`      | The DSPy signature for which to format the few-shot examples.                                                        | *required* |
| `demos`     | `list[dict[str, Any]]` | A list of few-shot examples, each element is a dictionary with keys of the input and output fields of the signature. | *required* |

Returns:

| Type            | Description                        |
| --------------- | ---------------------------------- |
| `list[Message]` | A list of multiturn lm15 messages. |

Source code in `dspy/adapters/base.py`

```
def format_demos(self, signature: type[Signature], demos: list[dict[str, Any]]) -> list[Message]:
    """Format the few-shot examples.

    This method formats the few-shot examples as multiturn messages.

    Args:
        signature: The DSPy signature for which to format the few-shot examples.
        demos: A list of few-shot examples, each element is a dictionary with keys of the input and output fields of
            the signature.

    Returns:
        A list of multiturn lm15 messages.
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
        messages.append(user_message(self.format_user_message_content(signature, demo, prefix=incomplete_demo_prefix)))
        messages.append(Message.assistant(self.format_assistant_message_content(
            signature, demo, missing_field_message="Not supplied for this particular example. "
        )))

    for demo in complete_demos:
        messages.append(user_message(self.format_user_message_content(signature, demo)))
        messages.append(Message.assistant(self.format_assistant_message_content(
            signature, demo, missing_field_message="Not supplied for this conversation history message. "
        )))

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

#### `format_task_description(signature: Signature) -> str`

Create a description of the task based on the signature

Source code in `dspy/adapters/two_step_adapter.py`

```
def format_task_description(self, signature: Signature) -> str:
    """Create a description of the task based on the signature"""
    parts = []

    parts.append("You are a helpful assistant that can solve tasks based on user input.")
    parts.append("As input, you will be provided with:\n" + get_field_description_string(signature.input_fields))
    parts.append("Your outputs must contain:\n" + get_field_description_string(signature.output_fields))
    parts.append("You should lay out your outputs in detail so that your answer can be understood by another agent")

    if signature.instructions:
        parts.append(f"Specific instructions: {signature.instructions}")

    return "\n".join(parts)
```

#### `format_user_message_content(signature: type[Signature], inputs: dict[str, Any], prefix: str = '', suffix: str = '', main_request: bool = False) -> str`

Source code in `dspy/adapters/two_step_adapter.py`

```
def format_user_message_content(
    self,
    signature: type[Signature],
    inputs: dict[str, Any],
    prefix: str = "",
    suffix: str = "",
    main_request: bool = False,
) -> str:
    # `main_request` is accepted for compatibility with the base Adapter contract; the first-stage prompt has no
    # output-format reminder to add.
    parts = [prefix]

    for name in signature.input_fields.keys():
        if name in inputs:
            parts.append(f"{name}: {inputs.get(name, '')}")

    parts.append(suffix)
    return "\n\n".join(parts).strip()
```

#### `parse(signature: Signature, completion: str) -> dict[str, Any]`

Use a smaller LM (extraction_model) with chat adapter to extract structured data from the raw completion text of the main LM.

Parameters:

| Name         | Type        | Description                        | Default    |
| ------------ | ----------- | ---------------------------------- | ---------- |
| `signature`  | `Signature` | The signature of the original task | *required* |
| `completion` | `str`       | The completion from the main LM    | *required* |

Returns:

| Type             | Description                                            |
| ---------------- | ------------------------------------------------------ |
| `dict[str, Any]` | A dictionary containing the extracted structured data. |

Source code in `dspy/adapters/two_step_adapter.py`

```
def parse(self, signature: Signature, completion: str) -> dict[str, Any]:
    """
    Use a smaller LM (extraction_model) with chat adapter to extract structured data
    from the raw completion text of the main LM.

    Args:
        signature: The signature of the original task
        completion: The completion from the main LM

    Returns:
        A dictionary containing the extracted structured data.
    """
    # The signature is supposed to be "text -> {original output fields}"
    extractor_signature = self._create_extractor_signature(signature)

    try:
        # Call the smaller LM to extract structured data from the raw completion text with ChatAdapter
        parsed_result = ChatAdapter()(
            lm=self.extraction_model,
            lm_kwargs={},
            signature=extractor_signature,
            demos=[],
            inputs={"text": completion},
        )
        return parsed_result[0]

    except AdapterParseError as e:
        raise AdapterParseError(
            adapter_name="TwoStepAdapter",
            signature=signature,
            lm_response=completion,
            message=f"Failed to parse response from the original completion: {e}",
        ) from e
```

```
  March 11, 2026
```
