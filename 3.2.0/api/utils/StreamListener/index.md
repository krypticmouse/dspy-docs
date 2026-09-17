# dspy.streaming.StreamListener

## `dspy.streaming.StreamListener(signature_field_name: str, predict: Any = None, predict_name: str | None = None, allow_reuse: bool = False)`

Class that listens to the stream to capture the streeaming of a specific output field of a predictor.

Parameters:

| Name                   | Type   | Description                                                                                                                                                                    | Default                                                                                                                                                                    |
| ---------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `signature_field_name` | `str`  | The name of the field to listen to.                                                                                                                                            | *required*                                                                                                                                                                 |
| `predict`              | `Any`  | The predictor to listen to. If None, when calling streamify() it will automatically look for the predictor that has the signature_field_name in its signature.                 | `None`                                                                                                                                                                     |
| `predict_name`         | \`str  | None\`                                                                                                                                                                         | The name of the predictor to listen to. If None, when calling streamify() it will automatically look for the predictor that has the signature_field_name in its signature. |
| `allow_reuse`          | `bool` | If True, the stream listener can be reused for multiple streams. Please note that this could hurt the performance because the same stream chunk is sent to multiple listeners. | `False`                                                                                                                                                                    |

Source code in `dspy/streaming/streaming_listener.py`

```
def __init__(
    self,
    signature_field_name: str,
    predict: Any = None,
    predict_name: str | None = None,
    allow_reuse: bool = False,
):
    """
    Args:
        signature_field_name: The name of the field to listen to.
        predict: The predictor to listen to. If None, when calling `streamify()` it will automatically look for
            the predictor that has the `signature_field_name` in its signature.
        predict_name: The name of the predictor to listen to. If None, when calling `streamify()` it will
            automatically look for the predictor that has the `signature_field_name` in its signature.
        allow_reuse: If True, the stream listener can be reused for multiple streams. Please note that this could
            hurt the performance because the same stream chunk is sent to multiple listeners.
    """
    self.signature_field_name = signature_field_name
    self.predict = predict
    self.predict_name = predict_name

    self.field_start_queue = []
    self.field_end_queue = Queue()
    self.stream_start = False
    self.stream_end = False
    self.cache_hit = False
    self.allow_reuse = allow_reuse

    self.json_adapter_state = {"field_accumulated_messages": ""}

    self.adapter_identifiers = {
        "ChatAdapter": {
            "start_identifier": f"[[ ## {self.signature_field_name} ## ]]",
            "end_identifier": re.compile(r"\[\[ ## (\w+) ## \]\]"),
            "start_indicator": "[",
            "end_pattern_prefixes": ["[", "[[", "[[ ", "[[ #", "[[ ##"],
            "end_pattern_contains": "[[ ##",
        },
        "JSONAdapter": {
            "start_identifier": f'"{self.signature_field_name}":',
            "end_identifier": re.compile(r"\w*\"(,|\s*})"),
            "start_indicator": '"',
            "end_pattern_prefixes": ['"', '",', '" ', '"}'],
            "end_pattern_contains": "}",
        },
        "XMLAdapter": {
            "start_identifier": f"<{self.signature_field_name}>",
            "end_identifier": re.compile(rf"</{self.signature_field_name}>"),
            "start_indicator": "<",
            "end_pattern_prefixes": ["<", "</"],
            "end_pattern_contains": "</",  # Any closing tag start
        },
    }
```

### Functions

#### `finalize() -> StreamResponse | None`

Finalize the stream and flush any remaining buffered tokens.

This should be called when the stream ends. It ensures no tokens are lost from the buffer and marks the final chunk appropriately.

Returns:

| Type             | Description |
| ---------------- | ----------- |
| \`StreamResponse | None\`      |
| \`StreamResponse | None\`      |

Source code in `dspy/streaming/streaming_listener.py`

```
def finalize(self) -> StreamResponse | None:
    """Finalize the stream and flush any remaining buffered tokens.

    This should be called when the stream ends.
    It ensures no tokens are lost from the buffer and marks the final chunk appropriately.

    Returns:
        A StreamResponse with the remaining buffered tokens and is_last_chunk=True,
        or None if there are no buffered tokens or the stream hasn't started.
    """
    if self.stream_end or not self.stream_start:
        # Stream already ended or never started, nothing to finalize
        return None

    self.stream_end = True
    if self.field_end_queue.qsize() > 0:
        token = self.flush()
        if token:
            return StreamResponse(
                self.predict_name,
                self.signature_field_name,
                token,
                is_last_chunk=True,
            )
    return None
```

#### `flush() -> str`

Flush all tokens in the field end queue.

This method is called to flush out the last a few tokens when the stream is ended. These tokens are in the buffer because we don't directly yield the tokens received by the stream listener with the purpose to not yield the end_identifier tokens, e.g., "\[[ ## ... ## ]\]" for ChatAdapter.

Source code in `dspy/streaming/streaming_listener.py`

```
def flush(self) -> str:
    """Flush all tokens in the field end queue.

    This method is called to flush out the last a few tokens when the stream is ended. These tokens
    are in the buffer because we don't directly yield the tokens received by the stream listener
    with the purpose to not yield the end_identifier tokens, e.g., "[[ ## ... ## ]]" for ChatAdapter.
    """
    last_tokens = "".join(self.field_end_queue.queue)
    self.field_end_queue = Queue()
    if isinstance(settings.adapter, JSONAdapter):
        return last_tokens
    elif isinstance(settings.adapter, XMLAdapter):
        boundary_index = last_tokens.find(f"</{self.signature_field_name}>")
        if boundary_index == -1:
            boundary_index = len(last_tokens)
        return last_tokens[:boundary_index]
    elif isinstance(settings.adapter, ChatAdapter) or settings.adapter is None:
        boundary_index = last_tokens.find("[[")
        if boundary_index == -1:
            boundary_index = len(last_tokens)
        return last_tokens[:boundary_index]
    else:
        raise ValueError(
            f"Unsupported adapter for streaming: {settings.adapter}, please use one of the following adapters: "
            f"{', '.join([a.__name__ for a in ADAPTER_SUPPORT_STREAMING])}"
        )
```

#### `receive(chunk: ModelResponseStream)`

Source code in `dspy/streaming/streaming_listener.py`

```
def receive(self, chunk: ModelResponseStream):
    adapter_name = settings.adapter.__class__.__name__ if settings.adapter else "ChatAdapter"
    if adapter_name not in self.adapter_identifiers:
        raise ValueError(
            f"Unsupported adapter for streaming: {adapter_name}, please use one of the following adapters: "
            f"{', '.join([a.__name__ for a in ADAPTER_SUPPORT_STREAMING])}"
        )
    start_identifier = self.adapter_identifiers[adapter_name]["start_identifier"]
    end_identifier = self.adapter_identifiers[adapter_name]["end_identifier"]
    start_indicator = self.adapter_identifiers[adapter_name]["start_indicator"]

    if self.stream_end:
        if self.allow_reuse:
            # Clear up the state for the next stream.
            self.stream_end = False
            self.cache_hit = False
            self.field_start_queue = []
            self.field_end_queue = Queue()
            self.json_adapter_state["field_accumulated_messages"] = ""
            self.stream_start = False
        else:
            return

    # Handle custom streamable types
    if (
        self._output_type
        and inspect.isclass(self._output_type)
        and issubclass(self._output_type, Type)
        and self._output_type.is_streamable()
    ):
        if parsed_chunk := self._output_type.parse_stream_chunk(chunk):
            return StreamResponse(
                self.predict_name,
                self.signature_field_name,
                parsed_chunk,
                is_last_chunk=self.stream_end,
            )

    # For non-custom streamable types, the streaming chunks come from the content field of the ModelResponseStream.
    try:
        chunk_message = chunk.choices[0].delta.content
        if chunk_message is None:
            return
    except Exception:
        return

    if chunk_message and start_identifier in chunk_message and not isinstance(settings.adapter, JSONAdapter):
        # If the cache is hit, the chunk_message could be the full response. When it happens we can
        # directly end the stream listening. In some models like gemini, each stream chunk can be multiple
        # tokens, so it's possible that response only has one chunk, we also fall back to this logic.
        message_after_start_identifier = chunk_message[
            chunk_message.find(start_identifier) + len(start_identifier) :
        ]
        if re.search(end_identifier, message_after_start_identifier):
            self.cache_hit = True
            self.stream_start = True
            self.stream_end = True
            return

    if len(self.field_start_queue) == 0 and not self.stream_start and start_indicator in chunk_message:
        # We look for the pattern of start_identifier, i.e., "[[ ## {self.signature_field_name} ## ]]" for
        # ChatAdapter to identify the start of the stream of our target field. Once the start_indicator, i.e., "[["
        # for ChatAdapter, is found, we start checking the next tokens
        self.field_start_queue.append(chunk_message)
        return

    if len(self.field_start_queue) > 0 and not self.stream_start:
        # We keep appending the tokens to the queue until we have a full identifier or the concanated
        # tokens no longer match our expected identifier.
        self.field_start_queue.append(chunk_message)
        concat_message = "".join(self.field_start_queue)

        if start_identifier in concat_message:
            # We have a full identifier, we can start the stream.
            self.stream_start = True
            self.field_start_queue = []
            # Keep the part after the start_identifier from the concat_message, we need to write it to the buffer.
            value_start_index = concat_message.find(start_identifier) + len(start_identifier)
            chunk_message = concat_message[value_start_index:].lstrip()

            if isinstance(settings.adapter, JSONAdapter):
                # For JSONAdapter, we rely on partial json parsing to detect the end of the field we are listening
                # to, so we need to maintain a few extra states to help us with that.
                # We add an extra "{" to the beginning of the field_accumulated_messages, so we can detect the
                # appearance of the next key.
                self.json_adapter_state["field_accumulated_messages"] += "{" + start_identifier

        elif self._buffered_message_end_with_start_identifier(concat_message.strip(), start_identifier):
            # If the buffered message ends with part of the start_identifier, we keep looking for the
            # start_identifier from the token stream.
            return
        else:
            # Doesn't match the expected identifier, reset the queue.
            self.field_start_queue = []
            return

    if self.stream_start and chunk_message:
        # The stream is started, we keep returning the token until we see the start of the next field.
        self.field_end_queue.put(chunk_message)

        token = None
        concat_message = "".join(self.field_end_queue.queue).strip()

        if not self._could_form_end_identifier(concat_message, adapter_name):
            # Buffer cannot form end identifier, safe to flush out the tokens in the buffer.
            token = self.flush()
        elif self.field_end_queue.qsize() > 10:
            # We keep the last 10 tokens in the buffer if they can potentially form the end_identifier to avoid
            # sending the DSPy boilerplate tokens to users. 10 is a heuristic number that is sufficient to capture
            # the end_identifier for all LMs.
            token = self.field_end_queue.get()

        # TODO: Put adapter streaming handling into individial classes, e.g., `JSONAdapterStreamListener`,
        # `ChatAdapterStreamListener`, `XMLAdapterStreamListener` instead of having many adhoc code in the
        # `StreamListener` class.
        if isinstance(settings.adapter, JSONAdapter):
            # JSONAdapter uses partial json parsing to detect the end of the field we are listening to, instead of
            # relying on the end_identifier.
            return self._json_adapter_handle_stream_chunk(token, chunk_message)
        else:
            # Other adapters rely on the end_identifier to detect the end of the field we are listening to.
            return self._default_handle_stream_chunk(token, end_identifier)
```
