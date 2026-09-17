# dspy.Signature

## `dspy.Signature`

Bases: `BaseModel`

### Functions

#### `append(name, field, type_=None) -> type[Signature]`

Insert a field at the end of the `inputs` or `outputs` section.

Parameters:

| Name    | Type   | Description                                   | Default                                                                                         |
| ------- | ------ | --------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `name`  | `str`  | Field name to add.                            | *required*                                                                                      |
| `field` |        | InputField or OutputField instance to insert. | *required*                                                                                      |
| `type_` | \`type | None\`                                        | Optional explicit type annotation. If type\_ is None, the effective type is resolved by insert. |

Returns:

| Type              | Description                                    |
| ----------------- | ---------------------------------------------- |
| `type[Signature]` | A new Signature class with the field appended. |

Example

```
import dspy

class MySig(dspy.Signature):
    input_text: str = dspy.InputField(desc="Input sentence")
    output_text: str = dspy.OutputField(desc="Translated sentence")

NewSig = MySig.append("confidence", dspy.OutputField(desc="Translation confidence"))
print(list(NewSig.fields.keys()))
```

Source code in `dspy/signatures/signature.py`

````
@classmethod
def append(cls, name, field, type_=None) -> type["Signature"]:
    """Insert a field at the end of the `inputs` or `outputs` section.

    Args:
        name (str): Field name to add.
        field: `InputField` or `OutputField` instance to insert.
        type_ (type | None): Optional explicit type annotation. If `type_` is `None`, the effective type is
            resolved by `insert`.

    Returns:
        A new Signature class with the field appended.

    Example:
        ```python
        import dspy

        class MySig(dspy.Signature):
            input_text: str = dspy.InputField(desc="Input sentence")
            output_text: str = dspy.OutputField(desc="Translated sentence")

        NewSig = MySig.append("confidence", dspy.OutputField(desc="Translation confidence"))
        print(list(NewSig.fields.keys()))
        ```
    """
    return cls.insert(-1, name, field, type_)
````

#### `delete(name) -> type[Signature]`

Return a new Signature class without the given field.

If `name` is not present, the fields are unchanged (no error raised).

Parameters:

| Name   | Type  | Description           | Default    |
| ------ | ----- | --------------------- | ---------- |
| `name` | `str` | Field name to remove. | *required* |

Returns:

| Type              | Description                                                                          |
| ----------------- | ------------------------------------------------------------------------------------ |
| `type[Signature]` | A new Signature class with the field removed (or unchanged if the field was absent). |

Example

```
import dspy

class MySig(dspy.Signature):
    input_text: str = dspy.InputField(desc="Input sentence")
    temp_field: str = dspy.InputField(desc="Temporary debug field")
    output_text: str = dspy.OutputField(desc="Translated sentence")

NewSig = MySig.delete("temp_field")
print(list(NewSig.fields.keys()))

# No error is raised if the field is not present
Unchanged = NewSig.delete("nonexistent")
print(list(Unchanged.fields.keys()))
```

Source code in `dspy/signatures/signature.py`

````
@classmethod
def delete(cls, name) -> type["Signature"]:
    """Return a new Signature class without the given field.

    If `name` is not present, the fields are unchanged (no error raised).

    Args:
        name (str): Field name to remove.

    Returns:
        A new Signature class with the field removed (or unchanged if the field was absent).

    Example:
        ```python
        import dspy

        class MySig(dspy.Signature):
            input_text: str = dspy.InputField(desc="Input sentence")
            temp_field: str = dspy.InputField(desc="Temporary debug field")
            output_text: str = dspy.OutputField(desc="Translated sentence")

        NewSig = MySig.delete("temp_field")
        print(list(NewSig.fields.keys()))

        # No error is raised if the field is not present
        Unchanged = NewSig.delete("nonexistent")
        print(list(Unchanged.fields.keys()))
        ```
    """
    fields = dict(cls.fields)

    fields.pop(name, None)

    return Signature(fields, cls.instructions)
````

#### `dump_state()`

Source code in `dspy/signatures/signature.py`

```
@classmethod
def dump_state(cls):
    state = {"instructions": cls.instructions, "fields": []}
    for field in cls.fields:
        state["fields"].append(
            {
                "prefix": cls.fields[field].json_schema_extra["prefix"],
                "description": cls.fields[field].json_schema_extra["desc"],
            }
        )

    return state
```

#### `equals(other) -> bool`

Compare the JSON schema of two Signature classes.

Source code in `dspy/signatures/signature.py`

```
@classmethod
def equals(cls, other) -> bool:
    """Compare the JSON schema of two Signature classes."""
    if not isinstance(other, type) or not issubclass(other, BaseModel):
        return False
    if cls.instructions != other.instructions:
        return False
    for name in cls.fields.keys() | other.fields.keys():
        if name not in other.fields or name not in cls.fields:
            return False
        if cls.fields[name].json_schema_extra != other.fields[name].json_schema_extra:
            return False
    return True
```

#### `insert(index: int, name: str, field, type_: type | None = None) -> type[Signature]`

Insert a field at a specific position among inputs or outputs.

Negative indices are supported (e.g., `-1` appends). If `type_` is omitted, the field's existing `annotation` is used; if that is missing, `str` is used.

Parameters:

| Name    | Type   | Description                                                     | Default                            |
| ------- | ------ | --------------------------------------------------------------- | ---------------------------------- |
| `index` | `int`  | Insertion position within the chosen section; negatives append. | *required*                         |
| `name`  | `str`  | Field name to add.                                              | *required*                         |
| `field` |        | InputField or OutputField instance to insert.                   | *required*                         |
| `type_` | \`type | None\`                                                          | Optional explicit type annotation. |

Returns:

| Type              | Description                                    |
| ----------------- | ---------------------------------------------- |
| `type[Signature]` | A new Signature class with the field inserted. |

Raises:

| Type         | Description                                                    |
| ------------ | -------------------------------------------------------------- |
| `ValueError` | If index falls outside the valid range for the chosen section. |

Example

```
import dspy

class MySig(dspy.Signature):
    input_text: str = dspy.InputField(desc="Input sentence")
    output_text: str = dspy.OutputField(desc="Translated sentence")

NewSig = MySig.insert(0, "context", dspy.InputField(desc="Context for translation"))
print(list(NewSig.fields.keys()))

NewSig2 = NewSig.insert(-1, "confidence", dspy.OutputField(desc="Translation confidence"))
print(list(NewSig2.fields.keys()))
```

Source code in `dspy/signatures/signature.py`

````
@classmethod
def insert(cls, index: int, name: str, field, type_: type | None = None) -> type["Signature"]:
    """Insert a field at a specific position among inputs or outputs.

    Negative indices are supported (e.g., `-1` appends). If `type_` is omitted, the field's
    existing `annotation` is used; if that is missing, `str` is used.

    Args:
        index (int): Insertion position within the chosen section; negatives append.
        name (str): Field name to add.
        field: InputField or OutputField instance to insert.
        type_ (type | None): Optional explicit type annotation.

    Returns:
        A new Signature class with the field inserted.

    Raises:
        ValueError: If `index` falls outside the valid range for the chosen section.

    Example:
        ```python
        import dspy

        class MySig(dspy.Signature):
            input_text: str = dspy.InputField(desc="Input sentence")
            output_text: str = dspy.OutputField(desc="Translated sentence")

        NewSig = MySig.insert(0, "context", dspy.InputField(desc="Context for translation"))
        print(list(NewSig.fields.keys()))

        NewSig2 = NewSig.insert(-1, "confidence", dspy.OutputField(desc="Translation confidence"))
        print(list(NewSig2.fields.keys()))
        ```
    """
    # It's possible to set the type as annotation=type in pydantic.Field(...)
    # But this may be annoying for users, so we allow them to pass the type
    if type_ is None:
        type_ = field.annotation
    if type_ is None:
        type_ = str

    input_fields = list(cls.input_fields.items())
    output_fields = list(cls.output_fields.items())

    # Choose the list to insert into based on the field type
    lst = input_fields if field.json_schema_extra["__dspy_field_type"] == "input" else output_fields
    # We support negative insert indices
    if index < 0:
        index += len(lst) + 1
    if index < 0 or index > len(lst):
        raise ValueError(
            f"Invalid index to insert: {index}, index must be in the range of [{len(lst) - 1}, {len(lst)}] for "
            f"{field.json_schema_extra['__dspy_field_type']} fields, but received: {index}.",
        )
    lst.insert(index, (name, (type_, field)))

    new_fields = dict(input_fields + output_fields)
    return Signature(new_fields, cls.instructions)
````

#### `load_state(state)`

Source code in `dspy/signatures/signature.py`

```
@classmethod
def load_state(cls, state):
    signature_copy = Signature(deepcopy(cls.fields), cls.instructions)

    signature_copy.instructions = state["instructions"]
    for field, saved_field in zip(signature_copy.fields.values(), state["fields"], strict=False):
        field.json_schema_extra["prefix"] = saved_field["prefix"]
        field.json_schema_extra["desc"] = saved_field["description"]

    return signature_copy
```

#### `prepend(name, field, type_=None) -> type[Signature]`

Insert a field at index 0 of the `inputs` or `outputs` section.

Parameters:

| Name    | Type   | Description                                   | Default                                                                                         |
| ------- | ------ | --------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `name`  | `str`  | Field name to add.                            | *required*                                                                                      |
| `field` |        | InputField or OutputField instance to insert. | *required*                                                                                      |
| `type_` | \`type | None\`                                        | Optional explicit type annotation. If type\_ is None, the effective type is resolved by insert. |

Returns:

| Type              | Description                                          |
| ----------------- | ---------------------------------------------------- |
| `type[Signature]` | A new Signature class with the field inserted first. |

Example

```
import dspy

class MySig(dspy.Signature):
    input_text: str = dspy.InputField(desc="Input sentence")
    output_text: str = dspy.OutputField(desc="Translated sentence")

NewSig = MySig.prepend("context", dspy.InputField(desc="Context for translation"))
print(list(NewSig.fields.keys()))
```

Source code in `dspy/signatures/signature.py`

````
@classmethod
def prepend(cls, name, field, type_=None) -> type["Signature"]:
    """Insert a field at index 0 of the `inputs` or `outputs` section.

    Args:
        name (str): Field name to add.
        field: `InputField` or `OutputField` instance to insert.
        type_ (type | None): Optional explicit type annotation. If `type_` is `None`, the effective type is
            resolved by `insert`.

    Returns:
        A new `Signature` class with the field inserted first.

    Example:
        ```python
        import dspy

        class MySig(dspy.Signature):
            input_text: str = dspy.InputField(desc="Input sentence")
            output_text: str = dspy.OutputField(desc="Translated sentence")

        NewSig = MySig.prepend("context", dspy.InputField(desc="Context for translation"))
        print(list(NewSig.fields.keys()))
        ```
    """
    return cls.insert(0, name, field, type_)
````

#### `with_instructions(instructions: str) -> type[Signature]`

Return a new Signature class with identical fields and new instructions.

This method does not mutate `cls`. It constructs a fresh Signature class using the current fields and the provided `instructions`.

Parameters:

| Name           | Type  | Description                                      | Default    |
| -------------- | ----- | ------------------------------------------------ | ---------- |
| `instructions` | `str` | Instruction text to attach to the new signature. | *required* |

Returns:

| Type              | Description                                         |
| ----------------- | --------------------------------------------------- |
| `type[Signature]` | A new Signature class whose fields match cls.fields |
| `type[Signature]` | and whose instructions equal instructions.          |

Example

```
import dspy

class MySig(dspy.Signature):
    input_text: str = dspy.InputField(desc="Input text")
    output_text: str = dspy.OutputField(desc="Output text")

NewSig = MySig.with_instructions("Translate to French.")
assert NewSig is not MySig
assert NewSig.instructions == "Translate to French."
```

Source code in `dspy/signatures/signature.py`

````
@classmethod
def with_instructions(cls, instructions: str) -> type["Signature"]:
    """Return a new Signature class with identical fields and new instructions.

    This method does not mutate `cls`. It constructs a fresh Signature
    class using the current fields and the provided `instructions`.

    Args:
        instructions (str): Instruction text to attach to the new signature.

    Returns:
        A new Signature class whose fields match `cls.fields`
        and whose instructions equal `instructions`.

    Example:
        ```python
        import dspy

        class MySig(dspy.Signature):
            input_text: str = dspy.InputField(desc="Input text")
            output_text: str = dspy.OutputField(desc="Output text")

        NewSig = MySig.with_instructions("Translate to French.")
        assert NewSig is not MySig
        assert NewSig.instructions == "Translate to French."
        ```
    """
    return Signature(cls.fields, instructions)
````

#### `with_updated_fields(name: str, type_: type | None = None, **kwargs: dict[str, Any]) -> type[Signature]`

Create a new Signature class with the updated field information.

Returns a new Signature class with the field, name, updated with fields[name].json_schema_extra[key] = value.

Parameters:

| Name     | Type             | Description                      | Default                    |
| -------- | ---------------- | -------------------------------- | -------------------------- |
| `name`   | `str`            | The name of the field to update. | *required*                 |
| `type_`  | \`type           | None\`                           | The new type of the field. |
| `kwargs` | `dict[str, Any]` | The new values for the field.    | `{}`                       |

Returns:

| Type              | Description                                                                 |
| ----------------- | --------------------------------------------------------------------------- |
| `type[Signature]` | A new Signature class (not an instance) with the updated field information. |

Source code in `dspy/signatures/signature.py`

```
@classmethod
def with_updated_fields(cls, name: str, type_: type | None = None, **kwargs: dict[str, Any]) -> type["Signature"]:
    """Create a new Signature class with the updated field information.

    Returns a new Signature class with the field, name, updated
    with fields[name].json_schema_extra[key] = value.

    Args:
        name: The name of the field to update.
        type_: The new type of the field.
        kwargs: The new values for the field.

    Returns:
        A new Signature class (not an instance) with the updated field information.
    """
    fields_copy = deepcopy(cls.fields)
    # Update `fields_copy[name].json_schema_extra` with the new kwargs, on conflicts
    # we use the new value in kwargs.
    fields_copy[name].json_schema_extra = {
        **fields_copy[name].json_schema_extra,
        **kwargs,
    }
    if type_ is not None:
        fields_copy[name].annotation = type_
    return Signature(fields_copy, cls.instructions)
```

:::
