# dspy.BoundingBox

`BoundingBox` is a `dspy.Type` for an axis-aligned rectangle. Its four required fields are `top`, `bottom`, `left`, and `right`: finite numbers in [0, 1], measured from the image’s top-left corner. `top < bottom` and `left < right` are required. Invalid geometry raises a validation error; DSPy does not clamp or reorder it. Boxes are immutable, serialize as JSON objects, and can be input or output fields.

## Locate an object with Jev

Set `TYPESAFE_API_KEY`, then pass an image to a signature with a `BoundingBox` output:

```
import dspy

class Locate(dspy.Signature):
    photo: dspy.Image = dspy.InputField()
    visible: bool = dspy.OutputField(desc="Is there a coffee mug in inputs.photo?")
    mug: dspy.BoundingBox = dspy.OutputField(desc="The coffee mug in inputs.photo.")

lm = dspy.LM("typesafe:jev-1.14.0", engine="lm15")
predict = dspy.Predict(Locate)
with dspy.context(lm=lm):
    result = predict(photo=dspy.Image.from_path("kitchen.jpg"))

if result.visible:
    print(result.mug.model_dump())  # top, bottom, left, right
```

`Predict` selects the native Jev route automatically; no separate adapter is needed. For async calls, use `await predict.acall(...)` in the same context. There is **no not-found box**: ask a presence question and ignore the box when the target is absent. Invalid native geometry raises `dspy.AdapterParseError`.

To convert to pixels, multiply `left` and `right` by the image width, and `top` and `bottom` by its height. Use the dimensions of the image sent to the model; map cropped-image coordinates back to the original yourself.

### Jev requirements and limits

- Images and boxes are experimental and require Jev 1.14 or `jev-preview`. DSPy sends the required experimental-image header automatically.
- Use `Predict` with one top-level box field per target. Native Jev does not support nested, list, or nullable boxes, generated text, tools, or streaming.
- Boxes can accompany `bool`, `Literal`, and experimental `Noul`, `Choice`, or `Score` outputs. Decision calibration applies to those fields, not boxes; use `Score` for ordinal scoring, not an integer `Literal`.
- Load PNG/JPEG bytes with `Image.from_path` or `Image.from_url`; bare URLs are not fetched. DSPy does not resize or apply EXIF orientation here. See the [image limits](https://docs.typesafe.ai/images).
- For multiple images, name the target and its input image in the output field description. Images in inputs and demonstrations retain their named context.

Provider confidence stays in the LM response, separate from geometry; boxes have no entry in `response.probabilities`. Neither valid geometry nor high confidence guarantees accurate localization. See TypeSafe’s [Box guide](https://docs.typesafe.ai/primitives/box) for reference images, confidence thresholds, and tiling.

## Use with generative LMs

The same type works with `ChatAdapter`, `JSONAdapter`, `XMLAdapter`, and `TwoStepAdapter`, including nested models, `list[BoundingBox]`, and optional boxes, subject to provider support. Generated coordinates receive the same validation but have no native TypeSafe confidence.

## `dspy.BoundingBox`

Bases: `Type`

An axis-aligned rectangle in normalized image coordinates.

Edges are finite numbers in [0, 1], measured from the top-left corner. Multiply left/right by the image width and top/bottom by its height to obtain pixel coordinates. Use the dimensions of the upright image actually sent to the model, not those of an original image before resizing/cropping.

Boxes have positive area: top < bottom and left < right. Invalid outputs are rejected, never reordered or clamped. Provider confidence is not part of the geometry; native TypeSafe evidence remains in the LM response.

### Attributes

#### `model_config = ConfigDict(frozen=True, extra='forbid', strict=True)`

#### `top: float = Field(description='Top edge as a fraction of image height, from 0 to 1.')`

#### `bottom: float = Field(description='Bottom edge as a fraction of image height, from 0 to 1.')`

#### `left: float = Field(description='Left edge as a fraction of image width, from 0 to 1.')`

#### `right: float = Field(description='Right edge as a fraction of image width, from 0 to 1.')`

### Methods:

#### `validate_edge(value)`

Source code in `dspy/adapters/types/bounding_box.py`

```
@field_validator("top", "bottom", "left", "right")
@classmethod
def validate_edge(cls, value):
    # Runtime validation keeps the schema portable to providers that reject
    # numeric minimum/maximum constraints in structured outputs.
    if not 0 <= value <= 1:
        raise ValueError("Bounding box edges must be finite numbers in [0, 1].")
    return value
```

#### `validate_order()`

Source code in `dspy/adapters/types/bounding_box.py`

```
@model_validator(mode="after")
def validate_order(self):
    if self.top >= self.bottom or self.left >= self.right:
        raise ValueError("Bounding boxes require top < bottom and left < right.")
    return self
```

#### `serialize_model(handler)`

Source code in `dspy/adapters/types/bounding_box.py`

```
@model_serializer(mode="wrap")
def serialize_model(self, handler):
    # Geometry is structured JSON, not a media marker or a JSON-encoded string.
    return handler(self)
```

#### `format() -> str`

Source code in `dspy/adapters/types/bounding_box.py`

```
def format(self) -> str:
    return json.dumps(self.model_dump())
```
