"""Policy I/O transforms for the plate-memory base (no-memory) baseline.

Dataset keys (as written by examples/yam/convert_plate_memory_to_lerobot.py):
    top_image, left_image, right_image  -> uint8 (H, W, 3)
    state                                -> float32 (7,)
    actions                              -> float32 (7,)
    task                                 -> str  (per-frame segment instruction)

Camera mapping into pi0.5's three image slots:
    top_image    -> base_0_rgb         (third-person view)
    left_image   -> left_wrist_0_rgb
    right_image  -> right_wrist_0_rgb  (DISABLED for this baseline)

The right camera is intentionally disabled for this baseline:
the slot is filled with all-zero pixels and image_mask['right_wrist_0_rgb'] is
np.False_, matching the missing-camera convention used in droid_policy / aloha_policy.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_plate_memory_example() -> dict:
    """Random input example (used for tracing / smoke tests)."""
    return {
        "observation/state": np.random.rand(7),
        "observation/top_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "Observe and remember which object is on each colored plate.",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class PlateMemoryInputs(transforms.DataTransformFn):
    """Maps plate-memory dataset records into the π₀.₅ model input format with right cam disabled."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        top = _parse_image(data["observation/top_image"])
        left = _parse_image(data["observation/left_image"])
        right_blank = np.zeros_like(top)

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": top,
                "left_wrist_0_rgb": left,
                "right_wrist_0_rgb": right_blank,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PlateMemoryOutputs(transforms.DataTransformFn):
    """Trims the model's padded action vector back to the YAM 7-DoF action space."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
