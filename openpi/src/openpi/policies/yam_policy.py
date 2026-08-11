import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_yam_example() -> dict:
    """Creates a random input example for the bimanual YAM policy."""
    return {
        "observation/state": np.random.rand(14),
        "observation/image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "find the bin with banana",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    # single frame [c, h, w] or a step sequence [t, c, h, w] -> channels-last
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    elif image.ndim == 4 and image.shape[1] == 3:
        image = einops.rearrange(image, "t c h w -> t h w c")
    return image


@dataclasses.dataclass(frozen=True)
class YamInputs(transforms.DataTransformFn):
    """Converts bimanual YAM inputs to the model's expected format. Used for training and inference."""

    # Determines which model will be used. Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        base_image = _parse_image(data["observation/image"])
        left_wrist_image = _parse_image(data["observation/left_wrist_image"])
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])

        # The YAM setup has all three views (one third-person + two wrist), so none are padded.
        # Sequence samples ([T, h, w, c] images) carry a per-step mask to match the step axis.
        mask = np.ones(base_image.shape[0], dtype=bool) if base_image.ndim == 4 else np.True_
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": mask,
                "left_wrist_0_rgb": mask,
                "right_wrist_0_rgb": mask,
            },
        }

        # Pad actions to the model action dimension. Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # Pass the per-frame subtask label through (training only; consumed by the tokenizer).
        if "subtask" in data:
            inputs["subtask"] = data["subtask"]

        # Memory sequence-training extras pass through.
        for key in (
            "seq_step_mask",
            "seq_block_boundary",
            "seq_probe_labels",
            "seq_probe_mask",
            "seq_probe_visible",
        ):
            if key in data:
                inputs[key] = data[key]

        return inputs


@dataclasses.dataclass(frozen=True)
class YamOutputs(transforms.DataTransformFn):
    """Converts model outputs back to the bimanual YAM action format. Used for inference only."""

    def __call__(self, data: dict) -> dict:
        # Return only the first 14 actions -- the rest is padding to the model action dim.
        return {"actions": np.asarray(data["actions"][..., :14])}
