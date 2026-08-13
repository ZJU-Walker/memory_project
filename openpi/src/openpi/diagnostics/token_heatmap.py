"""Render per-token memory-write diagnostics on the exact top-camera model input.

The pi0.5 vision encoder receives a 224x224 letterboxed RGB image and emits a
16x16 patch grid in row-major order: token ``i`` covers row ``i // 16`` and
column ``i % 16`` of the model image. When a raw camera frame is available, this
module exactly inverts the letterbox geometry—cropping padding before projecting
the patch map back to raw pixels. It never stretches padded model coordinates
over the camera content.

The renderer is independent from the v3.1 evaluator.  Callers provide one
``TokenMetricFrame`` per measured write after calculating a non-negative metric
such as per-token associative error or per-token fast-weight gradient norm.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

MODEL_IMAGE_SIZE = 224
TOKEN_GRID_SIZE = 16
TOKEN_COUNT = TOKEN_GRID_SIZE * TOKEN_GRID_SIZE
PATCH_SIZE = MODEL_IMAGE_SIZE // TOKEN_GRID_SIZE
TOKEN_LAYOUT = "row-major (y, x): token i -> row i//16, column i%16; no transpose or flip"
SCALE_MODES = ("video", "per_frame_zscore", "per_token_history")
ScaleMode = Literal["video", "per_frame_zscore", "per_token_history"]
_LOG_FLOOR = 1e-12


@dataclasses.dataclass(frozen=True)
class LetterboxGeometry:
    """Geometry used by ``openpi.shared.image_tools.resize_with_pad``."""

    source_height: int
    source_width: int
    resized_height: int
    resized_width: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int
    target_height: int = MODEL_IMAGE_SIZE
    target_width: int = MODEL_IMAGE_SIZE

    def __post_init__(self) -> None:
        dimensions = (
            self.source_height,
            self.source_width,
            self.resized_height,
            self.resized_width,
            self.target_height,
            self.target_width,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("source, resized, and target dimensions must be positive")
        if any(value < 0 for value in (self.pad_top, self.pad_bottom, self.pad_left, self.pad_right)):
            raise ValueError("letterbox padding must be non-negative")
        if (self.target_height, self.target_width) != (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE):
            raise ValueError("token heatmaps require the model's exact 224x224 target")
        if self.resized_height + self.pad_top + self.pad_bottom != self.target_height:
            raise ValueError("resized height plus vertical padding does not equal target height")
        if self.resized_width + self.pad_left + self.pad_right != self.target_width:
            raise ValueError("resized width plus horizontal padding does not equal target width")

        ratio = max(self.source_width / self.target_width, self.source_height / self.target_height)
        expected_resized_height = int(self.source_height / ratio)
        expected_resized_width = int(self.source_width / ratio)
        expected_pad_top, remainder_height = divmod(self.target_height - expected_resized_height, 2)
        expected_pad_left, remainder_width = divmod(self.target_width - expected_resized_width, 2)
        expected = (
            expected_resized_height,
            expected_resized_width,
            expected_pad_top,
            expected_pad_top + remainder_height,
            expected_pad_left,
            expected_pad_left + remainder_width,
        )
        actual = (
            self.resized_height,
            self.resized_width,
            self.pad_top,
            self.pad_bottom,
            self.pad_left,
            self.pad_right,
        )
        if actual != expected:
            raise ValueError(f"letterbox geometry does not match resize_with_pad: expected {expected}, got {actual}")

    def to_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class NormalizationSpec:
    """One fixed scale shared by every frame in a video.

    A video-wide scale is essential: independently normalizing each frame would
    make a nearly uniform, tiny update look as salient as a concentrated update.
    Associative errors and gradient norms have a meaningful zero, so the default
    anchors the lower color limit at zero and clips only the upper tail.
    """

    lower_percentile: float = 0.0
    upper_percentile: float = 99.0
    anchor_zero: bool = True
    # Fit the scale using only tokens that cover real camera pixels. Attention in particular
    # dumps large "sink" mass onto letterbox padding, which is not part of the image at all; a
    # scale fitted over those tokens compresses everything happening on the actual scene into a
    # few colour steps. Padding values are still recorded in the NPZ and reported as statistics.
    exclude_letterbox_padding: bool = False

    def __post_init__(self) -> None:
        if not (0.0 <= self.lower_percentile < self.upper_percentile <= 100.0):
            raise ValueError("normalization percentiles must satisfy 0 <= lower < upper <= 100")


@dataclasses.dataclass(frozen=True)
class ColorScale:
    vmin: float
    vmax: float
    lower_percentile: float
    upper_percentile: float
    anchor_zero: bool
    excluded_letterbox_padding: bool = False

    def __post_init__(self) -> None:
        if not (math.isfinite(self.vmin) and math.isfinite(self.vmax)):
            raise ValueError("color scale limits must be finite")
        if self.vmin < 0 or self.vmax <= self.vmin:
            raise ValueError(f"invalid non-negative color scale [{self.vmin}, {self.vmax}]")

    def to_dict(self) -> dict[str, float | bool]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class TokenMetricFrame:
    """One model-aligned image and its 256 write-token measurements."""

    raw_frame: int
    policy_step: int
    model_image_rgb: np.ndarray
    token_values: np.ndarray
    write_count: int | None = None
    phase: str = ""
    timestamp_s: float | None = None
    metadata: Mapping[str, str | int | float | bool | None] = dataclasses.field(default_factory=dict)
    raw_image_rgb: np.ndarray | None = None
    letterbox_geometry: LetterboxGeometry | None = None

    def __post_init__(self) -> None:
        if self.raw_frame < 0 or self.policy_step < 0:
            raise ValueError("raw_frame and policy_step must be non-negative")
        if self.write_count is not None and self.write_count < 0:
            raise ValueError("write_count must be non-negative when present")
        if self.timestamp_s is not None and (not math.isfinite(self.timestamp_s) or self.timestamp_s < 0):
            raise ValueError("timestamp_s must be finite and non-negative when present")

        image = _validate_model_image(self.model_image_rgb)
        values = _validate_token_values(self.token_values)
        raw_image = _validate_raw_image(self.raw_image_rgb) if self.raw_image_rgb is not None else None
        geometry = self.letterbox_geometry
        if raw_image is None and geometry is not None:
            raise ValueError("letterbox_geometry requires raw_image_rgb")
        if raw_image is not None:
            expected_geometry = letterbox_geometry(raw_image.shape[0], raw_image.shape[1])
            if geometry is None:
                geometry = expected_geometry
            elif geometry != expected_geometry:
                raise ValueError(
                    f"letterbox_geometry does not match raw_image_rgb: expected {expected_geometry}, got {geometry}"
                )
        object.__setattr__(self, "model_image_rgb", image)
        object.__setattr__(self, "token_values", values)
        object.__setattr__(self, "raw_image_rgb", raw_image)
        object.__setattr__(self, "letterbox_geometry", geometry)
        object.__setattr__(self, "metadata", _validate_metadata(self.metadata))


def _validate_metadata(
    value: Mapping[str, str | int | float | bool | None],
) -> dict[str, str | int | float | bool | None]:
    result = dict(value)
    for key, item in result.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metadata keys must be non-empty strings")
        if not isinstance(item, str | int | float | bool | type(None)):
            raise TypeError(f"metadata value {key!r} is not a JSON scalar")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"metadata value {key!r} must be finite")
    return result


def _validate_model_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.shape != (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE, 3):
        raise ValueError(f"model_image_rgb must be the exact 224x224 top-camera model input; got shape {array.shape}")
    if array.dtype != np.uint8:
        raise TypeError(f"model_image_rgb must be uint8 RGB, got {array.dtype}")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _validate_raw_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"raw_image_rgb must have shape (H, W, 3), got {array.shape}")
    if array.dtype != np.uint8:
        raise TypeError(f"raw_image_rgb must be uint8 RGB, got {array.dtype}")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _validate_token_values(values: np.ndarray | Sequence[float]) -> np.ndarray:
    array = np.asarray(values)
    if array.shape not in ((TOKEN_COUNT,), (TOKEN_GRID_SIZE, TOKEN_GRID_SIZE)):
        raise ValueError(f"token values must have shape (256,) or (16, 16), got {array.shape}")
    if array.dtype.kind not in "fiu":
        raise TypeError(f"token values must be numeric, got {array.dtype}")
    array = np.asarray(array, dtype=np.float64).reshape(TOKEN_COUNT)
    if not np.all(np.isfinite(array)):
        raise ValueError("token values contain NaN or infinity")
    if np.any(array < 0):
        raise ValueError("associative errors and gradient norms must be non-negative")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def token_grid(values: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return the row-major 16x16 grid used by SigLIP's flattened patch tokens."""

    return _validate_token_values(values).reshape(TOKEN_GRID_SIZE, TOKEN_GRID_SIZE)


def letterbox_geometry(source_height: int, source_width: int) -> LetterboxGeometry:
    """Reproduce the integer geometry in ``resize_with_pad`` without touching pixels."""

    if source_height <= 0 or source_width <= 0:
        raise ValueError("source dimensions must be positive")
    ratio = max(source_width / MODEL_IMAGE_SIZE, source_height / MODEL_IMAGE_SIZE)
    resized_height = int(source_height / ratio)
    resized_width = int(source_width / ratio)
    pad_top, remainder_height = divmod(MODEL_IMAGE_SIZE - resized_height, 2)
    pad_left, remainder_width = divmod(MODEL_IMAGE_SIZE - resized_width, 2)
    return LetterboxGeometry(
        source_height=source_height,
        source_width=source_width,
        resized_height=resized_height,
        resized_width=resized_width,
        pad_top=pad_top,
        pad_bottom=pad_top + remainder_height,
        pad_left=pad_left,
        pad_right=pad_left + remainder_width,
    )


def raw_top_camera_to_model_rgb(raw_rgb: np.ndarray) -> tuple[np.ndarray, LetterboxGeometry]:
    """Apply the model's exact inference-time resize-with-pad to a raw uint8 RGB frame.

    This deliberately imports and calls the shared OpenPI implementation instead of
    approximating its interpolation in OpenCV.  Inference does not crop, rotate, or
    flip the image; those operations exist only in randomized training augmentation.
    """

    image = np.asarray(raw_rgb)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"raw top-camera image must have shape (H, W, 3), got {image.shape}")
    if image.dtype != np.uint8:
        raise TypeError(f"raw top-camera image must be uint8 RGB, got {image.dtype}")
    geometry = letterbox_geometry(image.shape[0], image.shape[1])
    from openpi.shared import image_tools  # Imported lazily so pure rendering stays CPU-light.

    resized = np.asarray(image_tools.resize_with_pad(image, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE))
    return _validate_model_image(resized), geometry


def normalized_model_top_camera_to_rgb(model_image: np.ndarray) -> np.ndarray:
    """Convert an already-preprocessed ``[-1, 1]`` model tensor to display RGB.

    A singleton batch axis is accepted.  No resizing is performed, which prevents a
    caller from accidentally changing the patch-to-pixel alignment after inference.
    """

    image = np.asarray(model_image)
    if image.shape == (1, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE, 3):
        image = image[0]
    if image.shape != (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE, 3):
        raise ValueError(f"normalized model image must have shape (224, 224, 3), got {image.shape}")
    if image.dtype.kind != "f" or not np.all(np.isfinite(image)):
        raise TypeError("normalized model image must contain finite floating-point values")
    tolerance = 1e-5
    if np.min(image) < -1.0 - tolerance or np.max(image) > 1.0 + tolerance:
        raise ValueError("normalized model image must lie in [-1, 1]")
    return np.rint((np.clip(image, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)


def _content_coverage_grid(geometry: LetterboxGeometry) -> np.ndarray:
    content_mask = np.zeros((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), dtype=np.float32)
    content_mask[
        geometry.pad_top : MODEL_IMAGE_SIZE - geometry.pad_bottom,
        geometry.pad_left : MODEL_IMAGE_SIZE - geometry.pad_right,
    ] = 1
    return content_mask.reshape(TOKEN_GRID_SIZE, PATCH_SIZE, TOKEN_GRID_SIZE, PATCH_SIZE).mean(axis=(1, 3))


def fit_color_scale(frames: Sequence[TokenMetricFrame], spec: NormalizationSpec | None = None) -> ColorScale:
    """Fit a robust, video-wide linear scale to all 256-token measurements."""

    if not frames:
        raise ValueError("at least one token-metric frame is required")
    if spec is None:
        spec = NormalizationSpec()
    if spec.exclude_letterbox_padding:
        geometries = {frame.letterbox_geometry for frame in frames}
        if geometries == {None}:
            raise ValueError(
                "exclude_letterbox_padding needs raw_image_rgb/letterbox_geometry to know which tokens are padding"
            )
        if len(geometries) != 1:
            raise ValueError("every frame in one video must share a letterbox geometry")
        content = _content_coverage_grid(next(iter(geometries))).reshape(TOKEN_COUNT) > 0
        if not np.any(content):
            raise ValueError("letterbox geometry leaves no content tokens to fit a scale on")
        values = np.concatenate([frame.token_values[content] for frame in frames])
    else:
        values = np.concatenate([frame.token_values for frame in frames])
    vmin = 0.0 if spec.anchor_zero else float(np.percentile(values, spec.lower_percentile))
    vmax = float(np.percentile(values, spec.upper_percentile))
    if vmax <= vmin:
        observed_max = float(np.max(values))
        if observed_max > vmin:
            vmax = observed_max
        elif observed_max > 0:
            # Constant positive values: preserve a meaningful zero-to-value scale.
            vmin, vmax = 0.0, observed_max
        else:
            # An all-zero metric should render uniformly at the bottom of a valid scale.
            vmin, vmax = 0.0, 1.0
    return ColorScale(
        vmin, vmax, spec.lower_percentile, spec.upper_percentile, spec.anchor_zero, spec.exclude_letterbox_padding
    )


def normalize_token_values(values: np.ndarray | Sequence[float], scale: ColorScale) -> np.ndarray:
    grid = token_grid(values)
    return np.clip((grid - scale.vmin) / (scale.vmax - scale.vmin), 0.0, 1.0).astype(np.float32)


def zscore_token_values(values: np.ndarray | Sequence[float], z_range: float = 3.0) -> np.ndarray:
    """Standardize one frame by its own token mean/std and map [-z_range, z_range] to [0, 1].

    This is the per-frame complement of the video-wide absolute scale: associative errors decay
    by orders of magnitude as the fast weights fit the scene, so late frames are unreadable on
    one fixed scale. Standardizing each frame shows *relative* within-frame structure at every
    point of the episode. It deliberately erases absolute magnitude, so renders must keep the
    frame's absolute mean/std visible: a near-uniform frame amplified this way shows noise, and
    only the printed statistics distinguish that from a genuinely concentrated write.
    A constant frame maps everywhere to the 0.5 midpoint.
    """

    if not math.isfinite(z_range) or z_range <= 0:
        raise ValueError("z_range must be finite and positive")
    array = _validate_token_values(values)
    std = float(np.std(array))
    z = np.zeros(TOKEN_COUNT, dtype=np.float64) if std == 0 else (array - float(np.mean(array))) / std
    grid = z.reshape(TOKEN_GRID_SIZE, TOKEN_GRID_SIZE)
    return np.clip((grid + z_range) / (2.0 * z_range), 0.0, 1.0).astype(np.float32)


@dataclasses.dataclass(frozen=True)
class TokenHistory:
    """Per-token baselines fitted across every write in one episode.

    ``per_frame_zscore`` answers "which patch is unusual *within this frame*", which is
    dominated by fixed scene layout: image edges and the robot base sit above the frame mean in
    nearly every frame, so the same border lights up throughout and genuine events are hard to
    separate from that constant backdrop.

    This baseline answers the complementary question: "is this patch unusual *for itself*, right
    now?" Two corrections make that comparison meaningful:

    1. **Log space.** Associative errors decay by orders of magnitude as the fast weights fit
       the scene, so ratios rather than differences are the natural unit.
    2. **Robust frame detrending.** Each frame's own log-*median* is removed before fitting, so
       the global decay does not swamp the per-token signal. What remains is each token's
       deviation from the frame's overall level -- its *share* of the write, independent of how
       large the write was. The median rather than the mean is essential: a handful of spiking
       tokens would drag a mean baseline, making every other token look anomalously low in
       exactly the frames that contain an event.

    A per-token standard-deviation floor keeps near-constant tokens neutral. Without it, a token
    that never varies would divide by ~0 and render as a spurious extreme.

    A token therefore turns red when it takes a larger share of this write than it usually does,
    which is what isolates events (an object appearing, an arm entering) from static layout.
    """

    log_mean: np.ndarray
    log_std: np.ndarray
    frame_count: int
    # Tokens quieter than this are treated as constant and render neutral instead of exploding.
    log_std_floor: float = 1e-3

    def __post_init__(self) -> None:
        for name, array in (("log_mean", self.log_mean), ("log_std", self.log_std)):
            values = np.asarray(array, dtype=np.float64)
            if values.shape != (TOKEN_COUNT,):
                raise ValueError(f"{name} must have shape (256,), got {values.shape}")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values")
            object.__setattr__(self, name, values)
        if np.any(self.log_std < 0):
            raise ValueError("log_std must be non-negative")
        if self.frame_count < 1:
            raise ValueError("frame_count must be positive")
        if not math.isfinite(self.log_std_floor) or self.log_std_floor <= 0:
            raise ValueError("log_std_floor must be finite and positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_count": int(self.frame_count),
            "log_std_median": float(np.median(self.log_std)),
            "log_std_min": float(np.min(self.log_std)),
            "log_std_max": float(np.max(self.log_std)),
            "log_std_floor": float(self.log_std_floor),
            "tokens_below_floor": int(np.count_nonzero(self.log_std < self.log_std_floor)),
            "detrend": "per-frame log-median (robust to spiking tokens)",
        }


def fit_token_history(frames: Sequence[TokenMetricFrame]) -> TokenHistory:
    """Fit each token's frame-detrended log baseline over a whole episode."""

    if not frames:
        raise ValueError("at least one token-metric frame is required")
    stacked = np.stack([np.asarray(frame.token_values, dtype=np.float64) for frame in frames])
    logs = np.log(np.maximum(stacked, _LOG_FLOOR))
    residual = logs - np.median(logs, axis=1, keepdims=True)
    return TokenHistory(
        log_mean=residual.mean(axis=0),
        log_std=residual.std(axis=0),
        frame_count=len(frames),
    )


def history_token_values(
    values: np.ndarray | Sequence[float],
    history: TokenHistory,
    frame_values_for_detrend: np.ndarray | Sequence[float] | None = None,
    z_range: float = 3.0,
) -> np.ndarray:
    """Score one frame against per-token baselines and map [-z_range, z_range] to [0, 1].

    Tokens whose history is degenerate (zero variance) map to the neutral midpoint rather than
    exploding, so a constant patch never renders as a false hotspot.
    """

    if not math.isfinite(z_range) or z_range <= 0:
        raise ValueError("z_range must be finite and positive")
    array = _validate_token_values(values)
    detrend_source = array if frame_values_for_detrend is None else _validate_token_values(frame_values_for_detrend)
    logs = np.log(np.maximum(array, _LOG_FLOOR))
    residual = logs - float(np.median(np.log(np.maximum(detrend_source, _LOG_FLOOR))))
    usable = history.log_std >= history.log_std_floor
    z = np.where(usable, (residual - history.log_mean) / np.maximum(history.log_std, history.log_std_floor), 0.0)
    grid = z.reshape(TOKEN_GRID_SIZE, TOKEN_GRID_SIZE)
    return np.clip((grid + z_range) / (2.0 * z_range), 0.0, 1.0).astype(np.float32)


def history_token_z(values: np.ndarray | Sequence[float], history: TokenHistory) -> np.ndarray:
    """Return raw (unclipped, uncolored) per-token history z-scores for analysis."""

    array = _validate_token_values(values)
    logs = np.log(np.maximum(array, _LOG_FLOOR))
    residual = logs - float(np.median(logs))
    usable = history.log_std >= history.log_std_floor
    return np.where(usable, (residual - history.log_mean) / np.maximum(history.log_std, history.log_std_floor), 0.0)


def _display_grid(
    values: np.ndarray | Sequence[float],
    scale: ColorScale | None,
    scale_mode: ScaleMode,
    zscore_range: float,
    history: TokenHistory | None = None,
) -> np.ndarray:
    if scale_mode == "video":
        if scale is None:
            raise ValueError("scale_mode='video' requires a fitted ColorScale")
        return normalize_token_values(values, scale)
    if scale_mode == "per_frame_zscore":
        return zscore_token_values(values, zscore_range)
    if scale_mode == "per_token_history":
        if history is None:
            raise ValueError("scale_mode='per_token_history' requires a fitted TokenHistory")
        return history_token_values(values, history, z_range=zscore_range)
    raise ValueError(f"unsupported scale_mode {scale_mode!r}; choose one of {SCALE_MODES}")


def metric_statistics(values: np.ndarray | Sequence[float]) -> dict[str, float]:
    """Spatial concentration statistics that make an all-token write easy to spot."""

    array = _validate_token_values(values)
    total = float(np.sum(array))
    mean = float(np.mean(array))
    std = float(np.std(array))
    if total > 0:
        probability = array / total
        positive = probability > 0
        entropy = -float(np.sum(probability[positive] * np.log(probability[positive])))
        normalized_entropy = entropy / math.log(TOKEN_COUNT)
        effective_tokens = math.exp(entropy)
        top_count = max(1, math.ceil(0.1 * TOKEN_COUNT))
        top_10pct_mass = float(np.sum(np.partition(array, -top_count)[-top_count:]) / total)
    else:
        normalized_entropy = 0.0
        effective_tokens = 0.0
        top_10pct_mass = 0.0
    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": mean,
        "median": float(np.median(array)),
        "std": std,
        "coefficient_of_variation": std / mean if mean > 0 else 0.0,
        "normalized_entropy": normalized_entropy,
        "effective_token_count": effective_tokens,
        "top_10pct_mass_fraction": top_10pct_mass,
    }


_COLORMAPS = {
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "turbo": cv2.COLORMAP_TURBO,
    "viridis": cv2.COLORMAP_VIRIDIS,
}


def _coolwarm_lut() -> np.ndarray:
    # Diverging blue -> light gray -> red for signed z-score views; 0.5 is the neutral midpoint.
    anchors = np.array([[59, 76, 192], [221, 221, 221], [180, 4, 38]], dtype=np.float64)
    positions = np.array([0.0, 0.5, 1.0])
    ramp = np.linspace(0.0, 1.0, 256)
    return (
        np.stack([np.interp(ramp, positions, anchors[:, channel]) for channel in range(3)], axis=-1)
        .round()
        .astype(np.uint8)
    )


_DIVERGING_COLORMAPS = {"coolwarm": _coolwarm_lut}
SUPPORTED_COLORMAPS = tuple(sorted((*_COLORMAPS, *_DIVERGING_COLORMAPS)))


def _colorize(normalized: np.ndarray, colormap: str) -> np.ndarray:
    gray = np.rint(np.clip(normalized, 0.0, 1.0) * 255).astype(np.uint8)
    if colormap in _DIVERGING_COLORMAPS:
        return _DIVERGING_COLORMAPS[colormap]()[gray]
    if colormap not in _COLORMAPS:
        raise ValueError(f"unsupported colormap {colormap!r}; choose one of {SUPPORTED_COLORMAPS}")
    bgr = cv2.applyColorMap(gray, _COLORMAPS[colormap])
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def heatmap_overlay(
    model_image_rgb: np.ndarray,
    values: np.ndarray | Sequence[float],
    scale: ColorScale | None,
    *,
    alpha: float = 0.58,
    colormap: str = "inferno",
    draw_grid: bool = True,
    scale_mode: ScaleMode = "video",
    zscore_range: float = 3.0,
    history: TokenHistory | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(overlay_rgb, normalized_16x16_grid)`` at 224x224 resolution."""

    image = _validate_model_image(model_image_rgb)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must lie in [0, 1]")
    normalized = _display_grid(values, scale, scale_mode, zscore_range, history)
    heat_grid = _colorize(normalized, colormap)
    heat = cv2.resize(heat_grid, (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    overlay = np.rint((1.0 - alpha) * image.astype(np.float32) + alpha * heat.astype(np.float32)).astype(np.uint8)
    if draw_grid:
        # Patch boundaries are exact multiples of 14 pixels in the 224x224 model image.
        for coordinate in range(PATCH_SIZE, MODEL_IMAGE_SIZE, PATCH_SIZE):
            cv2.line(overlay, (coordinate, 0), (coordinate, MODEL_IMAGE_SIZE - 1), (235, 235, 235), 1)
            cv2.line(overlay, (0, coordinate), (MODEL_IMAGE_SIZE - 1, coordinate), (235, 235, 235), 1)
    return overlay, normalized


def project_normalized_grid_to_raw(normalized_grid: np.ndarray, geometry: LetterboxGeometry) -> np.ndarray:
    """Invert letterboxing and project a model-aligned 16x16 map onto raw-camera pixels.

    Values covering only black padding are cropped, not smeared into the camera
    image. Partial edge patches retain their content portion. Nearest-neighbor
    interpolation preserves discrete patch attribution and orientation.
    """

    normalized = np.asarray(normalized_grid)
    if normalized.shape != (TOKEN_GRID_SIZE, TOKEN_GRID_SIZE):
        raise ValueError(f"normalized token grid must have shape (16, 16), got {normalized.shape}")
    if normalized.dtype.kind not in "fiu" or not np.all(np.isfinite(normalized)):
        raise ValueError("normalized token grid must contain finite numeric values")
    if np.min(normalized) < 0 or np.max(normalized) > 1:
        raise ValueError("normalized token grid must lie in [0, 1]")

    model_map = cv2.resize(
        normalized.astype(np.float32),
        (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )
    row_end = MODEL_IMAGE_SIZE - geometry.pad_bottom
    column_end = MODEL_IMAGE_SIZE - geometry.pad_right
    content = model_map[geometry.pad_top : row_end, geometry.pad_left : column_end]
    if content.shape != (geometry.resized_height, geometry.resized_width):
        raise ValueError(
            "letterbox content shape is inconsistent with geometry: "
            f"expected {(geometry.resized_height, geometry.resized_width)}, got {content.shape}"
        )
    return cv2.resize(
        content,
        (geometry.source_width, geometry.source_height),
        interpolation=cv2.INTER_NEAREST,
    )


def _draw_raw_patch_grid(image: np.ndarray, geometry: LetterboxGeometry) -> None:
    row_end = MODEL_IMAGE_SIZE - geometry.pad_bottom
    column_end = MODEL_IMAGE_SIZE - geometry.pad_right
    for model_x in range(PATCH_SIZE, MODEL_IMAGE_SIZE, PATCH_SIZE):
        if geometry.pad_left < model_x < column_end:
            raw_x = round((model_x - geometry.pad_left) * geometry.source_width / geometry.resized_width)
            cv2.line(image, (raw_x, 0), (raw_x, geometry.source_height - 1), (235, 235, 235), 1)
    for model_y in range(PATCH_SIZE, MODEL_IMAGE_SIZE, PATCH_SIZE):
        if geometry.pad_top < model_y < row_end:
            raw_y = round((model_y - geometry.pad_top) * geometry.source_height / geometry.resized_height)
            cv2.line(image, (0, raw_y), (geometry.source_width - 1, raw_y), (235, 235, 235), 1)


def raw_heatmap_overlay(
    raw_image_rgb: np.ndarray,
    values: np.ndarray | Sequence[float],
    scale: ColorScale | None,
    geometry: LetterboxGeometry,
    *,
    alpha: float = 0.58,
    colormap: str = "inferno",
    draw_grid: bool = True,
    scale_mode: ScaleMode = "video",
    zscore_range: float = 3.0,
    history: TokenHistory | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(raw_overlay, model_grid, projected_raw_map)`` in raw-camera space."""

    image = _validate_raw_image(raw_image_rgb)
    if image.shape[:2] != (geometry.source_height, geometry.source_width):
        raise ValueError("raw image dimensions do not match letterbox geometry")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must lie in [0, 1]")
    normalized = _display_grid(values, scale, scale_mode, zscore_range, history)
    projected = project_normalized_grid_to_raw(normalized, geometry)
    heat = _colorize(projected, colormap)
    overlay = np.rint((1.0 - alpha) * image.astype(np.float32) + alpha * heat.astype(np.float32)).astype(np.uint8)
    if draw_grid:
        _draw_raw_patch_grid(overlay, geometry)
    return overlay, normalized, projected


def raw_projection_statistics(values: np.ndarray | Sequence[float], geometry: LetterboxGeometry) -> dict[str, float]:
    """Quantify how much attribution lies on visible pixels versus letterbox padding."""

    grid = token_grid(values)
    total = float(np.sum(grid))
    visible_fraction = 0.0 if total == 0 else float(np.sum(grid * _content_coverage_grid(geometry)) / total)
    return {
        "visible_content_mass_fraction": visible_fraction,
        "letterbox_padding_mass_fraction": 1.0 - visible_fraction if total > 0 else 0.0,
    }


def render_annotated_frame(
    frame: TokenMetricFrame,
    scale: ColorScale | None,
    *,
    metric_name: str,
    alpha: float = 0.58,
    colormap: str = "inferno",
    display_scale: int = 2,
    scale_mode: ScaleMode = "video",
    zscore_range: float = 3.0,
    history: TokenHistory | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render original and aligned overlay side-by-side with a fixed color bar."""

    if not metric_name:
        raise ValueError("metric_name must be non-empty")
    if display_scale < 1:
        raise ValueError("display_scale must be positive")
    target_long_side = MODEL_IMAGE_SIZE * display_scale
    if frame.raw_image_rgb is not None:
        assert frame.letterbox_geometry is not None
        overlay, normalized, _ = raw_heatmap_overlay(
            frame.raw_image_rgb,
            frame.token_values,
            scale,
            frame.letterbox_geometry,
            alpha=alpha,
            colormap=colormap,
            scale_mode=scale_mode,
            zscore_range=zscore_range,
            history=history,
        )
        original = frame.raw_image_rgb
        raw_height, raw_width = original.shape[:2]
        display_ratio = target_long_side / max(raw_height, raw_width)
        panel_width = max(1, round(raw_width * display_ratio))
        panel_height = max(1, round(raw_height * display_ratio))
        original_label = f"original raw top camera ({raw_width}x{raw_height})"
        footer_height = 94
        original_interpolation = cv2.INTER_AREA if display_ratio < 1 else cv2.INTER_NEAREST
    else:
        overlay, normalized = heatmap_overlay(
            frame.model_image_rgb,
            frame.token_values,
            scale,
            alpha=alpha,
            colormap=colormap,
            scale_mode=scale_mode,
            zscore_range=zscore_range,
            history=history,
        )
        original = frame.model_image_rgb
        panel_width = panel_height = target_long_side
        original_label = "top camera: exact 224x224 model input"
        footer_height = 70
        original_interpolation = cv2.INTER_NEAREST
    original_panel = cv2.resize(original, (panel_width, panel_height), interpolation=original_interpolation)
    overlay_panel = cv2.resize(overlay, (panel_width, panel_height), interpolation=cv2.INTER_NEAREST)

    header_height = 54
    gap = 16
    colorbar_width = 36
    colorbar_labels_width = 130
    canvas_width = panel_width * 2 + gap + colorbar_width + colorbar_labels_width
    canvas_height = header_height + panel_height + footer_height
    # libx264/yuv420p requires even dimensions.
    canvas_width += canvas_width % 2
    canvas_height += canvas_height % 2
    canvas = np.full((canvas_height, canvas_width, 3), 20, dtype=np.uint8)
    canvas[header_height : header_height + panel_height, :panel_width] = original_panel
    overlay_x = panel_width + gap
    canvas[header_height : header_height + panel_height, overlay_x : overlay_x + panel_width] = overlay_panel

    gradient = np.linspace(1.0, 0.0, panel_height, dtype=np.float32)[:, None]
    colorbar = _colorize(np.broadcast_to(gradient, (panel_height, colorbar_width)), colormap)
    bar_x = overlay_x + panel_width
    canvas[header_height : header_height + panel_height, bar_x : bar_x + colorbar_width] = colorbar

    white = (245, 245, 245)
    muted = (185, 185, 185)
    cv2.putText(canvas, original_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.56, white, 1)
    title = f"{metric_name} | raw {frame.raw_frame} | step {frame.policy_step}"
    if frame.write_count is not None:
        title += f" | write {frame.write_count}"
    cv2.putText(canvas, title, (overlay_x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.56, white, 1)
    if frame.phase:
        cv2.putText(canvas, f"phase: {frame.phase}", (overlay_x, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.48, muted, 1)

    if scale_mode == "video":
        assert scale is not None
        bar_top_label = f"{scale.vmax:.4g}"
        bar_mid_label = f"{(scale.vmin + scale.vmax) / 2:.4g}"
        bar_bottom_label = f"{scale.vmin:.4g}"
        scale_note = f"fixed video scale q{scale.upper_percentile:g}"
        if scale.excluded_letterbox_padding:
            scale_note += " (fitted on camera content only; padding tokens colour-clip)"
    elif scale_mode == "per_token_history":
        bar_top_label = f"+{zscore_range:g} sd"
        bar_mid_label = "own typical"
        bar_bottom_label = f"-{zscore_range:g} sd"
        scale_note = f"vs each token's own episode baseline, +/-{zscore_range:g} sd (share of write, log-detrended)"
    else:
        bar_top_label = f"+{zscore_range:g} sd"
        bar_mid_label = "frame mean"
        bar_bottom_label = f"-{zscore_range:g} sd"
        scale_note = f"per-frame z-score +/-{zscore_range:g} sd (relative view; magnitude in footer)"
    label_x = bar_x + colorbar_width + 8
    cv2.putText(canvas, bar_top_label, (label_x, header_height + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.43, white, 1)
    cv2.putText(
        canvas,
        bar_mid_label,
        (label_x, header_height + panel_height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        white,
        1,
    )
    cv2.putText(
        canvas,
        bar_bottom_label,
        (label_x, header_height + panel_height - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        white,
        1,
    )

    stats = metric_statistics(frame.token_values)
    footer_y = header_height + panel_height + 24
    line1 = (
        f"mean {stats['mean']:.4g}  std {stats['std']:.4g}  CV {stats['coefficient_of_variation']:.3f}  "
        f"effective tokens {stats['effective_token_count']:.1f}/256"
    )
    line2 = (
        f"top 10% mass {stats['top_10pct_mass_fraction']:.3f}  normalized entropy "
        f"{stats['normalized_entropy']:.3f}  {scale_note}"
    )
    cv2.putText(canvas, line1, (10, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, white, 1)
    cv2.putText(canvas, line2, (10, footer_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, muted, 1)
    if frame.letterbox_geometry is not None:
        projection_stats = raw_projection_statistics(frame.token_values, frame.letterbox_geometry)
        line3 = (
            f"raw projection: visible-content mass {projection_stats['visible_content_mass_fraction']:.3f}  "
            f"letterbox-padding mass {projection_stats['letterbox_padding_mass_fraction']:.3f}; "
            "padding values remain in NPZ"
        )
        cv2.putText(canvas, line3, (10, footer_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, muted, 1)
    return canvas, normalized


def _validate_video_frames(frames: Sequence[np.ndarray]) -> tuple[int, int]:
    if not frames:
        raise ValueError("cannot encode an empty video")
    first = np.asarray(frames[0])
    if first.ndim != 3 or first.shape[-1] != 3 or first.dtype != np.uint8:
        raise ValueError("video frames must be uint8 RGB arrays")
    height, width = first.shape[:2]
    if height % 2 or width % 2:
        raise ValueError("MP4 RGB frame width and height must be even for yuv420p")
    for index, frame in enumerate(frames[1:], start=1):
        array = np.asarray(frame)
        if array.shape != first.shape or array.dtype != np.uint8:
            raise ValueError(f"video frame {index} does not match first frame shape/dtype")
    return height, width


def _encode_imageio(frames: Sequence[np.ndarray], path: Path, fps: float) -> str:
    import imageio.v2 as imageio

    writer = imageio.get_writer(
        path,
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None,
        ffmpeg_log_level="error",
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
    return "imageio-ffmpeg:libx264"


def _reencode_h264_in_place(path: Path) -> None:
    """Rewrite an MP4 as H.264/yuv420p with the bundled ffmpeg binary.

    Runs one file-to-file subprocess (fork+exec, no streaming pipes), which is safe
    after JAX initialization, unlike imageio-ffmpeg's long-lived writer process.
    """

    import subprocess

    import imageio_ffmpeg

    reencoded = path.with_name(f".{path.stem}.h264{path.suffix}")
    reencoded.unlink(missing_ok=True)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(reencoded),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        if not reencoded.is_file() or reencoded.stat().st_size == 0:
            raise RuntimeError("ffmpeg returned success but produced no output")
    except Exception:
        reencoded.unlink(missing_ok=True)
        raise
    os.replace(reencoded, path)


def _encode_opencv(frames: Sequence[np.ndarray], path: Path, fps: float) -> str:
    # Every delivered MP4 must be H.264: mp4v (MPEG-4 Part 2) does not play in browsers or
    # editor previews. When OpenCV lacks the avc1 codec, mp4v is only an intermediate that is
    # immediately re-encoded; failing that, encoding fails rather than delivering mp4v.
    height, width = frames[0].shape[:2]
    errors = []
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if not writer.isOpened():
            writer.release()
            errors.append(f"{codec}: unavailable")
            continue
        try:
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        if path.is_file() and path.stat().st_size > 0:
            if codec == "avc1":
                return "opencv:avc1"
            try:
                _reencode_h264_in_place(path)
            except Exception as exc:
                path.unlink(missing_ok=True)
                errors.append(f"{codec}: wrote non-H.264 video and ffmpeg re-encode failed ({exc})")
                break
            return "opencv:mp4v+ffmpeg:libx264"
        errors.append(f"{codec}: produced no output")
    raise RuntimeError("OpenCV could not encode an H.264 MP4 (" + "; ".join(errors) + ")")


def encode_mp4(
    frames: Sequence[np.ndarray],
    path: str | Path,
    fps: float,
    *,
    encoder: Literal["auto", "imageio", "opencv"] = "auto",
) -> str:
    """Encode RGB frames, preferring H.264 and falling back to OpenCV codecs.

    The file is first written beside the destination and atomically renamed only
    after a non-empty video exists.  The returned string identifies the encoder.
    """

    _validate_video_frames(frames)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if encoder not in ("auto", "imageio", "opencv"):
        raise ValueError("encoder must be one of 'auto', 'imageio', or 'opencv'")
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing video: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.encoding{destination.suffix}")
    temporary.unlink(missing_ok=True)
    candidates = {
        "auto": (_encode_imageio, _encode_opencv),
        "imageio": (_encode_imageio,),
        "opencv": (_encode_opencv,),
    }[encoder]
    failures = []
    for encode in candidates:
        try:
            name = encode(frames, temporary, fps)
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("encoder returned without creating a non-empty file")
            os.replace(temporary, destination)
            return name
        except Exception as exc:
            failures.append(f"{encode.__name__}: {exc}")
            temporary.unlink(missing_ok=True)
    raise RuntimeError("all MP4 encoders failed: " + " | ".join(failures))


def _write_png(path: Path, rgb: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"OpenCV failed to write PNG {path}")


def _safe_metric_stem(metric_name: str) -> str:
    if not metric_name:
        raise ValueError("metric_name must be non-empty")
    stem = "".join(character if character.isalnum() or character in "-_" else "_" for character in metric_name)
    stem = stem.strip("_-")
    if not stem or stem in (".", ".."):
        raise ValueError(f"cannot form a safe filename from metric_name {metric_name!r}")
    return stem


def export_heatmap_video(
    frames: Sequence[TokenMetricFrame],
    output_dir: str | Path,
    *,
    metric_name: str,
    fps: float,
    normalization: NormalizationSpec | None = None,
    alpha: float = 0.58,
    colormap: Literal["inferno", "magma", "turbo", "viridis", "coolwarm"] = "inferno",
    display_scale: int = 2,
    video_encoder: Literal["auto", "imageio", "opencv"] = "auto",
    scale_mode: ScaleMode = "video",
    zscore_range: float = 3.0,
) -> dict[str, object]:
    """Write PNG/NPZ for every measured write plus one annotated MP4 and manifest."""

    if not frames:
        raise ValueError("at least one token-metric frame is required")
    metric_stem = _safe_metric_stem(metric_name)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if display_scale < 1:
        raise ValueError("display_scale must be positive")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must lie in [0, 1]")
    if colormap not in _COLORMAPS and colormap not in _DIVERGING_COLORMAPS:
        raise ValueError(f"unsupported colormap {colormap!r}; choose one of {SUPPORTED_COLORMAPS}")
    if scale_mode not in SCALE_MODES:
        raise ValueError(f"unsupported scale_mode {scale_mode!r}; choose one of {SCALE_MODES}")
    if not math.isfinite(zscore_range) or zscore_range <= 0:
        raise ValueError("zscore_range must be finite and positive")
    raw_presence = {frame.raw_image_rgb is not None for frame in frames}
    if len(raw_presence) != 1:
        raise ValueError("every frame in one video must consistently provide raw_image_rgb or consistently omit it")
    render_spaces = {
        frame.raw_image_rgb.shape[:2] if frame.raw_image_rgb is not None else (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)
        for frame in frames
    }
    if len(render_spaces) != 1:
        raise ValueError(f"every frame in one video must use the same raw/model image dimensions, got {render_spaces}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    png_dir = output / "frames"
    data_dir = output / "data"
    png_dir.mkdir()
    data_dir.mkdir()

    scale = fit_color_scale(frames, normalization) if scale_mode == "video" else None
    history = fit_token_history(frames) if scale_mode == "per_token_history" else None
    rendered = []
    records = []
    for index, frame in enumerate(frames):
        canvas, normalized = render_annotated_frame(
            frame,
            scale,
            metric_name=metric_name,
            alpha=alpha,
            colormap=colormap,
            display_scale=display_scale,
            scale_mode=scale_mode,
            zscore_range=zscore_range,
            history=history,
        )
        stem = f"{index:06d}_raw_{frame.raw_frame:06d}_step_{frame.policy_step:06d}"
        png_relative = Path("frames") / f"{stem}.png"
        data_relative = Path("data") / f"{stem}.npz"
        _write_png(output / png_relative, canvas)
        arrays = {
            "model_image_rgb": frame.model_image_rgb,
            "token_values": frame.token_values,
            "token_grid": frame.token_values.reshape(TOKEN_GRID_SIZE, TOKEN_GRID_SIZE),
            "normalized_grid": normalized,
            # Unclipped scores let downstream analysis rank hotspots past the +/-z_range cutoff.
            **(
                {"history_z": history_token_z(frame.token_values, history).reshape(TOKEN_GRID_SIZE, TOKEN_GRID_SIZE)}
                if history is not None
                else {}
            ),
            "raw_frame": np.asarray(frame.raw_frame, dtype=np.int64),
            "policy_step": np.asarray(frame.policy_step, dtype=np.int64),
        }
        geometry_dict = None
        statistics = metric_statistics(frame.token_values)
        if frame.raw_image_rgb is not None:
            assert frame.letterbox_geometry is not None
            projected = project_normalized_grid_to_raw(normalized, frame.letterbox_geometry)
            arrays.update(
                {
                    "raw_image_rgb": frame.raw_image_rgb,
                    "projected_normalized_raw_map": projected,
                    "source_shape": np.asarray(frame.raw_image_rgb.shape[:2], dtype=np.int64),
                    "resized_shape": np.asarray(
                        [frame.letterbox_geometry.resized_height, frame.letterbox_geometry.resized_width],
                        dtype=np.int64,
                    ),
                    "letterbox_padding": np.asarray(
                        [
                            frame.letterbox_geometry.pad_top,
                            frame.letterbox_geometry.pad_bottom,
                            frame.letterbox_geometry.pad_left,
                            frame.letterbox_geometry.pad_right,
                        ],
                        dtype=np.int64,
                    ),
                }
            )
            geometry_dict = frame.letterbox_geometry.to_dict()
            statistics.update(raw_projection_statistics(frame.token_values, frame.letterbox_geometry))
        np.savez_compressed(output / data_relative, **arrays)
        rendered.append(canvas)
        records.append(
            {
                "index": index,
                "raw_frame": frame.raw_frame,
                "policy_step": frame.policy_step,
                "write_count": frame.write_count,
                "phase": frame.phase,
                "timestamp_s": frame.timestamp_s,
                "png": png_relative.as_posix(),
                "data": data_relative.as_posix(),
                "rendered_on_raw_image": frame.raw_image_rgb is not None,
                "letterbox_geometry": geometry_dict,
                "statistics": statistics,
                "metadata": dict(frame.metadata),
            }
        )

    video_relative = Path(f"{metric_stem}_heatmap.mp4")
    encoder = encode_mp4(rendered, output / video_relative, fps, encoder=video_encoder)
    manifest: dict[str, object] = {
        "schema_version": "openpi.v31.token_heatmap.v1",
        "metric_name": metric_name,
        "metric_semantics": "non-negative per-token associative error or fast-weight gradient norm",
        "frame_count": len(frames),
        "fps": float(fps),
        "model_image_size": [MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE],
        "token_grid_size": [TOKEN_GRID_SIZE, TOKEN_GRID_SIZE],
        "patch_size_pixels": [PATCH_SIZE, PATCH_SIZE],
        "token_layout": TOKEN_LAYOUT,
        "image_alignment": {
            "model": (
                "256 values form the exact row-major 16x16 grid over the inference-time 224x224 "
                "letterboxed model input; inference applies no crop, rotation, transpose, or flip"
            ),
            "raw_projection": (
                "when raw_image_rgb is present, black letterbox padding is cropped from the 224x224 map and "
                "the remaining content is nearest-neighbor projected back to the original raw-camera geometry; "
                "padding-token values remain in the model-aligned 16x16 NPZ grid"
            ),
            "fallback": "when raw_image_rgb is absent, PNG/video panels use the 224x224 model input",
        },
        "scale_mode": scale_mode,
        "normalization": (
            {
                **scale.to_dict(),
                "scope": (
                    "one fixed scale across the entire video, fitted only on tokens covering real camera "
                    "pixels; letterbox-padding tokens are still recorded but may colour-clip"
                    if scale.excluded_letterbox_padding
                    else "one fixed scale across the entire video"
                ),
                "clipping": "values outside [vmin, vmax] are color-clipped; raw NPZ values are unchanged",
            }
            if scale is not None
            else {
                "scope": (
                    "per-token history: every frame's log values are detrended by that frame's own log-mean, "
                    "then each token is standardized against its own baseline over all "
                    f"{history.frame_count} writes in this episode; colors show whether a token takes a larger "
                    "or smaller SHARE of this write than it usually does, so static scene layout cancels out "
                    "and events stand out. Absolute magnitude is NOT encoded in color"
                    if history is not None
                    else "per-frame z-score: each frame is standardized by its own token mean/std, so colors show "
                    "relative within-frame structure only; absolute magnitude is NOT comparable across frames"
                ),
                "z_range": zscore_range,
                "clipping": (
                    f"z-values outside [-{zscore_range:g}, {zscore_range:g}] are color-clipped; "
                    "raw NPZ values are unchanged"
                ),
                **({"token_history": history.to_dict()} if history is not None else {}),
            }
        ),
        "alpha": alpha,
        "colormap": colormap,
        "encoder": encoder,
        "requested_video_encoder": video_encoder,
        "video": video_relative.as_posix(),
        "records": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
