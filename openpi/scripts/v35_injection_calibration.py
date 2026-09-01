"""Calibrate the frozen v3.5 memory injection from train-only replay statistics.

This utility intentionally does not load a model or a dataset.  Its input is a replay-stat
``.npz`` produced by a separate, hash-pinned model replay.  The required arrays are:

``episode_stable_id`` (N,)
    Unique, non-empty strings.
``episode_split`` (N,)
    Strings.  Every value must be exactly ``"train"``.
``clean_raw_retrieved`` (N, 16, D), float32
    The 16 real decision-query raw-read slots per episode after a single direct evidence
    commit, evaluated before delay decay.  Slots are never averaged before production pinning.
``layer8_residual`` (N, D), float32
    The aligned layer-8 residual used as the injection-scale reference.
``n_delay`` (N,), integer
    Valid sampled transitions between the evidence commit and decision read.
``alpha_step`` scalar
    The fixed per-transition output decay.
``memory_inject_w`` (D,), float32
    Raw gate parameters.  Their FP32 tanh must be 0.5 in every channel.
``noise_raw_retrieved`` (M, D), float32
    Raw real-noise/control reads.
``noise_episode_index`` (M,), integer
    Index of the aligned episode/residual for each noise read.
``noise_kind`` (M,)
    ``low_cos_query``, ``mixed_precision_residual``, or
    ``synthetic_orthogonal_query``.  At least one query control and one mixed-precision
    residual are required.
``noise_query_cosine`` (M,)
    Query-to-stored-key cosine.  Query controls must be finite and <= 0.1.  Values for
    mixed-precision residuals are ignored and may be NaN.
``source_sha256``, ``official_base_source_sha256``, ``dataset_sha256``, ``split_sha256``,
``replay_protocol_sha256``, ``collector_source_sha256``, ``preflight_sha256`` scalars
    Lower-case SHA256 provenance digests supplied by the authenticated replay producer.

The production denominator is deliberately computed across *all* channels before the gate
is applied.  ``c`` is then calibrated on open channels only.  A passing output is a canonical
JSON envelope whose ID is the SHA256 of its canonical ``payload`` object.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import dataclasses
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from numpy import typing as npt

from openpi.shared import project_paths

SCHEMA_VERSION = "openpi.v35.injection-calibration.v1"
TARGET_EFFECTIVE_GATE = np.float32(0.5)
GATE_ATOL = float(2 * np.finfo(np.float32).eps)
OPEN_GATE_THRESHOLD = np.float32(0.1)
CLEAN_AMPLITUDE_RANGE = (0.7, 0.8)
NOISE_P95_RATIO_MAX = 0.10
P90_DELAY_AMPLITUDE_MIN = 0.40
LOW_COSINE_MAX = 0.10
CLEAN_SLOT_COUNT = 16
_EPS = np.float32(1e-12)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_QUERY_NOISE_KINDS = frozenset(("low_cos_query", "synthetic_orthogonal_query"))
_ALLOWED_NOISE_KINDS = _QUERY_NOISE_KINDS | {"mixed_precision_residual"}

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.integer[Any]]


class CalibrationError(ValueError):
    """Raised when replay statistics cannot produce a sealed calibration artifact."""


@dataclasses.dataclass(frozen=True)
class ReplayStats:
    episode_stable_id: tuple[str, ...]
    episode_split: tuple[str, ...]
    clean_raw_retrieved: FloatArray
    layer8_residual: FloatArray
    n_delay: IntArray
    alpha_step: float
    memory_inject_w: FloatArray
    noise_raw_retrieved: FloatArray
    noise_episode_index: IntArray
    noise_kind: tuple[str, ...]
    noise_query_cosine: npt.NDArray[np.floating[Any]]
    source_sha256: str
    official_base_source_sha256: str
    dataset_sha256: str
    split_sha256: str
    replay_protocol_sha256: str
    collector_source_sha256: str
    preflight_sha256: str


@dataclasses.dataclass(frozen=True)
class LoadedReplayStats:
    stats: ReplayStats
    input_sha256: str
    npz_keys: tuple[str, ...]


def canonical_json_bytes(value: Any) -> bytes:
    """Return the single canonical JSON representation used for artifact identities."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(name: str, value: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise CalibrationError(f"{name} must be a lower-case 64-character SHA256 digest")
    return value


def _text_scalar(array: npt.NDArray[Any], name: str) -> str:
    if array.size != 1 or array.dtype.kind not in ("S", "U"):
        raise CalibrationError(f"{name} must be one string scalar")
    value = array.reshape(-1)[0]
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CalibrationError(f"{name} is not valid UTF-8") from exc
    return str(value)


def _text_vector(array: npt.NDArray[Any], name: str) -> tuple[str, ...]:
    if array.ndim != 1 or array.dtype.kind not in ("S", "U"):
        raise CalibrationError(f"{name} must be a one-dimensional string array")
    values = []
    for raw in array:
        if isinstance(raw, bytes):
            try:
                values.append(raw.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise CalibrationError(f"{name} contains invalid UTF-8") from exc
        else:
            values.append(str(raw))
    return tuple(values)


def _float32_matrix(array: npt.NDArray[Any], name: str) -> FloatArray:
    if array.dtype != np.dtype(np.float32):
        raise CalibrationError(f"{name} must be float32; got {array.dtype}")
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise CalibrationError(f"{name} must have non-empty shape (samples, channels)")
    if not np.all(np.isfinite(array)):
        raise CalibrationError(f"{name} contains NaN or infinite values")
    return np.asarray(array, dtype=np.float32)


def _float32_clean_tensor(array: npt.NDArray[Any]) -> FloatArray:
    if array.dtype != np.dtype(np.float32):
        raise CalibrationError(f"clean_raw_retrieved must be float32; got {array.dtype}")
    if array.ndim != 3 or array.shape[1] != CLEAN_SLOT_COUNT or not array.shape[0] or not array.shape[2]:
        raise CalibrationError(
            f"clean_raw_retrieved must have non-empty shape (episodes, {CLEAN_SLOT_COUNT}, channels); got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise CalibrationError("clean_raw_retrieved contains NaN or infinite values")
    return np.asarray(array, dtype=np.float32)


def _integer_vector(array: npt.NDArray[Any], name: str) -> IntArray:
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise CalibrationError(f"{name} must be a one-dimensional integer array")
    return array


def _scalar_float(array: npt.NDArray[Any], name: str) -> float:
    if array.size != 1 or not np.issubdtype(array.dtype, np.floating):
        raise CalibrationError(f"{name} must be one floating-point scalar")
    value = float(array.reshape(-1)[0])
    if not np.isfinite(value):
        raise CalibrationError(f"{name} must be finite")
    return value


def load_replay_stats(path: Path) -> LoadedReplayStats:
    """Load a strict, pickle-free NPZ and compute its exact input-byte digest."""
    path = Path(path)
    input_sha256 = _sha256_bytes(path.read_bytes())
    required = {
        "episode_stable_id",
        "episode_split",
        "clean_raw_retrieved",
        "layer8_residual",
        "n_delay",
        "alpha_step",
        "memory_inject_w",
        "noise_raw_retrieved",
        "noise_episode_index",
        "noise_kind",
        "noise_query_cosine",
        "source_sha256",
        "official_base_source_sha256",
        "dataset_sha256",
        "split_sha256",
        "replay_protocol_sha256",
        "collector_source_sha256",
        "preflight_sha256",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = tuple(sorted(archive.files))
            missing = sorted(required - set(keys))
            if missing:
                raise CalibrationError(f"replay-stat NPZ is missing required keys: {missing}")
            stats = ReplayStats(
                episode_stable_id=_text_vector(archive["episode_stable_id"], "episode_stable_id"),
                episode_split=_text_vector(archive["episode_split"], "episode_split"),
                clean_raw_retrieved=_float32_clean_tensor(archive["clean_raw_retrieved"]),
                layer8_residual=_float32_matrix(archive["layer8_residual"], "layer8_residual"),
                n_delay=_integer_vector(archive["n_delay"], "n_delay"),
                alpha_step=_scalar_float(archive["alpha_step"], "alpha_step"),
                memory_inject_w=np.asarray(archive["memory_inject_w"]),
                noise_raw_retrieved=_float32_matrix(archive["noise_raw_retrieved"], "noise_raw_retrieved"),
                noise_episode_index=_integer_vector(archive["noise_episode_index"], "noise_episode_index"),
                noise_kind=_text_vector(archive["noise_kind"], "noise_kind"),
                noise_query_cosine=np.asarray(archive["noise_query_cosine"]),
                source_sha256=_text_scalar(archive["source_sha256"], "source_sha256"),
                official_base_source_sha256=_text_scalar(
                    archive["official_base_source_sha256"], "official_base_source_sha256"
                ),
                dataset_sha256=_text_scalar(archive["dataset_sha256"], "dataset_sha256"),
                split_sha256=_text_scalar(archive["split_sha256"], "split_sha256"),
                replay_protocol_sha256=_text_scalar(archive["replay_protocol_sha256"], "replay_protocol_sha256"),
                collector_source_sha256=_text_scalar(archive["collector_source_sha256"], "collector_source_sha256"),
                preflight_sha256=_text_scalar(archive["preflight_sha256"], "preflight_sha256"),
            )
    except (OSError, ValueError) as exc:
        if isinstance(exc, CalibrationError):
            raise
        raise CalibrationError(f"cannot load replay-stat NPZ {path}: {exc}") from exc
    return LoadedReplayStats(stats=stats, input_sha256=input_sha256, npz_keys=keys)


def _validate_stats(stats: ReplayStats) -> tuple[int, int, int]:
    clean = stats.clean_raw_retrieved
    residual = stats.layer8_residual
    noise = stats.noise_raw_retrieved
    if clean.dtype != np.dtype(np.float32):
        raise CalibrationError("clean_raw_retrieved must remain float32 during calibration")
    if residual.dtype != np.dtype(np.float32):
        raise CalibrationError("layer8_residual must remain float32 during calibration")
    if noise.dtype != np.dtype(np.float32):
        raise CalibrationError("noise_raw_retrieved must remain float32 during calibration")
    if clean.ndim != 3 or clean.shape[1] != CLEAN_SLOT_COUNT or not clean.shape[0] or not clean.shape[2]:
        raise CalibrationError(f"clean_raw_retrieved must have shape (episodes, {CLEAN_SLOT_COUNT}, channels)")
    n_episode, _, channels = clean.shape
    if residual.shape != (n_episode, channels):
        raise CalibrationError(f"layer8_residual shape {residual.shape} must be ({n_episode}, {channels})")
    if noise.ndim != 2 or noise.shape[1] != channels or not noise.shape[0]:
        raise CalibrationError(
            f"noise_raw_retrieved must have non-empty shape (samples, {channels}); got {noise.shape}"
        )
    n_noise = noise.shape[0]
    if len(stats.episode_stable_id) != n_episode or len(stats.episode_split) != n_episode:
        raise CalibrationError("stable IDs and split labels must have one entry per clean episode")
    if any(not stable_id.strip() for stable_id in stats.episode_stable_id):
        raise CalibrationError("episode_stable_id contains an empty ID")
    if len(set(stats.episode_stable_id)) != n_episode:
        raise CalibrationError("episode_stable_id values must be unique")
    leaked = [
        stable_id
        for stable_id, split in zip(stats.episode_stable_id, stats.episode_split, strict=True)
        if split != "train"
    ]
    if leaked:
        raise CalibrationError("train-only calibration rejected non-train episodes: " + ", ".join(leaked[:8]))
    if stats.n_delay.shape != (n_episode,) or np.any(stats.n_delay < 0):
        raise CalibrationError("n_delay must contain one non-negative integer per episode")
    if not 0.0 <= stats.alpha_step < 1.0:
        raise CalibrationError(f"alpha_step must lie in [0, 1); got {stats.alpha_step}")
    if not np.all(np.isfinite(clean)) or not np.all(np.isfinite(residual)) or not np.all(np.isfinite(noise)):
        raise CalibrationError("replay vectors contain NaN or infinite values")

    gate_w = stats.memory_inject_w
    if gate_w.dtype != np.dtype(np.float32) or gate_w.shape != (channels,):
        raise CalibrationError(f"memory_inject_w must be float32 with shape ({channels},)")
    if not np.all(np.isfinite(gate_w)):
        raise CalibrationError("memory_inject_w contains NaN or infinite values")
    if stats.noise_episode_index.shape != (n_noise,):
        raise CalibrationError("noise_episode_index must have one integer per noise sample")
    if np.any(stats.noise_episode_index < 0) or np.any(stats.noise_episode_index >= n_episode):
        raise CalibrationError("noise_episode_index contains an out-of-range episode index")
    if len(stats.noise_kind) != n_noise:
        raise CalibrationError("noise_kind must have one label per noise sample")
    unknown_kinds = sorted(set(stats.noise_kind) - _ALLOWED_NOISE_KINDS)
    if unknown_kinds:
        raise CalibrationError(f"noise_kind contains unsupported labels: {unknown_kinds}")
    present_kinds = set(stats.noise_kind)
    if not (present_kinds & _QUERY_NOISE_KINDS):
        raise CalibrationError("at least one low-cosine or synthetic orthogonal query control is required")
    if "mixed_precision_residual" not in present_kinds:
        raise CalibrationError("at least one mixed_precision_residual noise sample is required")
    cosine = np.asarray(stats.noise_query_cosine)
    if cosine.shape != (n_noise,) or not np.issubdtype(cosine.dtype, np.floating):
        raise CalibrationError("noise_query_cosine must have one floating-point value per noise sample")
    query_mask = np.asarray([kind in _QUERY_NOISE_KINDS for kind in stats.noise_kind])
    query_cosine = cosine[query_mask]
    if not np.all(np.isfinite(query_cosine)):
        raise CalibrationError("query-control cosine values must be finite")
    if np.any(query_cosine < -1.0) or np.any(query_cosine > LOW_COSINE_MAX):
        raise CalibrationError(f"query-control cosine must be in [-1, {LOW_COSINE_MAX}]")
    _require_sha256("source_sha256", stats.source_sha256)
    _require_sha256("official_base_source_sha256", stats.official_base_source_sha256)
    _require_sha256("dataset_sha256", stats.dataset_sha256)
    _require_sha256("split_sha256", stats.split_sha256)
    _require_sha256("replay_protocol_sha256", stats.replay_protocol_sha256)
    _require_sha256("collector_source_sha256", stats.collector_source_sha256)
    _require_sha256("preflight_sha256", stats.preflight_sha256)
    return n_episode, n_noise, channels


def _rms_fp32(array: FloatArray, *, channel_mask: npt.NDArray[np.bool_] | None = None) -> FloatArray:
    selected = array if channel_mask is None else array[..., channel_mask]
    squared = np.square(selected, dtype=np.float32)
    mean_square = np.mean(squared, axis=-1, dtype=np.float32)
    rms = np.sqrt(mean_square).astype(np.float32, copy=False)
    if not np.all(np.isfinite(rms)):
        raise CalibrationError("FP32 RMS arithmetic overflowed or became non-finite")
    return rms


def _episode_slot_rms_fp32(array: FloatArray, *, channel_mask: npt.NDArray[np.bool_] | None = None) -> FloatArray:
    """RMS over slot and channel axes after all per-slot operations have completed."""
    if array.ndim != 3:
        raise CalibrationError(f"episode-slot aggregation requires [episodes,slots,channels], got {array.shape}")
    selected = array if channel_mask is None else array[..., channel_mask]
    squared = np.square(selected, dtype=np.float32)
    mean_square = np.mean(squared, axis=(1, 2), dtype=np.float32)
    rms = np.sqrt(mean_square).astype(np.float32, copy=False)
    if not np.all(np.isfinite(rms)):
        raise CalibrationError("FP32 episode-slot RMS arithmetic overflowed or became non-finite")
    return rms


def _production_slot_rms_fp32(raw: FloatArray) -> FloatArray:
    """The exact RMS used by Pi0._v32_inject_memory, including its FP32 epsilon."""
    mean_square = np.mean(np.square(raw, dtype=np.float32), axis=-1, dtype=np.float32)
    rms = np.sqrt(np.add(mean_square, _EPS, dtype=np.float32)).astype(np.float32, copy=False)
    if not np.all(np.isfinite(rms)):
        raise CalibrationError("production FP32 slot RMS overflowed or became non-finite")
    return rms


def production_pin_fp32(raw: FloatArray, gate: FloatArray, c: np.float32, tau: np.float32) -> FloatArray:
    """Mirror production pinning exactly: normalize each real retrieved slot independently."""
    if raw.ndim not in (2, 3) or raw.shape[-1] != gate.shape[0]:
        raise CalibrationError(f"production pin expects [samples,D] or [episodes,slots,D], got {raw.shape}")
    # The all-channel, per-slot denominator is a production invariant.  Do not average slots
    # before this point: pin(mean(raw)) is not mean(pin(raw)).
    denominator = np.maximum(_production_slot_rms_fp32(raw), tau).astype(np.float32, copy=False)
    gate_shape = (1,) * (raw.ndim - 1) + (gate.shape[0],)
    # Match `_v32_inject_memory` operation order, including FP32 roundoff: first form
    # `normed = raw * (c / denominator)`, then multiply the separate tanh gate.
    scale = np.divide(c, denominator, dtype=np.float32)
    normed = np.multiply(raw, scale[..., None], dtype=np.float32)
    return np.multiply(gate.reshape(gate_shape), normed, dtype=np.float32)


def _quantile(values: npt.NDArray[np.floating[Any]], q: float, *, method: str = "linear") -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method=method))


def _summary(values: npt.NDArray[np.floating[Any]]) -> dict[str, float | int]:
    values64 = np.asarray(values, dtype=np.float64)
    if values64.ndim != 1 or not values64.size or not np.all(np.isfinite(values64)):
        raise CalibrationError("cannot summarize an empty or non-finite distribution")
    return {
        "count": int(values64.size),
        "min": float(np.min(values64)),
        "p05": _quantile(values64, 0.05),
        "p10": _quantile(values64, 0.10),
        "p50": _quantile(values64, 0.50),
        "p90": _quantile(values64, 0.90),
        "p95": _quantile(values64, 0.95),
        "max": float(np.max(values64)),
    }


def _per_channel_rms_ratio(injected: FloatArray, residual: FloatArray) -> tuple[list[float | None], list[int]]:
    reduction_axes = tuple(range(injected.ndim - 1))
    numerator = np.sqrt(np.mean(np.square(injected, dtype=np.float32), axis=reduction_axes, dtype=np.float32))
    denominator = np.sqrt(np.mean(np.square(residual, dtype=np.float32), axis=0, dtype=np.float32))
    undefined = np.flatnonzero(denominator <= _EPS).astype(int).tolist()
    undefined_set = set(undefined)
    ratios = [
        None if index in undefined_set else float(numerator[index] / denominator[index])
        for index in range(numerator.shape[0])
    ]
    return ratios, undefined


def _artifact_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    digest = _sha256_bytes(canonical_json_bytes(payload))
    return {
        "artifact_sha256": digest,
        "calibration_id": f"sha256:{digest}",
        "hash_scope": "SHA256 of canonical_json($.payload)",
        "payload": payload,
    }


def verify_artifact(artifact: dict[str, Any]) -> bool:
    """Verify the embedded canonical payload digest without trusting serialization whitespace."""
    payload = artifact.get("payload")
    digest = artifact.get("artifact_sha256")
    calibration_id = artifact.get("calibration_id")
    if not isinstance(payload, dict) or not isinstance(digest, str):
        return False
    actual = _sha256_bytes(canonical_json_bytes(payload))
    return digest == actual and calibration_id == f"sha256:{actual}"


def calibrate_injection(
    stats: ReplayStats,
    *,
    input_sha256: str,
    npz_keys: Sequence[str] = (),
) -> dict[str, Any]:
    """Calibrate ``c`` and ``tau`` and return a passing, self-identifying artifact.

    All structural and numerical gates raise :class:`CalibrationError`; a failing result can
    therefore never be mistaken for a usable calibration artifact.
    """
    n_episode, n_noise, channels = _validate_stats(stats)
    _require_sha256("input_sha256", input_sha256)

    gate = np.tanh(stats.memory_inject_w.astype(np.float32)).astype(np.float32, copy=False)
    gate_error = np.abs(gate - TARGET_EFFECTIVE_GATE)
    max_gate_error = float(np.max(gate_error))
    if not np.all(gate_error <= GATE_ATOL):
        raise CalibrationError(
            "effective tanh(memory_inject_w) must equal 0.5 channelwise within "
            f"{GATE_ATOL:.9g}; max error={max_gate_error:.9g}"
        )
    open_mask = np.abs(gate) >= OPEN_GATE_THRESHOLD
    if not np.all(open_mask):
        closed = np.flatnonzero(~open_mask).astype(int).tolist()
        raise CalibrationError(f"fixed-gate initialization left closed channels: {closed[:16]}")

    clean = stats.clean_raw_retrieved
    residual = stats.layer8_residual
    clean_slot_raw_rms = _production_slot_rms_fp32(clean)
    clean_episode_raw_rms = _episode_slot_rms_fp32(clean)
    median_clean_raw_rms = float(np.median(clean_slot_raw_rms))
    if median_clean_raw_rms <= float(_EPS):
        raise CalibrationError("median clean raw-read RMS is zero or too small")
    tau = np.float32(median_clean_raw_rms / 0.75)
    if not np.isfinite(tau) or tau <= _EPS:
        raise CalibrationError("calibrated tau is zero or non-finite")

    clean_slot_amplitude = np.minimum(clean_slot_raw_rms / tau, np.float32(1.0)).astype(np.float32)
    clean_episode_amplitude = np.median(clean_slot_amplitude, axis=1).astype(np.float32)
    median_clean_amplitude = float(np.median(clean_episode_amplitude))
    if not CLEAN_AMPLITUDE_RANGE[0] <= median_clean_amplitude <= CLEAN_AMPLITUDE_RANGE[1]:
        raise CalibrationError(
            "median clean retained amplitude must be in "
            f"[{CLEAN_AMPLITUDE_RANGE[0]}, {CLEAN_AMPLITUDE_RANGE[1]}]; "
            f"got {median_clean_amplitude:.9g}"
        )

    residual_open_rms = _rms_fp32(residual, channel_mask=open_mask)
    if np.any(residual_open_rms <= _EPS):
        bad = np.flatnonzero(residual_open_rms <= _EPS).astype(int).tolist()
        raise CalibrationError(f"layer-8 residual RMS is zero for episodes {bad[:16]}")
    unit_clean = production_pin_fp32(clean, gate, np.float32(1.0), tau)
    unit_clean_open_rms = _episode_slot_rms_fp32(unit_clean, channel_mask=open_mask)
    median_signal_open_rms = float(np.median(unit_clean_open_rms))
    if median_signal_open_rms <= float(_EPS):
        raise CalibrationError("open-channel clean signal denominator is zero or too small")
    c = np.float32(float(np.median(residual_open_rms)) / median_signal_open_rms)
    if not np.isfinite(c) or c <= _EPS:
        raise CalibrationError("calibrated c is zero or non-finite")

    clean_injected = production_pin_fp32(clean, gate, c, tau)
    residual_all_rms = _rms_fp32(residual)
    clean_injected_episode_rms = _episode_slot_rms_fp32(clean_injected)
    clean_ratio = np.divide(clean_injected_episode_rms, residual_all_rms, dtype=np.float32)
    prototype_injected_rms_target = float(np.median(clean_injected_episode_rms))

    alpha = np.float32(stats.alpha_step)
    retention = np.asarray(np.power(np.float32(1.0) - alpha, stats.n_delay), dtype=np.float32)
    decayed = np.multiply(clean, retention[:, None, None], dtype=np.float32)
    decayed_injected = production_pin_fp32(decayed, gate, c, tau)
    decayed_ratio = np.divide(_episode_slot_rms_fp32(decayed_injected), residual_all_rms, dtype=np.float32)
    decayed_slot_amplitude = np.minimum(_production_slot_rms_fp32(decayed) / tau, np.float32(1.0)).astype(np.float32)
    decayed_amplitude = np.median(decayed_slot_amplitude, axis=1).astype(np.float32)

    p90_delay = int(np.quantile(stats.n_delay, 0.90, method="higher"))
    p90_retention = np.float32(np.power(np.float32(1.0) - alpha, p90_delay))
    p90_decayed = np.multiply(clean, p90_retention, dtype=np.float32)
    p90_decayed_injected = production_pin_fp32(p90_decayed, gate, c, tau)
    p90_slot_amplitude = np.minimum(_production_slot_rms_fp32(p90_decayed) / tau, np.float32(1.0)).astype(np.float32)
    p90_amplitude = np.median(p90_slot_amplitude, axis=1).astype(np.float32)
    p90_median_amplitude = float(np.median(p90_amplitude))
    if p90_median_amplitude < P90_DELAY_AMPLITUDE_MIN:
        raise CalibrationError(
            f"p90-delay median retained amplitude {p90_median_amplitude:.9g} is below "
            f"{P90_DELAY_AMPLITUDE_MIN} at n_delay={p90_delay}"
        )

    noise_episode_index = stats.noise_episode_index.astype(np.int64, copy=False)
    noise_residual = residual[noise_episode_index]
    noise_injected = production_pin_fp32(stats.noise_raw_retrieved, gate, c, tau)
    noise_ratio = np.divide(
        _rms_fp32(noise_injected),
        _rms_fp32(noise_residual),
        dtype=np.float32,
    )
    noise_p95_higher = _quantile(noise_ratio, 0.95, method="higher")
    if noise_p95_higher > NOISE_P95_RATIO_MAX:
        raise CalibrationError(
            f"real-noise injected/residual RMS p95 {noise_p95_higher:.9g} exceeds "
            f"{NOISE_P95_RATIO_MAX} (empirical higher quantile)"
        )

    clean_per_channel, clean_undefined = _per_channel_rms_ratio(clean_injected, residual)
    decayed_per_channel, decayed_undefined = _per_channel_rms_ratio(decayed_injected, residual)
    noise_per_channel, noise_undefined = _per_channel_rms_ratio(noise_injected, noise_residual)
    membership = [
        {"split": split, "stable_id": stable_id}
        for stable_id, split in zip(stats.episode_stable_id, stats.episode_split, strict=True)
    ]
    noise_kinds = np.asarray(stats.noise_kind)
    noise_by_kind = {kind: _summary(noise_ratio[noise_kinds == kind]) for kind in sorted(set(stats.noise_kind))}
    episode_metrics = [
        {
            "clean_injected_to_residual_rms": float(clean_ratio[index]),
            "clean_episode_raw_rms": float(clean_episode_raw_rms[index]),
            "clean_slot_raw_rms_p50": float(np.median(clean_slot_raw_rms[index])),
            "decayed_injected_to_residual_rms": float(decayed_ratio[index]),
            "decayed_retained_amplitude": float(decayed_amplitude[index]),
            "n_delay": int(stats.n_delay[index]),
            "stable_id": stats.episode_stable_id[index],
        }
        for index in range(n_episode)
    ]

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "provenance": {
            "dataset_sha256": stats.dataset_sha256,
            "input_npz_sha256": input_sha256,
            "npz_keys": sorted(npz_keys),
            "observed_membership_sha256": _sha256_bytes(canonical_json_bytes(membership)),
            "official_base_source_sha256": stats.official_base_source_sha256,
            "source_sha256": stats.source_sha256,
            "split_sha256": stats.split_sha256,
            "replay_protocol_sha256": stats.replay_protocol_sha256,
            "collector_source_sha256": stats.collector_source_sha256,
            "preflight_sha256": stats.preflight_sha256,
        },
        "population": {
            "channel_count": channels,
            "clean_slots_per_episode": CLEAN_SLOT_COUNT,
            "episode_count": n_episode,
            "noise_sample_count": n_noise,
            "noise_kind_count": {
                kind: int(sum(label == kind for label in stats.noise_kind)) for kind in sorted(set(stats.noise_kind))
            },
            "split": "train",
            "stable_ids": list(stats.episode_stable_id),
        },
        "parameters": {
            "alpha_step": float(alpha),
            "memory_injection_c": float(c),
            "memory_injection_tau": float(tau),
            # Correct oracle scale: the median episode RMS after every one of the 16 real
            # slots has passed through the exact production pin independently.
            "prototype_injected_rms_target": prototype_injected_rms_target,
        },
        "gate": {
            "effective_tanh_gate_max": float(np.max(gate)),
            "effective_tanh_gate_min": float(np.min(gate)),
            "max_abs_error_from_0_5": max_gate_error,
            "open_channel_count": int(np.sum(open_mask)),
            "open_channel_threshold": float(OPEN_GATE_THRESHOLD),
            "raw_w_sha256": _sha256_bytes(stats.memory_inject_w.tobytes(order="C")),
            "required_atol": GATE_ATOL,
            "target_effective_tanh_gate": float(TARGET_EFFECTIVE_GATE),
        },
        "statistics": {
            "clean_raw_slot_rms": _summary(clean_slot_raw_rms.reshape(-1)),
            "clean_raw_episode_rms": _summary(clean_episode_raw_rms),
            "clean_retained_episode_median_amplitude": _summary(clean_episode_amplitude),
            "injected_to_layer8_residual_rms": {
                "clean": _summary(clean_ratio),
                "decayed_at_episode_delay": _summary(decayed_ratio),
                "noise": _summary(noise_ratio),
                "noise_by_kind": noise_by_kind,
                "per_channel_rms_ratio": {
                    "clean": clean_per_channel,
                    "clean_undefined_residual_channels": clean_undefined,
                    "decayed_at_episode_delay": decayed_per_channel,
                    "decayed_undefined_residual_channels": decayed_undefined,
                    "noise": noise_per_channel,
                    "noise_undefined_residual_channels": noise_undefined,
                },
            },
            "episode_delay_retained_amplitude": _summary(decayed_amplitude),
            "p90_delay": {
                "n_delay": p90_delay,
                "quantile_method": "higher",
                "retention_factor": float(p90_retention),
                "retained_amplitude": _summary(p90_amplitude),
                "injected_to_layer8_residual_rms": _summary(
                    np.divide(_episode_slot_rms_fp32(p90_decayed_injected), residual_all_rms, dtype=np.float32)
                ),
            },
            "episode_metrics": episode_metrics,
        },
        "gates": {
            "all_channels_open": True,
            "all_episodes_train": True,
            "clean_median_amplitude_in_0_7_to_0_8": True,
            "fixed_effective_gate_is_0_5": True,
            "noise_p95_higher": noise_p95_higher,
            "noise_p95_max": NOISE_P95_RATIO_MAX,
            "noise_p95_pass": True,
            "p90_delay_median_amplitude": p90_median_amplitude,
            "p90_delay_min": P90_DELAY_AMPLITUDE_MIN,
            "p90_delay_pass": True,
            "passes": True,
        },
        "arithmetic": {
            "calibration_vector_dtype": "float32",
            "clean_read_shape": "[episode,16 real retrieved slots,channel]",
            "c_definition": (
                "median_episode(RMS_open(layer8_residual))/median_episode(RMS_slots,open(unit_clean_injection))"
            ),
            "injection_definition": (
                "per_slot(gate*(raw*(c/max(sqrt(mean(raw_slot**2)+1e-12),tau)))); "
                "all arithmetic and operation order mirror production FP32"
            ),
            "noise_gate_quantile_method": "higher",
            "prototype_scale_definition": ("median_episode(RMS_slots,channels(production_pin(clean_raw_retrieved)))"),
            "tau_definition": ("median_over_episode_slots(sqrt(mean(clean_raw_read_slot**2)+1e-12))/0.75"),
        },
    }
    return _artifact_envelope(payload)


def write_canonical_artifact(path: Path, artifact: dict[str, Any], *, overwrite: bool = False) -> None:
    """Write one canonical JSON document, refusing accidental replacement by default."""
    if not verify_artifact(artifact):
        raise CalibrationError("refusing to write an artifact with an invalid payload SHA256")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if overwrite else "xb"
    with path.open(mode) as stream:
        stream.write(canonical_json_bytes(artifact) + b"\n")


def _resolve_project_cli_path(path: Path, *, name: str) -> Path:
    """Resolve one CLI path while forbidding machine-local or escaping spellings."""
    raw = Path(path)
    if raw.is_absolute():
        raise CalibrationError(f"{name} must be relative to memory_project, got {str(raw)!r}")
    if ".." in raw.parts:
        raise CalibrationError(f"{name} must not escape memory_project, got {str(raw)!r}")
    try:
        return project_paths.project_path(raw)
    except project_paths.ProjectRootError as exc:
        raise CalibrationError(f"invalid {name}: {exc}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Every path argument must be relative to memory_project; absolute paths and '..' are rejected.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Train-only replay-stat NPZ, relative to memory_project.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New canonical calibration JSON, relative to memory_project.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        project_paths.configure_v35_runtime_environment()
        input_path = _resolve_project_cli_path(args.input, name="input")
        output_path = _resolve_project_cli_path(args.output, name="output")
        loaded = load_replay_stats(input_path)
        artifact = calibrate_injection(
            loaded.stats,
            input_sha256=loaded.input_sha256,
            npz_keys=loaded.npz_keys,
        )
        write_canonical_artifact(output_path, artifact)
    except (CalibrationError, project_paths.ProjectRootError, FileExistsError, OSError) as exc:
        parser.error(str(exc))
    print(f"wrote {args.output} ({artifact['calibration_id']})")


if __name__ == "__main__":
    main()
