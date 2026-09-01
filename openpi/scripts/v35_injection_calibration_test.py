import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from openpi.shared import project_paths

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    import v35_injection_calibration as calibration
finally:
    sys.path.remove(str(_SCRIPTS_DIR))


def _valid_arrays(*, n_delay: np.ndarray | None = None) -> dict[str, np.ndarray]:
    episode_count = 5
    channels = 4
    clean = np.ones((episode_count, calibration.CLEAN_SLOT_COUNT, channels), dtype=np.float32)
    residual = np.full((episode_count, channels), 0.3, dtype=np.float32)
    low_cos = np.full((episode_count, channels), 0.01, dtype=np.float32)
    mixed_precision = np.full((episode_count, channels), 0.005, dtype=np.float32)
    return {
        "episode_stable_id": np.asarray([f"train-episode-{i}" for i in range(episode_count)]),
        "episode_split": np.asarray(["train"] * episode_count),
        "clean_raw_retrieved": clean,
        "layer8_residual": residual,
        "n_delay": (
            np.asarray([10, 20, 30, 40, 50], dtype=np.int32) if n_delay is None else np.asarray(n_delay, dtype=np.int32)
        ),
        "alpha_step": np.asarray(0.01, dtype=np.float32),
        "memory_inject_w": np.full((channels,), np.arctanh(np.float32(0.5)), dtype=np.float32),
        "noise_raw_retrieved": np.concatenate((low_cos, mixed_precision), axis=0),
        "noise_episode_index": np.tile(np.arange(episode_count, dtype=np.int32), 2),
        "noise_kind": np.asarray(["low_cos_query"] * episode_count + ["mixed_precision_residual"] * episode_count),
        "noise_query_cosine": np.asarray([0.05] * episode_count + [np.nan] * episode_count),
        "source_sha256": np.asarray("1" * 64),
        "official_base_source_sha256": np.asarray("4" * 64),
        "dataset_sha256": np.asarray("2" * 64),
        "split_sha256": np.asarray("3" * 64),
        "replay_protocol_sha256": np.asarray("5" * 64),
        "collector_source_sha256": np.asarray("6" * 64),
        "preflight_sha256": np.asarray("7" * 64),
    }


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _portable_project(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "memory_project"
    (root / "openpi/src/openpi").mkdir(parents=True)
    (root / "openpi/pyproject.toml").touch()
    monkeypatch.setenv(project_paths.MEMORY_PROJECT_ROOT_ENV, str(root))
    for name, value in project_paths.v35_runtime_environment().items():
        monkeypatch.setenv(name, value)
    return root


def _load_and_calibrate(path: Path) -> dict:
    loaded = calibration.load_replay_stats(path)
    return calibration.calibrate_injection(
        loaded.stats,
        input_sha256=loaded.input_sha256,
        npz_keys=loaded.npz_keys,
    )


def test_calibration_uses_all_channel_denominator_and_emits_verifiable_artifact(tmp_path: Path) -> None:
    source = tmp_path / "stats.npz"
    _write_npz(source, _valid_arrays())

    artifact = _load_and_calibrate(source)
    payload = artifact["payload"]

    assert calibration.verify_artifact(artifact)
    assert artifact["calibration_id"] == f"sha256:{artifact['artifact_sha256']}"
    assert payload["status"] == "pass"
    assert payload["population"]["split"] == "train"
    assert payload["population"]["episode_count"] == 5
    assert payload["parameters"]["memory_injection_tau"] == pytest.approx(1.0 / 0.75)
    assert payload["parameters"]["memory_injection_c"] == pytest.approx(0.8)
    assert payload["parameters"]["prototype_injected_rms_target"] == pytest.approx(0.3)
    assert payload["population"]["clean_slots_per_episode"] == 16
    assert payload["gates"]["noise_p95_pass"]
    assert payload["gates"]["p90_delay_pass"]
    assert payload["statistics"]["p90_delay"]["n_delay"] == 50
    assert payload["provenance"]["input_npz_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()

    tampered = copy.deepcopy(artifact)
    tampered["payload"]["parameters"]["memory_injection_c"] = 123.0
    assert not calibration.verify_artifact(tampered)


def test_cli_writes_canonical_json_and_refuses_existing_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _portable_project(monkeypatch, tmp_path)
    source = root / "v35/diagnostics/stats.npz"
    output = root / "v35/diagnostics/nested/calibration.json"
    _write_npz(source, _valid_arrays())

    calibration.main(
        [
            "--input",
            "v35/diagnostics/stats.npz",
            "--output",
            "v35/diagnostics/nested/calibration.json",
        ]
    )

    raw = output.read_bytes()
    artifact = json.loads(raw)
    assert raw == calibration.canonical_json_bytes(artifact) + b"\n"
    assert calibration.verify_artifact(artifact)
    with pytest.raises(SystemExit) as exc_info:
        calibration.main(
            [
                "--input",
                "v35/diagnostics/stats.npz",
                "--output",
                "v35/diagnostics/nested/calibration.json",
            ]
        )
    assert exc_info.value.code == 2


@pytest.mark.parametrize("bad_input", ["../stats.npz", "/iris/u/kewalk/stats.npz"])
def test_cli_rejects_unconfined_paths(
    bad_input: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _portable_project(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        calibration.main(
            [
                "--input",
                bad_input,
                "--output",
                "v35/diagnostics/calibration.json",
            ]
        )

    assert exc_info.value.code == 2


def test_cli_has_no_overwrite_escape_hatch() -> None:
    with pytest.raises(SystemExit) as exc_info:
        calibration.main(
            [
                "--input",
                "v35/diagnostics/stats.npz",
                "--output",
                "v35/diagnostics/calibration.json",
                "--overwrite",
            ]
        )

    assert exc_info.value.code == 2


def test_non_train_episode_is_rejected_without_an_artifact(tmp_path: Path) -> None:
    arrays = _valid_arrays()
    arrays["episode_split"][2] = "dev"
    source = tmp_path / "leaked.npz"
    _write_npz(source, arrays)

    with pytest.raises(calibration.CalibrationError, match="rejected non-train episodes"):
        _load_and_calibrate(source)


def test_effective_gate_must_be_half_in_every_channel(tmp_path: Path) -> None:
    arrays = _valid_arrays()
    arrays["memory_inject_w"][1] = np.float32(0.0)
    source = tmp_path / "bad_gate.npz"
    _write_npz(source, arrays)

    with pytest.raises(calibration.CalibrationError, match=r"must equal 0\.5 channelwise"):
        _load_and_calibrate(source)


def test_noise_p95_gate_fails_closed(tmp_path: Path) -> None:
    arrays = _valid_arrays()
    arrays["noise_raw_retrieved"][:] = np.float32(1.0)
    source = tmp_path / "noisy.npz"
    _write_npz(source, arrays)

    with pytest.raises(calibration.CalibrationError, match="real-noise injected/residual RMS p95"):
        _load_and_calibrate(source)


def test_p90_delay_retention_gate_fails_closed(tmp_path: Path) -> None:
    arrays = _valid_arrays(n_delay=np.asarray([100, 100, 100, 100, 100]))
    arrays["alpha_step"] = np.asarray(0.02, dtype=np.float32)
    source = tmp_path / "long_delay.npz"
    _write_npz(source, arrays)

    with pytest.raises(calibration.CalibrationError, match="p90-delay median retained amplitude"):
        _load_and_calibrate(source)


def test_query_controls_require_verified_low_cosine(tmp_path: Path) -> None:
    arrays = _valid_arrays()
    arrays["noise_query_cosine"][0] = 0.11
    source = tmp_path / "bad_cosine.npz"
    _write_npz(source, arrays)

    with pytest.raises(calibration.CalibrationError, match="query-control cosine"):
        _load_and_calibrate(source)


def test_production_pin_is_per_slot_and_never_pin_of_slot_mean() -> None:
    raw = np.empty((1, calibration.CLEAN_SLOT_COUNT, 2), dtype=np.float32)
    raw[:, :8] = np.asarray([10.0, 0.0], dtype=np.float32)
    raw[:, 8:] = np.asarray([0.1, 0.0], dtype=np.float32)
    gate = np.full((2,), 0.5, dtype=np.float32)

    pinned_slots = calibration.production_pin_fp32(raw, gate, np.float32(1.0), np.float32(1.0))
    pinned_mean = calibration.production_pin_fp32(np.mean(raw, axis=1), gate, np.float32(1.0), np.float32(1.0))

    assert not np.allclose(np.mean(pinned_slots, axis=1), pinned_mean)
    # The large slot normalizes to unit-RMS before the 0.5 gate; the small slot remains below
    # the tau floor.  This is the exact production ordering the reducer must retain.
    assert pinned_slots[0, 0, 0] == pytest.approx(np.sqrt(0.5))
    assert pinned_slots[0, -1, 0] == pytest.approx(0.05)


def test_clean_reads_must_retain_all_sixteen_slots(tmp_path: Path) -> None:
    arrays = _valid_arrays()
    arrays["clean_raw_retrieved"] = arrays["clean_raw_retrieved"].mean(axis=1)
    source = tmp_path / "mean_only.npz"
    _write_npz(source, arrays)

    with pytest.raises(calibration.CalibrationError, match="episodes, 16, channels"):
        calibration.load_replay_stats(source)
