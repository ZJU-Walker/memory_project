"""v3.3 gradient-flow check (V33_MEMORY_HANDOFF section 16).

Builds real MEMORY-CRITICAL sequence batches from the training dataset (start shortly before
`inspect both bins`, truncated inside the waiting phase), loads a checkpoint, and measures

    g_tau = || d CE(endpoint) / d z_tau^w ||

for every step of the history via `Pi0.v33_endpoint_gradient_step`. The check passes when the
waiting-endpoint subtask CE reaches the evidence-phase writes (nonzero g_tau there). Two modes:

  * --gate_override 0.05   probe the PATHWAY on an untrained/early checkpoint (the zero-init
                           content gate would otherwise make every g_tau trivially zero);
  * (no override)          measure the realized gradient on a trained checkpoint.

Usage:
    uv run scripts/v33_gradient_flow_check.py \
        --checkpoint checkpoints/pi05_yam_mem_v33/<exp>/250 \
        --gate_override 0.05
"""

# ruff: noqa: I001 - data_loader (torch) must import before config (tensorflow), or the
# interpreter segfaults; same pinned order as data_loader_test.py.
import dataclasses
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import tyro

import openpi.training.data_loader as _data_loader
import openpi.training.config as _config
from openpi.diagnostics.v32_checkpoint import _align_params  # shared checkpoint-loading shim
from openpi.models import model as _model
from openpi.training import weight_loaders


@dataclasses.dataclass(frozen=True)
class Options:
    checkpoint: Path
    config: str = "pi05_yam_mem_v33"
    output_dir: Path = Path("diagnostic_outputs/v33_gradient_flow")
    batch_size: int = 4
    gate_override: float | None = None
    seed: int = 0


def _memory_critical_batch(config: _config.TrainConfig, batch_size: int, seed: int):
    """One collated sequence batch whose samples all start inside the memory-critical window,
    spread over the (instruction, side) cells, plus per-sample phase bookkeeping."""
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = _data_loader.transform_dataset(dataset, data_config)

    # the constructed dataset already carries its metadata; a fresh LeRobotDatasetMetadata
    # would touch the hugging-face hub again (fails on offline compute nodes)
    meta = _data_loader._unwrap_lerobot(dataset).meta  # noqa: SLF001
    info = _data_loader._episode_info_table(dataset, meta, data_config)  # noqa: SLF001
    windows = _data_loader._memory_critical_windows(info, data_config)  # noqa: SLF001
    if windows is None:
        raise ValueError(f"config {config.name} does not define memory-critical phases.")

    episode_starts = np.concatenate([[0], np.cumsum(info["length"])[:-1]])
    rng = np.random.default_rng(seed)
    usable = [e for e in range(len(windows)) if windows[e, 0] >= 0]
    chosen = rng.permutation(usable)[:batch_size]
    if len(chosen) < batch_size:
        raise ValueError(f"only {len(chosen)} episodes have usable phases; lower --batch_size.")

    # (endpoints are deterministic per start frame and memory-critical samples carry no TBPTT
    # fence, so nothing below consumes global np.random state)
    items, bookkeeping = [], []
    for e in chosen:
        start = int(rng.integers(windows[e, 0], windows[e, 1] + 1))
        items.append(dataset[int(episode_starts[e] + start)])
        bookkeeping.append(
            {
                "episode": int(e),
                "start_frame": start,
                "evidence": [int(info["evidence_start"][e]), int(info["evidence_end"][e])],
                "memory": [int(info["memory_lo"][e]), int(info["memory_hi"][e])],
                "side": int(info["side"][e]),
            }
        )
    batch = _data_loader._collate_fn(items)  # noqa: SLF001
    return _model.Observation.from_dict(batch), bookkeeping, data_config


def main(options: Options):
    config = _config.get_config(options.config)
    observation, bookkeeping, _ = _memory_critical_batch(config, options.batch_size, options.seed)

    import flax.nnx as nnx
    import jax

    raw_params = _model.restore_params(options.checkpoint / "params", dtype=jnp.float32)
    abstract_state = nnx.state(nnx.eval_shape(config.model.create, jax.random.key(0))).to_pure_dict()
    aligned = _align_params(abstract_state, raw_params)
    # _align_params restores structure (orbax drops flax None-bias slots) but not MISSING
    # subsystems: probing straight from a pre-memory base checkpoint (pi05_base) needs the
    # memory/compressor/conditioner params at their fresh initialization, exactly like
    # PartialCheckpointWeightLoader at training start.
    init_params = nnx.state(config.model.create(jax.random.key(0))).to_pure_dict()
    model = config.model.load(
        weight_loaders._merge_params(aligned, init_params, missing_regex=".*"),  # noqa: SLF001
        remove_extra_params=False,
    )
    model.eval()

    result = model.v33_endpoint_gradient_step(observation, gate_override=options.gate_override)
    grad = np.asarray(result["write_grad_norm"], dtype=np.float64)  # [b, t]
    valid = np.asarray(result["step_valid"], dtype=bool)

    stride = config.data.base_config.memory_stride_frames if hasattr(config.data, "base_config") else 15
    report = {"gate_override": options.gate_override, "samples": []}
    print(f"\n{'sample':>6} {'phase':>12} {'steps':>6} {'mean g_tau':>12} {'max g_tau':>12}")
    for i, book in enumerate(bookkeeping):
        t_valid = int(valid[i].sum())
        frames = book["start_frame"] + np.arange(t_valid) * stride
        phases = {
            "evidence": (frames >= book["evidence"][0]) & (frames <= book["evidence"][1]),
            "retention": (frames > book["evidence"][1]) & (frames < book["memory"][0]),
            "waiting": frames >= book["memory"][0],
        }
        sample = {**book, "valid_steps": t_valid, "g_tau": grad[i, :t_valid].tolist(), "phase_stats": {}}
        for name, mask in phases.items():
            g = grad[i, :t_valid][mask]
            sample["phase_stats"][name] = {
                "steps": int(mask.sum()),
                "mean": float(g.mean()) if len(g) else 0.0,
                "max": float(g.max()) if len(g) else 0.0,
            }
            print(
                f"{i:>6} {name:>12} {int(mask.sum()):>6} "
                f"{sample['phase_stats'][name]['mean']:>12.3e} {sample['phase_stats'][name]['max']:>12.3e}"
            )
        report["samples"].append(sample)

    evidence_max = max(s["phase_stats"]["evidence"]["max"] for s in report["samples"])
    endpoint_zero = all(s["g_tau"][-1] < 1e-12 for s in report["samples"] if s["g_tau"])
    report["evidence_max"] = evidence_max
    report["endpoint_is_zero"] = endpoint_zero  # causal order sanity: the endpoint's own write gets no gradient
    verdict = "PASS" if evidence_max > 0 else "FAIL"
    print(f"\nendpoint g == 0 (causal-order sanity): {endpoint_zero}")
    print(
        f"evidence-phase max g_tau: {evidence_max:.3e} -> {verdict}: "
        f"{'the waiting CE reaches the evidence-phase writes' if evidence_max > 0 else 'NO credit reaches the evidence phase'}"
    )

    options.output_dir.mkdir(parents=True, exist_ok=True)
    out = options.output_dir / "gradient_flow.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(tyro.cli(Options))
