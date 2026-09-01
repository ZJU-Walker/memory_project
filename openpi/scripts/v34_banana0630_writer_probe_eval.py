"""Score run5 writer features on the June-30 banana-only collection (model-level held-out).

For one checkpoint this evaluator:

1. reconstructs the exact run5 inference runtime via ``v34_fixed_writer_probe_eval`` (same
   transforms, fresh M0 per batch, zero writes, same float32 L2(mean-slot) writer pooling);
2. extracts writer features on ``observe bins`` (task_index==0) frames of every episode of
   ``yam/bin_memory_banana_subtask`` (30 episodes the model NEVER trained on, collected
   2026-06-30 with different lids/scene from the 0816 training set) at a fixed frame stride;
3. scores each episode-mean feature with two heads:
   * ``fresh``: the deterministic fit-all-56 logistic head refit from the checkpoint's
     completed fixed-probe artifact (features.npz), reproducing the training-set protocol;
   * ``online``: the checkpoint's own stored ladder writer head (logit right minus left).

The side label comes from the episode's terminal phase (task_index 2 = right, 1 = left).
This is an out-of-distribution transfer probe, not part of the preregistered v3.4 ladder.
"""

import argparse
import dataclasses
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v34_fixed_writer_probe_eval as fixed  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pandas as pd  # noqa: E402

OBSERVE_TASK_INDEX = 0
LEFT_TASK_INDEX = 1
RIGHT_TASK_INDEX = 2
PROMPT = "find the banana"
TRAIN_DATASET_ROOT = Path("/iris/u/kewalk/.cache/huggingface/lerobot/yam/bin_memory_0816_subtask")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe-artifact-dir", type=Path, required=True,
                        help="completed fixed-probe raw dir for the same checkpoint (features.npz)")
    parser.add_argument("--banana-root", type=Path,
                        default=Path("/iris/u/kewalk/.cache/huggingface/lerobot/yam/bin_memory_banana_subtask"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", default=fixed.RUN5_CONFIG)
    parser.add_argument("--parameter-source", choices=("raw", "ema"), default="raw")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--limit-episodes", type=int, default=0, help="smoke: only first N episodes")
    parser.add_argument("--frame-mode", choices=("observe", "inspect", "full"), default="observe",
                        help="observe: legacy task_index==0 window; inspect: the relabeled "
                             "'inspect both bins' window from the raw demo subtask_labels.json; "
                             "full: every frame (for per-frame score videos)")
    parser.add_argument("--raw-labels-root", type=Path,
                        default=Path("/iris/u/kewalk/memory_project/data/bin_memory_banana"))
    parser.add_argument("--save-frame-scores", action="store_true",
                        help="write per-episode npz with per-frame fresh/online scores")
    parser.add_argument("--episodes", default="",
                        help="comma-separated episode indices to process (default: all)")
    return parser.parse_args(argv)


def _fresh_head(probe_artifact_dir: Path):
    """Refit the deterministic fit-all-56 head from the artifact and sanity-check it."""
    report = json.loads((probe_artifact_dir / "report.json").read_text())
    with np.load(probe_artifact_dir / "features.npz", allow_pickle=False) as arrays:
        feats = np.asarray(arrays["episode_writer_evidence"], dtype=np.float64)
        labels = np.asarray(arrays["episode_labels"], dtype=np.int64)
        heldout = np.asarray(arrays["episode_heldout"], dtype=bool)
    fit_mask = ~heldout
    if int(fit_mask.sum()) != 56 or int(heldout.sum()) != 4:
        raise ValueError(f"unexpected split in features.npz: fit={fit_mask.sum()} heldout={heldout.sum()}")
    model = fixed._fit_probe(feats[fit_mask], labels[fit_mask])
    # Reproduce the artifact's fit-all-56 heldout balanced accuracy as a wiring check.
    scores4 = model.scores(feats[heldout])
    acc4 = float(np.mean((scores4 > 0).astype(np.int64) == labels[heldout]))
    reported = report["fresh_probe_streams"]["writer"]["fit_all_56_test_exact_heldout_4"][
        "evidence_metrics"]["accuracy"]
    if abs(acc4 - float(reported)) > 1e-9:
        raise RuntimeError(f"fresh-head refit does not reproduce the artifact: {acc4} != {reported}")
    return model


def main(argv=None) -> int:
    args = _parse_args(argv)
    out = args.output_dir / args.parameter_source
    if (out / "COMPLETE").exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)

    runtime_args = fixed.Args(
        checkpoint=args.checkpoint,
        dataset_root=TRAIN_DATASET_ROOT,
        output_dir=out / "_runtime_preflight",
        config=args.config,
        parameter_source=args.parameter_source,
        batch_size=args.batch_size,
    )
    runtime = fixed._Run5Runtime(runtime_args)
    fresh = _fresh_head(args.probe_artifact_dir)
    online = runtime._online_writer_head

    parquets = sorted((args.banana_root / "data" / "chunk-000").glob("episode_*.parquet"))
    if args.limit_episodes:
        parquets = parquets[: args.limit_episodes]
    if args.episodes:
        wanted = {int(x) for x in args.episodes.split(",")}
        parquets = [q for q in parquets if int(q.stem.split("_")[-1]) in wanted]
    if not parquets:
        raise FileNotFoundError(args.banana_root)

    episodes = []
    all_feats = []
    for path in parquets:
        episode = int(path.stem.split("_")[-1])
        df = pd.read_parquet(path, columns=[
            "image", "left_wrist_image", "right_wrist_image", "state", "task_index", "frame_index"])
        tasks = df["task_index"].to_numpy()
        if not np.array_equal(df["frame_index"].to_numpy(), np.arange(len(df))):
            raise ValueError(f"episode {episode}: frame_index not contiguous")
        terminal = set(np.unique(tasks)) - {OBSERVE_TASK_INDEX}
        if terminal == {LEFT_TASK_INDEX}:
            side, label = "left", 0
        elif terminal == {RIGHT_TASK_INDEX}:
            side, label = "right", 1
        else:
            raise ValueError(f"episode {episode}: ambiguous terminal phases {terminal}")
        if args.frame_mode == "inspect":
            import json as _json
            raw = _json.loads((args.raw_labels_root / f"demo{episode + 1}" /
                               "subtask_labels.json").read_text())
            if raw[-1]["end"] + 1 != len(df):
                raise ValueError(f"episode {episode}: raw labels cover {raw[-1]['end'] + 1} "
                                 f"frames but parquet has {len(df)}")
            seg = next(s for s in raw if s["task"] == "inspect both bins")
            window = np.arange(seg["start"], seg["end"] + 1)
        elif args.frame_mode == "full":
            window = np.arange(len(df))
        else:
            observe = np.flatnonzero(tasks == OBSERVE_TASK_INDEX)
            if observe.size == 0 or observe[0] != 0 or not np.array_equal(
                    observe, np.arange(observe[0], observe[-1] + 1)):
                raise ValueError(f"episode {episode}: observe phase is not a contiguous prefix")
            window = observe
        frames = window[:: args.frame_stride]

        pooled_frames = []
        rows = [df.iloc[int(f)] for f in frames]
        for start in range(0, len(rows), args.batch_size):
            live_rows = rows[start:start + args.batch_size]
            padded, live = fixed._pad_batch(live_rows, args.batch_size)
            observations = []
            for row in padded:
                frame = int(row["frame_index"])
                observation, _ = runtime.observation(row, frame, PROMPT)
                observations.append(observation)
            batched = runtime.batch_observations(observations)
            fresh_m0 = runtime.model.memory.init_state(args.batch_size)
            output = runtime._interface(batched, fresh_m0)
            jax.block_until_ready(output)
            pooled = fixed._online_pool(output["write_tokens"])
            pooled_frames.extend(pooled[:live])
        feats = np.stack(pooled_frames).astype(np.float32)
        episode_mean = np.mean(feats, axis=0, dtype=np.float32)
        all_feats.append(episode_mean)

        if args.save_frame_scores:
            frame_fresh = fresh.scores(feats.astype(np.float64))
            frame_logits = np.asarray(online(jnp.asarray(feats)))
            frame_online = (frame_logits[:, 1] - frame_logits[:, 0]).astype(np.float64)
            np.savez(out / f"episode_{episode:06d}_frame_scores.npz",
                     frames=frames.astype(np.int32),
                     task_index=tasks[frames].astype(np.int16),
                     fresh_scores=frame_fresh,
                     online_scores=frame_online,
                     label=np.int64(0 if side == "left" else 1))

        fresh_logit = float(fresh.scores(episode_mean[None, :])[0])
        logits = np.asarray(online(jnp.asarray(episode_mean[None, :].astype(np.float32))))
        if logits.shape != (1, 2) or not np.all(np.isfinite(logits)):
            raise ValueError(f"episode {episode}: invalid online-head logits {logits}")
        online_logit = float(logits[0, 1] - logits[0, 0])
        episodes.append({
            "episode": episode,
            "side": side,
            "label": label,
            "window_frames_total": int(window.size),
            "window_frames_used": int(frames.size),
            "fresh_logit_right_minus_left": fresh_logit,
            "fresh_prediction": "right" if fresh_logit > 0 else "left",
            "fresh_correct": bool((fresh_logit > 0) == bool(label)),
            "online_logit_right_minus_left": online_logit,
            "online_prediction": "right" if online_logit > 0 else "left",
            "online_correct": bool((online_logit > 0) == bool(label)),
        })
        print(f"ep{episode:02d} {side:>5}: fresh={fresh_logit:+8.3f} "
              f"({'ok' if episodes[-1]['fresh_correct'] else 'WRONG'})  "
              f"online={online_logit:+8.3f} ({'ok' if episodes[-1]['online_correct'] else 'WRONG'})",
              flush=True)

    def _balanced(key):
        accs = []
        for value in (0, 1):
            group = [e for e in episodes if e["label"] == value]
            accs.append(np.mean([e[key] for e in group]) if group else np.nan)
        return float(np.mean(accs))

    summary = {
        "schema": "openpi.v34.banana0630_writer_probe.v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(args.checkpoint.name),
        "parameter_source": args.parameter_source,
        "probe_artifact_dir": str(args.probe_artifact_dir),
        "banana_root": str(args.banana_root),
        "prompt": PROMPT,
        "frame_stride": args.frame_stride,
        "batch_size": args.batch_size,
        "fresh_m0_zero_writes": True,
        "n_episodes": len(episodes),
        "n_left": sum(1 for e in episodes if e["label"] == 0),
        "n_right": sum(1 for e in episodes if e["label"] == 1),
        "fresh_accuracy": float(np.mean([e["fresh_correct"] for e in episodes])),
        "fresh_balanced_accuracy": _balanced("fresh_correct"),
        "online_accuracy": float(np.mean([e["online_correct"] for e in episodes])),
        "online_balanced_accuracy": _balanced("online_correct"),
        "episodes": episodes,
    }
    suffix = f"_{args.episodes.replace(',', '-')}" if args.episodes else ""
    np.savez(out / f"features{suffix}.npz",
             episode_ids=np.asarray([e["episode"] for e in episodes], dtype=np.int32),
             episode_labels=np.asarray([e["label"] for e in episodes], dtype=np.int64),
             episode_writer_observe=np.stack(all_feats))
    summary["frame_mode"] = args.frame_mode
    (out / f"summary{suffix}.json").write_text(json.dumps(summary, indent=1))
    (out / f"COMPLETE{suffix}").write_text("ok\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
