"""Writer-probe transfer eval on the 0830_eval collection (fresh 0816-setup episodes).

The decisive generalization test the June-30 data could not provide: 10 never-trained episodes
recorded in the SAME setup/vocabulary as training (open both lids -> inspect both bins ->
close/reset; no terminal open, so each episode is scored under BOTH instructions):

* prompt "find the banana"        -> truth = the banana's bin side;
* prompt "find the grey pepper box" -> truth = the opposite side.

Per episode x prompt x checkpoint this evaluator extracts writer features (exact run5
transforms, fresh M0, zero writes) on the task_index==4 inspect frames AND a matched
equal-length approach window immediately before them (phase control), then scores episode-mean
features with the checkpoint's fit-all-56 fresh head and its stored online head.

Banana sides were determined visually from mid-inspect frames (see the companion grid in the
report folder): episodes 0-4 banana LEFT, 5-9 banana RIGHT.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v34_fixed_writer_probe_eval as fixed  # noqa: E402
import v34_banana0630_writer_probe_eval as ban  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pandas as pd  # noqa: E402

INSPECT_TASK = 4
PROMPTS = ("find the banana", "find the grey pepper box")
BANANA_SIDE = {0: "left", 1: "left", 2: "left", 3: "left", 4: "left",
               5: "right", 6: "right", 7: "right", 8: "right", 9: "right"}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe-artifact-dir", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path,
                        default=Path("/iris/u/kewalk/.cache/huggingface/lerobot/yam/bin_memory_0830_eval_subtask"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", default=fixed.RUN5_CONFIG)
    parser.add_argument("--parameter-source", choices=("raw", "ema"), default="raw")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--full-frame-scores", action="store_true",
                        help="score EVERY frame under both prompts and save per-episode npz (videos)")
    parser.add_argument("--episodes", default="", help="comma-separated episode subset")
    return parser.parse_args(argv)


def _features(runtime, rows, prompt, batch_size):
    pooled = []
    for start in range(0, len(rows), batch_size):
        live_rows = rows[start:start + batch_size]
        padded, live = fixed._pad_batch(live_rows, batch_size)
        obs = [runtime.observation(row, int(row["frame_index"]), prompt)[0] for row in padded]
        batched = runtime.batch_observations(obs)
        out = runtime._interface(batched, runtime.model.memory.init_state(batch_size))
        jax.block_until_ready(out)
        pooled.extend(fixed._online_pool(out["write_tokens"])[:live])
    return np.stack(pooled).astype(np.float32)


def main(argv=None) -> int:
    args = _parse_args(argv)
    out = args.output_dir / args.parameter_source
    if (out / "COMPLETE").exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    runtime_args = fixed.Args(checkpoint=args.checkpoint, dataset_root=ban.TRAIN_DATASET_ROOT,
                              output_dir=out / "_runtime_preflight", config=args.config,
                              parameter_source=args.parameter_source, batch_size=args.batch_size)
    runtime = fixed._Run5Runtime(runtime_args)
    fresh = ban._fresh_head(args.probe_artifact_dir)
    online = runtime._online_writer_head

    def score(feats):
        mean = np.mean(feats, axis=0, dtype=np.float32)
        fr = float(fresh.scores(mean[None, :])[0])
        lg = np.asarray(online(jnp.asarray(mean[None, :])))
        return fr, float(lg[0, 1] - lg[0, 0])

    parquets = sorted((args.eval_root / "data" / "chunk-000").glob("episode_*.parquet"))
    if args.episodes:
        wanted = {int(x) for x in args.episodes.split(",")}
        parquets = [q for q in parquets if int(q.stem.split("_")[-1]) in wanted]
    records = []
    for path in parquets:
        episode = int(path.stem.split("_")[-1])
        df = pd.read_parquet(path, columns=["image", "left_wrist_image", "right_wrist_image",
                                            "state", "task_index", "frame_index"])
        tasks = df["task_index"].to_numpy()
        inspect = np.flatnonzero(tasks == INSPECT_TASK)
        if inspect.size == 0 or not np.array_equal(inspect, np.arange(inspect[0], inspect[-1] + 1)):
            raise ValueError(f"episode {episode}: no contiguous inspect block")
        count = inspect.size
        if inspect[0] < count:
            raise ValueError(f"episode {episode}: not enough pre-inspect frames for the control")
        approach = np.arange(inspect[0] - count, inspect[0])
        banana = BANANA_SIDE[episode]
        for prompt in PROMPTS:
            truth = banana if prompt == PROMPTS[0] else ("left" if banana == "right" else "right")
            label = 1 if truth == "right" else 0
            if args.full_frame_scores:
                frames = np.arange(len(df))
                feats = _features(runtime, [df.iloc[int(f)] for f in frames], prompt, args.batch_size)
                fs = fresh.scores(feats.astype(np.float64))
                lg = np.asarray(online(jnp.asarray(feats)))
                os_ = (lg[:, 1] - lg[:, 0]).astype(np.float64)
                tag = "banana" if prompt == PROMPTS[0] else "pepper"
                np.savez(out / f"episode_{episode:06d}_{tag}_frame_scores.npz",
                         frames=frames.astype(np.int32), task_index=tasks.astype(np.int16),
                         fresh_scores=fs, online_scores=os_, label=np.int64(label))
                ins_feats, app_feats = feats[inspect], feats[approach]
            else:
                ins_feats = _features(runtime, [df.iloc[int(f)] for f in inspect], prompt, args.batch_size)
                app_feats = _features(runtime, [df.iloc[int(f)] for f in approach], prompt, args.batch_size)
            fi, oi = score(ins_feats)
            fa, oa = score(app_feats)
            records.append({
                "episode": episode, "prompt": prompt, "banana_side": banana, "truth": truth,
                "label": label, "inspect_frames": int(count),
                "fresh_inspect": fi, "online_inspect": oi,
                "fresh_approach": fa, "online_approach": oa,
                "fresh_correct": bool((fi > 0) == bool(label)),
                "online_correct": bool((oi > 0) == bool(label)),
            })
            print(f"ep{episode} {'BAN' if prompt == PROMPTS[0] else 'PEP'} truth={truth:<5} "
                  f"fresh={fi:+7.2f}({'ok' if records[-1]['fresh_correct'] else 'X'}) "
                  f"online={oi:+7.3f}({'ok' if records[-1]['online_correct'] else 'X'}) "
                  f"approach: fresh={fa:+7.2f} online={oa:+7.3f}", flush=True)

    def bal(key, subset):
        accs = []
        for value in (0, 1):
            group = [r for r in subset if r["label"] == value]
            accs.append(np.mean([r[key] for r in group]) if group else np.nan)
        return float(np.mean(accs))

    flips = []
    for episode in sorted({r["episode"] for r in records}):
        pair = {r["prompt"]: r for r in records if r["episode"] == episode}
        if len(pair) == 2:
            a, b = pair[PROMPTS[0]], pair[PROMPTS[1]]
            flips.append({"episode": episode,
                          "fresh_delta_banana_minus_pepper": a["fresh_inspect"] - b["fresh_inspect"],
                          "online_delta_banana_minus_pepper": a["online_inspect"] - b["online_inspect"],
                          "expected_delta_sign": "negative" if a["banana_side"] == "left" else "positive"})
    suffix = f"_{args.episodes.replace(',', '-')}" if args.episodes else ""
    summary = {
        "schema": "openpi.v34.eval0830_writer_probe.v1",
        "checkpoint_step": int(args.checkpoint.name),
        "parameter_source": args.parameter_source,
        "banana_sides": BANANA_SIDE,
        "n_trials": len(records),
        "fresh_balanced_accuracy": bal("fresh_correct", records),
        "online_balanced_accuracy": bal("online_correct", records),
        "fresh_bal_acc_banana_prompt": bal("fresh_correct", [r for r in records if r["prompt"] == PROMPTS[0]]),
        "fresh_bal_acc_pepper_prompt": bal("fresh_correct", [r for r in records if r["prompt"] == PROMPTS[1]]),
        "online_bal_acc_banana_prompt": bal("online_correct", [r for r in records if r["prompt"] == PROMPTS[0]]),
        "online_bal_acc_pepper_prompt": bal("online_correct", [r for r in records if r["prompt"] == PROMPTS[1]]),
        "prompt_flips": flips,
        "records": records,
    }
    (out / f"summary{suffix}.json").write_text(json.dumps(summary, indent=1))
    (out / f"COMPLETE{suffix}").write_text("ok\n")
    print(json.dumps({k: v for k, v in summary.items() if "accuracy" in k or k == "n_trials"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
