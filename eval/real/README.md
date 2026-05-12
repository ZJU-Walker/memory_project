# Real-robot deploy — `eval/real/`

Two clients live here:

- **`run_eval.py`** — CUP single-task client (3 cameras, fixed prompt, 10-step chunks).
- **`run_long_task.py`** — long-horizon plate-memory client (2 cameras + memory channel, 8 stages, 50-step chunks). **This README focuses on the long-task flow**; CUP usage is unchanged from prior versions.

Both are pure inference clients: they collect observations on the robot computer and send them over WebSocket to a `serve_policy.py` running on a workstation. Neither client loads model weights.

---

## Long-task client overview

The trained policy `pi05_long_task_mem_lora` operates on an 8-stage episode driven by the operator:

| stage_id | name | instruction template | memory |
|---|---|---|---|
| 0 | observe     | "Observe and remember which object is on each colored plate." | OFF |
| 1 | mix         | "Move all objects to the right side of the table." | OFF |
| 2 | delay       | "Wait briefly before continuing." | OFF |
| 3..6 | query1..4 | "Place the object originally on the {src} plate onto the {tgt} plate." | ON |
| 7 | return_home | "Return arm to home position." | OFF |

Live cameras: top + left wrist only. The policy's `right_wrist_0_rgb` slot is filled at training time by `YamLongTaskInputs` with `observation/memory_image`, which the deploy `StageManager` owns:

- Stage 0..2, 7: `memory_image = zeros(480,640,3)`, `has_memory = 0.0`.
- Stage 3..6 (queries): `memory_image = observe-stage keyframe`, `has_memory = 1.0`. The keyframe is the live top frame captured at the moment the operator presses `n` to leave stage 0.

---

## Running

### 1. Workstation (separate machine — runs the model)

```bash
cd /home/kewalk/memory_project/openpi
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_long_task_mem_lora \
    --policy.dir=checkpoints/pi05_long_task_mem_lora/long_v1/<step>
```

Confirm in the log:

- WebSocket listening on port `8000`.
- Norm-stats line ending in `pi05_long_task_mem_lora/...`.

Note the workstation's Tailscale IP (`tailscale ip -4`) — you'll pass it to the client as `--host`.

### 2. Robot computer (this directory)

CAN bus up:
```bash
sudo ip link set can_left up type can bitrate 1000000
```

Activate the venv:
```bash
source /home/david/openpi/.venv/bin/activate
cd /home/david/ke/memory_project
```

Dry-run first (pre-flight + 1 inference, no motion):
```bash
python -m eval.real.run_long_task --host <workstation_tailscale_ip> --dry-run
```

The client prints the required physical setup (plate -> object mapping, query order) and waits for Enter before opening cameras. Place objects exactly as listed. The pre-flight then prints camera shapes, channel means (sanity-check RGB ordering), and the first-inference chunk shape. On a clean dry-run you'll see `latest_top.jpg`, `latest_left.jpg`, `latest_memory.jpg` (zeros at this point) under `eval/real/runs/<ts>_long/`.

Live run (slow-mode, recommended for first runs):
```bash
python -m eval.real.run_long_task --host <ip> --hz 15 --max-joint-delta 0.2
```

Bump `--hz` only after a clean run.

### 3. Stage protocol (operator's view of a live run)

```
[stage] 0 (observe): "Observe and remember which object is on each colored plate."
   ... robot scans / holds ...
   [press n]   <- captures observe keyframe
[stage] 1 (mix): "Move all objects to the right side of the table."  memory=OFF
   ... robot mixes ...
   [press n]
[stage] 2 (delay): "Wait briefly before continuing."  memory=OFF
   [press n]
[stage] 3 (query1): "Place the object originally on the orange plate onto the green plate."  memory=ON
   ... pick + place ...
   [press n]
[stage] 4..6 ...
[stage] 7 (return_home): "Return arm to home position."  memory=OFF
   [press n]   <- triggers graceful shutdown after a short window
[done] log_dir=eval/real/runs/<ts>_long
```

Press `n` in the small pygame window (click it first to give it focus). `n` is a hard interrupt — the current chunk is dropped and the next inference uses the new stage's prompt and memory.

### 4. Abort

| input | effect |
|---|---|
| `q` in pygame window | graceful — finish current chunk, hold last pose 0.5s, exit |
| Ctrl-C | graceful (twice = hard kill) |
| `--max-steps` / `--max-time-s` | timeout exit |
| `pkill -9 -f run_long_task` | last resort from another shell |

The arm holds the last commanded pose on shutdown — it does **not** zero-torque, so a grasped object stays grasped.

---

## CLI flags (long-task client)

| flag | default | notes |
|---|---|---|
| `--host` | required | workstation Tailscale IP |
| `--port` | 8000 | server WebSocket |
| `--config` | `configs/yam_long_task_eval.yaml` | CAN channel + 2 RealSense serials |
| `--query-mode` | `fixed` | `fixed` / `random` / `manual` |
| `--seed` | 0 | only used when `--query-mode random` |
| `--hz` | **15** | slow-mode default; raise after clean run |
| `--chunk-len` | 25 | execute first N of 50 returned actions per inference |
| `--max-joint-delta` | 0.2 | per-step joint movement clip (rad) |
| `--start-steps` | 60 | smooth-move steps to chunk[0] |
| `--max-steps` | 5400 | control-step cap |
| `--max-time-s` | 480 | wall-clock cap (~8 min) |
| `--no-video` | off | skip MP4 recording |
| `--no-save-frames` | off | skip per-chunk JPG dump |
| `--dry-run` | off | pre-flight + 1 inference, no motion |
| `--ckpt-tag` | `""` | free-form tag written to metadata.json |

---

## Log layout

Each run lands in `eval/real/runs/<YYYYMMDD_HHMMSS>_long/`:

```
metadata.json            # CLI args, server metadata, query plan, stage events, git hash
trace.npz                # per control step: t_wall, state, executed_action, chunk_idx, idx_in_chunk, stage_id
chunks.npz               # per inference call: chunk_t_wall, chunk_obs_state, chunk_actions, infer_ms
events.log               # text events (stage transitions, joint-delta clips, infer timings)
top_camera_rgb.mp4       # streamed at native 30 fps from background grabber
left_camera_rgb.mp4
memory_keyframe.jpg      # the observe-stage keyframe used on stages 3..6
frames/
    chunk_NNNN_top.jpg
    chunk_NNNN_left.jpg
    chunk_NNNN_memory.jpg
latest_{top,left,memory}.jpg   # overwritten each chunk for live monitoring
```

Per-step `stage_id` lets you split metrics by stage in post-hoc analysis (mirrors `eval/offline_eval_long.py`).

---

## Query modes

- **`fixed`** (default) — deterministic plate->object mapping and identical 4-query order across runs. Best for clean retrieval-method comparisons.
- **`random`** — seeded random plan; queries cover all 4 source plates exactly once, targets sampled with replacement (matches dataset distribution).
- **`manual`** — operator types the mapping and 4 queries at boot.

Run `python -m eval.real.stage_machine` for a logic-only smoke check (no hardware needed).

---

## Files (long-task)

| file | purpose |
|---|---|
| `configs/yam_long_task_eval.yaml` | CAN channel + top/left RealSense serials |
| `stage_machine.py` | `QueryPlan` + `StageManager` (pure logic, unit-testable) |
| `long_task_env.py` | `LongTaskPolicyEnv` — YAM + 2 cameras + StageManager-driven obs dict |
| `video_recorder.py` | `_CameraStreamer` (shared with CUP client) |
| `logger.py` | `RolloutLogger` (extended; backwards-compat with CUP) |
| `run_long_task.py` | client entrypoint |

---

## Troubleshooting

- **Pre-flight: ABORT — first commanded pose differs by >0.3 rad.** Server is wired to a different checkpoint than expected, the arm isn't at home, or the StageManager is in the wrong stage. Check the printed stage and prompt; if both look right, recheck the served checkpoint dir.
- **Camera channel means look swapped (R << B).** `gello.cameras.realsense_camera.RealSenseCamera.read()` already returns RGB, so this would indicate a code regression in the streamer — re-check `video_recorder.py:90`.
- **`n` press not registering.** The pygame window needs focus — click it once, then press `n`. SSH users need X11 forwarding (`ssh -Y`).
- **YAM error `file descriptor cannot be a negative integer (-1)` on shutdown.** Raise `time.sleep(0.1)` in `LongTaskPolicyEnv.close` to give the i2rt CAN background thread more time to drain.
- **`ModuleNotFoundError: gello`.** Activate the openpi venv (`source /home/david/openpi/.venv/bin/activate`) — that's where gello is installed.
