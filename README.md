# memory_project

Workspace for fine-tuning Physical Intelligence's **π₀.₅ (pi-0.5)** on YAM single-arm datasets.

See [`plan_claude.md`](./plan_claude.md) for the running execution plan and status.

## Layout

```
memory_project/
├── dataset/CUP_TASK_0428/   # raw YAM teleop demos for the cup-grasping task (40 episodes, 30 Hz, 3 cameras)
├── dataset/PLATE_TASK_1/    # raw YAM teleop demos for the plate-memory task (37 long episodes, 30 Hz)
├── openpi/                  # cloned openpi repo + venv (gitignored)
├── i2rt/                    # cloned i2rt repo (YAM URDF, FK, MuJoCo SimRobot)
├── eval/                    # offline-eval scripts + outputs (npz, figs)
├── retriever/               # coarse-to-fine attention retriever (SigLIP + cross-attn reranker)
├── rewriter/                # Gemini VLM prompt rewriter over retrieved keyframes
├── memory_writer/           # attention-distilled keyframe writer (pi0.5 teacher → CLIP+MLP student)
├── test/                    # standalone validation scripts
└── plan_claude.md           # living plan & status tracker
```

## Install

The project depends on two upstream repos cloned as siblings of this one — **openpi** (training/inference) and **i2rt** (YAM driver, URDF, MuJoCo sim). Each ships its own venv; this project reuses the openpi venv for everything.

Tested on Ubuntu 22.04 with CUDA 12.x and a single NVIDIA GPU (≥ 24 GB VRAM for training, any modern GPU for inference). Python 3.11.

### 1. Clone

`openpi/` and `i2rt/` are vendored as subdirectories of this repo, so a single clone gives you everything:

```bash
cd /home/kewalk
git clone <this repo> memory_project
cd memory_project
```

### 2. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 3. Set up the openpi venv (training, eval, attention viz, serve)

```bash
cd /home/kewalk/memory_project/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Creates `openpi/.venv/` with JAX + Flax + LeRobot + transformers. Pretrained π₀.₅ weights download on first use into `~/.cache/openpi/`.

### 4. Install i2rt into the same venv (sim playback, FK, real-robot driver)

```bash
source /home/kewalk/memory_project/openpi/.venv/bin/activate
cd /home/kewalk/memory_project/i2rt
sudo apt install -y build-essential python3-dev linux-headers-$(uname -r)
uv pip install -e .
```

### 5. Extra deps used by `eval/` scripts

Usually pulled in transitively. If any are missing:

```bash
uv pip install imageio[pyav] scipy matplotlib tqdm einops mujoco
```

### 6. (Real-robot only) CAN bus

```bash
sudo ip link set can0 up type can bitrate 1000000             # one-shot
sudo sh /home/kewalk/memory_project/i2rt/devices/install_devices.sh   # auto-up on boot
```

### 7. Verify

```bash
source /home/kewalk/memory_project/openpi/.venv/bin/activate
python -c "import jax; print('jax devices:', jax.devices())"
python -c "from openpi.training.config import get_config; print(get_config('pi05_yam_cup_lora').name, get_config('pi05_plate_task').name)"
python -c "from i2rt.robots.get_robot import get_yam_robot; print('i2rt OK')"
```

You should see a `gpu(id=0)` device and no import errors.

## Setup

Activate the openpi venv before running anything:

```bash
source /home/kewalk/memory_project/openpi/.venv/bin/activate
```

## Cup task

Single-arm pick-and-place. 40 teleop demos at 30 Hz with three cameras (top, left wrist, right wrist) and 7-DoF joint-position actions. All three image slots are populated; the prompt is a fixed string ("pick up the cup with yellow object inside").

### Convert raw cup data → LeRobot dataset

Smoke test on a single demo first:

```bash
cd /home/kewalk/memory_project/openpi
python examples/yam/convert_cup_to_lerobot.py \
    --data-dir /home/kewalk/memory_project/dataset/CUP_TASK_0428 \
    --limit 1
```

Full conversion of all 40 demos:

```bash
cd /home/kewalk/memory_project/openpi
python examples/yam/convert_cup_to_lerobot.py \
    --data-dir /home/kewalk/memory_project/dataset/CUP_TASK_0428
```

### Compute normalization stats

```bash
cd /home/kewalk/memory_project/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_yam_cup_lora
```

### Train (LoRA fine-tune of π₀.₅)

Smoke run (200 steps) before committing to the full run:

```bash
cd /home/kewalk/memory_project/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/train.py pi05_yam_cup_lora \
    --exp-name=cup_smoke_test --num-train-steps=200 --overwrite
```

Full run (~12–24 h on a 4090):

```bash
cd /home/kewalk/memory_project/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/train.py pi05_yam_cup_lora \
    --exp-name=cup_v1 --overwrite
```

Checkpoints land in `openpi/checkpoints/pi05_yam_cup_lora/<exp-name>/`.

### Offline eval — replay a demo through the trained policy

```bash
cd /home/kewalk/memory_project
python eval/offline_eval.py --demo demo1
```

Saves `eval/results_demo1.npz` (state, gt_action, pred_first, pred_chunks). Prints per-joint MAE/RMSE/max vs ground truth.

Flags:
- `--demo demo5` — pick a different episode
- `--ckpt-step 25000` — load an earlier checkpoint (default 29999)
- `--max-steps 20` — quick smoke test on a slice

### Sim playback — drive YAM `SimRobot` from saved predictions

```bash
cd /home/kewalk/memory_project
python eval/sim_playback.py --demo demo1
```

Opens a passive MuJoCo viewer and executes the predicted actions in real time (30 fps). Default `--chunk-stride 10` plays each predicted action chunk in full (smooth); use `--chunk-stride 1` for per-step replay (jittery, mirrors live policy.infer at every step).

Flags:
- `--no-viewer` — headless run (logs sim trajectory to `eval/sim_<demo>.npz`)
- `--speed 2.0` — 2× real-time
- press **`R`** inside the viewer → uncheck **Skybox** for a flat background

### FK + plot + error metrics

```bash
cd /home/kewalk/memory_project
python eval/plot_eef.py --demo demo1
```

Loads `eval/results_<demo>.npz` (and `eval/sim_<demo>.npz` if it exists), runs YAM FK, prints per-joint MAE/RMSE/max and EEF mean/max in mm, saves three figures under `eval/figs/`:

- `<demo>_eef.png` — xyz time-series + 3D EEF trajectory (state / gt / pred / sim)
- `<demo>_gripper.png` — gripper joint angle over time
- `<demo>_error.png` — per-joint absolute error + EEF position error per step

### Attention visualization video

```bash
cd /home/kewalk/memory_project

# Recommended: language→image grounding (cup + object localize)
python eval/attention_viz.py --demo demo1 --queries prompt --layers 8,9,10,11

# Action-query attention (arm/gripper-dominant; useful for "what drives execution")
python eval/attention_viz.py --demo demo1 --queries actions --layers 14,15,16,17

# Smoke test on a slice
python eval/attention_viz.py --demo demo1 --queries prompt --layers 8,9,10,11 --max-steps 20
```

Captures attention from prompt tokens (or action tokens) onto image patches at every frame, overlays heatmaps on the 3 cameras, encodes to `eval/figs/attention_<demo>_q-<queries>.mp4`. Requires the gemma.py `self.sow` edit to be in place.

Flags:
- `--queries {prompt,actions}` — which token group to slice queries from. **`prompt` + middle layers (8–11)** is where PaliGemma's language→image grounding lives; **`actions` + late layers (14–17)** shows what the action heads attend to (gripper/arm). Default `actions`.
- `--layers 8,9,10,11` — comma-separated layer indices to average (all 18 transformer layers available).
- `--infer-every K` — run inference every K frames; reuse the heatmap in between (default 10, ~10× speedup).
- `--max-steps N` — quick smoke on the first N frames.

### Serve the trained policy (real-robot use)

WebSocket inference server on port 8000. Run on the workstation; the robot computer connects over Tailscale (`tailscale ip -4` on the workstation gives the host).

```bash
cd /home/kewalk/memory_project/openpi
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_yam_cup_lora \
    --policy.dir=checkpoints/pi05_yam_cup_lora/cup_0429_v1/29999
```

Run inside `tmux new -s server` so it survives terminal disconnects. From the robot computer, sanity-check connectivity with `curl http://<workstation_tailscale_ip>:8000/healthz`.

## Plate task

Long-horizon plate-memory task. Each demo is one long episode with `observe → mix → delay → query1..4 → return_home` segments. Top + left cameras carry the current view; `right_wrist_0_rgb` is zero-padded and masked off (single-arm YAM, no real right wrist). The per-frame instruction already names the resolved object (e.g. *"put banana on the light blue plate"*) — generated from each demo's `metadata.json` `episode_plan` at training time, and produced by an upstream VLM at inference (oracle / manual until a retriever is built). Demos 1-35 train, 36-37 held out.

### Convert raw plate data → LeRobot dataset

Smoke test on a single demo first:

```bash
cd /home/kewalk/memory_project/openpi
uv run examples/yam/convert_plate_task_to_lerobot.py \
    --data-dir /home/kewalk/memory_project/dataset/PLATE_TASK_1 \
    --limit 1
```

Full conversion of demos 1-35:

```bash
cd /home/kewalk/memory_project/openpi
uv run examples/yam/convert_plate_task_to_lerobot.py \
    --data-dir /home/kewalk/memory_project/dataset/PLATE_TASK_1 \
    --push_to_hub
```

Append `--push-to-hub` to also publish to `kewalk123/plate_task` on HF (required for Modal training; optional locally).

### Compute normalization stats

```bash
cd /home/kewalk/memory_project/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_plate_task
```

### Train (LoRA fine-tune of π₀.₅)

Smoke run (200 steps):

```bash
cd /home/kewalk/memory_project/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/train.py pi05_plate_task \
    --exp-name=plate_smoke --num-train-steps=200 --overwrite
```

Full run (~12–24 h on a 4090):

```bash
cd /home/kewalk/memory_project/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/train.py pi05_plate_task \
    --exp-name=plate_oracle_v2_0510 --overwrite
```

Checkpoints land in `openpi/checkpoints/pi05_plate_task/<exp-name>/`.

### Serve the trained policy (real-robot use)

WebSocket inference server on port 8000. Run on the workstation; the robot computer connects over Tailscale (`tailscale ip -4` on the workstation gives the host).

```bash
cd /home/kewalk/memory_project/openpi
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_plate_task \
    --policy.dir=checkpoints/pi05_plate_task/plate_oracle_v2_0510/<step>
```

Confirm via the log lines:
- `Restoring checkpoint from .../plate_oracle_v2_0510/<step>/params`
- `Loaded norm stats from .../pi05_plate_task/...`
- `server listening on 0.0.0.0:8000`

Run inside `tmux new -s server` so it survives terminal disconnects. From the robot computer, sanity-check connectivity with `curl http://<workstation_tailscale_ip>:8000/healthz`.

### Realtime eval — robot computer

Three eval clients live in `eval/real/`. All connect to the policy server via WebSocket and read the YAM joint state + 3 RealSense streams. Run each on the **robot computer** (not the workstation that's serving the policy). Stop signals are identical: `'q'` in the pygame window, Ctrl-C (twice = hard kill), or `--max-steps` / `--max-time-s` timeouts.

| Script | Prompt source |
|---|---|
| `run_eval.py` | Static `--prompt`, never updated. Baseline. |
| `run_eval_fix_image.py` | Press `n` to advance stage 0→7. Stages 3-6 rewrite with the **first top-camera frame** (snapshot at preflight); stage 7 rewrites with the current frame; stages 1, 2 leave the prompt untouched. No retriever / SigLIP. |
| `run_eval_retriever.py` | Press `n` to advance stage 0→7. Builds a live SigLIP keyframe bank from the top camera at `--memory-sample-hz` (default 1.0) **across all stages**; on each `n` past stage 0, retriever picks top-1 from the bank, rewriter generates a new prompt. |

#### Prereqs on the robot computer

1. **openpi venv + i2rt + CAN bus** — install steps 3, 4, 6 from the [Install](#install) section above. `run_eval_retriever.py` additionally loads PaliGemma SigLIP locally, so the robot box needs a CUDA GPU (≥ 8 GB VRAM free; tested on a 4090).

2. **Activate venv:**
   ```bash
   source /home/kewalk/memory_project/openpi/.venv/bin/activate
   cd /home/kewalk/memory_project
   ```

3. **Camera serials** in `eval/real/configs/yam_eval.yaml` must match the RealSenses plugged into this box. Defaults are set for the workstation rig — edit if needed:
   ```bash
   ls /dev/v4l/by-id/ | grep RealSense   # quick sanity check
   ```

4. **(retriever variant only) Local PaliGemma + retriever checkpoints.** Defaults in `eval/real/configs/yam_eval.yaml`:
   ```yaml
   retriever:
     ckpt: retriever/runs/v1_0511/best.pt
     openpi_ckpt_dir: openpi/checkpoints/pi05_plate_task/plate_oracle_v2_0510/29999
   ```
   Both paths must exist locally. Copy from the workstation if needed (e.g. `rsync -av <workstation>:/home/kewalk/memory_project/retriever/runs/v1_0511 retriever/runs/`).

5. **(fix-image + retriever variants only) Gemini API key** for the rewriter:
   ```bash
   export GEMINI_API_KEY="..."
   ```
   Add to `~/.bashrc` to persist.

6. **Server reachable** from this box. Check before each rollout:
   ```bash
   curl http://<workstation_tailscale_ip>:8000/healthz
   ```

#### Run

Always start with `--dry-run` once per session to validate the full path (preflight + 1 inference + retrieval pipeline) before the arm moves:

```bash
python eval/real/run_eval_retriever.py \
    --host <workstation_tailscale_ip> \
    --prompt "Place the object originally on the pink plate onto the pink plate." \
    --dry-run
```

Then live:

```bash
python eval/real/run_eval_retriever.py \
    --host <workstation_tailscale_ip> \
    --prompt "Place the object originally on the pink plate onto the pink plate."
```

Click the small pygame window once to give it focus, then:
- `n` — advance stage (0→1→…→7, clamped). Past stage 0 each press triggers retriever + rewriter; the new prompt is logged and used for the next chunk's `policy.infer`.
- `q` — graceful stop (holds last pose ~0.5 s, closes cameras, flushes logs).

The fix-image variant is a drop-in replacement of the script name; same flags otherwise:

```bash
python eval/real/run_eval_fix_image.py \
    --host <workstation_tailscale_ip> \
    --prompt "Place the object originally on the pink plate onto the pink plate."
```

#### Logs

Each rollout writes to `eval/real/runs/<timestamp>/`:
- `top_camera_rgb.mp4` / `left_camera_rgb.mp4` / `right_camera_rgb.mp4`
- `events.log` — every stage transition, retriever pick, rewriter output, server `infer_ms`
- `chunks/`, `frames/` — per-chunk JPGs + action arrays
- `metadata.json` — CLI args, prompt, server metadata, stop reason
- (`run_eval_fix_image.py` only) `first_image.jpg` — the snapshot fed to the rewriter for query stages

Useful flags (all three scripts):
- `--max-steps 600 --max-time-s 30` — cap rollout length.
- `--no-video` / `--no-save-frames` — skip MP4 / JPG dumps.
- `--chunk-len 5` — execute only the first 5 of each 10-step action chunk before re-inferring (more reactive, more server traffic).

Retriever-only flags:
- `--memory-sample-hz 0.5` — bank sample rate (default 1.0; bank caps at `N_MAX=240` in `retriever/dataset.py`).
- `--retriever-ckpt`, `--openpi-ckpt-dir`, `--rewriter-model`, `--retriever-device` — override the yaml.

## Memory writer (attention-distilled keyframe selector)

Two-stage method. **Offline teacher**: replay each demo through π₀.₅ every 30 frames (1 Hz at 30 Hz video), extract middle-layer language→image attention (layers 8-11, top camera, prompt-pooled → 256-D), and run a greedy attention-change loop to produce pseudo write/skip labels. **Online student**: a frozen CLIP ViT-B/32 image+text encoder feeds a tiny MLP head that predicts write probability — no π₀.₅ at inference time. Standalone module under `memory_writer/`; independent of `retriever/` and `rewriter/`.

Plate is the primary testbed (long episodes, memory-relevant); cup is a single-instruction sanity baseline. Activate the openpi venv first:

```bash
source /home/kewalk/memory_project/openpi/.venv/bin/activate
cd /home/kewalk/memory_project
```

### 1. Teacher pass — extract π₀.₅ attention per sampled frame

Heavy step (~5 s per π₀.₅ forward → ~10 min per plate demo at 1 Hz sampling, ~6 h for all 37 demos on a 4090). Saves `<demo>.npz` (signatures + visual features) and `<demo>_labels.npz` (pseudo write labels) under `memory_writer/cache/teacher/<task>/`. The CLI skips per-demo if both files already exist, so it's safe to interrupt and resume — run inside `tmux new -s teacher`.

Smoke on one demo first:

```bash
python -m memory_writer.build_teacher --task plate --demos 36 \
    --ckpt /home/kewalk/memory_project/openpi/checkpoints/pi05_plate_task/plate_oracle_v2_0510/29999 \
    --config pi05_plate_task
```

Then sanity-check the output:

```bash
python -c "from memory_writer.labels import inspect; inspect('memory_writer/cache/teacher/plate/demo36.npz')"
```

Expect S ≈ 100 sampled frames (≈ 2900 / 30), signature rows sum to 1, multiple unique prompts, positive fraction in [10%, 40%].

Full plate run (resumable — already-cached demos are skipped):

```bash
python -m memory_writer.build_teacher --task plate --demos 1-37 \
    --ckpt /home/kewalk/memory_project/openpi/checkpoints/pi05_plate_task/plate_oracle_v2_0510/29999 \
    --config pi05_plate_task
```

Cup equivalent uses `--task cup --config pi05_yam_cup_lora` and `checkpoints/pi05_yam_cup_lora/cup_0429_v1/29999`.

Flags:
- `--sample-every K` — π₀.₅ forward stride in frames (default 10).
- `--attention-threshold T` — cosine-distance cutoff for the pseudo-label loop (default 0.25).
- `--min-gap-steps N` — min raw-frame gap between consecutive kept keyframes (default 15 for plate, 10 for cup).
- `--overwrite` — redo demos that already have cache + labels.

### 2. CLIP feature cache — student-facing image+text features

Downloads `openai/clip-vit-base-patch32` on first run. Reads each demo's teacher cache and writes parallel `<demo>.npz` files of CLIP image + per-frame prompt embeddings under `memory_writer/cache/clip/<task>/`.

```bash
python -m memory_writer.build_clip_cache --task plate --demos 1-37
```

Fast (~1 min total for plate on a 4090). After this, training never touches images directly — all features are cached.

### 3. Train the student

BCE on pseudo write labels + λ·MSE regression to the teacher attention-change score. AdamW + WeightedRandomSampler keeps positive/negative balanced. Frozen encoders; only the ~470 k-param head is trained.

```bash
python -m memory_writer.train --task plate \
    --train-demos 1-35 --val-demos 36-37 \
    --out memory_writer/cache/student_runs/plate_v1
```

Saves `best.pt` (by val F1), `last.pt`, `train_log.jsonl`, and `args.json`. Converges in a few minutes.

Flags:
- `--epochs 30 --batch-size 256 --lr 3e-4 --lambda-score 0.5` — defaults.
- `--hidden 256` — MLP width.

### 4. Run the student on held-out demos → memory bank

Student-only: frozen CLIP + trained head, no π₀.₅. Replays each demo, emits an `AttentionMemoryBank` `.npz` plus per-keyframe PNG dumps under `memory_writer/cache/debug/<task>/<demo>/`.

```bash
python -m memory_writer.run_online --task plate --demos 36-37 \
    --ckpt memory_writer/cache/student_runs/plate_v1/best.pt \
    --out memory_writer/cache/banks/plate \
    --debug-dir memory_writer/cache/debug/plate
```

Flags:
- `--write-threshold 0.5` — sigmoid cutoff for write decisions.
- `--min-gap-steps 10` — min raw-frame gap.
- `--check-every-n-steps 3` — student is invoked every N frames (cheaper).
- `--redundancy-threshold 0.95` — skip if cosine-sim to last kept kf exceeds this.
- `--no-save-images` — skip the PNG dumps (banks only).

### 5. Evaluate the bank

Generic metrics (size, per-stage coverage, temporal coverage, visual diversity, redundancy) and teacher-agreement P/R/F1 if a teacher cache is provided.

```bash
python -m memory_writer.eval_bank \
    --bank memory_writer/cache/banks/plate/demo36.npz \
    --teacher memory_writer/cache/teacher/plate/demo36.npz \
    --out memory_writer/cache/banks/plate/demo36_eval.json
```

### 6. Debug timeline plot

Overlays student write events (colored by reason) on the teacher attention-change curve.

```bash
python -m memory_writer.viz \
    --bank memory_writer/cache/banks/plate/demo36.npz \
    --teacher memory_writer/cache/teacher/plate/demo36.npz \
    --out memory_writer/cache/debug/plate/demo36_timeline.png
```
