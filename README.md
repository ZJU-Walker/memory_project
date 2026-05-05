# memory_project

Workspace for fine-tuning Physical Intelligence's **π₀.₅ (pi-0.5)** on a YAM single-arm dataset.

See [`plan_claude.md`](./plan_claude.md) for the running execution plan and status.

## Layout

```
memory_project/
├── dataset/CUP_TASK_0428/   # raw YAM teleop demos (40 episodes, 30 Hz, 3 cameras)
├── openpi/                  # cloned openpi repo + venv (gitignored)
├── i2rt/                    # cloned i2rt repo (YAM URDF, FK, MuJoCo SimRobot)
├── eval/                    # offline-eval scripts + outputs (npz, figs)
├── test/                    # standalone validation scripts
├── test.py                  # quick scratch / probe script
└── plan_claude.md           # living plan & status tracker
```

## Install

The project depends on two upstream repos cloned as siblings of this one — **openpi** (training/inference) and **i2rt** (YAM driver, URDF, MuJoCo sim). Each ships its own venv; this project reuses the openpi venv for everything.

Tested on Ubuntu 22.04 with CUDA 12.x and a single NVIDIA GPU (≥ 24 GB VRAM for training, any modern GPU for inference). Python 3.11.

### 1. Clone

```bash
cd /home/kewalk
git clone <this repo> memory_project
cd memory_project
git clone https://github.com/Physical-Intelligence/openpi.git
git clone https://github.com/i2rt-robotics/i2rt.git
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

This creates `openpi/.venv/` with JAX + Flax + LeRobot + transformers and downloads pretrained π₀.₅ weights on first use into `~/.cache/openpi/`.

### 4. Install i2rt into the same venv (sim playback, FK, real-robot driver)

```bash
source /home/kewalk/memory_project/openpi/.venv/bin/activate
cd /home/kewalk/memory_project/i2rt
sudo apt install -y build-essential python3-dev linux-headers-$(uname -r)
uv pip install -e .
```

### 5. Extra deps used by `eval/` scripts

These are pulled in transitively by openpi/i2rt, but if any are missing:

```bash
uv pip install imageio[pyav] scipy matplotlib tqdm einops mujoco
```

### 6. (Real-robot only) CAN bus

```bash
sudo ip link set can0 up type can bitrate 1000000   # one-shot
sudo sh /home/kewalk/memory_project/i2rt/devices/install_devices.sh   # auto-up on boot
```

### 7. Verify

```bash
source /home/kewalk/memory_project/openpi/.venv/bin/activate
python -c "import jax; print('jax devices:', jax.devices())"
python -c "from openpi.training.config import get_config; print(get_config('pi05_yam_cup_lora').name)"
python -c "from i2rt.robots.get_robot import get_yam_robot; print('i2rt OK')"
```

You should see a `gpu(id=0)` device and no import errors.

## Setup

Activate the openpi venv before running anything:

```bash
source /home/kewalk/memory_project/openpi/.venv/bin/activate
```

## Commands

### Convert raw YAM data → LeRobot dataset

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

### Serve the trained policy (deferred — for real-robot use)

```bash
cd /home/kewalk/memory_project/openpi
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_yam_cup_lora \
    --policy.dir=checkpoints/pi05_yam_cup_lora/cup_0429_v1/29999
```
