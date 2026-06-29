# memory_project

Workspace for fine-tuning Physical Intelligence's **π₀.₅ (pi-0.5)** on the YAM single-arm long-horizon plate task.

## Layout

```
memory_project/
├── dataset/LONG_TASK_1/    # raw YAM teleop demos for the long-horizon plate task (37 long episodes, 30 Hz)
├── openpi/                 # cloned openpi repo + venv (gitignored), with YAM data config & policy transforms
└── i2rt/                   # cloned i2rt repo (YAM URDF, FK, MuJoCo SimRobot)
```

YAM-specific code lives under `openpi/`:

- `openpi/examples/yam/` — dataset converter (`convert_plate_task_to_lerobot.py`) and `plate_prompts.py`.
- `openpi/src/openpi/policies/yam_policy.py` — input/output transforms (`YamPlateTaskInputs`, `YamOutputs`).
- `openpi/src/openpi/training/config.py` — data config (`LeRobotYamPlateTaskDataConfig`) and train config (`pi05_plate_task`).

The dataset is a standard **LeRobot** dataset (written via `LeRobotDataset.create`), with custom flat feature keys (`top_image`, `left_image`, `state`, `actions`, `task`) bridged into openpi's expected names by a `RepackTransform`.

## Install

The project depends on two upstream repos vendored as subdirectories — **openpi** (training/inference) and **i2rt** (YAM driver, URDF, MuJoCo sim). Each ships its own venv; this project reuses the openpi venv for everything.

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

### 3. Set up the openpi venv (training, eval, serve)

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

### 5. (Real-robot only) CAN bus

```bash
sudo ip link set can0 up type can bitrate 1000000             # one-shot
sudo sh /home/kewalk/memory_project/i2rt/devices/install_devices.sh   # auto-up on boot
```

### 6. Verify

```bash
source /home/kewalk/memory_project/openpi/.venv/bin/activate
python -c "import jax; print('jax devices:', jax.devices())"
python -c "from openpi.training.config import get_config; print(get_config('pi05_plate_task').name)"
python -c "from i2rt.robots.get_robot import get_yam_robot; print('i2rt OK')"
```

You should see a `gpu(id=0)` device and no import errors.

## Setup

Activate the openpi venv before running anything:

```bash
source /home/kewalk/memory_project/openpi/.venv/bin/activate
```

## Plate task

Long-horizon plate task. Each demo is one long episode with `observe → mix → delay → query1..4 → return_home` segments. Top + left cameras carry the current view; `right_wrist_0_rgb` is zero-padded and masked off (single-arm YAM, no real right wrist). The per-frame instruction already names the resolved object (e.g. *"put banana on the light blue plate"*), generated from each demo's `metadata.json` at training time. Demos 1-35 train, 36-37 held out.

> **Note:** the raw demos now live under `dataset/LONG_TASK_1/`; the converter's docstring/defaults still reference the old `dataset/PLATE_TASK_1/` path — pass `--data-dir` explicitly.

### Convert raw plate data → LeRobot dataset

Smoke test on a single demo first:

```bash
cd /home/kewalk/memory_project/openpi
uv run examples/yam/convert_plate_task_to_lerobot.py \
    --data-dir /home/kewalk/memory_project/dataset/LONG_TASK_1 \
    --limit 1
```

Full conversion of demos 1-35:

```bash
cd /home/kewalk/memory_project/openpi
uv run examples/yam/convert_plate_task_to_lerobot.py \
    --data-dir /home/kewalk/memory_project/dataset/LONG_TASK_1 \
    --push_to_hub
```

Append `--push_to_hub` to also publish to `kewalk123/plate_task` on HF (required for Modal training; optional locally).

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
