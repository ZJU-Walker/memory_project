# memory_project

Workspace for fine-tuning Physical Intelligence's **π₀.₅ (pi05)** on the **bimanual YAM** `bin_memory_banana` task.

## Layout

```
memory_project/
├── data/bin_memory_banana/   # raw bimanual YAM teleop demos (30 episodes, ~30 Hz)
├── openpi/                    # cloned openpi repo + venv (gitignored), with YAM data config & policy transforms
└── i2rt/                      # cloned i2rt repo (YAM URDF, FK, MuJoCo SimRobot)
```

YAM-specific code lives under `openpi/`:

- `openpi/examples/yam/convert_yam_data_to_lerobot.py` — dataset converter.
- `openpi/src/openpi/policies/yam_policy.py` — input/output transforms (`YamInputs`, `YamOutputs`).
- `openpi/src/openpi/training/config.py` — data config (`LeRobotYamDataConfig`) and train config (`pi05_yam`).
- `openpi/src/openpi/models/pi0_memory.py` + the `pi05_yam_memory` config — the episodic-memory variant (see [Memory-as-Context](#memory-as-context-episodic-neural-memory) below).

## Data format

Each `data/bin_memory_banana/demoN/` folder contains, at ~30 Hz:

- `{left,right}_joint_positions.npy` — `(T, 7)` follower state (6 arm joints + 1 gripper per arm).
- `{left,right}_control.npy` — `(T, 7)` leader/teleop command (the action target).
- `top_camera_rgb.mp4`, `left_camera_rgb.mp4`, `right_camera_rgb.mp4` — 3 cameras (640×480).
- `metadata.json`.

The converter builds a standard **LeRobot** dataset:

- `state` = `concat(left_joint_positions, right_joint_positions)` → **14-dim** (actual follower state).
- `actions` = `concat(left_control, right_control)` → **14-dim** (leader command).
- `image` / `left_wrist_image` / `right_wrist_image` = top / left / right cameras.

The `pi05_yam` config trains with **delta actions** on the 6 arm joints of each arm and keeps the **grippers absolute** (mask `[6 delta, 1 abs, 6 delta, 1 abs]`), which handles the leader/follower gap when the gripper grasps an object. There is no language instruction in the data, so a fixed prompt `"find the bin with banana"` is injected.

## Setup

Activate the openpi venv before running anything:

```bash
source /iris/u/kewalk/memory_project/openpi/.venv/bin/activate
```


## Pi05 steps

### 1. Convert raw demos → LeRobot dataset

```bash
cd /iris/u/kewalk/memory_project/openpi
uv run examples/yam/convert_yam_data_to_lerobot.py \
    --data_dir /iris/u/kewalk/memory_project/data/bin_memory_banana
```

Writes the dataset to `$HF_LEROBOT_HOME/yam/bin_memory_banana`. Append `--push_to_hub` to also publish it to the Hugging Face Hub.

### 2. Compute normalization stats

```bash
cd /iris/u/kewalk/memory_project/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_yam
```

### 3. Train (fine-tune π₀.₅ from `pi05_base`)

```bash
cd /iris/u/kewalk/memory_project/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/train.py pi05_yam --exp-name=yam_banana_pi05 --overwrite

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train_memory.py pi05_yam_memory \
    --exp-name=yam_banana_pi05_mem_400m_v3 --overwrite
```

Checkpoints land in `openpi/checkpoints/pi05_yam/yam_banana_pi05/`. Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` so JAX can use up to 90% of GPU memory.

### 4. Serve the trained policy for inference

WebSocket inference server on port 8000:

```bash
cd /iris/u/kewalk/memory_project/openpi
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_yam \
    --policy.dir=/iris/u/kewalk/memory_project/openpi/checkpoints/pi05_yam/yam_banana_pi05/6000

# v1 — 3-camera memory encoding, forced via mem_camera=all:
uv run scripts/serve_policy_memory_v1.py \
--policy.dir=checkpoints/pi05_yam_memory/yam_banana_memory_400m_v1/<step>

# v2 — top-camera-only (training config default, nothing forced):
uv run scripts/serve_policy_memory_v2.py \
--policy.dir=checkpoints/pi05_yam_memory/yam_banana_memory_400m_v2/<step>
```

`<step>` is the checkpoint iteration to load (e.g. `29999` for the final step of a 30k-step run). Run inside `tmux` so it survives terminal disconnects.


## Memory-as-Context (episodic neural memory)

A MAC/Titans-style **online episodic memory** on top of `pi05` for the hidden-bin task: during one
episode the robot opens both bins (sees the banana), closes the lids, then must open the *remembered*
bin. A small memory MLP whose weights update **online within the episode** stores the banana location
during inspection and is read back at recall; its read/write projections are trained end-to-end by
the action loss. Everything is **additive** — the `pi05_yam` baseline above is left untouched.

New pieces (baseline `Pi0` / `pi05_yam` / `scripts/train.py` unchanged):

- `openpi/src/openpi/models/pi0_memory.py` — `Pi0Memory`: memory module + the causal split-cost
  unroll (write-only memory frames that skip the LLM, then one full action forward at the target).
- `openpi/src/openpi/models/pi0_config.py` — `Pi0MemoryConfig` (`d_mem`, `mem_eta/theta/alpha`,
  `freeze_vision`).
- `openpi/src/openpi/training/config.py` — `LeRobotYamMemoryDataConfig` + the `pi05_yam_memory` `TrainConfig`.
- `openpi/src/openpi/training/memory_data_loader.py` — episode-sequence loader (N causal frames + 1 action chunk).
- `openpi/scripts/train_memory.py` — trainer (reuses the baseline `train_step`; bakes in a cuBLAS GEMM flag).

### 1. Compute normalization stats

Same transforms as `pi05_yam`, so the stats are identical (already copied into
`openpi/assets/pi05_yam_memory/`). To recompute:

```bash
cd /iris/u/kewalk/memory_project/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_yam_memory
```

### 2. Train (joint fine-tune `pi05` + memory from `pi05_base`)

```bash
cd /iris/u/kewalk/memory_project/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/train_memory.py pi05_yam_memory --exp-name=yam_banana_memory --overwrite
```

Checkpoints land in `openpi/checkpoints/pi05_yam_memory/yam_banana_memory/` — **saved every 1000
steps, kept permanently every 5000** (same as `pi05_yam`). The SigLIP vision tower is frozen; the
Gemma LLM + action expert + memory module train jointly. The first compile of the `n_mem=16` episode
unroll takes a few minutes (cached afterwards under `$JAX_COMPILATION_CACHE_DIR`).

**Key knob — `mem_theta`** (the online memory learning rate, default `1e-2`): too small (e.g. the
original `1e-5`) and the memory barely moves within an episode and stores nothing. Sweep `~[1e-3, 1e-1]`
and watch `||M - M_0||` grow without blowing up; add `mem_alpha > 0` (forgetting) if it drifts.
`n_mem` (default 16) is the number of causal frames the memory unrolls over per item.

### Validation (smoke-tested 2026-06-30, single H200, from `pi05_base`)

```text
# weight loader keeps the 12 memory params; the rest load from pi05_base
memory keys in ref: 12 | present in loaded: 12
loss: 1.15  finite: True
nonzero memory grads: 12/12          # every memory param learns; frozen vision gets 0 grad

# data loader yields obs [B, N, 224, 224, 3] + a single action chunk [B, H, 32]
actions: (2, 50, 32) | image base_0_rgb: (2, 16, 224, 224, 3) | state: (2, 16, 32)

# jitted train_step on real batches (n_mem=4 for a fast smoke)
step 1: loss=0.0945 grad_norm=2.6329 param_norm=1803.75
step 2: loss=0.1106 grad_norm=3.3861 param_norm=1803.75
step 3: loss=0.0698 grad_norm=2.7867 param_norm=1803.75
```
