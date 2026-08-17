# v3.2 Checkpoint Diagnostics — Report for Step 5000

**Date:** 2026-08-15
**Checkpoint:** `checkpoints/pi05_yam_mem_v32/dualquery16_l8_s15_d6_t40_b14-27-40_tb25_bs12_bf16fast_save250_seed42/5000`
(training was at ~step 5,650/20,000 when this evaluation ran; losses at the time: CE ≈ 0.021, flow ≈ 0.0018, no instabilities)
**Config:** `pi05_yam_mem_v32` | **Episodes evaluated:** 0 (left-banana, 941 frames), 2 (right-banana, 841 frames), stride 15 → 63 + 57 = 120 policy steps
**Tooling:** `scripts/v32_checkpoint_diagnostics.py` → `src/openpi/diagnostics/v32_checkpoint.py` (new), plus two read-only model hooks in `src/openpi/models/pi0.py`
**Outputs:** `diagnostic_outputs/v32_ckpt5000/episode_000000/` and `.../episode_000002/`

---

## 1. Background

### Task
Bimanual YAM robot, "find the bin with the banana." Episode structure: two closed bins on the table → bins are opened, revealing a banana in one of them (left or right) → bins are closed again → the robot must open the bin that contains the banana. After closure the top camera is ambiguous; solving the task requires remembering which side held the banana. Subtask labels per frame: `observe bins` during the observation phase, then `open left bin` / `open right bin`. Dataset: 30 training episodes (16 left / 14 right), LeRobot format, 3 cameras (top + 2 wrist) + 14-D bimanual joint state.

### v3.2 architecture (what is being tested)
- Layer-8 top-camera hidden states `h8_top` [256 patches, 2048] are the **single source** for both memory read and write.
- Two **independent learned 16-token query banks** cross-attend over `h8_top`:
  - *read queries* → query vectors for the Titans fast-weight memory read ("what do I need to recall");
  - *write queries* → the 16 tokens written into memory after prediction ("what is worth storing").
- Retrieved 16 tokens are multiplied by a learned `memory_gate` vector and injected after layer 8, flowing through blocks 9–17 to influence subtask decoding and action denoising.
- Causal order per timestep: read M_{t−1} → predict subtask + actions → write M_t. Write content structurally cannot contain retrieved memory (this removes v3.1's echo path).
- Trained fresh from pi05_base. This is the **first trained v3.2 checkpoint ever evaluated**.

### Why these tests
v3.1 diagnostics showed: writer attention was near-uniform (cross-token CV ≈ 0.005), the writer went idle after bin closure (attention collapsing onto a prompt sink), and one head routed retrieved memory back into the write (echo). v3.2's dual-query design was built to fix these. The tests below check, at step 5000: (a) does the policy causally depend on memory content, (b) did the query banks specialize, (c) is writing event-selective.

---

## 2. Method

All tests run in **one replay pass per episode**. At every sampled frame (stride 15) with the *same pre-write memory state*:

1. `v32_query_attention_step` (new read-only hook): runs the split prefix exactly like inference (blocks 0–8 → dual-query read → gated inject → blocks 9–17) and returns the FP32 query→patch softmax maps for both banks, the retrieved tokens, the write tokens, and per-slot write diagnostics. Commits nothing.
2. **Normal sample** (`zero_read=False, allow_write=True`): free-decodes the subtask, denoises 10-step actions, then commits the frame's memory write. This is the state that carries forward — replay dynamics match deployment.
3. **Zero-read counterfactual** (`zero_read=True, allow_write=False`): identical call, identical RNG/noise, but retrieved memory is replaced by zeros. Any output difference vs. call 2 is *causally attributable to retrieved memory content*.
4. **Teacher-forced log-prob**, both with normal read and zero read (`allow_write=False`): exact summed log-probability of the ground-truth subtask string under each condition.
5. Fast-weight drift metrics (on device): ‖M_t‖, ‖M_t − M_{t−1}‖, ‖M_t − M_0‖, ‖M_t − M_anchor‖ (anchor = first frame ≥ 300), separately for fast weights and momentum.

Notes:
- Same RNG key and same noise tensor for the normal/zero pair, so action differences are not sampling noise.
- Model runs in FP32 params / BF16 compute as in training; batch 1; 10 denoise steps; greedy subtask decode (max 10 tokens).
- Actions are compared in **robot space** (un-normalized, delta-decoded), overall and split by arm (dims 0–6 left, 7–13 right).

### Metric definitions
- `action_mse_robot`: mean squared difference between normal and zero-read action chunks in robot units.
- `gt_logp_normal/zero`: summed log-prob of the GT subtask string (teacher-forced). `dlogp = normal − zero` > 0 means memory *helps* the GT subtask.
- `decoded mismatch`: fraction of frames where the free-decoded subtask string differs between normal and zero-read.
- Attention **normalized entropy** per query slot: H(p)/log(256) over the 256 patches; 1.0 = uniform, 0 = single patch. **Effective token count** = exp(H): 256 = uniform.
- `eta, theta, alpha, surprise, grad_norm`: Titans write internals (learning-rate gate, momentum, forgetting, associative error, write gradient magnitude).
- `write_slot_norm_cv`: coefficient of variation of the 16 write-token norms within a frame (selectivity across slots).
- `memory_gate_norm`: L2 norm of the learned gate vector that scales retrieved tokens before injection.

---

## 3. Results

### 3.1 Zero-read counterfactual (handoff §12.6) — **memory content is almost never used**

| Metric | Episode 0 (left) | Episode 2 (right) |
|---|---|---|
| Free-decode mismatch (normal vs zero-read) | 0 / 63 frames | 0 / 57 frames |
| Decoded-vs-GT accuracy, normal read | 98.4% | 98.2% |
| Decoded-vs-GT accuracy, zero read | 98.4% | 98.2% |
| `action_mse_robot`, median | 6.8e−7 | 8.6e−7 |
| `action_mse_robot`, max (frame) | 0.037 (frame 60) | 0.058 (frame 60) |
| `dlogp`, magnitude everywhere except decision frame | ≤ 4e−5 | ≤ 1.6e−4 |
| `dlogp` at the decision frame 480 | **−1.09** | **−0.91** |

- After the first ~5 memory steps, zeroing retrieval changes essentially nothing: actions to ~1e−7 MSE, GT log-prob to ~1e−6.
- The only meaningful signal in the whole run: at **frame 480** — exactly where the decoded subtask flips from `observe bins` to `open {left,right} bin`, one label-frame *before* GT flips at 495 — memory shifts subtask logits by ~1 nat in both episodes. Sign: with memory, the model is *more* committed to `open X bin` (so the stale `observe bins` GT label gets ~1 nat less likely). This is the single trace of memory influencing the decision computation.
- The early-episode action MSE spikes (frames 0–60, up to 0.058) occur while memory is nearly blank and surprise is huge (1.9e7 decaying); they reflect sensitivity of a not-yet-settled policy to small perturbations, not meaningful memory use.

### 3.2 Query→patch attention maps (handoff §12.1–12.2) — **no specialization; write "selectivity" is a padding sink**

- **Read bank:** every one of the 16 slots has normalized entropy 0.992–0.999 on *every* frame of both episodes (effective token count ≈ 250/256). Retrieval is a near-uniform average pool of the image. No slot specialization; no behavior change after bin closure.
- **Write bank:** baseline entropy ≈ 0.99 (same uniform wash), but dips to 0.60–0.75 at some frames (min at raw frame 390 in ep0, 195 in ep2). Rendering those frames shows the concentrated mass lands on the **letterbox padding band at the bottom of the image** — not on bins, banana, or grippers — and **all 16 slots highlight the same region** (slot-wise entropy minima are within 0.05–0.2 of each other). The dips are an attention-sink artifact, not event encoding.
- The two banks *are* numerically distinct (different maps, unit test confirms distinct parameters), but neither has learned semantic structure at step 5000.

### 3.3 Write/retrieval statistics over time (handoff §12.3–12.4) — **writes are magnitude-uniform; note the gates are frozen by design**

| Statistic | Episode 0 | Episode 2 |
|---|---|---|
| `eta` (write gate) | **0.9005 on every frame** (min=med=max) | same |
| `surprise` | 1.9e7 (frame 0, blank memory) decaying to O(0.1–1); med 36.5 | med 108 |
| `write_slot_norm_cv` | 0.0008–0.05, med 0.002 | 0.0005–0.06, med 0.005 |
| `retrieval_norm` (RMS) | 0.04–1.17, med 0.14 | 0.07–1.32, med 0.19 |
| `memory_gate_norm` | **0.0452, constant** | same |

- **Correction to an earlier verbal readout:** eta being identical on every frame is *by design in this run*, not a learned pathology. The training config freezes the Titans gate parameters (`freeze_filter=".*memory/gate.*"`); the gate Linear has a zero kernel and fixed biases, so θ = sigmoid(−2.2) ≈ 0.10, η = sigmoid(2.2) ≈ 0.90, α = sigmoid(−4.6) ≈ 0.01 are data-independent constants. Consequently, per-frame write selectivity in v3.2 can only arise through the *content* of the 16 write tokens and the surprise/gradient magnitude — and those are what the CV and entropy rows above show to be near-uniform.
- Cross-slot write CV ≈ 0.002–0.005 median: the 16 written tokens are near-identical in magnitude (echoes v3.1's CV ≈ 0.005 uniform-write finding, now across query slots instead of patch tokens).
- The retrieved signal entering the network is `gate ⊙ retrieved`: with ‖gate‖ = 0.045 and retrieval RMS ≈ 0.14–0.19, the injected memory tokens are **under ~1% of prefix-token scale** — the read pathway is strangled at the gate. This is the proximate cause of the zero-read no-op.
- Fast-weight drift curves (in `curves.png` / `summary.json`): monotone growth of ‖M_t − M_0‖ with per-frame step deltas that do not spike at reveal or closure — writes accumulate uniformly rather than episodically.

### 3.4 Side-information leak (critical evaluation caveat, discovered incidentally)

At the decision frame 480 the bins are **visibly closed** in the model's own 224×224 input (verified from stored images in `arrays.npz`), the arms have not yet committed to a side — and the model **decodes the correct side with memory zeroed, in both episodes**. Therefore the side information reaches the policy through non-memory channels:

1. **Proprioceptive/postural leak:** the 14-D joint state (and arm silhouettes in the image) during/after the lid-closing choreography can correlate with the banana side in the training data;
2. **Training-episode memorization:** 30 episodes, heavily resampled during training, 3.4B model — per-episode cues may be memorized.

Consequences: (a) subtask accuracy on training episodes can never demonstrate memory use; (b) the trustworthy probes are counterfactuals (zero-read, memory-swap) evaluated in the closure→decision window; (c) CE being nearly solved (0.021) through these leak channels means **gradient pressure to open the memory pathway is weak** — a plausible root cause for the gate staying at 0.045.

---

## 4. Engineering notes

- **v3.2 checkpoints do not load via `Pi0Config.load`:** flax 0.10.2 keeps `bias: None` state entries for `nnx.Linear(use_bias=False)` (all 8 query-compressor projections); orbax silently drops `None` on save, so restored params fail the strict structure check ("symmetric difference {'bias'}"). Workaround implemented in `v32_checkpoint.py::_align_params` (project restored params onto the abstract state, refill exactly-`None` slots, call `load(..., remove_extra_params=False)`). **Any other v3.2 checkpoint consumer (e.g. `serve_yam_memory.py`) will need the same fix.**
- New read-only model hooks (unit-tested in `pi0_v32_diagnostics_test.py`, no effect on training): `MemoryQueryCompressor.attention_probs` (exact softmax weights of the forward attention) and `Pi0.v32_query_attention_step` (per-frame diagnostic bundle).
- Videos are H.264 (libx264, enforced), 5 fps, one video frame per policy step (15 raw frames ≈ 0.5 s robot time).

## 5. Artifact inventory (for further analysis)

Per episode directory (`diagnostic_outputs/v32_ckpt5000/episode_000000/`, `episode_000002/`):

- `summary.json` — per-frame records: `raw_frame, gt_subtask, decoded_normal, decoded_zero, action_mse_{model,robot,robot_left,robot_right}, action_maxabs_robot, gt_logp_{normal,zero}, gt_mean_logp_{normal,zero}, retrieval_norm, read_query_norm, write_token_norm, memory_gate_norm, eta, theta, alpha, surprise, grad_norm, write_slot_{norm,error}_cv, {read,write}_entropy_mean, drift_*, anchor_*`; plus episode aggregates (`mean_all`, `mean_pre_anchor`, `mean_post_anchor` with anchor=300).
- `arrays.npz` — `raw_frames` [T]; `model_images` [T,224,224,3] u8 (exact model inputs); `read_attention`/`write_attention` [T, 8 heads, 16 slots, 256 patches] fp16 (softmax weights; head-mean was used for videos/entropy); `retrieved`/`write_tokens` [T,16,2048] fp16; `actions_normal`/`actions_zero` [T,horizon,action_dim]; `robot_actions_normal`/`robot_actions_zero` [T,horizon,14].
- `read_attention.mp4` / `write_attention.mp4` — 4×4 grid, one tile per query slot, jet overlay normalized per-tile-per-frame (shows *where*, not *how strongly* — trust the `eff`/`H` labels for magnitude), header shows mean entropy.
- `curves.png` — six panels: zero-read action divergence, GT log-probs (red lines = free-decode mismatch, none occurred), retrieval/write magnitudes + retrieved cos(t,t−1), write gates/selectivity, attention entropies, fast-weight drift.
- `../run.json` — full option record (checkpoint, stride, seed, anchor, denoise/decode steps).

Reproduce / extend to any future checkpoint:

```bash
.venv/bin/python scripts/v32_checkpoint_diagnostics.py \
  --checkpoint checkpoints/pi05_yam_mem_v32/<run>/<step> \
  --dataset-root ~/.cache/huggingface/lerobot/yam/bin_memory_banana_subtask \
  --episode-indices 0 2 --anchor-frame 300 \
  --output-dir diagnostic_outputs/v32_ckpt<step>
```

(~20 min on one H200 for two episodes, batch 1.)

## 6. Open questions and suggested next analyses

1. **Trajectory over training:** rerun at 10k/15k/20k. Watch: `memory_gate_norm` (0.045 → growing?), zero-read `dlogp` at/around the decision frame (does the −1 nat pulse widen into a window?), read-slot entropy (does any slot break from uniform?), `eta` (does it ever deviate from 0.9005?).
2. **Memory-swap test (handoff §12.5):** build M from a left episode and a right episode up to matched post-closure frames, swap them under a fixed ambiguous observation, and check whether the decoded side follows the memory. This cleanly separates memory from the proprioceptive leak — the most important missing test.
3. **Quantify the leak:** train/evaluate the decision from proprioceptive state alone (e.g., logistic probe on the 14-D state at frames 400–495 across the 30 episodes) to establish how much side information proprioception carries.
4. **Padding-sink hypothesis:** the write-entropy dips target letterbox padding. Check whether the dips coincide with global image statistics changes (lid motion covering the visual field) and whether masking padding patches from the compressor keys removes the artifact.
5. **Gate-freezing trade-off:** with θ/η/α frozen (see §3.3 correction), the memory writes with the same learning rate on every frame by construction. Analyze whether the frozen α ≈ 0.01 forgetting combined with ~40+ uniform writes per episode dilutes the reveal-time content by the decision frame (the drift curves in `summary.json` contain the data for this).
6. **Weak-pressure hypothesis:** CE ≈ 0.021 with leaks ⇒ little gradient for memory. Consider (only as analysis, per handoff §9 no new mechanisms mid-run): measuring CE restricted to decision frames with proprioception ablated/perturbed, to estimate the residual loss that only memory could remove.
