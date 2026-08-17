# v3.2 Episodic Fast-Weight Memory for π0.5 — Complete Method Handover

**Purpose of this document:** a self-contained description of the v3.2 memory method — architecture, training procedure, data pipeline, implementation layout, current status, and known issues — for a reader with **no prior context** on this project. Everything needed to reason about or extend the method is either stated here or pointed to by file path (repo root: `openpi/`).

**Companion document:** `docs/v32_ckpt5000_diagnostics.md` — evaluation results for the first trained checkpoint (step 5000).

---

## 1. Project context

### 1.1 The task

A bimanual YAM robot (two 7-DoF arms, 14-D joint state) must "find the bin with the banana." Episode structure:

1. Two closed bins sit on a table (top camera + left/right wrist cameras).
2. The bins are opened; a banana is visible in exactly one bin (left or right).
3. The bins are closed again — after this, the top camera is **ambiguous**: both bins look identical.
4. The robot must open the bin that contains the banana.

Per-frame subtask labels: `observe bins` during phases 1–3, then `open left bin` / `open right bin`. Solving phase 4 requires remembering which side held the banana — this is the memory probe the whole project is built around.

**Dataset:** `yam/bin_memory_banana_subtask` (LeRobot v2.1 format), 30 episodes (16 left / 14 right), 638–1002 frames each (~22,705 frames total), 3 RGB cameras + 14-D state + 14-D actions, subtask string per frame via `task_index`. Banana reveal happens around raw frame 300 (per-episode reveal frames can be supplied via `assets/pi05_yam/reveal_frames.json`; default 300).

### 1.2 The base policy: π0.5

π0.5 (openpi) is a vision-language-action model:

- **VLM backbone:** PaliGemma — SigLIP vision encoder (each 224×224 camera image → 256 patch tokens) + Gemma-2B LLM (width 2048, **18 transformer layers**, `paligemma_variant="gemma_2b"`).
- **Action expert:** a smaller Gemma (300M) running alongside the VLM through shared attention (two-config `gemma.Module`), producing flow-matching action denoising. Actions: horizon 50, dim 32 (model space), decoded to 14-D robot deltas.
- **Subtask prediction (`predict_subtask=True`):** the model autoregressively decodes the current subtask string (+ FAST-tokenized action hints) from a "causal" token segment before denoising actions; trained with cross-entropy (CE), teacher-forced.
- **pi05 mode:** adaRMS conditioning of the action expert on the diffusion timestep.

### 1.3 Version history (why v3.2 exists)

- **v3:** Titans fast-weight memory attached to π0.5; memory read/written with the 256 layer-8 top-camera hidden states directly (dense 256-token interface). Writer source = layer-8 hidden states (`memory_write_source="layer_hidden"` era).
- **v3.1:** writer source switched to the *final* contextualized memory-token outputs c_t (`memory_write_source="post_attention"`), plus RTC and sequence training. Diagnostics on trained v3.1 checkpoints found:
  1. writer attention spent 68–85% of its mass on a prompt-token sink and went idle after bin closure (memory-block attention mass collapsed 38–50% → 5–16% at closure);
  2. writes were near-uniform across the 256 tokens (cross-token CV ≈ 0.005) on every frame — no event selectivity;
  3. one attention head routed up to 49% of its mass from the writer tokens into the injected (retrieved) memory block — an "echo" path where the writer could store retrieved memory instead of new observations;
  4. subtask rows attended ≤ 0.4% to the memory block at layer 8 — retrieved memory barely used.
- **v3.2 (this method):** a clean redesign of the memory *interface only* (spec: `../V32_MEMORY_HANDOFF.md`). One-sentence definition:

> **v3.2 uses layer-8 top-camera hidden states as a shared visual source, compresses them with 16 learned read queries to retrieve 16 fast-memory tokens for the remaining VLM/action pathway, and independently compresses the same layer-8 source with 16 learned write queries to update the fast-weight memory for future timesteps.**

Explicit non-goals (spec §9): no new write gates, no "write only when banana visible" heuristics, no probe losses, no ROI supervision, no new forgetting rules — the experiment isolates *dense 256-token interface → 16-token learned dual-query bottleneck*.

---

## 2. The Titans fast-weight memory (unchanged from v3/v3.1)

File: `src/openpi/models/memory.py` (class `TitansMemory`, based on the Titans paper's neural memory).

- **State (per episode, per sample):** `MemoryState(fast_weights, momentum)` — the weights of a small MLP ("the memory") plus their momentum buffer. MLP dims: **512 → 1024 → 1024 → 1024 → 2048** (`d_key=512, hidden_dims=(1024,1024,1024), d_value=2048`). All FP32. Initialized fresh at episode start (`init_state`).
- **Outer (slow, backprop-trained) parameters:** input projections `W_K, W_V, W_Q` mapping the 2048-wide writer/reader tokens into key space (512) and value space (2048), plus the gate Linear below.
- **Write** (`write(state, h)` where `h` is [B, n, 2048]; in v3.2 n = 16 writer tokens):
  1. keys/values: `K = W_K(h)`, `V = W_V(h)` (normalized);
  2. associative loss per token: `e_i = ||M(K_i) − V_i||²`; frame loss = mean; its gradient w.r.t. fast weights is the "surprise" signal;
  3. per-sample global-norm clip at 1.0 (`max_grad_norm`) — direction kept, step length bounded;
  4. update with momentum and forgetting (Titans eqs. 11–14): `momentum ← η·momentum − θ·grad`, `fast_weights ← (1 − α)·fast_weights + momentum`;
  5. gates **θ (inner LR), η (momentum), α (forgetting)** are per-frame scalars: `sigmoid(gate_linear(mean over tokens of h))`. The gate Linear has a **zero-initialized kernel** and fixed biases θ_bias=−2.2, η_bias=2.2, α_bias=−4.6, i.e. until trained the gates are constants **θ≈0.10, η≈0.90, α≈0.01**.
  6. Returned aux: `surprise, grad_norm, theta, eta, alpha`.
- **Read** (`read(state, q)`): normalize `W_Q(q)` into key space, run the memory MLP: returns [B, n, 2048] retrieved tokens.
- **Important v3.2 training decision:** the gate Linear is **frozen** during v3.2 training (`freeze_filter=PathRegex(".*memory/gate.*")` in the train config) — "the Titans write gates stay frozen at their measured stable operating point." So in this run θ/η/α are *data-independent constants by design*; write selectivity can only come from write-token content. (History: v1/v2 of the 400M-scale memory diverged to NaN; the frozen operating point + clip + zero-init content gate are the stability fixes.)

---

## 3. The v3.2 memory interface

Files: `src/openpi/models/pi0.py` (class `Pi0`, search "v32"), `src/openpi/models/pi0_config.py`.

### 3.1 Source representation

For each timestep, run Gemma blocks **0..8** (`memory_layer=8`) over the prefix (images + prompt). Take the hidden states at the 256 token positions of the **top camera** (the first camera's SigLIP tokens): `h8_top` [B, 256, 2048], cast to FP32. This is the *single* source for both read and write. (Caveat from spec §8: layer-8 top-camera positions may already mix information from other tokens via attention — this is a "controlled source," not a guarantee of modality isolation.)

### 3.2 Dual query banks — `MemoryQueryCompressor`

Two **independent** instances (`read_query_compressor`, `write_query_compressor`), each:

- a learned query bank [16, 2048] (`memory_query_tokens=16`), initialized N(0, 1/√d);
- 8-head cross-attention (`memory_query_heads=8`, head_dim 256) over the 256 source tokens: Q from the query bank, K/V from `h8_top`;
- four bias-free Linear projections (query/key/value/output), FP32 master params, **BF16 compute** (`compute_dtype` = model dtype) with FP32 softmax logits; output cast back to FP32 before entering TitansMemory;
- output: 16 tokens [B, 16, 2048].

Read bank output `q_t` = "what do I need to recall"; write bank output `z_t` = "what is worth storing." Same source, separate parameters, no weight sharing.

### 3.3 Read path and injection

1. `retrieved = memory.read(M_{t−1}, q_t)` — read the **pre-write** state (16 tokens, 2048-wide).
2. Gate: `memory_tokens = memory_gate ⊙ retrieved`, where `memory_gate` is a learned per-channel vector [2048], **zero-initialized** — an untrained/empty memory injects exactly zero content. (This vector is trainable — *not* covered by the freeze filter.)
3. The 16 gated tokens are appended after the prefix and processed by Gemma blocks **9..17** together with the prefix hidden states (single pass; no second full-VLM pass). They are attendable by the subtask decode and the action suffix, so retrieved memory can influence both.

### 3.4 Write path

After the frame's prediction is finished, `M_t = memory.write(M_{t−1}, z_t)`. Structural properties:

- `z_t` is computed from `h8_top` (blocks 0–8 only), which **cannot contain retrieved memory** — memory tokens are injected only after layer 8. v3.1's echo path is removed by construction, not by masking.
- No use of final-layer states (c18) or retrieved tokens as writer content.

### 3.5 Causal ordering (per timestep)

read M_{t−1} → inject → decode subtask → denoise actions → write M_t → carry M_t to t+1. Invariant: the current action never sees a state containing the current frame's write. (Regression-tested; see §7.)

### 3.6 Execution details (how one timestep actually runs)

Token layout (single KV cache with fixed capacity = prefix + 16 + 128):

```
[ 3×256 image tokens | prompt tokens (≤ max_token_len=80) | 16 memory tokens | ≤128 causal (subtask) tokens | action suffix ]
```

- **Split prefix:** Gemma runs with per-block `active_layers` masks (nn.cond inside the scanned blocks — `gemma.py` "partial_layers"): first call runs blocks 0–8 over the prefix (no final norm), producing `h8_all`; after read/injection, a second call runs blocks 9–17 over `[h8_all, memory_tokens]`. Both calls share one pre-allocated KV cache (`_v32_empty_cache`), so keys/values for layers ≤8 exist only for prefix positions and for layers >8 also for memory positions (`_v32_layer_mask` gives the two layer groups different key visibility).
- **Subtask decode:** greedy, autoregressive from the last prefix hidden state, up to `max_decode_steps` (10 at inference; buffer `causal_token_len=128`), stop at newline/EOS. Teacher-forcing path (`forced_subtask_tokens`) exists for exact log-prob evaluation.
- **Action denoising:** flow matching, 10 steps at inference, suffix attends to prefix + memory + live causal tokens.
- **RTC (real-time chunking):** `simulated_delay=6` — during training/inference the first ≤6 actions of the chunk can be frozen to the previously-executed prefix (`rtc.py`: `condition_action_prefix` / `restore_action_prefix`), matching deployment where inference latency overlaps execution.
- Inference entry point: `Pi0.sample_with_memory(...)` → `_sample_with_memory_v32(...)`, with diagnostic knobs `zero_read` (replace retrieved with zeros) and `allow_write` (compute but do not commit the write).

---

## 4. Training procedure

Config: `pi05_yam_mem_v32` in `src/openpi/training/config.py`. Loss: `Pi0._compute_sequence_loss_v32` in `pi0.py`.

### 4.1 Sequence objective

Training operates on **sequences of memory steps** to teach the recurrent write/read dynamics:

- A training example = `memory_seq_steps=40` (T40) consecutive policy steps sampled every `memory_stride_frames=15` raw frames (S15) → covers up to 600 raw frames (~20 s), essentially the same physical horizon as the earlier T60/S10 with one-third fewer recurrent writes.
- Per step t (inside a `lax.scan`, each step wrapped in `jax.checkpoint` for memory): run the v3.2 split prefix with the *carried* memory state; compute
  - **CE loss** on the teacher-forced causal segment (subtask string + FAST action tokens; `subtask_from_task=True, subtask_lookahead=15`), from the causal decode logits (BF16 vocab projection, `bf16_vocab_projection=True`);
  - **flow-matching action loss** on the suffix — computed **behind `stop_gradient(kv)`** so action gradients do not flow back into the VLM prefix/memory (CE is the pathway that trains the memory interface);
  - then **write** M_t (after the losses — matches inference ordering).
- **TBPTT:** `memory_block_steps=25` — gradients truncate (stop-gradient on the carried state) at 25-step block boundaries within the 40-step sequence.
- Steps beyond an episode's end are masked (`seq_step_mask`; logged as `sequence_valid_fraction` ≈ 0.94).
- Memory probe: disabled (`memory_probe_weight=0.0`) — the `probe_head` module exists only for checkpoint compatibility.
- Image augmentation on the sequence images during training.

### 4.2 Data pipeline

`src/openpi/training/data_loader.py` + transforms (`BuildMemorySequence`, `TokenizeMemorySubtaskInputs`, YAM repack/delta/normalize):

- **Sequence sampling:** with p=0.5 a full-trajectory start (frame 0; 30 such starts), else a random slice start (11,685 candidates; frames too close to episode end or in a dead zone are excluded). Slices give the model sequences that *begin mid-episode* (memory starts blank there — decision-relevant frames get oversampled via the reveal-frame geometry).
- **Sequence buckets** `(14, 27, 40)` steps with `memory_min_slice_steps=14`: each batch is bucketed to one length (`SequenceBucketBatchSampler`) to bound padding waste; logged `sequence_bucket_steps` ≈ 38.5 average.
- Prompt: `"find the bin with banana"` injected as default; `max_token_len=80` (audited max context 69), `causal_token_len=128` (audited max 123).

### 4.3 Optimization

- Batch 12 (sequences) — note each sequence is 40 steps × 4 LLM calls, so the effective compute is large.
- AdamW, `clip_gradient_norm=1.0`; cosine schedule warmup 200 → peak LR 5e-5 → decay over 30k to 5e-5 (flat 5e-5 in effect); EMA 0.999.
- Init: `pi05_base` pretrained weights via partial loader (new modules — compressors, memory, gate — fresh). Trains **from scratch as a memory model**: never resumes v3/v3.1 checkpoints.
- Frozen: only `.*memory/gate.*` (Titans θ/η/α gate Linear). SigLIP and everything else train.
- 20,000 steps, checkpoint every 250 (orbax keeps the latest + every 5000th), seed 42.
- Model size: **3.40B params**. Throughput: ~26 s/step on 2×H200 (FSDP, bs12); ~35 s/step single H200. Compute-bound (GPU util 100%); double rematerialization (per-block `nothing_saveable` + per-step `jax.checkpoint`) — `dots_saveable` OOMs because one step makes four 18-layer LLM calls (early prefix, late prefix, causal, suffix).

---

## 5. Current status (2026-08-15)

- Training run `dualquery16_l8_s15_d6_t40_b14-27-40_tb25_bs12_bf16fast_save250_seed42` (wandb `openpi/6ly6zd6w`): step ~5,650/20,000, CE ≈ 0.021, flow ≈ 0.0018, grad norm 0.3–1.0, no instabilities. ETA ~4 days.
- Checkpoints on disk: step 5000 (permanent) and the rolling latest; 10000/15000/20000 will persist.
- **Step-5000 evaluation done** (see `docs/v32_ckpt5000_diagnostics.md` for full numbers). Headlines:
  1. **Memory pathway effectively inert:** zero-read counterfactual changes nothing (0/120 decoded-subtask flips, action MSE ~1e−7, GT log-prob Δ ~1e−6) — *except* a consistent ~1-nat logit shift at exactly the decision frame in both test episodes.
  2. `memory_gate` norm has grown from 0 to only 0.045 → injected memory ≈ <1% of token scale.
  3. Read queries are uniform (entropy ≈ 0.998, all 16 slots identical behavior); write queries mostly uniform with occasional dips that focus on **letterbox padding**, not on bins/banana.
  4. Write magnitudes near-uniform across slots (CV ≈ 0.002–0.005). (θ/η/α constant by design — frozen.)
  5. **Evaluation caveat discovered:** the model decodes the correct side *with memory zeroed* even at post-closure ambiguous frames → side information leaks via proprioceptive state/arm posture and/or train-episode memorization. Only counterfactual probes (zero-read, memory-swap) in the closure→decision window are trustworthy evidence of memory use; train-episode subtask accuracy is not.
  6. Working hypothesis: CE is nearly solved through leak channels → weak gradient pressure to open the memory gate. Watch gate norm and the decision-frame divergence at 10k/15k/20k.

---

## 6. Implementation map (where everything lives)

| Component | Location |
|---|---|
| v3.2 spec (design intent, invariants, required tests) | `../V32_MEMORY_HANDOFF.md` |
| Model: compressors, split prefix, sequence loss, sampling | `src/openpi/models/pi0.py` — `MemoryQueryCompressor`, `_v32_prepare_memory_prefix`, `_compute_sequence_loss_v32`, `_sample_with_memory_v32`, `v32_query_attention_step` (diagnostic) |
| Config knobs + validation | `src/openpi/models/pi0_config.py` (`memory_architecture="v32_layer8_dual_query"` requires memory_layer=8, query_compressed writer, 16 query tokens) |
| Titans memory | `src/openpi/models/memory.py` |
| Partial-layer Gemma + fixed KV cache + remat knob | `src/openpi/models/gemma.py` |
| RTC (action-prefix conditioning) | `src/openpi/models/rtc.py` |
| Train config `pi05_yam_mem_v32` | `src/openpi/training/config.py` |
| Sequence building / bucketing / oversampling | `src/openpi/training/data_loader.py`, `src/openpi/transforms.py` |
| Regression tests (8, incl. the 6 spec-required) | `src/openpi/models/pi0_v32_test.py`; diagnostics hooks: `pi0_v32_diagnostics_test.py` |
| Checkpoint diagnostics runner (zero-read, attention maps, write stats) | `scripts/v32_checkpoint_diagnostics.py` → `src/openpi/diagnostics/v32_checkpoint.py` |
| Step-5000 results | `docs/v32_ckpt5000_diagnostics.md`, artifacts in `diagnostic_outputs/v32_ckpt5000/` |

### Known gotchas

1. **v3.2 checkpoints fail `Pi0Config.load`** with a `{'bias'}` structure mismatch: flax 0.10.2 keeps `bias: None` state entries for the bias-free compressor Linears; orbax drops `None` on save. Fix: project restored params onto the abstract state and refill exactly-`None` slots, then `load(..., remove_extra_params=False)` — implemented as `_align_params` in `v32_checkpoint.py`. Any new checkpoint consumer needs this.
2. Spec §9 forbids adding mechanisms mid-run (write gates, probes, heuristics) — keep the first v3.2 run a clean comparison.
3. The seven post-training diagnostics the spec asks for (§12): read/write query maps ✅, retrieval over time ✅, writer norms ✅, zero-read ✅ (all in the runner); left/right **memory swap**, and **freeze-after-visible** are still to be built.

---

## 7. Regression tests guarding the invariants

`pi0_v32_test.py` (all passing): config validation (layer-8/query-compressed/16-token enforced); dual banks distinct and emit 16 tokens; write tokens depend only on current `h8_top` while read uses the pre-write state; retrieved tokens exist only after layer 8 and change the late representation; prediction identical before commit and only `allow_write` changes state; gradients reach both query banks; end-to-end sequence CE reaches queries and slow memory; BF16 compute keeps FP32 masters.

## 8. Success criteria (spec §13) and open questions

Implementation criteria (all verified): dual 16-slot banks over layer-8 top-camera source, independent parameters, 16 retrieved tokens injected after layer 8 surviving to the action pathway, no c18/retrieved-memory writer source, causal read-before-write, v3.1 mechanics otherwise unchanged. **Task success is an empirical question** — currently open. The live questions for analysis:

1. Does `memory_gate` keep growing over training, and does the zero-read divergence widen from the single decision frame into the whole post-closure window? (Rerun the diagnostics at 10k/15k/20k — one command, §5 of the diagnostics doc.)
2. Does the side decision *causally* come from memory? Needs the left/right memory-swap test at post-closure frames (separates memory from the proprioceptive leak).
3. How much side information does proprioception alone carry? (Probe the 14-D state at frames 400–495.)
4. With α ≈ 0.01 frozen and ~40–60 uniform writes/episode, does reveal-time content survive to the decision frame? (Fast-weight drift data already collected.)
5. If CE is solved via leaks and the gate stays closed, the dataset/objective — not the architecture — may be the binding constraint; the decision-frame CE (oversampled) is the pressure point to analyze.
