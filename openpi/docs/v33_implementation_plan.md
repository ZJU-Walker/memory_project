# v3.3 implementation plan (grounded in this repo)

Source: `V33_MEMORY_HANDOFF.md` (the design intent). This document maps that design onto the
actual v3.2 code and data, corrects the points where the handoff's assumptions don't match the
repo, and defines the concrete changes. Status: **approved and implemented** (converter,
sampler, conditioner, configs, §16 probe, tests); see §9 for the remaining launch steps.
Serving note: `pi05_yam_mem_v33` has no `default_prompt` — the client/eval script must send
the task's instruction ("find the banana" / "find the grey pepper box") with every request.

## 0. What v3.3 is

Two focused changes over `pi05_yam_mem_v32`, trained fresh from `pi05_base` on the new
60-episode two-task dataset:

1. **Task-conditioned write queries** — the 16 learned write queries are shifted by a
   zero-init cross-attention over the instruction/context token hidden states at layer 8,
   so the writer can select task-relevant content ("find the banana" vs "find the grey
   pepper box" over the same scene).
2. **Memory-critical sampling** — a new sampling branch (~50 % of draws) that starts just
   before `inspect both bins` and *ends at a random point inside the neutral wait phase*,
   so the ordinary subtask CE at the endpoint can only be solved from memory.

No new losses, no task-specific heads, gates stay frozen, read path unchanged.

## 1. Dataset facts (verified against the labels on disk)

- 60 demos, all labeled, perfectly balanced: banana 15 L / 15 R, grey box 15 L / 15 R.
- Every episode follows the 5-phase schema in order; sided labels are consistent.
- Episode lengths 713–1115 frames; inspect spans 28–75 frames; wait spans 13–337 frames
  (mean 63). **One wait span (13) is shorter than the 15-frame subtask lookahead** → the
  endpoint window needs a clamp rule (§4).
- `0816_grey_box/demo30`: videos truncated to 740 frames but the labels were made on that
  timeline and end at 739 — usable as-is, only some execution frames are short.
- The old `bin_memory_banana` dataset is NOT mixed in. v3.3 trains on the new 60 only.

## 2. Corrections / gaps vs the handoff

| # | Handoff says | Repo reality → what we do |
|---|---|---|
| 1 | "condition on H_I, hidden representation of the instruction" | `_v32_prepare_memory_prefix` already computes layer-8 hiddens for **all** prefix tokens; the non-image slice `h8_all[:, num_img:]` (instruction + state tokens, with `prefix_mask`) is currently discarded. Conditioning on it is free — no extra transformer pass. It includes state tokens; both are inference-available, and the §17.3 factorial diagnostic isolates the instruction effect. |
| 2 | implicit: prompt exists per task | **The pipeline has ONE constant prompt** (`default_prompt` + `InjectDefaultPrompt`; the LeRobot `task` field is occupied by the subtask). v3.3 needs per-episode prompts: converter writes a `meta/episode_prompts.json` sidecar, a new raw-item transform injects `prompt` per episode (InjectDefaultPrompt already no-ops when `prompt` is present; `YamInputs` already passes it through). |
| 3 | "sample t_q, end inside waiting" | Our sampler is **start-anchored**: weights over start frames, sequences run to episode end, bucketed by length. Memory-critical = new weight branch over starts in `[inspect_start − pad, inspect_start]` + per-draw endpoint truncation of `seq_step_mask` in `BuildMemorySequence`. Geometry: 17–27 steps at stride 15 → existing bucket 27 fits; no bucket changes. |
| 4 | §16 "verify gradient reaches inspect writes" | **The current TBPTT would break it by construction**: `memory_block_steps=25` with a random shift cuts backprop through the memory state mid-sequence, severing wait→inspect credit in most bucket-27 samples. Decision (user): keep the fences for the memory/compute profile of normal T40 samples, and emit **no fence inside memory-critical samples** — `seq_block_boundary` is already per-sample and data-driven, and at ≤27 valid steps a memory-critical sample's differentiated chain is no longer than a normal sample's 25-step block. The §16 check verifies the credit path empirically. |
| 5 | reveal-frames dead zone | `_episode_info_table`'s `switch` = first label change = start of *inspect* under 5-phase labels — the old reveal/switch dead-zone rule is semantically wrong for the new data. Replace with a label-derived rule: slice starts forbidden in `(evidence_end, action_start)` (blank memory + side labels ⇒ grading teaches guessing). `memory_reveal_frames_path` is not used for v3.3. |
| 6 | §15 store `memory_required` per frame | Not needed: derive phases from the label strings via two config lists (`memory_required_subtasks`, `evidence_subtasks`). Same generic effect, no conversion-format coupling. |
| 7 | token budget unstated | v3.2's `max_token_len=80` / `causal_token_len=128` were audited on the OLD dataset. New labels are longer ("close both lids and reset arms", "wait; target bin is …") and prompts differ → re-audit over the converted dataset and set both accordingly. |
| 8 | — | New norm stats + assets dir for the combined repo (`./assets/pi05_yam_0816`), computed with a plain non-memory config, mirroring how `pi05_yam` assets serve v3/v3.1/v3.2. |
| 9 | — | Serving/eval must send the right instruction per task (client-side `--prompt`); flag when touching eval scripts. |

## 3. Data conversion (first, it's the slow step)

**`examples/yam/convert_yam_data_to_lerobot.py`**
- Accept multiple sources: `--data_dirs <banana_dir> <grey_dir> --instructions "find the banana" "find the grey pepper box" --repo_name yam/bin_memory_0816_subtask`.
  (Back-compat single-dir usage kept.)
- All 60 episodes go into ONE repo; after each `save_episode`, record
  `episode_index → instruction`; at the end write `meta/episode_prompts.json` into the
  dataset directory.
- Everything else (features, FPS=30 convention, `_load_frame_subtasks` contract) unchanged.

Commands:
```bash
uv run examples/yam/convert_yam_data_to_lerobot.py \
  --data_dirs /iris/u/kewalk/memory_project/data/0816_banana \
              /iris/u/kewalk/memory_project/data/0816_grey_box \
  --instructions "find the banana" "find the grey pepper box" \
  --repo_name yam/bin_memory_0816_subtask
# token-length audit (new small script; sets max_token_len / causal_token_len)
uv run scripts/v33_audit_token_lengths.py --config-name pi05_yam_0816
# norm stats with the plain config
uv run scripts/compute_norm_stats.py --config-name pi05_yam_0816
```

## 4. Sampling: memory-critical branch

**`src/openpi/training/config.py` — `DataConfig` gains:**
- `memory_required_subtasks: tuple[str, ...] = ()` — the wait labels.
- `evidence_subtasks: tuple[str, ...] = ()` — `("inspect both bins",)`.
- `memory_critical_prob: float = 0.0` — mass of the new branch (0.5 for v3.3).
- `memory_critical_start_pad: int = 75` — start window size before inspect (5 steps).
- `episode_prompts_path: str | None = None` — the sidecar from §3.

**`data_loader._episode_info_table`** additionally derives, per episode, from the label
strings + config lists: `evidence_start/end`, `memory_lo/hi` (first/last wait frame),
`action_start = memory_hi + 1`, and the balance cell = (instruction, side). Episodes missing
a phase are excluded from the branch with a warning.

**`data_loader._sequence_sampling_info`** builds three-branch weights:
- full trajectories: `(1 − slice − mc)` mass on frame-0 starts (unchanged);
- slices: `slice` mass, allowed starts now exclude the new dead zone `(evidence_start,
  memory_hi]` and the memory-critical window. The dead zone starts at the FIRST evidence
  frame (adversarial-review fix): a slice starting mid-inspection may already have missed
  the revealing glimpse, yet the waiting labels ahead still grade the side — partial
  evidence teaches guessing exactly like none. Episodes whose memory-required labels are
  not contiguous (stray label mid-execute) are excluded entirely, since a stretched
  window would let endpoints grade "memory" on visible evidence;
- memory-critical: `mc` mass split **equally over the 4 (task, side) cells**, then equally
  over episodes in the cell, then uniformly over starts in
  `[max(1, evidence_start − pad), evidence_start]`.
  `valid_steps` for these starts = `memory_critical_endpoint(start) + 1` — the endpoint is
  DETERMINISTIC per start frame (shared helper with `BuildMemorySequence`), so bucket
  assignment is exact; endpoint diversity comes from the uniformly drawn start.
- Mixture for v3.3: full 0.25 / slice 0.25 / memory-critical 0.50.

**`transforms.MemoryEpisodeInfo`** attaches the per-episode phase bounds;
**`transforms.BuildMemorySequence`**: when the item's start frame is inside the
memory-critical window, truncate `seq_step_mask` at `memory_critical_endpoint(start)` — the
**deterministic** stride-grid step whose observation lies in
`[memory_lo, memory_hi − lookahead]` (fallback tiers cover the 13-frame wait and grids that
straddle a short wait). Steps beyond `t_q` are loss-masked and their writes are no-ops,
exactly like end-of-episode padding today.

Endpoints must be deterministic per start frame: the bucket sampler assigns each start's
exact valid length ahead of time, and a per-draw random endpoint mixes lengths inside one
bucket batch whenever a long wait crosses a bucket boundary (this tripped
`_sequence_bucket_collate_fn`'s homogeneity check on the first launch — waits reach 337
frames). Endpoint DIVERSITY (§11) comes from stratification instead: consecutive starts
cycle through the eligible waiting steps, and the start itself is drawn uniformly from the
window. The identical helper runs in the sampler (for `valid_steps`) and in the transform
(for the mask), with a regression test asserting they agree on every window start.

**TBPTT:** `memory_block_steps=25` stays for normal samples; `BuildMemorySequence` emits no
fence for memory-critical samples (see correction #4), so
`L_wait → m_tq → M → z_inspect → Q^w(I)` is differentiable end-to-end exactly where §16
requires it.

## 5. Model: task-conditioned write queries

**`pi0_config.Pi0Config`**: `memory_task_conditioned_write: bool = False`.

**`pi0.py` — new `MemoryQueryConditioner(nnx.Module)`**:
`Q(I) = Q0 + out_proj(CrossAttn(Q0 → H_ctx, ctx_mask))` with
- `H_ctx = h8_all[:, num_img:]`, `ctx_mask = prefix_mask[:, num_img:]` (padding masked in
  the softmax);
- q/k/v projections like `MemoryQueryCompressor` (FP32 params, bf16 compute), **out_proj
  zero-init** → at initialization `Q(I) ≡ Q0` and the whole model is *exactly* v3.2
  (unit-tested equality). Same stability discipline as the memory gate / adaLN-zero.

**`MemoryQueryCompressor.__call__`/`attention_probs`** gain an optional `queries=` argument
(default: broadcast the bank, current behavior).

**`_v32_prepare_memory_prefix`** (shared by training loss, serving, and all diagnostics —
one wiring point): when enabled, compute `Q(I)` and pass it to the **write** compressor
only. Read queries and everything downstream unchanged.

## 6. Train config `pi05_yam_mem_v33`

Copy of `pi05_yam_mem_v32` with: repo `yam/bin_memory_0816_subtask`; assets
`./assets/pi05_yam_0816`; `memory_task_conditioned_write=True`; `memory_block_steps=25`
(fences skipped per-sample on the memory-critical branch);
re-audited `max_token_len`/`causal_token_len`; data fields
`memory_required_subtasks=("wait; target bin is left", "wait; target bin is right")`,
`evidence_subtasks=("inspect both bins",)`, `memory_critical_prob=0.5`,
`episode_prompts_path=<sidecar>`; no `default_prompt` (per-episode prompts), no
`memory_reveal_frames_path`. Unchanged: T40/S15, buckets (14,27,40), lookahead 15,
batch 12, lr 5e-5 cosine, gate freeze, probe off, save_interval 250, fresh
`PartialCheckpointWeightLoader(pi05_base)`. Plus a plain `pi05_yam_0816` config for norm
stats. Optional-but-cheap: `remat_policy="dots_saveable"` on H200 (committed knob).

## 7. Verification before launch (§16 first)

1. **Unit tests** (CPU): conditioner zero-init ⇒ bitwise v3.2 equality; masked-softmax
   correctness; sampler cell balance / dead zone / window; endpoint truncation respects
   `[memory_lo, memory_hi − lookahead]`, cap, and the 13-frame-wait clamp; prompt sidecar
   injection (banana vs grey episodes get different prompts, InjectDefaultPrompt untouched
   behavior elsewhere); converter multi-dir round trip; `debug_mem`-style CPU run through
   the v3.3 code path.
2. **Gradient-flow check (§16)** — new `scripts/v33_gradient_flow_check.py`: on real
   memory-critical batches, add a zero "tap" to each step's write tokens and report
   `g_τ = ‖∂CE(t_q)/∂z_τ‖` across the history. Must be nonzero at inspect-phase steps.
   Run on the test GPU before interpreting any training.

## 8. Diagnostics at checkpoints (mapped to existing tools)

- **Zero-read at wait frames** (§17.1): existing `retrieved_token_ablation_step` via
  `scripts/v32_checkpoint_diagnostics.py`, pointed at wait-phase frames of the new data.
  Success: zero-read degrades side prediction.
- **Memory swap** (§17.2): existing `memory_swap_read_step` — prediction should follow the
  swapped memory under a fixed neutral wait observation.
- **Instruction-factorial writer** (§17.3, new small step method): same inspect frame,
  both prompts → write-token cosine separation must be ≫ frame jitter (the v3.2
  side-separability harness `v32_side_separability.py` reused for the comparison).
- **Attention maps + slot/similarity stats** (§17.4): existing `v32_query_attention_step`.
- **Gradient-over-time** (§17.5): the §16 script rerun at checkpoints.

## 9. Order of work

1. Converter extension + run conversion (slow; start first).
2. Data-side code: sidecar prompt injection, phase table, 3-branch sampler, endpoint
   truncation + all data tests.
3. Model-side: conditioner + wiring + equality tests.
4. Token audit → norm stats → `pi05_yam_0816` / `pi05_yam_mem_v33` configs.
5. §16 gradient check on GPU.
6. Launch training; run §8 diagnostics at ckpt 1000/2500/5000.

## 10. Success criteria (§18, unchanged in substance)

Writer separates by instruction on identical frames; retention survives close/reset (wait
CE with memory ≫ zero-read); memory swap flips the predicted side; and the sampled wait
subtask accuracy at long retention horizons tracks memory, not shortcuts.
