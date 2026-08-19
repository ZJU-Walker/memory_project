# π0.5 + Titans episodic memory — v3.3 status report

**Run**: `pi05_yam_mem_v33` / `v33_run1` (wandb `7my510hb`), 2×H200, fresh from `pi05_base`.
**Report date**: 2026-08-19, training at step ~6250 of 20 000.
**Scope**: what v3.3 changes relative to v3.2, how it is trained and verified, and what five
checkpoints of writer-attention diagnostics show.

The short version: **training converges well (loss 7.85 → 0.062, CE-dominated) and the subtask
policy is strong (waiting-phase side correct 18/18), but the episodic memory is not being used
to produce that answer.** Zeroing the memory read leaves side accuracy at 18/18; a
contradicting donor memory never flips a prediction; the read moves the left-vs-right decision
by ~0.2–0.4 nats against margins of 15–22 nats. In parallel the writer's spatial attention went
anchor → collapse (bin focus 0.36 at step 4250 → 0.001 at 6250). The likely root cause is that
the waiting observation leaks the answer — proprioception alone recovers the side at 70% — so
nothing pressures the model to remember. Evidence in §6.2 and §6.4; consequences in §7.

---

## 1. Problem setting

Bimanual YAM robot, "which bin holds the object I was asked to find?" Each episode:

| phase | subtask label | what happens |
|---|---|---|
| approach | `open both lids` | arms open both bin lids |
| **evidence** | `inspect both bins` | both bins visible and open; **the answer is observable here** |
| retention | `close both lids and reset arms` | lids close; the answer is no longer visible |
| **waiting** | `wait; target bin is left/right` | neutral pose; the label encodes the remembered answer |
| execute | `open left bin` / `open right bin` | acts on the remembered side |

Both objects are always present, one per bin (a banana and a grey pepper box). The
instruction — "find the banana" or "find the grey pepper box" — is per-episode, and the
`inspect` → `close` → `wait` structure means the side label during waiting can only be
produced from memory of the evidence phase.

Dataset: `yam/bin_memory_0816_subtask`, 60 episodes / 53 593 frames, 7-label subtask
vocabulary, per-episode instructions in `meta/episode_prompts.json`. All 60 episodes have
contiguous, correctly ordered phases; the four (instruction × side) cells are exactly
balanced at 15 episodes each.

## 2. Architecture recap (what v3.2 already did)

`memory_architecture = "v32_layer8_dual_query"` — a Titans-style fast-weight memory attached
to a π0.5 VLA at transformer block 8:

1. blocks 0–8 run on the prefix → `h8_all`; the 256 top-camera slots are `h8_top`;
2. two learned 16-query banks cross-attend to `h8_top`: a **read** bank and a **write** bank;
3. the read queries retrieve from the Titans memory; the retrieval passes a **zero-initialized
   content gate** and is injected as 16 extra tokens visible to blocks 9–17;
4. the model predicts (subtask CE + action flow), and only *then* the write tokens update the
   fast weights — the causal order read → predict → write, which is what makes credit
   assignment through the memory meaningful;
5. Titans write gates are frozen (`freeze_filter = PathRegex(".*memory/gate.*")`) at their
   measured stable operating point; the zero-init content gate keeps initialization identical
   to a memory-free model.

Sequence training: one sample is T consecutive prediction steps at stride 15 frames, scanned
with rematerialization so GPU memory does not grow with T.

## 3. What v3.3 adds

### 3.1 Task-conditioned write queries

v3.2's write queries were a fixed learned bank — the writer stored the same *kind* of summary
regardless of the task. v3.3 conditions them on the instruction:

```
Q(I) = Q0 + out_proj( CrossAttn( Q0 → h8_all[:, num_img:], prefix_mask[:, num_img:] ) )
```

- the context is the **non-image prefix** (tokenized instruction + state), whose layer-8
  hidden states blocks 0–8 already computed — no extra transformer pass;
- `out_proj` is **zero-initialized**, so at step 0 v3.3 is bitwise identical to v3.2 and the
  pathway opens only as far as the objective pushes it (same discipline as the content gate);
- padding is masked out of the softmax, with an all-padding guard.

Implementation: `MemoryQueryConditioner` in `src/openpi/models/pi0.py`, enabled by
`Pi0Config.memory_task_conditioned_write` (validated to require the v3.2 architecture).

### 3.2 Memory-critical trajectory sampling

The v3.2 sampler drew full trajectories and random slices. Neither reliably produced the one
thing this task needs: a sequence that *starts before the evidence* and *ends inside the
waiting phase*, so the endpoint's subtask CE is answerable only from memory. v3.3 adds a third
branch (50% of samples):

- start uniformly in `[evidence_start − 75, evidence_start]` (clipped at frame 1);
- truncate at a waiting-phase endpoint;
- mass balanced equally over the four (instruction, side) cells, then episodes, then starts;
- these samples carry **no TBPTT fence**, so the endpoint CE backpropagates through the whole
  wait → retention → evidence chain (normal samples keep `memory_block_steps=25` for the
  memory/compute profile).

**Endpoints are deterministic per start frame.** This is not cosmetic: the bucket sampler must
know each sample's exact length in advance, and a per-draw random endpoint mixes bucket lengths
inside a batch — that bug crashed the first launch with
`sequence bucket batch is not homogeneous`. Endpoint variety instead comes from the uniformly
drawn start; consecutive starts stratify across the eligible waiting-phase grid steps
(`transforms.memory_critical_endpoint`, shared by the sampler and the fetch-time transform,
with a property test walking every window start of every episode).

### 3.3 Supporting changes

- **Per-episode prompts**: the LeRobot `task` field is occupied by the subtask, so instructions
  live in a `meta/episode_prompts.json` sidecar written by the converter and injected by
  `InjectPromptFromEpisode` (strict — a missing entry is an error, not a fallback).
  *Serving consequence*: v3.3 has **no `default_prompt`**; clients must send the instruction.
- **Label-derived phases**: `evidence_subtasks` / `memory_required_subtasks` config lists
  replace the old reveal-frame JSON; phase bounds come from the labels themselves.
- **Token budget re-audit** over all 53 593 frames: context max 68, causal max 122 →
  `max_token_len=80`, `causal_token_len=128` (same static shapes as v3.2).

## 4. Verification performed

**Adversarial review** (commit `2f10f99`) found and fixed three real training bugs:

1. **Mid-inspect slice contamination** — the slice dead zone started at `evidence_end`, so a
   slice could start mid-inspection having missed the reveal yet still be graded on the side
   labels ahead. That is training to guess. Dead zone now starts at `evidence_start`
   (3 052 contaminated starts on this dataset, ~9% of the slice branch).
2. **Correlated worker RNG** — `_worker_init_fn` never reseeded numpy after fork, so all 12
   dataloader workers shared one stream (TBPTT fence shifts identical across workers).
3. **Unguarded non-contiguous wait labels** — a stray waiting label mid-execute would stretch
   the memory window over frames where the answer is visible. Such episodes are now excluded
   with a warning (this dataset is clean; the guard protects future data).

**Test suite**: 68 tests green, including a property test asserting the sampler's predicted
length equals the transform's actual mask sum for every memory-critical start under four
hostile geometries that all occur in the real data (wait shorter than the stride; episode
ending inside the wait; pad-clipped window; non-contiguous labels).

**§16 gradient-flow check** (`scripts/v33_gradient_flow_check.py`) on real memory-critical
batches: the endpoint's own write receives **exactly zero** gradient (confirming causal order
read → predict → write), and the waiting endpoint's CE reaches the **evidence-phase writes with
the largest gradients in the history** (max 0.57, ~2–4× the retention steps) through 13–15
fence-free steps. The credit path v3.3 depends on is real.

**End-to-end**: 20 real loader batches collate homogeneously; a 6-step real `train.py` run
compiled all three bucket shapes, loaded `pi05_base` with the new conditioner grafted, and
checkpointed cleanly.

## 5. Training progress

| step window | total loss | CE | flow | grad norm |
|---|---|---|---|---|
| 0–250 | 7.847 | 7.813 | 0.034 | 66.8 |
| 2500–2750 | 0.164 | 0.161 | 0.0032 | 1.93 |
| 3250–3500 | 0.118 | 0.115 | 0.0028 | 1.33 |
| 4000–4250 | 0.097 | 0.095 | 0.0026 | 1.48 |
| 4750–5000 | 0.078 | 0.075 | 0.0024 | 1.22 |
| 6000–6250 | 0.062 | 0.060 | 0.0022 | 0.82 |

Healthy, monotone, CE-dominated (the subtask objective is what drives memory learning; the flow
loss sits behind `stop_gradient(kv)` by design). Sequence-valid fraction 0.907, mean bucket
length 34.4 steps.

## 6. Writer-attention diagnostics

### 6.1 Method

`scripts/v33_writer_attention.py` replays four episodes (one per instruction × side cell) at
the training write cadence, carrying the real memory state — each frame commits exactly the
`write_tokens` the model would write. Per frame it records the write-query attention over the
256 top-camera patches in three variants:

- **TRUE** — conditioned on the episode's real instruction;
- **CF** — identical frame and memory state, counterfactual instruction;
- **BASE (Q0)** — the unconditioned bank, the within-frame baseline.

Outputs per episode: a 4-panel H.264 video (camera | TRUE | CF | |TRUE−CF|), 16-tile per-slot
grid videos, an NPZ of head-averaged maps, and per-frame metrics.

**A note on reading the videos.** Attention here is extremely peaked — the single top patch
typically holds 35–40% of a frame's mass and the top five hold ~85%. Two renderings are
produced because they answer different questions:

- `*_slots_v32.mp4` — the v3.2 convention: each tile normalized by its own per-frame maximum,
  JET colormap, opacity proportional to intensity. Every slot's peak saturates red, so *where*
  a slot looks is obvious at a glance. Purely relative: it says nothing about magnitude.
- `*_slots.mp4` and the 4-panel video — one fixed scale across the episode, inferno, flat
  alpha, so frames are comparable over time. Against that fixed ceiling ~96% of patches render
  below 2% brightness, which makes a genuinely sharp map look like an almost untouched image.

Both agree with the numbers: in the v3.2-style rendering at 4250 every slot shows a saturated
blob on the right bin, while at 6250 the blobs sit on empty table and letterbox padding.

`scripts/v33_writer_correctness.py` grounds those maps in scene geometry. The bins occupy fixed
patch regions of the letterboxed 16×16 grid — rows 6–9, left bin cols 5–7, right bin cols 8–10,
calibrated visually against evidence frames. (Caution for anyone recalibrating: transparent bin
*lids* lie on the table beside the bins during open phases; an earlier cols-4–6 box measured the
lid, not the bin.) It reports attention mass inside the bins, target-vs-distractor preference,
and TRUE-vs-CF steering.

### 6.2 Results across five checkpoints

| step | gate ‖·‖ | conditioner ‖out_proj‖ | evidence JS(TRUE,CF) | evidence ‖ΔQ‖ | **evidence bins focus** | waiting bins focus |
|---|---|---|---|---|---|---|
| 2750 | 0.0356 | 2.96 | 0.062 | 83 | 0.207 | 0.078 |
| 3500 | 0.0427 | 3.71 | 0.101 | 142 | 0.265 | 0.003 |
| 4250 | 0.0502 | 4.31 | 0.173 | 281 | **0.355** | 0.131 |
| 5000 | 0.0567 | 4.77 | 0.213 | 403 | 0.185 | 0.015 |
| 6250 | 0.0630 | 5.30 | 0.127 | 569 | **0.001** | 0.000 |

Uniform-attention baseline for bins focus is 0.094 (24 of 256 patches); the unconditioned Q0
bank stays at that baseline (0.093–0.100) at *every* checkpoint — all bin-seeking behavior came
from the conditioning pathway.

**Three findings.**

**(a) The conditioning pathway is alive and monotonically strengthening.** `out_proj` grows
2.96 → 5.30 and the instruction moves the query bank ever harder (‖ΔQ‖ 83 → 569, a 6.9×
increase). Swapping the instruction changes the written tokens by 15–40% of their norm. The
v3.3 mechanism is doing *something* substantial and growing.

**(b) It found the bins, then lost them.** Bins focus rose to 3.8× uniform at 4250, then
collapsed to 0.001 — 100× *below* uniform — at 6250. This is not one noisy slot: at 2750–5000
**all 16** write slots put >5% of their mass in the bin regions; at 6250 **zero of 16** do. The
top attended patches moved from the right-bin object (patch (8,8), consistent across episodes)
to empty table, letterbox padding, and arm hardware ((10,3), (0,9), (12,1)) — classic
attention-sink positions. Verified by direct per-patch rendering on the raw frames.

**(c) Even at its best, the spatial behavior was a shortcut, not object selection.** At 4250
the anchor was the *same* patch in every episode regardless of instruction or object placement
— it sat on the grey box in one episode and on the banana in another — and TRUE-vs-CF steering
of bin choice was ≈0 (win rate 0.42, i.e. chance). Because both objects are always present, one
per bin, "always encode the right bin" is a *sufficient statistic* for the side answer: knowing
the grey box is on the right plus the instruction "find the banana" implies left. The task
structure permits this shortcut; nothing in the objective demands genuine instruction-driven
looking.

### 6.3 What this does and does not mean

The loss keeps falling *while* attention leaves the bins, so the model is solving the subtask
objective by some route that no longer requires spatially attending the evidence. Candidate
explanations, in rough order of my confidence:

1. **The information is already elsewhere in the token stream.** The write tokens are built
   from `h8_top`, but blocks 0–8 have already mixed information globally; a "sink" patch can
   carry a summary of the scene. Attention position then stops tracking semantics. This would
   make the collapse benign-but-uninformative rather than harmful.
2. **The content gate is still nearly closed** (0.063 after 6250 steps). The predictor consumes
   little retrieval, so pressure on *what* the writer looks at is weak; the writer may be
   drifting under the weaker instruction-conditioning gradient rather than being shaped by
   downstream usefulness.
3. **Shortcut consolidation**: with the right-bin statistic sufficient, the model may have
   compressed it into a representation that no longer needs the bin patches at all.

**These cannot be distinguished from attention maps alone.** The decisive test is the zero-read
ablation at waiting frames: if zeroing the retrieval collapses the side prediction, memory is
carrying the answer regardless of where attention points; if it does not, the model is solving
the task without memory and the whole pipeline needs rethinking. That test is implemented for
v3.2 (`v32_checkpoint.py`, section 12.6) and is the top-priority next diagnostic.

## 6.4 Offline eval: the memory is not being used (decisive)

`scripts/v33_offline_eval.py` answers the question §6.3 left open. It replays the same four
episodes through `sample_with_memory` at the write cadence and **scores every prediction
against the ground-truth label**, with three memory conditions at each step: normal (reads the
live memory, commits its write), **zero-read** (retrieval zeroed), and **swap** (reads a donor
episode's memory built by replaying an episode with the *opposite* side through its evidence
phase). It also teacher-forces both side answers to measure how many nats the read moves the
left-vs-right decision — catching influence below the argmax threshold.

Results at step 6250 (4 episodes, 243 scored predictions):

| phase | n | exact | exact (zero-read) | side acc | side acc (zero-read) | follows swap | mean abs margin shift |
|---|---|---|---|---|---|---|---|
| approach | 73 | 0.95 | 0.95 | — | — | 0.05 | 0.343 |
| evidence | 12 | 0.67 | 0.67 | — | — | 0.00 | 0.293 |
| retention | 58 | 0.93 | 0.93 | — | — | 0.00 | 0.202 |
| **waiting** | 18 | 0.83 | 0.67 | **1.00** | **1.00** | **0.00** | 0.379 |
| execute | 82 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.142 |

**The subtask policy is good.** Execute is perfect, retention 0.93, and the waiting-phase side
is correct in **18 of 18** steps. The residual "errors" are one-step phase-boundary timing
(emitting `inspect both bins` one step before the label switches), not semantic mistakes.

**The memory contributes nothing to the answer.** Three independent measurements agree:

1. **Zero-read keeps side accuracy at 18/18.** The apparent +0.17 exact-match gain in the
   waiting phase is *entirely* timing: every normal-vs-zero-read disagreement is
   `wait; target bin is right` vs `open right bin` — the same side, a different phase word.
   Not one disagreement is a side flip.
2. **A contradicting memory never changes the answer** (`follows_swap = 0.00` in every phase
   after approach). Feeding a frame the memory of an opposite-side episode leaves the
   prediction exactly where it was.
3. **The margins say why.** The forced log-probability gap between the two side answers is
   **15–22 nats**, while zeroing the read moves it by only **0.1–1.0 nats**. The decision is
   made ~20× more strongly than anything the memory could influence — consistent with a
   content gate still at 0.063.

**Where the answer actually comes from.** The waiting-phase observation is not as neutral as the
task design assumes: a leave-one-out logistic regression on the robot's own joint state during
waiting recovers the side at **70%** (chance 50%), and one gripper joint separates the classes
by ~20 pooled SD within any single episode's wait window. So proprioception leaks a substantial
part of the answer, and the wrist cameras plausibly supply the rest — the model can be right
without remembering anything. This also explains §6.2's attention collapse: if the current
observation carries the answer, there is no gradient pressure to keep attending the evidence,
and none to open the content gate.

Artifacts: `diagnostic_outputs/v33_offline_eval/6250_v2/episode_*_eval.mp4` (H.264, 30 fps,
672×928) burn in ground truth, all three decodes, the margin shift, and a running waiting-phase
scoreboard; `episode_*.json` holds every per-step record.

## 6.5 Write-token probe: was the answer ever written down?

§6.4 proved the *policy* ignores the memory. That is ambiguous between two failures needing
opposite fixes:

* **never written** — the side never entered `W_t`, so the writer is the broken component;
* **written but never read** — `W_t` holds the side and the read/gate path is the broken link.

`scripts/v33_write_token_probe.py` discriminates by fitting a **diagnostic-only** linear readout
(no loss, no gradient reaches the model) on the write tokens `W_t ∈ R^{16×d}` harvested at the
training write cadence while carrying real memory state:

    W_t  ->  {left, right}

**Methodology (the result is worthless without it).** Frames inside one episode share lighting,
object placement and arm pose, all correlated with side, so a frame-level split lets the probe
memorize episode identity and report a meaningless ~1.0. The split is therefore
**leave-one-EPISODE-out**, enforced in code, with one mean vector per episode per phase (60
episodes, 4 balanced cells of 15). Sampling runs at `--sample_stride 5` while writes stay on the
training grid of 15: at stride 15 the two phases that carry the question are the thinnest
(evidence 3.5 frames/episode, min 2; waiting 4.2, min 1), and finer sampling lifts them to ~10.4
and ~12.8 without moving the fast weights off-distribution.

Every probe is reported against three controls, because an AUC in isolation is uninterpretable:

| control | rules out |
|---|---|
| `shuffled` labels | the null *this estimator at this sample size* produces |
| `read` tokens | what the policy consumes vs. what was written |
| `state` (raw proprioception) | the known leak — joint state alone already gives ~70% |

**A write-token AUC that does not clear both the shuffled null and the `state` baseline is not
evidence of encoded memory.** The `approach` phase is an internal negative control: it precedes
the reveal, so signal there means the probe is reading a nuisance rather than the evidence.

These controls earned their keep immediately. An 8-episode smoke run drew its episodes as a
prefix, and because the dataset is stored in cell order (15 episodes per cell) that gave a 7
left / 1 right design where "always guess left" scores 0.875. The evidence-phase write tokens
duly reported accuracy 0.88 / AUC 1.00 — which reads as a triumphant result and means nothing:
the shuffled null also reached AUC 1.00, so no number cleared its own baseline. The episode cap
is now stratified across the four cells, and `analyze()` emits an explicit warning whenever the
majority-class rate exceeds 0.65. The full 60-episode run is balanced by construction (4 cells
of 15) and was never affected, but the episode is the reason this report quotes nulls beside
every probe rather than bare accuracies.

The counterfactual-instruction test from the same replay asks whether the instruction steers
*content*: fit on true-instruction tokens, then score the same frames re-encoded under the other
instruction. `cf_flip_rate` is the fraction of held-out episodes whose decoded side flips when
only the prompt changes.

Status: implemented and unit-tested (17 tests); the checkpoint-6500 run is pending a free GPU.

## 7. Recommended next steps

§6.4 already ran the decisive ablation, so these are ordered by what its result implies.

1. **Fix the leak, or the memory can never be needed** (blocking). The waiting observation
   carries the answer: joint state alone gives 70% and the wrist cameras likely supply more. As
   long as that holds, no amount of training will make the model remember, because remembering
   buys nothing. Options, roughly in order of cost: (a) randomize/neutralize the arm pose during
   the wait so proprioception is uninformative; (b) drop or mask the wrist cameras during the
   wait phase; (c) lengthen the wait and re-home the arms first. This is a *data/protocol*
   change, not a modeling one.
2. **Quantify the remaining leak before retraining.** Run the same LOO probe on wrist-camera
   embeddings, not just joint state, so we know exactly which channels to close.
3. **Then re-measure with the existing tooling.** `v33_offline_eval.py` gives the pass/fail
   signal directly: memory works when zero-read *drops* side accuracy and `follows_swap` rises
   above ~0.5.
4. **Only after that, revisit the architecture.** The conditioning pathway is healthy
   (`out_proj` 2.96 → 5.30) and the §16 credit path is verified, so there is no evidence the
   mechanism is broken — it is unexercised. If the gate still refuses to open on non-leaky data,
   then consider an explicit read bottleneck or a stronger memory-critical mixture.
5. **Note for interpreting §6.2**: the attention collapse is most likely a *consequence* of the
   leak rather than an independent failure — with the answer available in the current frame,
   nothing keeps the writer looking at the evidence.

## 8. Reproducing

```bash
# training (2×H200)
uv run scripts/train.py pi05_yam_mem_v33 --exp_name=v33_run1

# writer-attention diagnostic at any checkpoint (~2 min GPU, forward only)
uv run scripts/v33_writer_attention.py \
  --checkpoint checkpoints/pi05_yam_mem_v33/v33_run1/<step> \
  --dataset_root ~/.cache/huggingface/lerobot/yam/bin_memory_0816_subtask \
  --output_dir diagnostic_outputs/v33_writer_attention/<step>

# geometry-grounded correctness analysis (CPU; pass several runs for the trend)
uv run scripts/v33_writer_correctness.py \
  --run_dirs diagnostic_outputs/v33_writer_attention/{2750,3500,4250,5000,6250} \
  --stills --dataset_root ~/.cache/huggingface/lerobot/yam/bin_memory_0816_subtask

# per-slot grids: v3.2-style (readable) or other variants, CPU-only from saved NPZ
uv run scripts/v33_render_slot_grids.py \
  --run_dirs diagnostic_outputs/v33_writer_attention/<step> \
  --dataset_root ~/.cache/huggingface/lerobot/yam/bin_memory_0816_subtask \
  --style v32 [--variant cf]

# section-16 gradient-flow check
uv run scripts/v33_gradient_flow_check.py \
  --checkpoint checkpoints/pi05_yam_mem_v33/v33_run1/<step> [--gate_override 0.05]
```

Key commits: `b4cb198` (v3.3 core), `75d80d3` (deterministic endpoints), `2f10f99` (adversarial
fixes), `4d0fdb8` (writer-attention diagnostic), `7f098d8` (per-slot grids), `abd6570`
(correctness analysis).

## 9. Environment notes (hard-won)

- **Import order is load-bearing twice.** `torch` must import before `tensorflow`
  (`openpi.training.config` pulls TF in; the reverse order segfaults with exit 139 and no
  output), and `pyarrow.parquet` must import before the openpi stack (otherwise every read of
  this dataset's inline-image parquet files segfaults the same silent way).
- **GPU access via `sc`**: `srun --jobid=<id> --overlap` inherits the AFS `$HOME`, which breaks
  every cache (lerobot/HF/uv). Job scripts must `export HOME=/iris/u/kewalk` and add
  `/iris/u/kewalk/.local/bin` to `PATH`.
- Checkpoint loading from a pre-memory base (`pi05_base`) needs the checkpoint merged over
  *materialized* init params; `_align_params` only restores flax `None`-bias slots, it does not
  graft missing subsystems.
