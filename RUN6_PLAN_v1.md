# RUN6 PLAN — v3.5 executable specification v1

**Date:** 2026-08-30  
**Repository state reviewed:** `main@db3fa38` plus the current uncommitted research artifacts  
**Experiment name:** `v3.5 / run6`  
**Status:** specification proposed; **not approved for a 10k launch**  
**Source draft:** `RUN6_PLAN_draft_v0.md` (preserved unchanged)

---

## 0. Executive decision

Run6 should continue with the draft's central scientific question: install the evidence-time side percept, then test the memory chain link by link. The v0 implementation, however, must not be used directly. It assumes one key/value pair per write while production writes 16 pairs, misstates the current augmentation and checkpoint status, and does not yet close the decision-frame shortcut.

The v1 decisions are:

1. Keep the claim narrow: an **installed**, label-gated side percept and a causally tested memory chain. Do not claim emergent writing, learned write timing, or cross-collection generalization.
2. Use **one pooled association per frame**, not a 16-token pseudo-exact update:
   `k_bar = L2Norm(mean_i k_i)`, `v_bar = L2Norm(mean_i v_i)`.
3. Add a default-off `delta_output` write rule. In this rule only `w3` is fast: hidden fast leaves never update or decay; `b3` and all output momentum stay exactly zero; alpha decays only `w3`.
4. Set `delta_rate=1.0`. This is an explicit delta assignment, not the old inner-loss gradient step. Do not silently fall back to 0.5; a failure is a stop/branch condition.
5. Main-run writes are **E-only**. All other valid frames are output-decay-only; padding is a no-op.
6. Initial frame definitions are conservative: E=`inspect both bins`; O=the whole `close both lids and reset arms`; D=static-trimmed waiting only. Set `W=0` for the early `open {side}` extension until a real motion boundary exists.
7. Add non-detached `L_write` on pooled values and `L_read` on pooled raw, pre-pin reads. Drop the old 7-way auxiliary loss; retain detached ladder heads as telemetry.
8. Replace run5's per-frame augmentation with **time-consistent augmentation at the same initial strength**. Do not start with the draft's aggressive 0.85 crop or spatial wrist transforms until visibility is audited.
9. Do not reuse the existing `pi05_yam_mem_v34_run6_staticwait`; it remains the data-only control. Add a new `pi05_yam_mem_v35_run6` config.
10. Do not resume a Run5 optimizer state. Partial-load selected Run5 model parameters, fresh-initialize new heads, reset Adam/EMA/global step, and record an explicit graft manifest.
11. Treat 15750 as the characterized baseline and 18000 as the latest candidate. The provisional source is **15750/raw**; promote 18000/raw only if the completed fixed probe, late causal test, and retention test are non-inferior. Never switch silently.
12. Pass the data/leak and mechanism gates before any main H100 run. First authorize a 1k pilot; extend to 2.5k and then 10k only rung by rung.

---

## 1. Current status that the plan must use

### 1.1 Code and baseline

- `main` and `origin/main` are at `db3fa38`.
- The verified v3.4.1 baseline already passed the existing unit/stage-0/4-GPU smoke suites.
- `pi05_yam_mem_v34_run6_staticwait` changes only the loader trim. It is not a Run6 model and must remain an untouched control.
- The launch snapshot/provenance, not today's dirty worktree alone, is the authoritative description of Run5.

### 1.2 Checkpoints

- Run5 has a complete checkpoint at step 18000; training stopped after that save.
- Live retention removed 15750, but an archived 15750 copy exists under `openpi/diagnostic_checkpoints/v34_run5_eta0_resume_copies/15750`.
- Step 18000 training metrics are finite, but the single writer-ladder point fell to about 0.695 after mostly 0.84–0.89 late-run values. This may be probe noise, so recency alone is not a selection rule.
- The 18000 fixed probe was still running when this specification was frozen. G0 below resolves the source checkpoint.

### 1.3 Fresh 0830 evidence

Ten new same-setup episodes already exist. They cover open→inspect→close/reset, but do **not** contain waiting, decision, or terminal opening. Their side order is blocked rather than randomized.

Across the completed Run5 checkpoint sweep, fresh/online balanced writer accuracy is approximately 0.45–0.55. Therefore:

- call this set **P7a writer-generalization**, not a second memory-chain heldout;
- record the result as evidence that the Run5 writer is collection/session keyed;
- do not use P7a to make a full-chain claim;
- a full-chain **P7b** needs randomized/interleaved episodes containing E→O→D→terminal action.

### 1.4 Known decision shortcut

Current waiting-frame probes decode side from image at about 0.92 episode-OOF and from state at about 0.70. Static motion trimming does not remove the collection/session, lid, arm-pose, or state confound. P1 is therefore a real launch blocker, not paperwork.

---

## 2. Scientific scope and preregistered claims

### 2.1 Primary question

With the side percept directly supervised at evidence time and direct decision-frame shortcuts controlled, can the fast-weight system:

1. commit the evidence value;
2. retain it through at most 40 valid memory transitions;
3. retrieve it from a decision-time query;
4. causally change the policy's token/action choice?

### 2.2 Allowed claim if Run6 passes

Within the 0816 training collection and its held-out episodes, an installed side representation can be committed, retained, read, and causally used by a label-gated fast-weight memory.

### 2.3 Claims explicitly excluded

- The writer content emerged solely from downstream task demand.
- The model learned when to write without labels.
- The mechanism generalizes to a new collection or scene.
- A multi-item associative memory was solved; Run6 deliberately reduces each frame to one association.

P7b can expand the third claim only if it is collected with randomized side order and evaluated as a full chain.

---

## 3. Why the v0 commit rule must change

Production `write_kv` receives tensors shaped `[batch, 16, d]`. Its associative objective is the mean over 16 squared errors, without a factor of one half, and currently differentiates, clips, decays, and updates every hidden/output fast leaf including `b3`.

Consequences:

- A rank-one update is exact for one association, not for all 16 associations jointly.
- If the update were interpreted as an ordinary gradient step even for one pair, the old loss convention adds a factor of two; `1 / ||h||^2` is not that gradient step's exact learning rate.
- Values are already unit-L2; the v0 `||v|| >> 1` regime cannot occur as written.
- The reported historical per-write delta is a whole-state max-absolute statistic, not `||Delta w3||_F`; it cannot support the v0 arithmetic.
- Setting hidden theta to zero does not freeze hidden features under today's code because alpha still decays them.

A 16-pair joint exact solve would require a regularized pseudoinverse of the 16×16 hidden Gram matrix. That adds rank-collapse, conditioning, differentiation, and sharding risk while the task needs only one side bit and intentionally tolerates last-write-wins behavior. It is out of scope for Run6.

---

## 4. Memory-core specification

### 4.1 Association pooling

For every eligible frame:

```text
k_bar = L2Norm(mean over the 16 projected key slots)
v_bar = L2Norm(mean over the 16 projected value slots)
```

Log both pre-normalization norms. A near-zero pooled norm is a mechanism error and must not be hidden by epsilon normalization.

The same pooled `v_bar` is the input to `L_write`; the same one-association geometry is used by the commit diagnostic.

### 4.2 New write rule

Add a static configuration choice:

```text
write_rule = "gradient" | "delta_output"
association_mode = "tokens" | "pooled_frame"
delta_rate = 1.0
freeze_output_bias = true
decay_output_only = true
```

All old configs default to `gradient/tokens`, with a bit-exact regression test. Only the new v3.5 config selects `delta_output/pooled_frame`.

For row-vector code (`prediction = h @ W`), the v3.5 update is:

```text
W_decay = (1 - alpha) * W
h       = hidden(k_bar)                         # fixed hidden fast leaves
r       = v_bar - h @ W_decay
W_new   = W_decay + delta_rate * outer(h, r) / max(dot(h, h), eps)
```

At `delta_rate=1`, the immediate own-key residual should be numerical zero. At a general rate, `r_post = (1-delta_rate) r`.

### 4.3 Fast-state invariants

In `delta_output` mode:

- `w0..w2` and `b0..b2`: no content update, no alpha decay, no momentum update;
- `w3`: alpha decay before residual computation, then direct delta assignment;
- `b3`: fast value and momentum are exactly zero after init, commit, decay-only, padding, checkpoint save, and restore;
- output momentum: exactly zero; eta is unused in this branch;
- the update remains differentiable; do not stop-gradient through `h`, `k_bar`, or `v_bar` merely because state assignment is closed form.

### 4.4 Per-frame transition semantics

| frame state | transition |
|---|---|
| valid and E-eligible | output decay + content commit |
| valid and not E-eligible | output decay only |
| padding / invalid | exact no-op |
| episode reset | blank v3.5 memory state |

Training and evaluation must share these semantics. The current inference names map as follows: v3.5 non-E is `dynamics_only`, not `frozen`.

### 4.5 Telemetry

Log per pooled write and aggregate by episode/phase:

- pre-pool key/value norm and post-pool norm;
- `||h||`, pre/post residual and residual ratio;
- `||Delta W3||_F`, `||W3||_F`, and `maxabs(W3)`;
- output decay count since last E commit;
- raw own-key read RMS, cosine, and norm ratio;
- non-finite count;
- hidden-leaf and `b3` invariant violations.

Do not reuse the old severe-inner-clip alarm for the direct output assignment; it does not measure this branch.

---

## 5. Data, phase masks, and sampling

### 5.1 Initial phase definitions

Use only boundaries the current labels can support:

- **E:** all valid frames labeled `inspect both bins`.
- **O:** the entire segment labeled `close both lids and reset arms`.
- **D:** static-trimmed waiting frames only.
- **Early terminal open:** not in D in v3.5-v1 (`W=0`). The existing `open {side} bin` begins at the first side-specific motion, so “before arm commitment” cannot be recovered from that label.

Do not include `open both lids` in E until a versioned fully-open detector/sidecar has been validated. Do not infer physical both-lids-closed onset from the coarse close/reset label.

All masks and write eligibility must use the unshifted current-frame field `subtask_now`. The existing `subtask` field is a `t+15` look-ahead target and would leak future phase information into the write gate.

### 5.2 Versioned boundary artifact

Generate a per-episode artifact containing:

- raw-frame and memory-step E/O/D intervals;
- static-trim endpoints and detector version;
- task, side, collection, and heldout status;
- final valid E anchor;
- visual grid review result and reviewer timestamp;
- source labels/data hashes.

Hand-check all four heldouts and at least one episode per task×side cell; automatically reject interval inversions, overlap, missing anchors, or out-of-bounds indices. Episode 26 is a known candidate anomaly (execution tail labeled as waiting) and must be corrected or excluded from P1/P2/training before the artifact is approved.

### 5.3 Evidence-write coverage

The current stride/alignment does not naturally yield three E writes in every sequence; roughly 8% of current critical start grids have fewer than three, spanning 11 episodes.

Change the loader so every memory-critical sequence contains the final valid E anchor in addition to its regular sampled E frames.

- hard gate: at least one prior E commit for every D loss frame;
- target: at least two distinct E commits when the labeled interval permits;
- report the full per-episode distribution rather than pretending `>=3` is guaranteed.

One correct final E commit is logically sufficient for this last-write-wins task.

### 5.4 State validity versus gradient reach

Track two separate bits:

- `read_state_valid`: a prior E commit exists in this sequence/state;
- `read_credit_reachable`: after the most recent TBPTT boundary, an earlier valid E commit exists and gradient can reach it.

Apply the main `L_read` only when both bits are true. Log state-valid but unreachable D frames as evaluation/coverage telemetry, not training examples. Existing memory-critical samples intentionally omit TBPTT fences, so low reach would be a loader regression, not an established explanation of Run5.

### 5.5 Leak gate and remediation

P1 uses actual D frames after all v3.5 preprocessing and evaluates:

- each camera separately, all cameras jointly;
- robot state separately;
- images + state jointly;
- raw and augmented inputs;
- episode-level out-of-fold balanced accuracy, per-task/cell results, and a bootstrap interval.

**Launch gate:** no policy-visible D modality or joint probe may exceed 0.60 balanced accuracy, and reset-memory native side margin must be statistically near zero.

If P1 fails:

1. do not launch the main pilot;
2. prefer randomized/interleaved recollection with a common neutral waiting pose;
3. if robot access is unavailable, define a separate `v35_scaffold` branch using paired, task-matched side-neutral D observations and state neutralization, then repeat P1;
4. label that branch as counterfactual/scaffolded and do not convert its result into a natural-scene claim;
5. lid-only masking is not accepted unless the all-camera+state audit passes afterward.

---

## 6. Objectives and gradient wiring

### 6.1 `L_write`

```text
z_write(t) = g_w(v_bar_t)
L_write(ep) = mean CE(z_write(t), side_ep) over E frames in that episode
```

- `v_bar` is post-projection, pre-commit, pooled, and non-detached.
- Aggregate per episode first, then across the batch, so long E segments do not dominate.
- Gradient should reach `g_w`, value projection/writer compressor, and the intended backbone layers.
- Frames outside E contribute exactly zero.

### 6.2 `L_read`

```text
r_bar(t) = mean over the 16 raw retrieved slots, before RMS pin/injection
z_read(t) = g_r(r_bar(t))
L_read(ep) = mean CE(z_read(t), side_ep) over valid D frames
```

- Require both `read_state_valid` and `read_credit_reachable`.
- Aggregate per episode, side-balanced across the batch.
- Gradient should reach `g_r` and `project_q`; on reachable critical sequences it must also reach the evidence value path through the differentiable commit.
- Memory reset, donor swap, and injection ablation are evaluation interventions, not training labels.

Name the trainable heads `memory_write_side_head` and `memory_read_side_head` so they are included in the existing memory-path optimizer/preclip classification and are not mistaken for detached ladder probes.

### 6.3 Loss set

Main pilot starts with:

```text
L = L_flow + existing subtask CE + 0.3 * L_write + 0.3 * L_read
```

- Drop the 7-way memory auxiliary objective.
- Keep detached ladder/probe heads as telemetry only.
- Do not add `L_align` in the main branch. Add a separately named branch at weight 0.1 only if commit and retention pass but natural read fails.

Before fixing the 0.3 weights, P5 measures loss-only gradient norms on the writer, query projector, memory parameters, and backbone. Adjust once before the pilot if either side loss dominates action gradients or is numerically absent; record the decision in provenance.

Each new masked loss must expose a global numerator and valid-count denominator. Wire those quantities into both the ordinary training path and the gradient-accumulation path; averaging per microbatch or bucket would change the effective weights when mask counts differ.

### 6.4 Gradient guards

`L_write` and `L_read` can bypass the current K/V cotangent guard when attached directly to their features. Add a branch-local feature cotangent cap (initial cap 1.0) and retain the existing global gradient clip. P5 must test both nonzero reach and bounded magnitude; “gradient exists” alone is insufficient.

---

## 7. Augmentation

Run5 did not have “no augmentation.” It currently applies frame-independent top-camera 95% crop/resize plus ±5° rotation, and color jitter to all cameras; wrists do not receive spatial transforms.

For the first v3.5 pilot:

- keep those magnitudes and camera-specific policies;
- sample parameters once per sequence and camera, then reuse across time;
- use independent RNG across batch samples and cameras;
- no horizontal flip;
- verify identical temporal geometry with a deterministic test;
- audit that both-bin evidence remains visible in at least 99% of augmented E frames.

The draft's 0.85 crop, ±8% translation, hue change, and spatial wrist augmentation become a later preregistered ablation, not an unmeasured simultaneous change.

---

## 8. Checkpoint selection and initialization

### G0 selection rule

Evaluate archived 15750/raw and 18000/raw with the same:

1. fixed episode-OOF evidence writer/value probe and approach control;
2. own-key residual/retention battery;
3. late raw+EMA causal token/action test;
4. three-way frozen / decay-only / normal interference diagnostic;
5. P7a fresh writer probe.

Use 18000/raw only if it is non-inferior on the in-domain fixed probe and causal/retention diagnostics. Otherwise use 15750/raw. Fresh P7a remains secondary because both checkpoints may be at chance there.

### Initialization contract

- Load only compatible model parameters through the partial loader.
- Source raw parameters from `train_state/params`; do not confuse the standalone `params` tree with raw, because it is EMA.
- Fresh-initialize `g_w`, `g_r`, and any new gate/telemetry parameters.
- Ignore obsolete auxiliary-head parameters explicitly and record them.
- Reset Adam, EMA accumulator, optimizer count, and global training step to zero.
- Save a graft manifest with source checkpoint path, raw/EMA choice, source and destination tree hashes, matched/missing/extra leaves, data/norm/config hashes, git commit, and dirty-tree inventory.
- Do not use `--resume`.

---

## 9. Preflight gates and execution order

No main training starts until G0–G5 pass.

### G0 — Freeze source and provenance

- Complete the 18000 probe and paired 15750/18000 diagnostics.
- Select the source using Section 8.
- Snapshot the exact code/data/config state without folding unrelated dirty files into the attribution.
- Output: `openpi/docs/run6_preflight.md`, checkpoint graft manifest, and immutable source paths.

### G1 — Data boundaries and leak

- Produce/version E/O/D artifacts and frame grids.
- Guarantee the evidence anchor; report writes, E→D gap, state-valid, and credit-reachable distributions.
- Run P1 across all modalities.
- Stop if the ≤0.60 leak gate fails; choose recollection or the separately named scaffold branch.

### G2 — Memory core

- Implement pooling and `delta_output` behind default-off flags.
- Preserve the old rule bit-exactly.
- Add telemetry and invariant checks.
- Pass all synthetic and production-shape tests in Section 10.

### G3 — Loss, masks, and augmentation

- Add `g_w/g_r`, episode-balanced losses, masks, branch-local cotangent caps, and shared-across-time augmentation.
- Verify zero loss outside masks and correct train/eval transition semantics.
- Calibrate the two side-loss weights from loss-only gradient measurements.

### G4 — Injection calibration

Do not assume reads become 10× larger. On realistic committed states measure:

- raw read RMS and cosine;
- `tanh(gate)` distribution;
- RMS after the existing pin;
- actual injected RMS and residual-stream ratio;
- saturation fraction.

Keep the existing residual target scale `c` unless these measurements show the injected/residual ratio changed. Recalibrate `tau` first if raw reads moved.

### G5 — Smoke and step-0 mechanism evaluation

- Single-device forward/backward/update and checkpoint round trip.
- 4-device B12 FSDP production-shape step with existing `w3` sharding.
- No NaN/Inf, unexpected all-gather, or memory regression.
- At step 0, commit and 40-step retention gates pass before learning.

### G6 — Authorized 1k pilot only

- Save/evaluate at steps 0, 250, 500, and 1000.
- Run the full ladder at 1000.
- If mechanism or leak controls fail, stop; do not “give it more steps.”

### G7 — Conditional extension

- If 1k passes, authorize 2500.
- If 2500 passes write/read/action gates, extend to 5000.
- Continue to 7500/10000 only while action quality and causal gates remain passing.
- Save at 250/500/1000 initially, then every 500 so the 2.5k and 7.5k rungs actually exist; preserve every rung checkpoint used in a report.
- Under the current training loop, 10,000 updates end at step id 9999. Reports and launchers must distinguish “10k updates” from a literal checkpoint directory named 10000, or explicitly change/test the save convention.

---

## 10. Required tests

### 10.1 Core numerical tests

1. One pooled pair, random nonzero old `W`, alpha>0: immediate residual within FP32 tolerance.
2. `delta_rate=0.5`: post/pre residual ratio ≈0.5, preventing a false “exact” claim.
3. Repeated commit and decay-only transitions: `b3` fast value/momentum remain bitwise zero.
4. Hidden fast leaves remain bitwise identical across commit, decay-only, padding, and reset cycles.
5. Eligible=decay+commit; ineligible=decay-only; padding=no-op; train and inference agree.
6. Pooled norm near zero raises telemetry/failure instead of silently producing a plausible write.
7. Old `gradient/tokens` path is bit-exact against v3.4.1 fixtures.

### 10.2 Gradient tests

1. `L_write` reaches value projection, writer compressor, and intended backbone layers.
2. `L_write` is exactly zero outside E.
3. `L_read` reaches query projection/compressor.
4. On a reachable critical sequence, `L_read` reaches the evidence value path.
5. On a fenced sequence, the reach telemetry is false and no false long-range claim is made.
6. State/KV and new side-feature cotangent caps stay within configured bounds.

### 10.3 Retention and interference tests

- Evaluate immediate, 1, 10, and 40 decay steps.
- Record cosine **and** norm/projection gain; scalar alpha decay preserves cosine and can fool a cosine-only gate.
- Frozen is the retention upper bound, decay-only is the production E-only path, and normal post-E writes are an interference stress. Do not require the v0 `Frozen < Dynamics-only < Normal` ordering.
- Include collapsed-key and eight-distractor curves as diagnostics, but do not require multi-item exactness.

### 10.4 Distributed/restore tests

- B12 one-step forward/backward/update on 4 devices.
- Preserve output-axis sharding for `w3`.
- Compare one-device and four-device results within tolerance.
- Verify partial load from both raw and EMA source trees, fresh heads, ignored old heads, optimizer reset, and exact save/restore invariants.

---

## 11. Evaluation ladder and gates

Use episode/cell macro metrics. Frames from one episode are not independent samples. Always report exact episode counts and uncertainty, not frame-only percentages.

| rung | measurement | pass criterion | failure interpretation |
|---|---|---|---|
| 0 mechanism | own-key pre/post residual at step 0 | pre ≥0.9; post ≤0.1, normally far lower | code/geometry bug; stop |
| 1 write | fixed episode-OOF pooled-value side probe; approach control | evidence ≥0.95 by 2.5k; control ≤0.60 | loss/mask/representation bug |
| 2 retain | own-key read after ≤40 production decay-only steps | cosine ≥0.80 and norm ratio ≥0.55; reset low | alpha or decay semantics |
| 3 read | natural / reset / opposite-donor `g_r` | natural correct on 4/4 heldouts; reset ≤0.60; donor follows donor on ≥3/4 | query alignment if rungs 0–2 pass |
| 4 use | provenance-bound token+action causal intervention | donor changes requested action side on ≥3/4 episodes/cells; reset ≤0.60; injection ablation removes effect | injection scale/placement if rung 3 passes |
| 5 task health | flow/subtask/action metrics at matched step | within ±10% of selected Run5 baseline | side objectives harming policy |

Additional rules:

- A >20% matched action-loss regression, any non-finite state, invariant violation, or leak-gate regression stops the run.
- Do not auto-change `delta_rate` in place. Any rate/alpha/loss change creates a new named branch and reruns step-0 gates.
- `L_align` is allowed only when commit/retain pass and read fails; it is not a generic rescue knob.
- Evaluate chosen training parameters and EMA at each major rung, but preregister which one supplies the final headline result.

---

## 12. Fresh and transfer evaluation

### P7a — Existing 0830 writer set

- Keep eval-only.
- Report current chance-level Run5 result as baseline.
- Use episode-macro results and disclose blocked side order and missing D/action segment.
- It may measure writer generalization only.

### P7b — Needed for a fresh-chain claim

Collect at least 8–12 complete episodes with:

- side randomized/interleaved in collection order;
- both tasks × both sides represented;
- E→close/reset→neutral waiting→decision→terminal open;
- common camera/robot reset protocol;
- labels and boundary artifact reviewed before evaluation.

If P7b does not exist, cross-collection results stay secondary and the final claim remains in-collection.

### June-30

Finish the 5-task LeRobot reconversion and keep it evaluation-only. This is not a launch gate for the in-collection mechanism pilot, but it is required before reporting transfer.

---

## 13. Concrete code-change map

### Preserve

- `openpi/src/openpi/training/config.py`: keep `pi05_yam_mem_v34_run6_staticwait` unchanged.
- Existing v3.4 evaluators and launch snapshots remain immutable references.
- Current user edits to converter/label tools remain separate from Run6 changes.

### Modify or add for v3.5

| area | files | required change |
|---|---|---|
| memory core | `openpi/src/openpi/models/memory.py` | pooled association support; default-off delta rule; output-only decay; invariants; telemetry |
| core tests | `openpi/src/openpi/models/memory_v34_test.py` and/or new `memory_v35_test.py` | exact commit, decay semantics, hidden/bias invariants, legacy regression |
| model config | `openpi/src/openpi/models/pi0_config.py` | static v3.5 flags and loss weights |
| sequence model | `openpi/src/openpi/models/pi0.py` | shared projection/pooling; E write gate; raw-read loss input; new heads; time-consistent augmentation |
| observations/masks | `openpi/src/openpi/models/model.py`, `openpi/src/openpi/transforms.py`, `openpi/src/openpi/policies/yam_policy.py` | versioned E/O/D, eligibility, D, state-valid, reach telemetry fields |
| loader | `openpi/src/openpi/training/data_loader.py` | boundary artifact; final-E anchor; critical sampling; episode-balanced metadata |
| train config | `openpi/src/openpi/training/config.py` | new `pi05_yam_mem_v35_run6`; partial initializer; 1k pilot schedule |
| training metrics | `openpi/scripts/train.py` | numerator/count aggregation in normal and gradient-accumulation paths, branch gradient metrics, commit/retention telemetry |
| diagnostics | new `openpi/scripts/v35_commit_diagnostic.py`, `v35_read_eval.py`, `v35_causal_memory_eval.py`, plus leak-audit v2 | gates G0–G5 and ladder |
| cluster/provenance | new `openpi/cluster_v35/` | snapshot, graft manifest, staged launchers, rung watcher |
| documentation | `openpi/docs/run6_preflight.md` | all preflight results and signed launch decision |

Re-use rather than rewrite the existing fixed writer probe, stage-0 memory battery, three-way retention evaluator, causal evaluator, leak scripts, and checkpoint snapshot machinery wherever their provenance checks remain valid.

---

## 14. Deliverables and stop points

### Deliverable A — preflight package

- this frozen v1 spec;
- G0 source-selection report;
- versioned E/O/D artifact and visual audit;
- P1 all-modality leak report;
- P7a baseline and P7b status;
- checkpoint graft manifest.

**Stop point:** no implementation launch if source or leak status is unresolved.

### Deliverable B — tested implementation

- legacy-compatible memory core;
- v3.5 config/loss/data wiring;
- numerical, gradient, retention, distributed, and restore tests;
- step-0 mechanism report.

**Stop point:** no GPU pilot if any invariant or mechanism gate fails.

### Deliverable C — 1k pilot

- checkpoints 0/250/500/1000;
- raw+EMA ladder results;
- matched Run5 task-health comparison;
- explicit go/stop decision for 2.5k.

### Deliverable D — conditional 10k run

Only after the 1k and 2.5k gates pass. Each parameter change starts a separately named branch; no undocumented in-place rescue.

---

## 15. Immediate next actions

1. Let the 18000 fixed probe finish; run matched late causal and retention diagnostics on 15750/raw and 18000/raw.
2. Freeze the source checkpoint with the Section 8 selection rule.
3. Build and audit the E/O/D boundary artifact; run the all-modality P1 leak gate.
4. If P1 fails, stop and choose interleaved recollection or the explicitly scaffolded D-neutral branch.
5. Only after P1 passes, implement the pooled output-delta core behind default-off flags and run Section 10 tests.
6. Wire `L_write/L_read`, masks, cotangent guards, and time-consistent augmentation; calibrate gradients and injection.
7. Pass single-device and 4-device smoke plus step-0 mechanism gates.
8. Launch only the 1k pilot, then decide whether Run6 earns more compute.

The mechanism experiment uses teacher E masks. A deployable learned gate is a separate follow-on deliverable: the current subtask output is look-ahead aligned and cannot safely gate the current frame. It requires a current-phase head, per-sample dynamic mask, fail-closed behavior, and a sticky no-content-write state from close/reset to episode end; teacher/predicted agreement must be reported before any deployment claim.

This order separates four possible failure classes—old checkpoint quality, data leakage, memory mechanics, and policy use—so a negative result has a specific interpretation.
