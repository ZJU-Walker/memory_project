# v3.5 Implementation Plan for Claude Review

Revision: 5.1, portable fresh-base bootstrap clarification after implementation audit.

Status: **Implementation is authorized and in progress under this frozen contract; training cannot launch until the release gates pass.** A separate session owns the 0816/0830 labeling, manual review, and conversion. This work will not modify those files.

## 1. Objective and Claim Boundary

v3.5 will build one directly testable memory chain:

- Write prompt-conditioned target-side evidence only during `inspect both bins` (E, evidence).
- Retain it through `close both lids and reset arms` (O, occlusion).
- Supervise reading during a strictly static `wait {side}` interval (D, decision).
- Use causal interventions to distinguish memory write, memory retention, memory read, and downstream memory use.

The stored value semantics are explicitly **prompt-conditioned target side**, not prompt-independent object layout. Switching the banana/gray-box prompt on the same image should therefore switch the writer's target-side representation.

The final claim is limited to an **0816+0830 within-collection held-out memory mechanism result**. It must not claim cross-session generalization, and open-loop arm-side steering must not be described as closed-loop task success.

## 2. Frozen Data Protocol

### 2.1 Collections

- 0816: 60 episodes.
- 0830: 30 additional episodes collected on August 30 in the same setup on a different day:
  - part1: 16 valid full-chain episodes;
  - part2: 14 valid full-chain episodes;
  - exclude `0830_bin_part2/demo14`, which has no terminal execute phase.
- Expected combined LeRobot dataset: **90 episodes** under a new versioned name.
- The June-30 collection remains a separate eval-only transfer probe and is not merged into training.
- Every episode must record stable ID, source path, collection/session ID, prompt, target side, label hash, include/exclude reason, and split.
- The launch manifest uses schema version 2. Included records additionally carry canonical `collection` (`0816` or `0830`), 0830 `part`, canonical object, converted episode index/frame count, manual E-visibility/contact-sheet provenance, and an independent hashed `D_valid` detector sidecar. A hashed block-confound audit is part of the manifest. Training rechecks prompt, label bytes, exact five-phase coverage, task vocabulary/order, and side consistency, but never recomputes `D_valid` from sealed final-test state.

### 2.2 Label and Boundary Rules

- E starts only after both objects inside the two bins are visible. Manual contact-sheet review is authoritative; a collection-calibrated detector is triage only.
- Exclude the last five raw E frames from write eligibility. Define the **final eligible E anchor** as the last sampled frame whose raw index is `<= semantic_E_end - 5` and whose contact sheet still shows both objects completely. This makes the last direct commit come from a clean visible frame rather than the E-to-O transition.
- D starts after both arms have returned to neutral and before target-side motion begins.
- Memory eligibility uses the stricter static core from the complete **14-dimensional robot state**.
- Keep semantic subtask labels separate from the strict `D_valid` sidecar. An automatic static detector must not rewrite neighboring semantic phases.
- Correct confirmed semantic errors, including 0816 episode 26, in the labeling session. Other boundaries require manual confirmation.
- Conversion must fail closed: no silent skips, exact five-phase coverage, fixed seven-task vocabulary, prompt/side consistency, and an episode end matching the shortest valid stream.
- Before freezing the manifest, verify that object and side assignments were randomized within each 0830 part rather than collected in side-specific time blocks.
- Report E-window length and E-to-D raw-frame-gap distributions separately for 0816, 0830 part1, and 0830 part2.

### 2.3 Split

- Train: 74 episodes = 56 from 0816 plus 18 from 0830.
- Development: eight episodes = the four existing stable-ID 0816 held-out episodes plus four 0830 episodes. Select the four new development episodes with `split_seed=35`, using manifest fields only, after enforcing object/side coverage and preserving at least one training episode in every `part x object x side` cell. Model outputs, images, and probe results cannot influence selection.
- Final test: one episode for every `part x object x target-side` cell in 0830, for eight episodes total.
- Freeze the exact assignment with algorithm `openpi.v35.sha256-ranked-manifest-fields.v1`: stable hash-ranking uses only seed 35, stable ID, part, object, and target side; it first chooses one final-test episode per 0830 cell, then one development episode per object-side pair while preserving at least one train episode in every 0830 cell. The four 0816 development stable IDs remain fixed. The manifest records the algorithm-spec hash, and every consumer recomputes the assignment.
- Final-test episodes must never affect training, normalization, threshold selection, branch selection, or exploratory analysis.
- Final-test labels may receive structural/integrity QA before the split is sealed; after sealing, their observations and derived features remain untouched until the final evaluation.
- Compute normalization statistics from the 74 training episodes only.

If all 30 new 0830 episodes are used for training, the project explicitly gives up a fresh full-chain final-test claim.

### 2.4 Non-Gating Fresh-Base Baseline

After labels and the split are frozen, record one FIG1-style step-0 writer baseline on the **74 training episodes only**, reporting 0816 and 0830 strata separately. The shared pretrained parameters come from the official fresh Pi0.5 base, while the v3.5 writer/value path and its side head are newly initialized. Evaluate the initialized online head and a fresh episode-level out-of-fold probe without touching development or final-test observations.

- This baseline documents the initial condition; it has no accuracy threshold and is not evidence that the writer already transfers or works.
- It cannot change augmentation, thresholds, branch selection, or launch eligibility.
- It is not a sweep over previous runs. The official base source, exact object ID or hash, initialization seed, and resulting parameter-tree hash must be recorded.

## 3. Memory Transition Clock and Decay

The canonical v3.5 memory clock is one sampled memory step every **15 raw dataset frames**, matching `memory_stride_frames=15`. References to a "frame transition" below mean a sampled memory step, not every raw video frame.

- Use `alpha_step = 0.01` per 15-frame memory step for the first pilot.
- Never apply `alpha=0.01` at every raw frame.
- The equivalent per-raw-frame rate is `1 - 0.99 ** (1/15) = 0.0006698`.
- If online execution updates memory after a different raw-frame interval `delta_f`, preserve the same physical decay rate with:

  `alpha(delta_f) = 1 - (1 - alpha_step) ** (delta_f / 15)`

- Sequence buckets, TBPTT boundaries, final-E anchoring, and reachability are defined in sampled steps, but all reports also show raw-frame durations.
- Define delay exactly as the number of valid sampled transitions after the final eligible E commit and before the first `D_valid` read.
- Freeze p50, p90, and maximum of `n_delay` from the training split, where `n_delay` is measured in valid sampled transitions. Retention evaluation replays this real transition count. The primary gate is evaluated at frozen p90 `n_delay`; corresponding raw-frame gaps and the full per-episode distribution are descriptive reports only.

### 3.1 Long-Delay Training Windows

Independently sampled training windows do **not** implicitly carry memory state from a previous window. To prevent long-delay episodes from silently losing all D supervision, train with two explicit window families:

1. **Natural windows:** ordinary contiguous sampled sequences. They preserve E/O dynamics and ensure O-phase observations are trained with memory present. A natural window may supervise D only when its final eligible E anchor and D steps fit within the configured bucket.
2. **Skip-O analytic windows:** run one or two eligible E anchor steps, obtain the post-commit `W_E`, skip the write-free interval analytically, and then run D steps:

   `W_D = (1 - alpha_step) ** n_delay * W_E`

   Hidden fast leaves remain unchanged, while `b3` and momentum remain zero. This is exactly equivalent to replaying the omitted valid non-write transitions under the v3.5 output-only fixed-alpha rules.

- Compute analytic decay in FP32.
- Represent every sparse jump explicitly as `seq_decay_gap_before[t]`, the number of omitted valid non-E transitions before step `t` is read. The scan order is `analytic_decay -> read -> current-step transition`; dense steps have gap zero. This prevents an off-by-one decay on the first D read.
- A skipped interval may contain only valid, non-E, non-writing steps and no memory-reset event. Violation fails closed.
- Skip-O windows are state-valid and credit-reachable by construction.
- Every training episode must contribute at least one valid skip-O D candidate; failure is a data/sampling gate, not a silent episode drop.
- Within memory-critical sampling, target a 50/50 mixture of natural and skip-O families, balanced by collection/object/side. Long-delay D supervision comes from skip-O; natural windows continue to cover the real O frames.
- Configuration validation must reject analytic skipping if a future branch allows hidden/bias/momentum changes or writes inside O.

A preliminary read-only audit of the current labels found `n_delay` p50/p90/max of approximately `14/16/20`, so no current episode appears to exceed the 40-step bucket. One known short-D episode has no legal same-residue grid. The final split artifact remains authoritative: such an episode must receive an explicit clock-aware sparse D step, be relabeled, or be excluded from memory supervision with a recorded reason; it cannot disappear silently.

## 4. Memory-Core Design

Production `write_kv` currently receives 16 key/value token pairs. A single rank-one update cannot exactly fit all 16 pairs simultaneously. The v3.5 pilot therefore deliberately reduces each frame to **one pooled association vector**:

1. Average the 16 projected keys and values independently and L2-normalize them to obtain `k_bar` and `v_bar`.
2. Update only the fast memory output matrix `w3`.
3. Do not update or decay hidden fast leaves. Slow/base hidden parameters remain trainable.
4. Force fast `b3` and every momentum leaf to exact zero.
5. At each valid sampled step, read before state transition. The first E step therefore reads empty memory and cannot use a same-frame write.
6. Transition rules:
   - eligible E step: decay `w3`, then direct delta commit;
   - non-E valid step: decay `w3` only;
   - padding step: strict no-op.
7. For the row-vector implementation `h @ W`:

   `W_dec = (1 - alpha_step) W`

   `r = v_bar - h @ W_dec`

   `W_new = W_dec + delta_rate * outer(h, r) / (h dot h)`

8. Use `delta_rate = 1.0` for the pilot. This is a direct delta assignment, not an exact gradient step under the existing `theta` loss.
9. Disable the drift trust region in delta mode. Preserve state and key/value cotangent guards.
10. A pooled norm or `h dot h` below the preregistered numerical floor fails closed and produces telemetry.
11. Keep alpha fixed/stop-gradient in this branch and test the read-before-write decay ordering directly.
12. Keep pooled keys/values, hidden activations used by memory, `h dot h`, fast `w3`, analytic powers/decay, commit residual/outer-product update, 16-slot raw reads, their mean, `L_read`, and calibration in FP32 regardless of the surrounding model dtype. Cast only the final pinned memory tokens when they enter the Transformer stream, and report both pre-cast and post-cast injected RMS.

An exact own-key commit does not guarantee successful natural reading from the 16 query tokens. Query/key alignment, production commit residual, and real-gap retention are separate gates.

If one vector per frame proves insufficient, a later capacity branch may use the regularized 16-pair least-squares update:

`W += H.T @ inverse(H @ H.T + lambda * I) @ (V - H @ W)`

That is not part of the first pilot.

## 5. Write and Read Training

- `seq_write_mask`: current-frame, non-lookahead, tail-guarded E eligibility. Only these steps may commit.
- `seq_decision_mask`: strict static D eligibility.
- `L_write`: predict target side from non-detached `v_bar`; initial weight `0.3`.
- `L_read`: average the 16 raw retrieved vectors before injection scaling, then predict target side; initial weight `0.3`.
- Disable the old seven-way memory auxiliary loss.
- Retain detached diagnostic ladder heads without backbone gradients.
- Define `L_read_mask = D_valid & valid_frame & read_state_valid`.
- `read_state_valid` means that at least one successful, non-degenerate eligible E commit occurred before the current read.
- Do **not** mask `L_read` with `read_credit_reachable`. That flag is logged as a credit-assignment diagnostic because `L_read` can still train `project_q` and its head when the earlier commit is behind TBPTT.
- Define `reachable_fraction = sum(L_read_mask & read_credit_reachable) / sum(L_read_mask)` and report it by collection, object, side, and E-to-D gap bucket.
- Use final-E anchoring as a sampling target. Every supervised D sequence must be state-valid; target at least one credit-reachable E write when possible.
- Make final-eligible-E anchoring a hard sampler rule for every training window containing `D_valid`. Resample an invalid window; as defense in depth, mask `L_read`, side-bearing subtask CE, and flow/action loss on any state-invalid D step that survives preprocessing, and require its training count to be zero.
- Define a **use-pressure step** as a state-valid D observation whose transformed action-target chunk overlaps the first side-specific execute motion. Report its count per episode and collection. Train flow on all state-valid D steps, but define open-loop side-steering evaluation only on use-pressure steps.
- Use natural and skip-O windows as defined in Section 3.1. The skip-O path must use the exact manifest-derived `n_delay`, not an approximate bucket length.
- Aggregate side losses per episode first, then macro/balance across `(collection, object, side)` cells.
- Gradient accumulation must use exact numerator/count denominators.
- Add a branch-local feature cotangent cap because the direct heads bypass the original memory key/value clip.

## 6. Sequence-Consistent Augmentation

Preserve the current initial strengths and camera policies while making random parameters time-consistent within a sequence:

- Top camera: 95% crop and resize, up to 5-degree rotation, and color jitter.
- Wrist cameras: no spatial transform; color jitter only.
- No horizontal flip.
- All frames from the same sample and camera reuse the same transform parameters.
- Different samples receive independently sampled transforms.

Tests must verify temporal consistency and cross-sample randomness. Freeze this augmentation recipe before step 0. The non-gating fresh-base writer baseline cannot change it; a stronger recipe, if later justified by a failed pilot, must be a separately named preregistered branch rather than an in-run adjustment.

## 7. Fresh-Base Initialization and Injection Calibration

The current implementation is:

`memory_tokens = tanh(w) * retrieved * c / max(rms(retrieved), tau)`

Implementation note: `tau` is an RMS floor, not the argument of the tanh gate. In v3.5, `tanh(w)` is a separate fixed per-channel injection gate, not a learned pilot parameter. Initialize and calibrate as follows:

1. Load shared pretrained parameters only from the official fresh Pi0.5 base at `gs://openpi-assets/checkpoints/pi05_base/params`. No run5 or v3.4 checkpoint participates. Record the resolved source object ID or hash. Each cluster may download this official asset into its own project-local `v35/cache`; the cache is not a required transfer artifact.
2. Freshly initialize the memory and query compressors, conditioner, slot embedding, state-null parameter, injection projection, detached ladder heads, and write/read side heads with a recorded seed. Initialize optimizer, EMA, and global step from scratch.
3. Set every `memory_inject_w` channel to `w = atanh(0.5) = 0.549306...`, verify `tanh(w) = 0.5` within FP32 tolerance, and freeze it before calibration and throughout the pilot. Any missing, overwritten, closed, or sign-flipped channel is an initialization failure, not a conditional reset case.
4. Replay real E-to-D sequences from the **74 training episodes only** after direct commits. Development and final-test episodes cannot tune `c`, `tau`, or any calibration statistic.
5. Measure raw-read RMS for a single clean evidence commit. Keep reset memory as an exact-zero sanity check, not as the noise distribution.
6. Start with `tau = median(signal_RMS) / 0.75`, then verify that `a(r) = min(RMS(r)/tau, 1)` is in `0.7-0.8` for the median clean committed read.
7. Define real noise using (a) reads whose query hidden vector has cosine `<= 0.1` to stored `h(k_bar)`, and (b) the mixed-precision post-commit residual read. If real queries do not provide enough low-cosine examples, add preregistered synthetic orthogonal-query controls and report the shortage.
8. Define open channels `O = {j: abs(tanh(w_j)) >= 0.1}`. Under the required initialization, `O` must contain every channel. Preserve the production all-channel denominator in `u = tanh(w) * retrieved / max(RMS_all(retrieved), tau)`, then set `c = median_train74(RMS_O(layer8_residual)) / median_signal_train74(RMS_O(u))`, using the same indices. Fail closed if any channel is unexpectedly closed or the denominator is near zero.
9. Require the real-noise injected-RMS to layer-8-residual-RMS ratio to have p95 `<= 0.10`.
10. At frozen p90 `n_delay`, require the episode-median retained signal amplitude `a(r) >= 0.40`; report episode p10 descriptively. If one `tau` cannot satisfy clean-signal, decayed-signal, and real-noise conditions simultaneously, fail injection calibration.
11. Freeze calibrated `c` and `tau` for the pilot. Record signal/noise raw RMS, full/per-channel injected-to-layer-8 ratios, the fixed `tanh(w)`, and pre/post-cast injected RMS on every rung.

This initialization deliberately preserves the official Pi0.5 shared backbone and action-policy parameters while giving every v3.5-specific memory, query, and diagnostic path a fresh start. Therefore, no step-0 memory competence is assumed. Run both writer-dependent direct-carry and train-side-prototype oracle controls at step 0 and every rung. They bypass memory state/query retrieval and diagnose whether the downstream consumer can use a consistently pinned side code. Apply the numerical exit rules in Gate D before classifying the consumer as the bottleneck. A consumer-recovery branch may reinitialize only separable v3.5-specific consumption parameters or introduce a small memory-specific adapter. Do **not** blindly reinitialize all shared layers 9-18 and destroy the official-base policy.

Use the standard official checkpoint-loading path with an explicit shared-versus-fresh allowlist. Fail closed on a missing or shape-mismatched shared leaf and on any unexpected loaded v3.5-specific leaf. Record loaded shared leaves, freshly initialized leaves, the initialization seed, source hash, and final parameter-tree hash in an initialization manifest. A custom raw-checkpoint transplant is outside the v3.5 launch scope.

Because calibration and the step-0 release gates both depend on the exact same fresh parameter tree, initialization uses a two-phase, zero-update bootstrap rather than weakening the training gate. The initialize phase creates an audited raw step-0 state without constructing a data loader or drawing a batch. After train-only calibration, the finalize phase authenticates the same parameter-tree hash, binds calibrated `c`, `tau`, `alpha_step`, the fixed gate, data/norm identities, and saves checkpoint 0 with the untouched optimizer plus the exact initial sampler/RNG state. Gate A, Gate B, and step-0 Gate C/task health then consume that finalized checkpoint and produce the pilot authorization. `train.py` never starts v3.5 fresh: it must authenticate the external pilot authorization and resume finalized checkpoint 0 before the first batch or optimizer update. Checkpoints from update 250 onward embed that authorization byte-for-byte.

## 8. Release Gates

### Gate A: Data

- Manual review is complete.
- The 90-episode count, source manifest, labels, stable split, and hashes are frozen.
- Task vocabulary and stable-ID mapping are exact.
- Object/side ordering within each 0830 part passes the block-confound audit.
- E-window and E-to-D gap distributions are reported by collection.
- Normalization is train-only.
- Every D loss step is `read_state_valid`; `read_credit_reachable` fraction is reported, not gated.
- Every training window containing `D_valid` has a final eligible E anchor; the state-invalid D side-loss/action-loss count is exactly zero.
- Use-pressure step counts are reported per episode, collection, object, and side.
- After the five-raw-frame tail guard and stride alignment, every training episode has at least one eligible sampled E step; target at least two. Report min/p10/p50/p90/max by collection. Zero eligibility fails Gate A and cannot be handled by silently dropping the episode.
- Report eligible raw-E frames, eligible sampled-E steps, and successful commit counts per episode and cell; explicitly list episodes with only one sampled E step.
- Every training episode has at least one skip-O D candidate. Report natural/skip-O sample counts and effective `n_delay` distributions.

### Gate B: Leakage

Run episode-grouped out-of-fold probes on final D frames with final preprocessing. Primary gates use the **74 training episodes only**; development results are descriptive and final-test observations remain sealed. Aggregate all `D_valid` logits inside an episode before producing one episode prediction; never treat D frames as independent samples.

Pre-register exactly two primary gating probes:

1. joint final images plus robot state;
2. concatenated fresh-v3.5-initialization `k_bar` plus `v_bar` features at step 0.

For the two probes across the two collections, define one four-test family:

- within each collection, compute left/right balanced accuracy inside each object and macro-average objects equally;
- shuffle side labels within each `collection x object` stratum for at least 1,000 episode-level permutations;
- recompute the complete episode-grouped OOF pipeline for every permutation; do not merely shuffle fixed predictions;
- compute `p_perm = (1 + count(T_perm >= T_observed)) / (B + 1)`;
- apply Bonferroni correction across the four primary tests: each observed statistic must not exceed its null 98.75th percentile, equivalently `p_perm > 0.0125`;
- on the pooled 74 training episodes, each primary probe must also have episode-macro balanced-accuracy point estimate `<= 0.62`;
- report confidence intervals, but never gate on CI bounds.

The `0.62` pooled point threshold is deliberately less brittle than `0.60`: at chance and n=74 it costs roughly 2% false stops per primary probe rather than roughly 4%. The family-corrected permutation tests remain the primary collection-level statistical gate.

Top-only, each wrist, all-images-only, state-only, prompt-only, layer-8-only, `k_bar`-only, and `v_bar`-only probes remain required descriptive diagnostics with CIs. Report per-side recall and per-cell results descriptively; balanced accuracy is undefined for a single-side subset.

If either primary probe fails its family-corrected collection test or pooled point-estimate gate, stop the natural branch. The next step is recollection or a separately named neutralized/scaffold branch. Descriptive-probe excursions trigger investigation but are not automatic stop rules.

### Gate C: Mechanism and Step 0

- Synthetic FP32 pooled own-key post-commit residual: `<= 1e-5`.
- Production mixed-precision relative commit residual `||r_plus|| / (||v_bar|| + 1e-8)` p95: `<= 1e-2`.
- `delta_rate = 0.5` closes exactly half the residual.
- Alpha ordering and 15-frame clock conversion are correct.
- Hidden, bias, and momentum invariants hold; padding is a strict no-op; legacy mode remains bit-exact.
- Dtype tests prove that fast state, pooling/normalization, memory hidden computation, direct commit, analytic decay, and raw retrieval remain FP32 under BF16/mixed-precision model execution.
- For `n_delay={0,1,40,100}`, dense repeated decay and analytic skip must match in FP32 forward state/read and gradients with relative error `<= 1e-6`. Tests also verify that the first D read occurs before the current D decay and that illegal skipped E/write/reset/invalid intervals fail closed.
- `L_write` gradients reach value projection/backbone.
- In a reachable case, `L_read` gradients reach query projection/head and the prior E commit.
- In an unreachable case, `L_read` still trains query projection/head but cannot cross the TBPTT boundary.
- State-invalid and padding cases are fully masked and have zero loss gradient.
- Injection calibration in Section 7 passes on real sequences.
- Under E-only output decay, own-key retention is closed-form: expected cosine is 1 and expected norm ratio is `rho_expected = (1-alpha_step) ** n_delay`. At frozen p90 `n_delay`, require cosine `>= 0.9999`, relative norm-ratio error versus `rho_expected <= 1e-3`, and absolute norm ratio `>= 0.55`. If the frozen p90 delay makes `rho_expected < 0.55`, revise alpha before launch rather than letting an impossible gate fail later. Report p50, maximum-delay, corresponding raw-frame gaps, and per-episode results.
- At frozen p90 `n_delay`, episode-median retained injection amplitude remains `>= 0.40`; real query-misalignment/mixed-precision noise remains below the Section 7 limit.
- Before calibration, verify that every channel has `w=atanh(0.5)` and `tanh(w)=0.5` within FP32 tolerance. Freeze `memory_inject_w`; any closed or sign-flipped channel fails Gate C rather than triggering a conditional checkpoint-leaf reset.
- Report actual D-query geometry per rung, by collection/object/side and delay bucket. For each of 16 queries compute hidden-space cosine to stored `h(k_bar_E)` and `beta_j = dot(h(q_j), h_E) / dot(h_E, h_E)`. Report mean/max cosine, low-alignment fraction, `beta_mean`, `mean(abs(beta))`, cancellation ratio, sign consistency, `cos(mean_raw_read, v_bar)`, and anchor-predicted versus actual raw-read residual. Own-key retention verifies implementation; learned writer quality and D-query alignment are the learned links.
- Oracle injection and action-to-memory attention are recorded at step 0 as consumer-path diagnostics, not as reasons to inspect final-test data.

### Gate D: 1k Pilot

- Run one 1,000-update main pilot without an in-run parameter sweep.
- Save fixed rungs at 0, 250, 500, and 1,000 completed updates.
- Do not select a checkpoint by maximizing accuracy on the eight development episodes. Rungs are judged by preregistered episode/cell-macro mechanism metrics; the 1k endpoint is the fixed pilot decision point.
- Log production commit residual, real-gap retention cosine/norm ratio, raw-read RMS, injected-token RMS, and reachable fraction at every rung.
- Produce one hard outcome per episode and condition after aggregating eligible E, D, or use-pressure frames inside that episode. All development thresholds below are episode counts.
- Separate read from use with eight matched conditions/measurements:
  - natural memory;
  - reset memory;
  - opposite-side donor memory;
  - zero injection;
  - oracle direct-carry injection: take the same episode's final eligible E `v_bar`, apply the frozen pin, and inject it while bypassing memory state and query retrieval;
  - train-side-prototype oracle: inject the frozen mean training `v_bar` for the requested side (leave-one-episode-out for training-set diagnostics), separating consumer ability from one episode's writer quality;
  - opposite-side prototype oracle-donor: inject the other side's prototype and measure whether the consumer follows the donor;
  - attention mass from action-expert/frame-token queries to memory-token keys during D.

Construct each side prototype by first averaging eligible `v_bar` values within episode, then averaging episodes within side and L2-normalizing. Training diagnostics use leave-one-episode-out prototypes, and every rung records the prototype artifact hash. Pin correct-side and opposite-side directions to the calibrated median clean-read **injected** RMS. Never inject them at raw `v_bar` magnitude.

Hard development thresholds at 1k:

- prompt-bound writer claim: under the FIG1 episode-level OOF protocol on the 74 training episodes, natural-prompt and counterfactual-prompt writer side accuracy are each `>= 0.90`; development writer correctness is at least `7/8`. This gates only writer claims, not the natural-prompt mechanism chain;
- read head: natural correct on at least `7/8`, reset target-side correct on at most `4/8`, and opposite-memory donor followed on at least `7/8`;
- open-loop action use: native correct on at least `7/8`, reset target-side correct on at most `4/8`, and opposite-memory donor followed on at least `7/8`;
- zero injection: target-side correct on at most `4/8`, and paired predicted-side verdict differs from reset on at most `1/8` episode. Also report, without a separate hard gate, `macro(s_zero - s_reset)`, where `s = (RMS(delta_right_6) - RMS(delta_left_6)) / (RMS(delta_right_6) + RMS(delta_left_6) + 1e-8)` uses the six non-gripper joints of each arm on use-pressure steps and is averaged within episode first;
- correct-side train-prototype oracle succeeds on at least `7/8`, opposite-side prototype oracle-donor follows the donor on at least `7/8`, and the paired action-side prediction flips on at least `7/8`;
- report writer-dependent direct-carry oracle separately and require at least `7/8` agreement once the writer gate passes;
- Gate C production commit and retention thresholds continue to pass.

Attention mass is a read/use attribution diagnostic. Report its enrichment relative to the uniform token baseline and its paired change from reset/zero injection; it is not causal evidence and has no hard gate.

Run the same causal battery on training episodes as supporting evidence only. A reset/donor effect demonstrates causal use on those training samples, but neither its presence nor absence establishes held-out generalization.

On the 74 training episodes, evaluate the correct-side and opposite-side prototype oracles with leave-one-episode-out prototypes. The opposite-side oracle-donor follow rate is the primary well-resolved consumer diagnostic because memorized actions resist rather than create an intervention-driven flip.

Define task health on a fixed, no-augmentation calibration suite:

- freeze the suite's stable IDs, frame indices, preprocessing/norm hash, flow timestep, action noise, and RNG before evaluation;
- instantiate v3.5 once with official Pi0.5 base shared parameters and fixed-seed fresh leaves. Before enabling memory transition/injection, record the **fresh official-base path** flow loss and subtask CE on this exact suite. Reuse this identical parameter tree for the calibrated v3.5 step-0 evaluation, so both references share the same fresh heads and differ only by the enabled v3.5 path;
- v3.5 step 0 must have flow loss `<= 1.10x` the source reference and subtask CE `<= source + 0.05`;
- no NaN/Inf in losses, gradients, parameters, or memory state;
- every rung must have flow loss `<= 1.10x` both the source reference and its v3.5 step-0 value;
- every rung must have subtask CE `<= source + 0.05` and `<= v3.5_step0 + 0.05`;
- severe-clip rate, defined as pre-clip global grad norm greater than `10x` the configured clip threshold, is `<= 1%` of optimizer steps.
- branch-local feature-cotangent-cap bind rate is `<= 5%` of eligible E/D loss terms.

Pilot exit rule:

- **Pass:** all hard thresholds pass. Continue to the frozen full budget of **10,000 completed updates**.
- **Inconclusive:** core, injection, retention, reset, and task health pass, but a learned natural/donor count is `5/8` or `6/8`; or each prototype-oracle count is at least `6/8` and has improved by at least one episode from step 0 but is below `7/8`. Extend the same run once to 2,500 updates.
- **Fail:** a numerical/invariant/retention gate fails; reset is correct on at least `5/8`; natural or donor succeeds on at most `4/8`; task health fails; or either prototype-oracle count is at most `5/8` (or remains `6/8` without at least one-episode improvement). Stop this branch and use the diagnosed core, writer, or consumer fallback.
- At 2,500 updates, either pass the same thresholds or stop; no second extension.

After a 1k pass, use fixed post-pilot rungs at 2,500, 5,000, and 10,000 updates. The raw 10,000-update checkpoint is the preregistered primary endpoint if all mechanism and task-health gates remain valid; do not choose a checkpoint by maximizing development accuracy.

Keep June-30 sealed during pilot and branch selection. If a transfer result is desired, run it once at the frozen final endpoint only; it is never a gate or source of normalization/calibration data.

Counterfactual prompt binding is a hard gate for making a **prompt-bound writer claim** on development episodes. It does not block the mechanism rungs, because the natural-prompt memory chain can still be evaluated independently. Final-test episodes remain untouched until branch and checkpoint policy are frozen.

## 9. Code Scope

- `models/memory.py`: pooled association, output-only direct delta, FP32 fast state/read/commit, analytic skip-O decay, invariants, and telemetry.
- `models/pi0.py` and `pi0_config.py`: E-only transition, `L_write`, `L_read`, state-valid/reachable tracking, state-invalid side-loss defense, injection calibration support, two oracle interventions, and time-consistent augmentation.
- `training/data_loader.py` and `transforms.py`: E/O/D sidecar, five-frame E tail guard, strict D mask, hard final-E anchoring, natural/skip-O sampling, use-pressure mask, and seeded stable-ID split.
- `training/config.py`: independent v3.5 config and dataset version.
- standard official-base checkpoint loading plus an explicit shared/fresh initialization allowlist; `scripts/v35_step0_bootstrap.py` owns the zero-update initialize/finalize boundary, `scripts/v35_train.py` installs calibration values from the sealed artifact, and `scripts/train.py` accepts only an authorized resume and handles exact loss denominators and completed-update checkpoint semantics. No custom raw-checkpoint transplant is required for launch.
- `scripts/v35_prepare_pilot.py` and `cluster_v35/prepare_pilot.sh` provide the create-only, resume-validating path from fresh step 0 through calibration and Gate A/B/C to pilot authorization; they never run an optimizer update.
- Pilot authorization binds the complete production Python source tree, v3.5 scripts, dependency lockfiles, and runtime package versions. Every verify or train entry reloads the current Gate A/B/C evidence and authenticates the live checkpoint contents; a source, environment, parameter, optimizer, iterator, or telemetry change invalidates authorization. Resume is allowed only from the externally sealed source rung named by an authorization; an intermediate crash checkpoint needs its own sealed rung binding or the run restarts from the prior authorized source.
- The pre-pilot entry verifies train/development parquet and metadata bytes against the frozen dataset inventory, checks sealed final-test files by path and size without opening their payload, and fixes both replay and leakage batch size at 8 so resumed evidence cannot mix collection protocols.
- New manifest-driven leakage, writer, retention, attention, and causal evaluators.
- New unit and integration tests.
- Do not modify labeling/conversion files owned by the separate data-preparation session.

## 10. Execution Order

1. User and Claude approve this revision, including the 10,000-update full budget.
2. Finish manual labels, side/block audit, manifest, split, conversion, and train-only normalization.
3. Run the zero-update initialize phase from the official Pi0.5 base, freshly initialize every v3.5-specific leaf, and freeze the exact step-0 tree identity. The official base may be downloaded independently into each cluster's project-local cache.
4. Implement the memory clock, pooled output-only delta core, and numerical tests.
5. Implement masks, final-E anchoring, natural/skip-O sampling, state-valid/reachable tracking, losses, and sequence-consistent augmentation.
6. Set and freeze `tanh(memory_inject_w)=0.5`, calibrate `c` and `tau` on the 74 training episodes only, and freeze them.
7. Finalize the same zero-update state with calibrated injection values and exact initial sampler/RNG state; do not draw a data batch.
8. Use the create-only, resume-validating pre-pilot orchestrator to authenticate the frozen dataset inventory, run Data, Leakage, and Step-0 gates, then seal and verify the pilot authorization and live checkpoint for that exact run identity without an optimizer update.
9. Start the 1k pilot only by authorized resume from finalized checkpoint 0 and only if all launch gates pass.
10. Apply the fixed pass/inconclusive/fail exit rule and run final test once only after the branch and reporting policy are frozen.

## 11. Decisions Resolved by Claude's Review

1. **Split:** use `74 train / 8 development / 8 final test`; 0830 is fresh same-setup data from a different day, while June-30 remains eval-only.
2. **Leak gate:** use two primary probes per collection, Bonferroni-corrected permutation tests across the four primary tests, and a pooled 74-episode point-estimate gate. Other modalities are descriptive.
3. **Memory core:** use one pooled vector per frame with output-only direct delta for the pilot; remove the exact-16-pair claim.
4. **Primary parameters:** use raw parameters as primary. EMA is only a consistency check because a reset EMA is not meaningful after 1k updates.
5. **Prompt binding:** make it a hard gate for the prompt-bound writer claim on development data, not a blocker for the natural-prompt mechanism rungs.
6. **Read supervision:** mask with `read_state_valid`, not `read_credit_reachable`; report reachability separately.
7. **Fresh-base initialization:** load shared pretrained parameters from the official Pi0.5 base; freshly initialize memory/query compressors, conditioner, slot embedding, state-null, injection projection, ladder heads, and side heads. Oracle injection diagnoses the new consumer path without assuming step-0 competence.
8. **Fixed injection gate:** initialize every channel with `w=atanh(0.5)`, verify `tanh(w)=0.5`, and freeze it before calibration and throughout the pilot; there is no conditional inherited-leaf reset.
9. **Long delays:** use explicit skip-O analytic decay alongside natural windows; do not assume hidden state carry across independent samples.
10. **Development gates:** use episode counts on eight development episodes, not infeasible decimal thresholds.
11. **Pooled leakage point gate:** use `0.62` and accept its preregistered false-stop tradeoff; family-corrected permutation tests remain primary.
12. **Injection scale:** with all fixed-gate channels required open, calibrate `c` and `tau` on the 74 training episodes only and freeze `memory_inject_w`, `c`, and `tau` for the pilot.
13. **Source reference:** compare v3.5 step 0 and every rung against the fresh official-base path on the identical no-augmentation suite; no previous-run checkpoint or custom raw-checkpoint transplant is part of launch.
14. **Portable bootstrap:** official Pi0.5 weights are downloaded per cluster into `memory_project/v35/cache` and need not be synchronized. All non-downloadable data, norm assets, manifests, gate evidence, checkpoints, and provenance use `memory_project`-relative identities. Optimizer training begins only by authorized resume from the finalized zero-update checkpoint.

Implementation follows this revision immediately. Training begins only after all launch gates pass.
