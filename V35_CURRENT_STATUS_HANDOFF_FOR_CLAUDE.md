# v3.5 Current Structure, Status, and Claude Handoff

Last updated: 2026-08-31, America/Los_Angeles.

Purpose: this is the operational handoff for the current v3.5 implementation. It describes what exists, what has been verified, the current failed pre-pilot attempt, the immediate blocker, and the rules for continuing without invalidating the scientific lineage.

This document reports the repository as it exists now. The longer frozen design contract remains in `V35_PLAN_FOR_CLAUDE_REVIEW.md`.

## 1. Executive Status

The v3.5 implementation is substantially complete, but it is **not authorized for optimizer training yet**.

The converted 90-episode dataset, frozen 74/8/8 split, train-only normalization, portable project layout, memory core, data masks and sampler, calibration collectors, Gate A/B/C/D producers and reducers, two-phase step-0 bootstrap, source identity freeze, live checkpoint authorization, and pre-pilot orchestrator are implemented.

A real four-H100 pre-pilot attempt named `v35_fresh_pilot_20260831_r2` was run on Slurm allocation `17130107`. It completed the expensive fresh Pi0.5 bootstrap and wrote the provisional step-0 identity. It then stopped before calibration because the generated `initialization_graft_manifest.json` is pretty-printed JSON, while the immutable-stage validator requires canonical compact JSON with exactly one trailing newline.

The failure was fail-closed and occurred before:

- calibration replay;
- Gate A, Gate B, or Gate C;
- pilot authorization;
- any optimizer update;
- any final-test access.

Therefore the scientific state is clean, but pre-pilot qualification is incomplete.

Immediate next action:

1. Make the graft-manifest writer emit the same canonical bytes expected by every immutable-stage consumer.
2. Add a focused writer-to-consumer regression test.
3. Run the focused bootstrap/orchestrator suites and the full v3.5 regression suite.
4. Sync the exact changed files to the main project.
5. Start a **new experiment name**, for example `v35_fresh_pilot_20260831_r3`.

Do not resume `r2` after changing source. Its pre-pilot source snapshot authenticates the old source tree, so a correct resume must reject the changed code.

## 2. Scientific Objective and Phase Names

v3.5 tests one explicit causal memory chain:

- **E, evidence:** `inspect both bins`. Eligible frames write prompt-conditioned target-side evidence.
- **O, occlusion:** `close both lids and reset arms`. No writes are allowed; memory only decays.
- **D, decision:** static `wait; target bin is left/right`. The policy must read previously stored evidence before the target-side execute motion.
- **X, execute:** `open left bin` or `open right bin`.

The stored value means the prompt-conditioned target side. It is not intended to be a prompt-independent object-location representation.

The final permitted claim is within-collection held-out memory behavior on the combined 0816+0830 data. It is not a cross-session generalization claim, and open-loop side steering is not closed-loop task success.

## 3. Portable Project Structure

The intended transfer unit is the complete `memory_project` directory.

```text
memory_project/
  data/
    lerobot/
      yam/bin_memory_0816_0830_v35_subtask/
    0816_0830_episode_manifest_v35_frozen.json
    0816_0830_episode_manifest_v35_frozen_*.json
  openpi/
    cluster_v35/
      env.sh
      sync_project.sh
      compute_norm_stats.sh
      step0.sh
      prepare_pilot.sh
      train.sh
    scripts/
      v35_step0_bootstrap.py
      v35_collect_calibration_replay.py
      v35_calibration_replay.py
      v35_injection_calibration.py
      v35_data_gate.py
      v35_leakage_features.py
      v35_leakage_gate.py
      v35_rung_collect.py
      v35_rung_eval.py
      v35_pilot_gate.py
      v35_training_authorization.py
      v35_prepare_pilot.py
      v35_train.py
    src/openpi/
      shared/project_paths.py
      models/memory.py
      models/pi0.py
      models/pi0_config.py
      training/config.py
      training/data_loader.py
      training/weight_loaders.py
      training/checkpoints.py
      training/v35_authorization.py
      transforms.py
  v35/
    assets/pi05_yam_0816_0830_v35/
    cache/
    checkpoints/
    diagnostics/
    tmp/
    wandb/
  V35_PLAN_FOR_CLAUDE_REVIEW.md
  V35_CURRENT_STATUS_HANDOFF_FOR_CLAUDE.md
```

All production v3.5 inputs and outputs are resolved through `MEMORY_PROJECT_ROOT`. Runtime launchers confine LeRobot data, model/tokenizer caches, JAX cache, assets, diagnostics, checkpoints, temporary files, and W&B files below the project.

The official Pi0.5 base remains the downloadable identity:

```text
gs://openpi-assets/checkpoints/pi05_base/params
```

The other cluster may download that asset independently. The official-base cache does not need to be synchronized. No run5 or v3.4 checkpoint is an input to v3.5.

The Python environment is cluster-specific. A supplied interpreter must import `openpi` from this same copied checkout; a foreign editable installation fails closed.

## 4. Frozen Data State

Converted dataset location:

```text
data/lerobot/yam/bin_memory_0816_0830_v35_subtask
```

Data population:

- 0816: 60 episodes.
- 0830 part 1: 16 included episodes.
- 0830 part 2: 14 included episodes.
- Excluded raw episode: `0830_bin_part2/demo14`, because it has no terminal execute phase.
- Converted total: 90 episodes.
- Frozen split: 74 train, 8 development, 8 sealed final test.
- Split seed: 35.
- Train-only normalization rows: 64,618.

Frozen identities:

| Artifact | SHA-256 |
| --- | --- |
| Dataset tree | `21b02ea4752280abee252535f1c519f611b7308a6540c73a699ee2bcbd47ed5f` |
| Relocation inventory | `6856b49a6910c9449c983af582a922ce037300bc51c01eea1ed3317f1f409b73` |
| Frozen episode manifest | `c3b41d3247204aee4b7428ffdcc80a21d699a62aadec478cc499b52b6881dd57` |
| Train-only norm stats | `b8ed515a495b04b7a58cdc6d18e18aeec888e6f6133dccf35635eb497dc9e3d7` |
| Portable norm provenance | `c46bb0e714b8657f08bc73f0b252d8f9b79f623fba9829623c249b9eb89511bf` |
| Train-storage seal | `2eb4eefccd1a03ab05f304a1add74382f2f4067aea0164ccff6e7a41dc6c9074` |
| Dataset/frame protocol | `c8f4ef48a4717e45c992ee456bdf7ec4220bf1db11d203a48ce71c1f3a1c96db` |

The project-local dataset copy contains 202 files and 58,534,560,294 logical bytes.

The original source dataset under the old Hugging Face cache is still being uploaded and must remain untouched:

```text
/iris/u/kewalk/.cache/huggingface/lerobot/yam/bin_memory_0816_0830_v35_subtask
```

No current setup, sync, or training operation is authorized to delete or mutate that source.

The frozen manifest includes stable IDs, raw source identity, converted indices and frame counts, prompt, object, target side, collection, 0830 part, split, label hashes, E-visibility review, independent strict `D_valid` evidence, and a block-confound audit.

Training revalidates exact task vocabulary, prompt/object mapping, side consistency, label bytes, and five-phase coverage. Final-test payload is not opened during pre-pilot work.

## 5. Memory-Core Implementation

The original rank-one idea could not exactly fit 16 simultaneous token associations. v3.5 therefore uses one pooled association per evidence frame.

Current rule:

1. Pool 16 projected keys and values to `k_bar` and `v_bar`.
2. Use an output-layer-only direct delta update.
3. Keep hidden fast leaves unchanged.
4. Keep fast output bias and momentum leaves exactly zero.
5. Read before the current step transition.
6. On an eligible E step: decay, then commit.
7. On a valid non-E step: decay only.
8. On padding or invalid transition: strict no-op.

For row-vector hidden state `h`:

```text
W_dec = (1 - alpha_step) * W
r = v_bar - h @ W_dec
W_new = W_dec + delta_rate * outer(h, r) / (h dot h)
```

Frozen pilot values:

- `delta_rate = 1.0`
- `alpha_step = 0.01`
- memory clock = one sampled step per 15 raw frames
- drift trust region disabled in delta mode

Fast `W`, commit arithmetic, analytic decay, hidden norm, raw retrieval, and calibration are FP32 regardless of surrounding model dtype. Degenerate pooled or hidden norms fail closed.

The own-key commit and natural query retrieval are tested separately. Exact commit does not imply that D queries align with the stored E key.

## 6. Data, Masks, and Sampling

Frozen sequence fields include:

- `seq_write_mask`: eligible current-frame E commits.
- `seq_decision_mask`: strict static D reads.
- `seq_occlusion_mask`: O frames.
- `seq_read_state_valid`: a successful eligible E commit exists before this read.
- `seq_read_credit_reachable`: the prior E commit is inside the current differentiable window.
- `seq_decay_gap_before`: omitted valid non-write sampled transitions before the current read.
- `seq_use_pressure_mask`: D steps whose action chunk reaches initial execute motion.
- `seq_sparse_skip_o`: analytic skip-O family marker.
- episode, collection, object, side, and memory-cell IDs.

Two memory-critical window families are implemented:

- natural contiguous E/O/D windows;
- skip-O windows that commit E, apply exact FP32 analytic decay, then evaluate D.

The sampler verifies that every omitted skip-O sampled frame belongs to semantic O and contains no write or reset event. Tail-E or non-O gaps are rejected before they receive sampling mass.

Side losses are aggregated per episode and macro-balanced by memory cell. Runtime successful-commit state is combined with data-provided state validity. State-invalid D objectives are removed from both numerator and denominator.

## 7. Model Training and Diagnostics

Implemented v3.5 objectives and controls include:

- write-side loss on non-detached pooled `v_bar`;
- read-side loss on mean raw retrieval;
- detached ladder diagnostics;
- branch-local feature cotangent cap;
- per-update severe global-clip accounting;
- finite-update accounting;
- query/key cosine telemetry per delay rung;
- direct-carry oracle;
- correct-side prototype oracle;
- opposite-side prototype oracle;
- opposite-side donor intervention;
- reset and zero-read controls;
- action-expert-to-memory attention diagnostics;
- paired no-memory versus calibrated-memory task-health checks.

Time-consistent augmentation reuses sampled transform parameters across time for the same sample and camera, while different samples use independent random parameters.

## 8. Initialization, Calibration, and Authorization

v3.5 starts from the official Pi0.5 base only. Shared pretrained leaves are loaded through an explicit audited allowlist. All v3.5-specific memory, query, consumer, diagnostic, and side-head leaves are freshly initialized. Optimizer, EMA, and global step are fresh.

`memory_inject_w` is initialized channelwise to `atanh(0.5)` and frozen. Calibration derives `tau` and `c` from train-74 only, using the exact 16-slot FP32 production pinning operation before aggregation.

The zero-update launch sequence is deliberately two-phase:

1. `initialize`: load the official base, create the exact fresh tree and raw checkpoint 0, and write provisional identities.
2. Collect train-only replay and calibrate injection.
3. `finalize`: verify the same step-0 tree, install frozen calibration, and write the finalized checkpoint 0 with untouched optimizer and exact initial iterator/RNG state.
4. Run Gate A, Gate B, and rung-0 Gate C/task health.
5. Seal pilot authorization.
6. Only then may training resume checkpoint 0.

The source/runtime snapshot is created before the first stage and includes production OpenPI Python source, v3.5 scripts, dependency lock files, and runtime package versions. Every stage execution, stage skip, authorization, verification, and training launch rechecks it.

Authorization v2 binds:

- complete source/runtime identity;
- semantic training config hash;
- manifest, norm, storage, and calibration identities;
- official-base source identity;
- actual step-0 parameter tree;
- actual train-state tree, including optimizer state;
- iterator/RNG state;
- cumulative telemetry state;
- externally sealed source rung.

`verify-only` and real training both restore the live Orbax checkpoint and compare it against authorization-linked evidence. Self-consistent but tampered checkpoint-owned JSON is insufficient.

## 9. Pre-Pilot Orchestrator

Primary entry point:

```text
openpi/cluster_v35/prepare_pilot.sh
```

It performs these stages without an optimizer update:

1. fresh bootstrap initialize;
2. calibration preflight;
3. frozen train-74 replay selection;
4. four parallel replay collectors;
5. replay seal;
6. injection calibration;
7. bootstrap finalize;
8. Gate A data decision;
9. Gate B feature extraction;
10. Gate B leakage decision;
11. rung selection;
12. rung-0 collection and sealing;
13. semantic config identity;
14. pilot authorization;
15. final authorized checkpoint-0 verification.

Gate B failure stops the natural branch before development payload is read. All production replay and leakage batches are fixed at 8; resume with a changed batch protocol is rejected.

Before GPU work, the orchestrator verifies train/development parquet and metadata bytes against the frozen inventory. Final-test parquet files are checked only for regular-file status and frozen size; their content is not read or hashed.

## 10. Current H100 and Slurm State

Allocation used for the real run:

- Slurm job allocation: `17130107`
- node: `iris-hgx-1`
- accelerators: 4 H100
- current allocation keeper: `cluster_scripts/train_hs.py`, approximately 1 GB per GPU

The independent Qwen training step `17130107.87` was stopped with `scancel 17130107.87` after user authorization. The parent allocation was retained. Job `17126912` on `iris-hgx-2` was not touched.

A four-GPU JAX smoke test saw all devices successfully.

The current failed pre-pilot command used:

```bash
openpi/cluster_v35/prepare_pilot.sh \
  --experiment-name v35_fresh_pilot_20260831_r2 \
  --gpus 0,1,2,3
```

Current `r2` output files on the main project include:

```text
v35/checkpoints/pi05_yam_mem_v35/v35_fresh_pilot_20260831_r2/
  initialization_graft_manifest.json
  step0_bootstrap_provisional.json
  prepilot_source_identity.json
```

The bootstrap itself completed. The orchestrator then returned exit code 2 with:

```text
immutable JSON is not canonical with one trailing newline:
.../initialization_graft_manifest.json
```

## 11. Immediate Blocker: Graft Manifest Encoding

Direct cause:

- `openpi/src/openpi/training/weight_loaders.py::_write_manifest()` currently serializes the graft manifest with `json.dumps(..., indent=2, sort_keys=True)`.
- `openpi/scripts/v35_prepare_pilot.py::_load_immutable_json()` accepts only canonical compact sorted JSON, with no extra whitespace and exactly one trailing newline.
- The bootstrap producer and orchestrator consumer therefore disagree on the byte-level immutable artifact contract.

Recommended correction:

- Define or reuse one canonical JSON byte function: sorted keys, compact separators, `allow_nan=False`, frozen `ensure_ascii` policy, exactly one trailing newline.
- Make `_write_manifest()` write those bytes create-only.
- Keep `manifest_sha256` semantics unchanged unless a schema migration is intentionally introduced.
- Add a regression that writes a real `GraftManifest` through `_write_manifest()` and immediately loads it through the production immutable-stage loader.
- Add a cold subprocess bootstrap-to-stage-validation test if practical.

Do not loosen the orchestrator to accept arbitrary pretty JSON. The stronger solution is to make the immutable producer obey the existing canonical contract.

After the source fix, do not use `--resume` on `r2`. Start a new experiment because the old run's frozen source identity must reject source drift.

## 12. Other Important Problems Already Found and Fixed

The following issues were discovered during implementation and have already been addressed in the current source tree unless noted otherwise:

1. **Invalid 16-token rank-one exact-fit claim.** Replaced by pooled-frame output-only direct delta.
2. **Gradient-step factor mismatch.** The new write rule is named and implemented as direct delta, not the old theta gradient step.
3. **Hidden fast-map drift.** Delta mode keeps hidden fast leaves fixed and disables the old drift trust region.
4. **FP16/BF16 precision risk.** Fast weights, commits, raw reads, and calibration are pinned to FP32.
5. **E/O/D clock ambiguity.** The production clock is exactly one memory step per 15 raw frames.
6. **Long-delay supervision loss.** Added analytic skip-O windows with explicit decay-gap accounting.
7. **Wait-motion leakage.** D uses an independent strict 14D static-valid sidecar instead of automatically rewriting semantic labels.
8. **Normalization leakage.** Norm statistics are computed from train-74 only and training validates their provenance and exact bytes.
9. **Manifest-side/object/prompt mismatch.** Active episodes require exact label, prompt, object, side, part, split, and stable-ID agreement.
10. **Silent sampler loss.** Gate A reports and validates per-episode E commits, natural candidates, skip-O candidates, D state validity, and use-pressure availability.
11. **Gate B false continuation.** A failing Gate B decision stops the pipeline before rung/dev evaluation.
12. **Inference memory no-op or uncontrolled writes.** v3.5 requires explicit transition-valid and write masks; omitted masks are a no-op.
13. **Norm producer/consumer path mismatch.** Both use project-local `v35/assets` and portable provenance.
14. **Source drift after partial pre-pilot.** A create-once full source/runtime snapshot is checked before every stage and skip.
15. **Foreign editable checkout.** Production entry points verify that imported OpenPI source belongs to `MEMORY_PROJECT_ROOT`.
16. **Symlink path escape.** Project-path resolution checks containment after resolution.
17. **Checkpoint-owned provenance forgery.** Verification restores live params, train state, iterator, runtime identity, and telemetry and compares them to an external authorized rung.
18. **Checkpoint-0 W&B resume failure.** The first real training start from bootstrap checkpoint 0 creates the W&B run; later rungs require strict resume.
19. **Replay batch drift across resume.** Production replay and leakage batch sizes are fixed at 8.
20. **Torch/config import-order segfault.** Gate A cold import loads the Torch/data-loader path before training config; subprocess help coverage exists.
21. **Frozen manifest pretty-print mismatch.** The already SHA-authenticated historical frozen episode manifest has a dedicated strict loader that accepts its frozen representation without pretending it is a canonical envelope.
22. **NNX structural `None` leaves.** Bias-free NNX linears contain real structural `None` leaves. Graft schema/hash logic now represents them explicitly as `dtype=none`, `shape=()`, `structural-none`, while rejecting arbitrary objects.

## 13. Real Structural-None Failure and Verification

The first real fresh bootstrap attempt exposed a bias-free NNX leaf:

```text
read_query_compressor/key_proj/bias = None
```

The audited loader previously treated every leaf as an array and failed before training. The fix added an explicit structural-None schema identity and preserved the target sentinel.

Evidence after the fix:

- focused weight-loader/bootstrap/calibration suite: 40 passed;
- exact weight-loader suite: 19 passed;
- a direct four-H100 bootstrap named `v35_diag_structnone3` succeeded and wrote its provisional step-0 artifact.

Retained diagnostic experiments:

- `v35_fresh_pilot_20260830`: failed before optimizer update on the original structural-None issue;
- `v35_diag_none_leaf2`: failed diagnostic that localized the exact path;
- `v35_diag_structnone3`: successful direct bootstrap smoke;
- `v35_fresh_pilot_20260831_r2`: bootstrap succeeded, then stopped on graft-manifest canonical encoding.

These directories are provenance and debugging evidence. Do not delete or silently reuse them.

## 14. Test Evidence

Recorded results before the latest real run:

- full v3.5-focused repository suite: 285 passed, 5 deselected;
- main orchestration/authorization/Gate-A focused suite: 49 passed;
- iris-ws-18 one-GPU memory/gradient contracts: 17 passed;
- latest structural-None focused suite: 40 passed;
- exact weight-loader suite: 19 passed;
- earlier Gate-C real gradient contracts on H100: 5 passed;
- broad merged authorization/exact-resume suites were green in their focused runs;
- Ruff, formatting checks, `git diff --check`, and shell syntax checks were green before the latest blocker.

Important qualification:

The full 285-test suite has not yet been rerun after the latest structural-None patch. Only focused suites and a real four-H100 bootstrap have validated that patch. After fixing graft-manifest serialization, rerun both the focused tests and the full v3.5 suite.

## 15. Worktree and Ownership Warning

The repository is intentionally dirty. It contains extensive user-owned labeling, conversion, report, v3.4, and v3.5 work. Do not reset, clean, checkout, or overwrite unrelated files.

The user-owned labeling/conversion files were hash-checked before and after the selective sync and remained unchanged. Continue using explicit file whitelists when copying source changes to the main project. Never use a broad sync with `--delete`.

Do not edit production Python source while a pre-pilot experiment is actively running. The source snapshot is intentionally strict; a code edit invalidates that experiment lineage. Documentation outside the hashed production selection may be edited safely.

## 16. Safe Continuation Procedure for Claude

Claude should follow this order:

1. Read this handoff, `V35_PLAN_FOR_CLAUDE_REVIEW.md`, `v35/README.md`, and `openpi/docs/v35_training_authorization.md`.
2. Confirm that `v35_fresh_pilot_20260831_r2` is stopped, not running.
3. Preserve all old experiment directories and the original Hugging Face upload source.
4. Fix only the graft-manifest producer's canonical byte encoding and add the cross-component regression.
5. Run focused tests covering `weight_loaders`, `v35_step0_bootstrap`, `v35_prepare_pilot`, calibration replay, and authorization.
6. Run Ruff, format check, `git diff --check`, and shell syntax checks.
7. Run the full v3.5 regression suite.
8. Selectively sync changed files to `/iris/u/kewalk/memory_project`; verify that user labeling/conversion hashes remain unchanged.
9. Use a new experiment name, such as `v35_fresh_pilot_20260831_r3`.
10. Run `--plan`, then the real four-H100 prepare command.
11. Do not start `train.sh` unless all 15 pre-pilot stages finish and final `verify-only` passes.
12. If any stage fails and source must change, preserve the failed experiment and use another new experiment name after the fix.

Suggested plan command from the project root:

```bash
openpi/cluster_v35/prepare_pilot.sh \
  --experiment-name v35_fresh_pilot_20260831_r3 \
  --gpus 0,1,2,3 \
  --plan
```

Suggested real pre-pilot command:

```bash
openpi/cluster_v35/prepare_pilot.sh \
  --experiment-name v35_fresh_pilot_20260831_r3 \
  --gpus 0,1,2,3
```

Only after final authorization and verification:

```bash
openpi/cluster_v35/train.sh \
  --experiment-name v35_fresh_pilot_20260831_r3 \
  --calibration v35/diagnostics/runs/v35_fresh_pilot_20260831_r3/calibration/injection_calibration.json \
  --pilot-authorization v35/diagnostics/runs/v35_fresh_pilot_20260831_r3/pilot_authorization.json \
  --target 1000 \
  --fsdp-devices 4
```

## 17. Remaining Release Risks

The main unresolved items are:

1. The graft-manifest producer/consumer canonical JSON mismatch blocks real pre-pilot stage 2.
2. A complete real train-74 calibration replay has not yet finished.
3. Real Gate A/B/C and rung-0 evidence have not yet been sealed for an authorized experiment.
4. No pilot authorization exists yet.
5. The first true 1,000-update pilot has not started.
6. The latest structural-None patch still needs the full regression suite, not only focused tests.
7. Real replay and rung collection may expose additional runtime or memory-pressure issues despite strong unit coverage.
8. Final-test observations remain sealed and must stay sealed until the preregistered final endpoint.

## 18. Definition of Ready for Training

v3.5 is ready for the 1,000-update pilot only when all of these are true:

- new experiment source identity is frozen and unchanged;
- official-base graft and actual step-0 tree are authenticated;
- train-74 replay and injection calibration pass;
- finalized checkpoint 0 matches the calibrated tree exactly;
- Gate A passes;
- Gate B passes and chooses the permitted branch;
- rung-0 Gate C and task health pass their required conditions;
- pilot authorization is canonical and externally linked;
- final `v35_train.py --verify-only` restores and authenticates the live checkpoint;
- no optimizer update has occurred before those checks;
- the launch uses the exact authorized source, dataset, norm, calibration, configuration, iterator state, and checkpoint.

Until then, status should be reported as **implementation present, pre-pilot qualification blocked, training not authorized**.

