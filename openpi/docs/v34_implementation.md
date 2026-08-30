# v3.4 implementation notes and runbook

> **Status note (2026-08-27):** the current user-designated v3.4.1 snapshot is documented in
> [`v3.4.1_claude_handoff.md`](v3.4.1_claude_handoff.md). That handoff incorporates the run4/run5
> evidence audit and supersedes this file wherever the two conflict. In particular, several
> causal claims and historical metric descriptions below are now known to be too strong.

Implements the historical v3.4 plan (v1.3), whose durable design decisions are folded into
this document and the v3.4.1 handoff. Everything is behind flags on the shared v3.2
skeleton; `pi05_yam_mem_v34` turns them on together. All defaults preserve v3.2/v3.3
bit-exactly (their tests and checkpoints still pass/load).

## Plan-to-code map

| Plan | Code |
|---|---|
| 5.1 aux demand (frame-invariant Q_aux, post-write read_key, class-balanced macro CE, margin A/B) | `pi0.py::_compute_sequence_loss_v32` (aux block), `memory_aux_*` params in `Pi0.__init__`, macro assembly in `scripts/train.py::_aux_macro_ce` |
| 5.2 input-level per-segment state masking (single view default, dual-view flag) | `Pi0._v32_apply_state_null`, seq-loss step; per-segment draw in `transforms.py::MemoryV34Labels`; state span from `tokenizer.py::FASTSubtaskTokenizer._state_span_mask` (sentencepiece byte offsets) |
| 5.3 blinded memory tokens | `Pi0._v32_split_late_mask` (`memory_blind_tokens`) |
| 5.4 CE re-seed from last valid non-memory token | `Pi0._v32_ce_seed_hidden` / `_v32_causal_seed`; used by the seq loss, `_sample_with_memory_v32` (scoring + greedy token0), `v33_endpoint_gradient_step` |
| 5.5 QK-norm + temperature + letterbox P_valid | `MemoryQueryCompressor` / `MemoryQueryConditioner` (`qk_norm`, `logit_scale` init 0.5·log d_head, clamp exp(λ)≤64); `letterbox_patch_valid` (480×640 → grid rows 2..13 valid); applied identically in `__call__` and `attention_probs` via the shared `_attention_logits` |
| 5.6 tanh(w)·RMS-pinned injection | `Pi0._v32_inject_memory` (`memory_injection_mode="tanh_rms"`; calibrated `memory_injection_c/tau` values live in the run configs and launch provenance) |
| 5.7 unit-L2 memory MLP + autodiff token diagnostics + drift region | `memory.py` (`MemoryConfig.mlp_l2norm`, He layer-0, `token_write_diagnostics` on per-token `jax.grad`, `drift_radius`) |
| 5.8 gates frozen | unchanged `freeze_filter=.*memory/gate.*` |
| 5.9 instruction-only conditioner context | `memory_conditioner_context="instruction_only"`; `token_state_mask` is REQUIRED at every `_v32_prepare_*` call site (fails loudly if missing) |
| 5.10 key-space core API | `memory.py`: `project_kv` / `project_q` / `read_key` / `write_kv`; `read`/`write` are thin wrappers (bit-identical, tested) |
| §6 writer/read probes | online stop-grad heads in the seq loss with isolated constant-LR SGD (`train.py`, `TrainConfig.probe_lr`); fixed-frame, freshly refit episode-split writer evaluation: `scripts/v34_fixed_writer_probe_eval.py` |
| 8.4 three-way retention | `TitansMemory.write_kv(zero_gradient=True)` / `decay_step`; `sample_with_memory(write_mode="normal"|"frozen"|"dynamics_only")`; eval: `scripts/v34_retention_eval.py` (prints the plan-8.4 table + the v3.4.1 trigger) |
| Stage 0 | `scripts/v34_stage0_memory_core.py` (PASS on A5000 and H100; `--legacy` reproduces the v3.3 failure) |
| Held-out episodes | `DataConfig.heldout_episodes=(15, 29, 44, 59)` — last episode of each (instruction × side) cell, zero sampling mass in every branch (`data_loader._sequence_sampling_info`) |

New Observation fields: `token_state_mask`, `seq_state_masked`, `seq_subtask_class`,
`seq_side_label`, `seq_evidence_mask`, `seq_waiting_mask` (wired through `MemoryV34Labels`,
`YamInputs`, repack structure, `_SEQUENCE_TIME_KEYS`, `inputs_spec`, FakeDataset).

Unit tests (plan 9.2 a–i): `memory_v34_test.py` (f, h, i + Stage-0 smoke),
`pi0_v34_test.py` (a, b, c, d, e, g + qk-norm, letterbox geometry, write_mode, aux credit),
`tokenizer_v34_test.py`, `transforms_v34_test.py`, `train_test.py::test_v34_*` (isolated
probe SGD inside the real train step), and `v34_fixed_writer_probe_eval_test.py`.

Deliberately deferred (plan-permitted): the optional layer-0-instruction-embedding
conditioning of Q_aux (5.1, "further flag, default off") and the per-slot injection scalar
(5.6 "optional factorization") are not implemented; the §8 action-level swap statistic S is
evaluated with `v33_offline_eval.py`'s swap machinery + the retention script at token level
first (extend to the action statistic at checkpoint-eval time).

## Cluster runbook

The retained launchers are the supported, configuration-specific entry points. Run them
inside an existing allocation according to their own fail-closed resource checks; do not
copy old job IDs or GPU bindings from historical logs.

1. `stage0.sh` — memory-core numerical checks.
2. `smoke_run5_eta0.sh`, `pilot_run5_eta0.sh`, `train_run5_eta0.sh` — the stable eta=0
   run5 sequence.
3. `smoke_run6_staticwait.sh`, `train_run6_staticwait.sh` — the static-wait causal-mask
   experiment.
4. `run_fixed_writer_probe_step_one_gpu.sh` and
   `run_evidence_instruction_probe_video_step_one_gpu.sh` — checkpoint-bound diagnostics.
5. `snapshot_checkpoint.py` — manifests and copies a finalized checkpoint without
   mutating it; its focused test is `test_snapshot_checkpoint.py`.

## v34_run1 postmortem (2026-08-24) and the two stability guardrails

v34_run1 entered a ~700-step explosion/collapse limit cycle starting ~step 1300: `grad_norm`
doubled every ~100 steps (5 → 50 → 2e5+), then every memory diagnostic collapsed (side/phase
accuracy to chance, read probe 1.0 → 0.8, CE rising) while the global clip scaled every
parameter's update by 1/norm; partial recovery, then repeat. Killed at ~2.8k.

Measured root cause in the preserved run1 diagnostic logs (ckpt 2750): outer training —
dominantly the 5.1 aux demand (aux-alone grads 10–14 vs CE-alone
1.5–2.8) — drove the recurrent write chain's BACKWARD pass expansive: median 1.16–1.20x per
step at ckpt 2750 vs contractive (0.89–1.05x) at init on identical captured inputs, i.e.
50–500x over one segment and far more at cycle peaks, landing in `memory.m0` (the deepest
parameter). Inner write grads sat 20x over their design point (45–53 vs 2.8 at init,
per-write clip saturated at 0.02). Not the L2Norm 1/||x|| singularity (pre-norm activations
healthy at 0.6–1.3); not a weight blow-up (all interface params near init).

Guardrails added (both sized from the measurements, both no-ops when not binding —
unit-tested bit-exact):

* `MemoryConfig.state_cotangent_clip = 10.0` — `_clip_state_cotangent` (custom_vjp identity)
  on `write_kv`'s incoming state: per-sample clip of the M_t → M_{t-1} backward chain,
  direction preserved. Healthy state cotangents measured 0.4–1.8 (depths 1–10); 10.0 binds
  only on the expansive tail. Same-step read cotangents are outside the clip.
* `TrainConfig.memory_grad_clip = 5.0` — group pre-clip of the memory-path gradients
  (`train.py MEMORY_PATH_FILTER`) BEFORE the shared global clip; logs `memory_grad_norm`.
  Memory-group norms measured 2–14 (median ~5) with the whole VLM at 1–2; the cap stops
  spikes from reaching every parameter through the global clip.

Escalation if cycles persist in v34_run2: enable `drift_radius` (plan-5.7 trust region;
measured segment drift reached 17.6 and was still growing). Watch `memory_grad_norm` and
`grad_norm` in wandb; the cycle signature is grad_norm doubling per 100-step window.

### v34_run2 (8ya7k5ir): the cycle returned through the VLM path — third guardrail

Run 2 (both first guardrails active) reproduced the cycle on schedule (~step 1200–1400): the
memory group stayed capped (peak 17.6 pre-clip) but total grad_norm hit 48 with the memory
group at 3.5 — the amplified backward escaped through the write's k/v inputs into the VLM,
a path the first two clips do not cover. Killed at ~1500.

* `MemoryConfig.kv_cotangent_clip = 1.0` — same custom-vjp construction, applied to the
  write's projected (k, v): caps, per sample per write, what one step may send backward
  toward the write tokens and the tower below.
* Telemetry: `diagnostic/write_inner_grad_norm` / `_max` (from the write aux, stop-gradient)
  — core steepness. Healthy 0.5–3; ramping toward ~50 precedes each cycle by several hundred
  steps. `drift_radius` was ruled out as an escalation: the fresh core's per-segment drift
  (p95 17.9 in the preserved diagnostic log) is indistinguishable from the pathological
  core's 17.6 — steepness, not drift magnitude, marks the disease. The one-off probe
  programs were retired after the result was incorporated here; their launch-time source
  remains in the provenance snapshot.

Run 3 (`v34_run3`, wandb nwpoji2d) carries all three clips + telemetry. If the cycle appears
a third time, the remaining lever is the pressure source itself (aux weight / TBPTT segment
structure) — a plan-level change.
