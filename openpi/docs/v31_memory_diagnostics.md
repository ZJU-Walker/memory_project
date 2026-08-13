# π0.5 Memory v3.1 diagnostics

**Document date:** 2026-08-12

**Artifact schema:** `openpi.v31.diagnostics.v1`

**Current concrete adapter:** fixed-observation, offline-only π0.5/YAM replay

This document describes the diagnostics that are implemented in the current repository. It is an
operator and data-contract reference, not a proposal. Where the generic framework exposes a future
extension point that the π0 adapter does not yet implement, that limitation is called out explicitly.

The relevant implementation is split across:

- [`src/openpi/diagnostics/v31.py`](../src/openpi/diagnostics/v31.py): shared experiment contracts,
  four test runners, safety checks, artifact writer, summaries, and CLI;
- [`src/openpi/diagnostics/v31_pi0.py`](../src/openpi/diagnostics/v31_pi0.py): concrete π0.5/YAM
  checkpoint loader, raw replay loader, canonical scoring, and model adapter;
- [`src/openpi/models/memory_diagnostics.py`](../src/openpi/models/memory_diagnostics.py): complete
  fast-memory clone/hash/snapshot persistence;
- [`scripts/v31_memory_diagnostics.py`](../scripts/v31_memory_diagnostics.py): thin executable entry
  point.

## 1. What is implemented now

The current implementation provides:

- deterministic, fixed-observation offline replay from raw YAM episode directories;
- exact full-sequence scores for `open left bin\n` and `open right bin\n`;
- a side-effect-free `evaluate_snapshot` contract with complete state cloning and hash checks;
- oracle-subtask, complete-state-swap, freeze-after-reveal/close, and temporal diagnostics;
- checkpoint-parameter-tree-compatible v3.1 loading with an explicit static config;
- task-configured action-side classification, with `undetermined` as the safe default;
- strict JSON/CSV/NPZ artifacts, complete state snapshots, aggregate summaries, and plots;
- fail-closed safety validation for offline, shadow, and future active modes.

The following are **not** implemented by the concrete π0 adapter:

- a live observation or robot adapter;
- WebSocket broker replay or asynchronous broker scheduling;
- hardware actuation of any kind;
- recorded RTC committed prefixes in the raw offline source;
- the expensive per-token calculations proposed for `--diagnostic-level tokens`.

Consequently, the current supported operating mode is offline shadow evaluation. The π0 adapter
raises `NotImplementedError` for every `realtime` request before opening hardware. Offline replay
also supplies no RTC prefix, so `committed_action` equals `pre_rtc_action`, the event warns that RTC
and broker state are unavailable, and an oracle result can establish action conditioning but cannot
claim that the asynchronous RTC/broker path passed.

### 1.1 Training probe versus this diagnostic suite

The current v3/v3.1 main training objective is:

\[
\mathcal{L}=\mathcal{L}_{\mathrm{flow}}+
1.0\,\mathcal{L}_{\mathrm{causal\ CE}}.
\]

Both configs use `memory_probe_weight=0.0` and disable probe diagnostics by default. The old binary
probe head remains in the parameter tree for checkpoint compatibility, but it does not contribute to
the main loss. If its separate training-time diagnostic flag is enabled, its outputs are detached and
its raw and EMA parameters remain frozen.

The four tests in this document do not use that binary probe as their evidence. They use canonical
policy sequence scores, generated subtasks, actions, and complete-state interventions. This matters
because a post-write binary probe can exploit the current query or visible state and is not by itself
evidence that historical memory causes the policy's later decision.

## 2. Non-negotiable evaluation contract

Every branch is evaluated through the shared functional interface:

```python
evaluate_snapshot(
    observation,
    fast_state,
    rtc_state,
    *,
    forced_subtask=None,
    zero_read=False,
    allow_write=False,
    seed=0,
)
```

The wrapper enforces the following properties:

1. It hashes the caller-owned complete state before evaluation.
2. It asks the adapter for an isolated state clone and deep-copies the observation and RTC state.
3. It verifies that the clone has the same content hash and write count.
4. It runs one branch using the requested seed and controls.
5. It verifies that the caller-owned state was not mutated.
6. It verifies the reported before/after hashes and write counts against the returned state.
7. If `allow_write=False`, any state/hash/count change is an error.
8. A reported write must increment the count by exactly one; a no-write result must retain the
   complete state exactly.

The model path remains read-before-write: the subtask and action at policy step `t` read
`M_(t-1)`, and an allowed write creates the state consumed at `t+1`. The concrete adapter refuses a
model whose `memory_write_source` is not `post_attention`.

Matched branches keep the checkpoint, transformed observation, state snapshot, decoder settings,
flow-noise seed, and requested RTC prefix fixed unless that branch explicitly changes one of them.
Recorded observations are never replaced by counterfactual predicted observations or actions.

## 3. Offline episode manifest

The episode manifest is strict JSON with version `yam-eval-v1`. Duplicate keys and non-finite
numbers are rejected. A complete example is:

```json
{
  "version": "yam-eval-v1",
  "control_hz": 30.0,
  "action_side": {
    "axis_index": 0,
    "axis_name": "left/right task axis",
    "coordinate_frame": "yam_base",
    "window_start": 0,
    "window_end": 50,
    "aggregation": "endpoint_delta",
    "positive_side": "right",
    "threshold": 0.05
  },
  "episodes": [
    {
      "episode_id": "demo_left_001",
      "path": "/absolute/path/to/demo_left_001",
      "ground_truth_side": "left"
    },
    {
      "episode_id": "demo_right_001",
      "path": "/absolute/path/to/demo_right_001",
      "ground_truth_side": "right"
    }
  ]
}
```

### 3.1 Root fields

| Field | Required | Meaning |
|---|---:|---|
| `version` | yes | Must equal `yam-eval-v1`. |
| `control_hz` | yes | Explicit, finite, positive replay control frequency. There is no silent default. |
| `action_side` | no | Robot/task-owned rule for converting a trajectory into left/right/undetermined. |
| `episodes` | yes | Non-empty list of unique episode entries. |

Each episode entry requires a non-empty unique `episode_id`, a directory `path`, and a
`ground_truth_side` equal to `left` or `right`.

### 3.2 Files required in each episode directory

```text
top_camera_rgb.mp4
left_camera_rgb.mp4
right_camera_rgb.mp4
left_joint_positions.npy       # [T, 7], finite
right_joint_positions.npy      # [T, 7], finite
```

Left and right joint arrays must have equal lengths. The usable raw length is the minimum of that
state length and the three video lengths. Every annotation frame must be within this usable range.
Replay samples raw frames `0, effective_stride, 2*effective_stride, ...`; therefore a very large
cadence override can leave no sampled observation in a narrow decision window.

### 3.3 Action-side rule

The framework deliberately does not invent a robot-independent left/right threshold. If
`action_side` is omitted, action arrays are still saved, but all automatic action-side results are
`undetermined` and tests that require them can be marked invalid.

Required rule fields are `axis_index`, `axis_name`, `coordinate_frame`, `positive_side`, and a
non-negative `threshold`. Optional fields are:

- `window_start`, default `0`;
- `window_end`, default the configured model action horizon;
- `aggregation`, one of `endpoint`, `mean`, or `endpoint_delta`, default `endpoint`.

The window must satisfy `0 <= window_start < window_end <= action_horizon`. A metric greater than
`threshold` maps to `positive_side`; a metric less than `-threshold` maps to the other side; values
inside the band are `undetermined`. `endpoint_delta` subtracts the matching raw robot-state axis,
whereas `endpoint` and `mean` operate directly on output actions.

The values in the JSON example are illustrative only. They must be reviewed for the YAM coordinate
frame and task before using action-side pass/fail labels.

## 4. Annotation file

Annotations use strict JSON version `manual-v1`:

```json
{
  "version": "manual-v1",
  "episodes": {
    "demo_left_001": {
      "reveal_frame": 210,
      "close_frame": 330,
      "decision_start_frame": 500,
      "decision_end_frame": 720
    },
    "demo_right_001": {
      "reveal_frame": 205,
      "close_frame": 325,
      "decision_start_frame": 495,
      "decision_end_frame": 700
    }
  }
}
```

Every episode in the manifest must have an annotation. `reveal_frame` and `close_frame` are
required. Decision bounds are optional, but oracle/state-swap require at least one sampled decision
observation, and freeze aggregation requires decision observations for every variant.

The validated ordering is:

```text
0 <= reveal_frame <= close_frame
close_frame <= decision_start_frame              # when provided
decision_start_frame <= decision_end_frame       # when end is provided
```

Frames are raw recording indices, not policy-step indices. The replay loader assigns phases as:

- `pre_reveal` before `reveal_frame`;
- `visible` from reveal up to, but not including, `close_frame`;
- `decision` inside the optional inclusive decision interval;
- `post_close` after close and before decision;
- `post_decision` after `decision_end_frame`.

Freeze alignment uses scheduled sampled writes rather than assuming an annotation itself lies on a
write. The reveal-aligned write is the first scheduled frame in the inclusive
`[reveal_frame, close_frame]` interval. The close-aligned write is the last scheduled frame at or
before `close_frame`. If there is no reveal-containing scheduled write, freeze and temporal reports
are explicitly invalid.

## 5. Canonical semantic scoring

The diagnostic does not compare one tokenizer-dependent token. It teacher-forces each complete
canonical training string, including its newline terminator:

```text
open left bin\n
open right bin\n
```

For each exact sequence it records total autoregressive log probability and token count. It also
records mean log probability to make differing token counts visible:

Let `x_L` and `x_R` denote the two exact newline-terminated strings above. Then:

\[
s_L = \log p(x_L\mid o_t,M),
\qquad
s_R = \log p(x_R\mid o_t,M),
\]

\[
\Delta_{LR}=s_L-s_R.
\]

Positive `delta_lr` favors left and negative favors right. The length-normalized counterpart is
`left_mean_logp - right_mean_logp`. For a labeled episode:

\[
\text{signed_margin} =
\begin{cases}
\Delta_{LR}, & \text{left ground truth},\\
-\Delta_{LR}, & \text{right ground truth}.
\end{cases}
\]

A positive signed margin favors the correct side. The free-decoded subtask is recorded separately;
a continuous score can move without crossing the greedy decode boundary.

## 6. Complete fast-state snapshots

A π0 diagnostic state contains more than the fast MLP weights. The complete reproducibility state
is:

```text
Pi0FastState
  memory_state.fast_weights       all batched float32 MLP leaves
  memory_state.momentum           matching Titans momentum/past-surprise leaves
  writes                          committed write count
  last_write_raw_frame            optional cadence metadata
  last_write_wall_time_s          optional cadence metadata
  state_hash                      SHA-256 of all fields above
```

The lower-level memory digest canonicalizes sorted leaf names, dtype, shape, and exact bytes for
both `fast_weights` and `momentum`. The adapter's complete hash then includes the write counter and
last-write metadata. Counterfactual state cloning covers every one of these fields.

Persisted state files are compressed, pickle-free NPZ archives. Their embedded JSON manifest uses
schema `openpi.fast_memory_state`, version `1`, and records:

- every leaf's group/name/NPZ key/dtype/shape;
- state hash and envelope hash;
- write count;
- adapter metadata containing complete-state hash and last-write frame/time.

Loading validates schema, exact key set, float32 dtypes, matching fast/momentum shapes, hashes, and
metadata before returning an independent NumPy or JAX-backed state. A snapshot file should be
treated as integrity checked, not as proof of which training recipe produced it.

## 7. Test 1: oracle subtask

Question: if the semantic choice is supplied, do left and right text conditions produce distinct,
correct actions?

For each episode, the current application selects the first sampled observation in the annotated
decision phase, reconstructs the complete state immediately before it, then evaluates:

- forced `open left bin`, no write;
- forced `open right bin`, no write.

Both branches use the same observation, state, and seed. Predicted pre-RTC and committed action
arrays are saved. With a valid `action_side` rule, the report can identify:

- `action_conditioning_failure` when forced left/right do not yield left/right pre-RTC actions;
- `rtc_broker_failure` only when an adapter actually evaluated broker/RTC output and it collapses;
- `pass` when both pre- and post-RTC paths separate correctly;
- `invalid_or_undetermined` when a required task-side metric is unavailable.

For the current offline π0 adapter, broker evaluation is unavailable and committed equals pre-RTC.
The strongest current success label is therefore `action_conditioning_pass_rtc_not_evaluated`, not
a full RTC/broker pass.

## 8. Test 2: counterfactual complete-state swap

Question: holding the current observation fixed, does replacing only episodic state causally change
the semantic margin, decoded subtask, or action?

The application chooses one left and one right episode whose pre-decision write counts are as close
as possible. It evaluates both fixed decision observations under:

- `left_state`;
- `right_state`;
- learned reset state `m0`;
- `zero_read` using the reset-state branch while preserving model layout.

This is the implemented 2×2 observation/state comparison plus reset controls. The report records
the source episode, ground-truth side, phase, complete hash, and write count. For each observation:

\[
\text{swap_effect}=\Delta_{LR}(\text{left_state})-
                    \Delta_{LR}(\text{right_state}).
\]

An exact write-count match is required before the generic runner can emit `strong_causal_pass` or
`partial_pass`; otherwise it reports `write_age_confounded` even when the raw margins move. Other
interpretations include `policy_ignores_memory`, `decoder_uses_memory_actions_ignore_it`,
`learned_right_prior`, and `mixed_or_inconclusive`.

The selected pre-decision left/right states and learned `m0` are persisted under `states/`.

## 9. Test 3: freeze after reveal/close

Question: is useful side information written and then overwritten or diluted by later writes?

Each episode is replayed three times over the same recorded observations and derived per-step seeds:

- `normal`: every scheduled write is enabled;
- `freeze_after_reveal`: allow writes through the first reveal-aligned write, then freeze;
- `freeze_after_close`: allow writes through the last scheduled write at or before close, then
  freeze.

Every branch starts from an isolated clone of the same episode reset state. No predicted action can
change a later observation. The runner summarizes signed margins over the annotated decision
window and reports interpretations such as `overwrite_or_interference`, `late_integration`,
`never_stored_or_used`, `no_observed_interference`, or `mixed_or_inconclusive`.

If cadence and annotations yield no valid reveal write, or if every variant lacks a decision
observation, the report remains in the artifacts but is marked invalid with its reason.

## 10. Test 4: temporal diagnostics

Question: when does side evidence appear, drift, or disappear relative to writes?

At every sampled policy step, the runner logs the prediction made from the incoming state and then,
when due, the resulting post-prediction write. Recorded fields include:

- episode/observation ID, raw frame, policy step, phase, wall time, and seed;
- both canonical scores, raw and length-normalized margins, signed margin, and decoded subtask;
- pre-RTC and committed actions plus configured action-side results;
- write due/done, count before/after, complete hash before/after, configured cadence, and actual
  write interval;
- retrieval norm, content-gate norm, associative surprise, inner gradient norm, and
  `theta`/`eta`/`alpha`;
- action horizon, control frequency, latency, RTC metadata, and warnings.

The first policy prediction that can use a reveal-aligned write occurs strictly after that write's
frame because the model is read-before-write. Per-episode interpretations include
`learned_then_lost`, `never_learned`, `post_close_right_favoring`, and `mixed_or_inconclusive`.
The cross-episode summary only calls a right-attractor pattern when valid temporal reports cover
both left- and right-banana episodes.

## 11. Command line

Run commands from the `openpi` repository directory. The checkpoint must be a local checkpoint-step
directory containing `params/` and the matching normalization assets under `assets/`.

### 11.1 Full offline replay

```bash
cd /iris/u/kewalk/memory_project/openpi

.venv/bin/python scripts/v31_memory_diagnostics.py offline \
  --checkpoint checkpoints/pi05_yam_mem_v31/EXPERIMENT/2500 \
  --config pi05_yam_mem_v31 \
  --adapter openpi.diagnostics.v31_pi0:create_application \
  --episodes /absolute/path/to/episode_manifest.json \
  --annotations /absolute/path/to/annotations.json \
  --tests oracle,state_swap,freeze,temporal \
  --seed 0 \
  --diagnostic-level basic \
  --output-dir diagnostics/v31_ckpt2500_seed0
```

The output directory must not already exist. A full replay is expensive: every diagnostic event
scores two complete canonical strings in addition to the free or forced action prediction.

### 11.2 Cadence override

Normally omit cadence and horizon overrides so the loaded config is authoritative. To deliberately
test a different raw-frame cadence:

```bash
.venv/bin/python scripts/v31_memory_diagnostics.py offline \
  --checkpoint checkpoints/pi05_yam_mem_v31/EXPERIMENT/2500 \
  --config pi05_yam_mem_v31 \
  --adapter openpi.diagnostics.v31_pi0:create_application \
  --episodes /absolute/path/to/episode_manifest.json \
  --annotations /absolute/path/to/annotations.json \
  --tests temporal \
  --write-every-frames 20 \
  --output-dir diagnostics/v31_stride20
```

The run manifest records both configured and effective cadence and emits a warning when they differ.
`--action-horizon` is only a provenance/check flag in the current fixed-shape model adapter; a
mismatch warns, and the checkpoint model's horizon remains unchanged.

### 11.3 Realtime is not currently supported

The generic parser exposes a future `realtime` contract, but this command with the current adapter
will intentionally fail before any hardware access:

```bash
.venv/bin/python scripts/v31_memory_diagnostics.py realtime \
  --checkpoint checkpoints/pi05_yam_mem_v31/EXPERIMENT/2500 \
  --config pi05_yam_mem_v31 \
  --adapter openpi.diagnostics.v31_pi0:create_application \
  --robot-config /path/to/robot_config.json \
  --tests temporal \
  --output-dir diagnostics/not_created
```

The current error is `NotImplementedError`: the concrete π0 adapter is offline-only and requests a
separately reviewed shadow-stream adapter. Do not interpret the presence of realtime/active CLI
flags in the generic framework as hardware support. There is no current diagnostic command that
actuates the robot.

## 12. Artifacts

Every successful run creates a new self-contained directory:

```text
RUN_DIR/
  run_manifest.json
  events.jsonl
  summary.json
  oracle_results.csv
  state_swap_results.csv
  freeze_results.csv
  temporal_results.csv
  actions/*.npz
  states/*.npz
  token_diagnostics/*.npz
  plots/margin_over_time.png
  plots/margin_vs_write_count.png
  plots/state_swap_effect.png
```

All four CSV files and all four subdirectories are created even when a test produces no rows. An
invalid report with no event records appears in `summary.json`, not necessarily as a CSV row.

`run_manifest.json` records code revision (including a `-dirty` suffix), params-only checkpoint
hash, complete runtime-selected training config, CLI overrides, input paths and SHA-256 hashes,
versions, seed, configured/effective cadence, action horizon, control rate, RTC availability,
enabled tests, diagnostic level, hardware/counterfactual mode, and safety flags/warnings.

`events.jsonl` is the lossless scalar event stream. Dense pre-/post-RTC actions are compressed in
`actions/*.npz` and referenced by relative path. `states/*.npz` stores integrity-checked complete
snapshots used by oracle/state-swap runs. CSV files provide a flat view for analysis. `summary.json`
retains valid/invalid report and record counts, every invalid reason, interpretation counts, and
cross-episode temporal analysis. Plots are generated from the same recorded events.

## 13. `diagnostic_level=tokens` status

The generic schema and artifact writer support `DiagnosticResult.token_diagnostics` and compressed
`token_diagnostics/*.npz` files. The CLI also accepts:

```text
--diagnostic-level tokens
```

However, the current `Pi0SnapshotEvaluator` does **not** calculate or populate per-token diagnostic
arrays. In particular, it does not yet export:

- per-token associative errors;
- per-token write-gradient norms;
- 16×16 token maps;
- writer-token cosine-similarity matrices or summaries;
- effective-rank/SVD diagnostics.

At present, selecting `tokens` with the concrete Pi0 adapter fails immediately, before checkpoint or
model loading, rather than silently producing an empty directory. Use `basic` for current experiments
unless implementing and validating the token calculations in a separate change.

## 14. Checkpoint and config provenance warning

OpenPI checkpoints persist parameter arrays, but the parameter tree does not encode every static
Python config choice that determines semantics. In particular, shape-compatible checkpoints do not
prove:

- whether their memory writer was trained as v3 `raw_hidden` or v3.1 `post_attention`;
- whether their training cadence was 50, 25, 10, or another frame interval;
- whether RTC delay simulation was enabled and with what range;
- whether the old weighted probe objective was active;
- which sequence buckets, TBPTT block, sampling recipe, or base initialization produced them.

For that reason the CLI requires an explicit adapter and exposes `--config`; the π0 adapter rejects
anything whose selected runtime config is not `post_attention`. This prevents an accidental raw
writer at evaluation time, but it cannot prove that the checkpoint was *trained* with that writer.
The manifest faithfully records the selected current config and params hash. It also records
`checkpoint_static_config_provenance_verified`. Legacy checkpoints whose Orbax
`custom_metadata` is empty remain usable, but that field is `false` and a warning states that the
training-time writer/config is unverified. Future checkpoints may provide both `config_name` and
`memory_write_source`; the adapter rejects partial or contradictory metadata. This validation is
still not a substitute for provenance of cadence, objective, data, and code revision.

For older checkpoints, recover provenance from the original launch command, immutable config/code
revision, job logs, and experiment metadata before interpreting results. Never relabel an old
probe-trained checkpoint as a clean no-probe model merely because it loads under today's
`memory_probe_weight=0.0` config. A clean no-probe ablation must start from the same `pi05_base`
initialization and train under the new objective.

## 15. Safety and interpretation rules

- Offline mode is always shadow-only. `--execute-actions` and active counterfactual mode are rejected.
- State swap, freeze, and temporal counterfactuals are shadow-only even in the generic realtime
  contract.
- The generic contract would permit only one active oracle side at a time, only in realtime active
  mode, and only after an exact operator-confirmation interlock. The current π0 adapter implements
  none of that actuation path and rejects realtime first.
- No counterfactual branch may control later offline observations; all replays use recorded frames.
- No action-side pass/fail claim is valid without a reviewed robot/task `action_side` rule.
- No RTC/broker pass claim is valid when `rtc_metadata.broker_simulated` and `broker_available` are
  both false. This is the current offline condition.
- A state-swap result with unequal left/right write counts is age-confounded and is reported as such.
- Invalid trials remain visible in `summary.json`; do not remove them from denominators without an
  explicit analysis rule.
- Existing probe-trained checkpoints can be diagnosed, but those runs locate failure mechanisms;
  they are not evidence for the clean no-probe training ablation.

## 16. Recommended first run

Before a large replay, use one left and one right episode, correct manual annotations, and no cadence
override. Confirm that:

1. both episodes contain a sampled decision frame;
2. the action-side rule is correct in robot coordinates, or intentionally omit it and accept
   `undetermined` action results;
3. `run_manifest.json` names `pi05_yam_mem_v31`, `post_attention`, configured/effective stride 10,
   and the intended checkpoint hash;
4. state hashes do not change on oracle and state-swap read-only branches;
5. temporal writes increment exactly once per sampled frame;
6. invalid reports and warnings are understood before scaling to all episodes.

Only after those checks should results be used to distinguish semantic memory failure, overwrite,
subtask-to-action conditioning failure, a right prior, or an RTC/broker issue.
