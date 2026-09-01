# v3.5 Training Authorization

`pi05_yam_mem_v35` is fail-closed. Calibration alone does not permit an optimizer
update. A fresh run also requires the canonical pilot authorization at:

```text
v35/diagnostics/authorization/pilot.json
```

All paths recorded by the reducer are relative to `memory_project`. The JSON is
canonical and self-hashed; changing whitespace or any linked artifact invalidates it.
Its exact bytes are copied into every checkpoint.

## Pilot authorization

The reducer requires:

- a passing `openpi.v35.data-gate-decision.v1` Gate-A artifact whose
  `detail_report` descriptor authenticates a canonical
  `openpi.v35.data-gate-report.v1` per-episode distribution report with the same
  frozen manifest and dataset identity;
- the passing Gate-B decision from `v35_leakage_gate.py`;
- the completed-update-0 Gate-C/task-health rung accepted by
  `v35_pilot_gate.py`;
- the frozen manifest, train-storage seal, train-only norm assets, official-base
  initialization identity, and passing calibration;
- the exact semantic training-config hash printed before authorization by
  `scripts/v35_train.py --print-semantic-config-sha256`.

The semantic hash includes the full training recipe and experiment name. It excludes
only the frozen continuation target/schedule, resume mode, and authorization-path
fields, so the same run can move from 1,000 to 2,500 or 10,000 updates without hiding a
model, optimizer, data, or runtime change.

```bash
CONFIG_SHA256="$(python scripts/v35_train.py \
  --experiment-name "$EXP_NAME" \
  --calibration v35/diagnostics/runs/"$EXP_NAME"/calibration/calibration.json \
  --print-semantic-config-sha256)"

python scripts/v35_training_authorization.py pilot \
  --manifest data/0830_0831_episode_manifest_v36_frozen.json \
  --manifest-sha256 "$MANIFEST_SHA256" \
  --gate-a v35/diagnostics/gates/gate_a.json \
  --gate-b v35/diagnostics/gates/gate_b.json \
  --step0-rung v35/diagnostics/rungs/0.json \
  --norm-stats v35/assets/pi05_yam_0830_0831_v36/yam/bin_memory_0830_0831_v36_subtask/norm_stats.json \
  --norm-provenance v35/assets/pi05_yam_0830_0831_v36/yam/bin_memory_0830_0831_v36_subtask/norm_stats_provenance.json \
  --config-name pi05_yam_mem_v35 \
  --experiment-name "$EXP_NAME" \
  --initialization-seed 42 \
  --semantic-training-config-sha256 "$CONFIG_SHA256" \
  --output v35/diagnostics/authorization/pilot.json
```

## Continuation authorization

The 1,000-update checkpoint may continue to 2,500 only when the deterministic Gate-D
decision is `inconclusive`. It may continue to 10,000 only when Gate D is `pass` at
1,000 or 2,500. A 2,500 pass must also retain the prior one-time extension
authorization. The reducer re-loads every rung and recomputes Gate D rather than
trusting the decision label alone.

```bash
python scripts/v35_training_authorization.py continuation \
  --manifest data/0830_0831_episode_manifest_v36_frozen.json \
  --manifest-sha256 "$MANIFEST_SHA256" \
  --pilot-authorization v35/diagnostics/authorization/pilot.json \
  --gate-d-decision v35/diagnostics/gates/gate_d_1000.json \
  --rung-result v35/diagnostics/rungs/0.json \
  --rung-result v35/diagnostics/rungs/250.json \
  --rung-result v35/diagnostics/rungs/500.json \
  --rung-result v35/diagnostics/rungs/1000.json \
  --endpoint 1000 \
  --target 2500 \
  --output v35/diagnostics/authorization/extend_2500.json
```

Pass the resulting project-relative path to `scripts/v35_train.py` as
`--continuation-authorization`. At the evaluated source rung, training hashes
the restored raw parameter tree and requires an exact match to Gate D. A later
crash-resume (for example at 5,000) is allowed only if that checkpoint already contains
the byte-identical continuation authorization.
