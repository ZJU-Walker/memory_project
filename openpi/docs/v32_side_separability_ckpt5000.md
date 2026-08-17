# v3.2 side-separability analysis — write vs retrieved tokens (ckpt 5000)

**Question.** The zero-read counterfactual showed the policy barely uses memory. That leaves
four candidate failure localizations along the pipeline

```
write z_t  →  fast weights M_t  →  retrieved m_t  →  gated injection → policy
```

1. the write never captures side → nothing downstream can have it;
2. good write → memory loses it;
3. good write → memory stores it → read fails;
4. good write → good storage → good read → gate/policy ignores it.

This analysis compares the stored per-frame **write tokens** `z_t` [16, 2048] and
**pre-gate retrieved tokens** `m_t` [16, 2048] between one left-banana episode
(`episode_000000`) and one right-banana episode (`episode_000002`), at matched phases, and
asks whether side can be decoded at each stage.

**Verdict: the chain fails at the first box — the write is side-blind.** During the reveal,
while the banana is visibly on opposite sides, `z_left` and `z_right` are ≈99 % cosine-similar
and no more different than consecutive frames of the *same* episode. Everything downstream is
therefore starved: the retrieved tokens carry only a ~10⁻³-cosine episode fingerprint, which
matches the previously observed ~1-nat logit nudge at the decision frame and is then crushed
by the 0.045-norm injection gate.

---

## 1. Data and method

- Source: `diagnostic_outputs/v32_ckpt5000/episode_00000{0,2}/arrays.npz` from the ckpt-5000
  diagnostics replay (stride 15, write committed each step; `retrieved` is saved **before**
  the `memory_gate` multiply, so this probes the memory+read pathway independent of the gate).
- The two episodes are phase-locked: GT subtask flips `observe bins → open {left,right} bin`
  at replay step 33 (raw frame 495) in both. Phase windows (verified against the stored
  224×224 top-camera inputs; transition steps with lids/arms in motion are excluded):

  | phase | steps | scene |
  |---|---|---|
  | pre_reveal | 0–8 | bins closed, arms mostly at rest |
  | reveal | 16–26 | bins open, **banana visible (left vs right)** |
  | post_closure | 30–32 | bins closed again, still "observe bins" |
  | decision_closed | 33–40 | GT subtask names the side; bins still closed |
  | execution_open | 44–56 | robot has re-opened the chosen bin |

- Tokens are mean-pooled over the 16 slots (lossless here: mean pairwise slot cosine is
  0.998–1.000 in every phase — total slot collapse, and per-slot cross-episode distances are
  uniform to ≤4×10⁻³, so no individual slot carries hidden signal).
- Metrics per phase and stage:
  - **cross-episode distance**: cosine distance between the two episodes at matched steps;
  - **within-episode jitter**: mean pairwise cosine distance among the phase's frames inside
    one episode (the null scale — a real side signal must exceed it);
  - **separation ratio** = cross / within;
  - **decode accuracy**: nearest-centroid left-vs-right, fit on even / eval on odd frames
    within the phase (`decode_within`), and fit on reveal / eval elsewhere
    (`decode_from_reveal` — the *persistence* test); chance = 0.5.
- Code: `src/openpi/diagnostics/v32_side_separability.py` (unit-tested on synthetic
  trajectories, `..._test.py`); outputs in
  `diagnostic_outputs/v32_ckpt5000/side_separability/`.

## 2. Results

### Write tokens `z_t`

| phase | cross | within | ratio | decode_within | decode_from_reveal |
|---|---|---|---|---|---|
| pre_reveal | 1.9e-2 | 2.4e-1 | 0.08 | 0.62 | 0.50 |
| **reveal** | **1.0e-2** | **9.4e-3** | **1.08** | **0.60** | **0.60** |
| post_closure | 1.6e-2 | 1.6e-2 | 1.06 | 0.50 | 0.50 |
| decision_closed | 8.1e-3 | 6.5e-3 | 1.26 | 0.75 | 0.44 |
| execution_open | 7.2e-2 | 6.6e-2 | 1.09 | 0.75 | 0.50 |

At the reveal — the one window where a side-selective writer *must* differ between the
episodes — the cross-episode distance (0.010) equals within-episode frame-to-frame jitter
(0.009) and decoding is at chance. The reveal-fitted left-minus-right axis does not transfer
to any later phase (0.44–0.50). For calibration: a mere arm-timing mismatch at step 5
produces a *larger* excursion (0.15) than the banana's side ever does, and only in the last
~4 steps of execution (persistently different scenes: robot inside the left vs right bin)
does the distance finally rise to 0.28. The writer tracks coarse global scene state; the
small, briefly visible object that decides the task moves it by ~1 % — consistent with the
uniform 256-patch write attention averaging the banana away.

The (pre_reveal within = 0.24) outlier is the memory/feature transient of the first few
steps, not signal.

### Retrieved tokens `m_t` (pre-gate)

| phase | cross | within | ratio | decode_within | decode_from_reveal | ‖m‖ (L/R) |
|---|---|---|---|---|---|---|
| pre_reveal | 5.4e-4 | 7.6e-1 | 0.00 | 0.50 | 0.56 | 25 / 31 |
| reveal | 1.1e-3 | 7.9e-4 | 1.34 | 1.00 | 1.00 | 10 / 12 |
| post_closure | 2.6e-4 | 3.9e-4 | 0.68 | 1.00 | 0.50 | 6.4 / 7.6 |
| decision_closed | 1.6e-3 | 6.8e-4 | 2.33 | 0.88 | 0.69 | 5.3 / 6.5 |
| execution_open | 1.1e-2 | 1.6e-3 | 6.99 | 1.00 | 0.62 | 3.2 / 4.9 |

Two observations:

1. **The read-out is nearly a constant.** Both episodes' retrieved tokens follow the *same*
   slow trajectory (projection plot: the two curves move in lockstep from −40 to +8), and its
   norm decays monotonically over the episode (≈25 → 10 → 7 → 6 → 4). The memory read
   converges toward an episode-independent attractor and fades.
2. **A microscopic but consistent episode fingerprint exists.** Cross-episode distance
   1.6e-3 at decision (ratio 2.3, decode_within 0.88) — the two episodes are 99.84 %
   cosine-similar yet stably distinguishable. This hairline offset is the natural source of
   the ~1-nat subtask-logit nudge previously measured at the decision frame. After the
   0.045-norm gate it is functionally irrelevant (injected magnitude ≪1 % of prefix-token
   scale). Note `decode_from_reveal` stays weak (0.50–0.69): the fingerprint is generic
   episode identity, not a persisting reveal-side code.

### Figures

- `separation_over_time.png` — cross-episode distance per step vs within-episode jitter,
  phase bands shaded (reveal = yellow, closure = grey, decision = blue).
- `lr_projection.png` — both episodes projected on the reveal-fitted L−R axis; a working
  side memory would separate the curves from the reveal onward; instead they overlap (z) or
  run in lockstep with a hairline offset (m).
- `pca_trajectories.png` — 2-D PCA trajectories per stage.

## 3. Interpretation for the boxed chain

```
side-blind write  ─X→  (memory)  ─→  (read)  ─→  (gate)
```

- **Box 1 fails.** Side information never enters the pipeline: z at reveal is
  indistinguishable across sides at the level of the writer's own temporal jitter. The v3.2
  writer has not failed to *store or transmit* — it has failed to *encode*.
- Boxes 2–4 are therefore untestable for side content from this data (nothing to lose,
  store, or read), but the constant-attractor + norm-decay behaviour of `m` and the strangled
  gate indicate they would each add further attenuation even if the write were fixed.
- This sharpens the earlier findings rather than contradicting them: uniform write attention
  (entropy ≈ 0.99, padding-band sinks) is the *mechanism*; side-blind write tokens are the
  *consequence* measured at the representation level.

## 4. Caveats

- **n = 1 episode per side**, so "side" is confounded with episode identity; that cuts the
  charitable way here — even episode *identity* is barely decodable from z at reveal, and a
  confound would inflate, not hide, separability. The phase-resolved profile (no separation
  pre-reveal *or* at reveal) is the meaningful signature.
- Replay stride 15 might skip the single most informative reveal frames, but 11 reveal steps
  with the bins fully open are included.
- Distances are on slot-mean tokens in fp16→fp32; slot collapse (0.998+ uniformity) makes
  this essentially exact.

## 5. Natural next steps (not yet run)

1. **Ceiling check at the writer's input:** capture `h8_top` for these frames and run the
   same analysis — confirms the side signal exists at layer 8 (it should: the images differ
   visibly) and pins the loss to the write compressor specifically.
2. Repeat at ckpt 6000+ to see whether write-stage separability improves with training.
3. If the writer input is confirmed separable: interventions target the write path (query
   diversity / attention selectivity / slot decorrelation), not the gate alone — an open gate
   injecting a side-blind read-out cannot solve the task.
