# v3.3 write-token probe: do the write tokens encode the target side?

**Checkpoint:** `checkpoints/pi05_yam_mem_v33/v33_run1/6500`
**Date:** 2026-08-19
**Artifact:** `diagnostic_outputs/v33_write_probe/6500/probe.json`
**Code:** `src/openpi/diagnostics/v33_write_token_probe.py`, `scripts/v33_write_token_probe.py`
**Schema:** `openpi.v33.write_token_probe.v1`

---

## 1. Why this test exists

The offline eval (`scripts/v33_offline_eval.py`, status report §6.4) established that the
**policy does not use the memory**: zeroing the read leaves waiting-phase side accuracy at 18/18,
a swapped opposite-side memory never flips a prediction (`follows_swap = 0.00`), and the read
moves the left-vs-right logit margin by 0.2–0.4 nats against margins of 15–22 nats.

That result is **ambiguous between two failures that need opposite fixes**:

| hypothesis | meaning | implied fix |
|---|---|---|
| **never written** | the side never entered the write tokens `W_t` | fix the writer / the task pressure |
| **written but never read** | `W_t` holds the side; retrieval or the gate drops it | fix the read path / the gate |

A behavioural ablation cannot separate these, because both produce an identical dead policy. So
this test looks *inside* the memory pipeline and asks the content question directly:

> Given the write tokens `W_t ∈ R^{16×d}` produced at a frame, can a linear readout recover the
> episode's target side?  `W_t → {left, right}`

Plus the instruction question:

> Re-encode the **same** visual frames under the counterfactual instruction. If the writer is
> instruction-conditioned, `W_t(banana)` and `W_t(box)` should encode different sides.

This is a **diagnostic-only probe**: no training loss, no gradient reaches the model. The probe is
fit *post hoc* on harvested tensors.

---

## 2. What was actually done

### 2.1 Harvest

Each of the **60 episodes** is replayed against checkpoint 6500 carrying **real memory state** —
every frame on the write grid commits exactly the `write_tokens` the model would write, so the
fast weights at frame *t* are the ones the model would actually have. Per sampled frame the runner
records three streams under **two instruction conditions** (true and counterfactual):

| stream | tensor | role |
|---|---|---|
| `write` | `write_tokens` `[16, 2048]` | **the question** — what the writer deposits |
| `read` | `retrieved` `[16, 2048]` | what the policy actually consumes (pre-gate) |
| `state` | raw proprioception | **the known leak** — the calibration baseline |

Frames are bucketed into four phases and averaged, giving **one vector per episode per phase per
stream**.

### 2.2 Sampling cadence (deliberately decoupled from the write cadence)

At the training stride of 15 the two phases that carry the scientific question are the thinnest:
evidence averages **3.5 frames/episode (min 2)** and waiting **4.2 (min 1)** — too few to average
over. The probe therefore samples every **5** frames while still **committing writes only on the
stride-15 grid**. Writing 3× more often than the model ever does would push the fast weights
off-distribution and make the retrieved tokens meaningless.

Realized frame counts (60 episodes):

| phase | mean | min | max |
|---|---|---|---|
| approach | 53.0 | 43 | 77 |
| evidence | **10.4** | 6 | 15 |
| retention | 42.5 | 32 | 60 |
| waiting | **12.8** | 3 | 78 |

### 2.3 The estimator — and why the split is the whole ballgame

**Leave-one-EPISODE-out**, enforced in code. Frames inside one episode share lighting, object
placement and arm pose, all of which correlate with side; a frame-level split lets the probe
memorize episode identity and report a meaningless ~1.0. Each episode contributes exactly one
training point per phase.

The classifier is a ridge-regularized logistic regression (dependency-free, deterministic, 400
full-batch GD steps, L2 = 1.0). Strong regularization matters: with n = 60 and d = 32,768 an
unregularized fit separates *any* labeling perfectly.

**Three controls accompany every probe**, because an AUC in isolation is uninterpretable:

| control | rules out |
|---|---|
| `shuffled` labels (20 permutations) | the null *this estimator at this sample size* produces |
| `read` stream | conflating "written" with "delivered to the policy" |
| `state` stream | the known proprioceptive leak (joint state alone ≈ 70%) |

**Decision rule stated in advance:** a write-token AUC counts as evidence of encoded memory only
if it clears **both** the shuffled null **and** the `state` baseline.

`approach` is the **internal negative control**: it precedes the reveal, so the side is not yet
observable by anyone. Signal there means the probe is reading a nuisance, not a memory.

### 2.4 Design integrity

Verified balanced, no warning raised:

```
balance {'right': 30, 'n': 60, 'majority_rate': 0.5}
cells:  {(banana, left): 15, (banana, right): 15, (box, left): 15, (box, right): 15}
```

The shuffled nulls came out at **AUC 0.44–0.52 mean** across all twelve (phase × stream) cells —
i.e. the estimator sits at chance when the labels carry no signal, which is the evidence that the
pipeline is not self-confirming.

> **A caught methodological failure, recorded because it nearly produced a fake result.** An early
> 8-episode smoke run drew its episodes as a prefix. The dataset is stored in cell order (15 per
> cell), so it got a **7 left / 1 right** design where "always guess left" scores 0.875. It duly
> reported evidence-phase **accuracy 0.88 / AUC 1.00** — which reads as a triumphant result and
> means nothing: the shuffled null *also* hit AUC 1.00, so no number cleared its own baseline. The
> episode cap is now stratified across the four cells, and `analyze()` warns whenever the
> majority-class rate exceeds 0.65. This is precisely what the controls are for.

---

## 3. Results

Pathway scalars at 6500: `conditioner_output_proj_norm = 5.399`, `memory_gate_norm = 0.0639`.
(The conditioner grew from 2.96 at ckpt 2750 — it is very much alive. The content gate has
stayed shut throughout.)

```
     phase   n  stream    acc    auc  null_acc  null_p95
  approach  60   write   0.83   0.92      0.48      0.63   <-- NEGATIVE CONTROL FIRES
  approach  60    read   0.63   0.64      0.49      0.67
  approach  60   state   0.93   0.94      0.48      0.57
  evidence  60   write   0.72   0.79      0.50      0.63
  evidence  60    read   0.63   0.64      0.51      0.94
  evidence  60   state   0.72   0.82      0.50      0.63
 retention  60   write   0.82   0.88      0.46      0.60
 retention  60    read   0.50   0.43      0.53      0.68
 retention  60   state   0.77   0.88      0.48      0.66
   waiting  60   write   0.90   0.95      0.51      0.61
   waiting  60    read   0.43   0.36      0.46      0.62   <-- READ AT/BELOW CHANCE
   waiting  60   state   0.70   0.73      0.47      0.66
```

Counterfactual-instruction transfer (fit on true-instruction tokens, score the same frames
re-encoded under the other instruction):

```
     phase  flip_rate  shift_toward_other  |shift|  |true_score|
  approach       0.03              +0.163    6.838        32.421
  evidence       0.25              +4.759   23.399        33.364
 retention       0.37              +9.020   72.502        76.218
   waiting       0.07              +2.260    4.815        18.804
```

---

## 4. Analysis

### 4.1 The write tokens are not encoding a memory

Write-token AUC clears the shuffled null in every phase (0.79–0.95 vs nulls ~0.60–0.63). Read
naively, that says "yes, the answer is written down." **It does not**, for two reasons that the
controls were built to expose.

**The negative control fires.** `approach` scores **AUC 0.92 — before the bins are ever opened.**
At that point the target side is not observable by any means; there is nothing legitimate to
encode. A probe that recovers the side from a pre-reveal frame is reading a nuisance correlate.
This single number invalidates the naive reading of the whole write column.

**The nuisance is identifiable.** The `state` row tracks the `write` row almost perfectly:

| phase | write AUC | state AUC |
|---|---|---|
| approach | 0.92 | 0.94 |
| evidence | 0.79 | **0.82** |
| retention | 0.88 | 0.88 |
| waiting | 0.95 | 0.73 |

To a good approximation the write tokens are **re-encoding proprioception**. This is the same
observation leak found in §6.4 (LOO logistic regression on waiting-phase joint state recovers the
side at 70%), now measured through a second, independent channel. Decisively: **evidence — the one
phase where a genuine memory should stand out — is the weakest write row (0.79) and does not beat
its own state baseline (0.82).**

Waiting is the one phase where write (0.95) clearly exceeds state (0.73). That gap is the only
candidate for real encoded content in the table, but it is not attributable to memory: by the
waiting phase the writer has seen the whole episode including the evidence, and the write tokens
are computed from the *current* frame's layer-8 features, which include the wrist cameras and the
full visual scene. The most economical explanation is a richer read of the same leaky observation,
not retrieval of a stored fact — and §4.2 shows the retrieval path is empty in exactly that phase.

### 4.2 "Written but never read" is ruled out

The `read` stream — the retrieved tokens, i.e. what the policy actually consumes — sits at
**AUC 0.43 in retention and 0.36 in waiting**, at or below chance, with negative class separation
(`sep = −500` and `−873`). Whatever the writer deposits, retrieval is not carrying side
information forward into the policy.

This is consistent with, and independently corroborated by, three prior measurements: the content
gate at 0.0639, `follows_swap = 0.00`, and the 0.2–0.4 nat margin shift. **Four independent
probes now agree.** So of the two hypotheses:

- **"never written"** — supported (no side content beyond the observation leak);
- **"written but never read"** — **not** supported.

### 4.3 The instruction perturbs the writer but does not steer an encoded side

`cf_flip_rate` peaks at **0.25 (evidence)** and **0.37 (retention)** but is **0.03 (approach)**
and **0.07 (waiting)**. The phase profile is right — the instruction matters most where evidence
is being observed and least before the reveal — so the v3.3 conditioner is doing *something*,
consistent with `out_proj` growing to 5.40.

But it is not flipping an encoded decision. In every phase `|shift| < |true_score|`
(e.g. evidence 23.4 vs 33.4; retention 72.5 vs 76.2), meaning the instruction moves the probe's
score without carrying it across the decision boundary in most episodes. If the writer were
encoding "the target side implied by *this* instruction," swapping the instruction on a frame
where banana and box imply different bins should flip the readout most of the time. It does not.

### 4.4 What this does *not* show

Stated explicitly to avoid overclaiming in the other direction:

- It does **not** prove the write tokens are empty of all memory content. This is a *linear* probe
  on *episode-mean* vectors; nonlinear structure, per-slot structure, or content that is present
  but not linearly decodable would be missed.
- It does **not** measure whether the architecture *could* store the side — only what this
  checkpoint, trained on this leaky dataset, actually does.
- The waiting-phase write/state gap (0.95 vs 0.73) is unexplained by the leak alone and deserves a
  follow-up before being dismissed.
- n = 60 episodes with LOO gives roughly ±0.06 resolution on AUC; differences smaller than that
  (e.g. retention write 0.88 vs state 0.88) should not be interpreted as meaningful.

---

## 5. Implication

The ambiguity that motivated this test is closed: **there is no evidence for a written-but-unread
memory, so repairing the read path or forcing the gate open is not the priority.** The blocking
problem remains the observation leak identified in §6.4 — the current frame carries the answer, so
remembering buys nothing and no gradient pressure ever requires it.

This *strengthens* status-report §7 rather than redirecting it. Ordered:

1. **Close the leak** (data/protocol, not modeling): neutralize or re-home the arm pose during the
   wait so proprioception is uninformative; mask the wrist cameras during the wait.
2. **Quantify the residual leak** with the same LOO probe on wrist-camera embeddings.
3. **Re-run this probe and the offline eval** on a retrained checkpoint. Pass criteria:
   - write-token evidence-phase AUC clears the `state` baseline **and** approach stays at chance;
   - `read` AUC rises above chance in waiting;
   - zero-read *drops* side accuracy and `follows_swap` climbs above ~0.5.
4. **Only then revisit the architecture.** The conditioning pathway is healthy and the §16 credit
   path is verified; there is still no evidence the mechanism is broken, only that it is
   unexercised.

---

## 6. Reproducing

```bash
# full run: 60 episodes, ~50 min on an idle A5000 (batch-1 inference, largely CPU-bound
# on inline-image parquet decode; an H200 does not speed this up proportionally)
uv run python scripts/v33_write_token_probe.py \
  --checkpoint checkpoints/pi05_yam_mem_v33/v33_run1/6500 \
  --dataset_root ~/.cache/huggingface/lerobot/yam/bin_memory_0816_subtask \
  --output_dir diagnostic_outputs/v33_write_probe/6500 \
  --sample_stride 5

# wrapper: stratified 8-episode smoke, then the full set
bash scripts/run_write_token_probe.sh 6500 5
```

Unit tests (21, no GPU): `uv run pytest src/openpi/diagnostics/v33_write_token_probe_test.py`.
They target the ways the estimator could lie — a leaky split, an AUC disagreeing with the
brute-force definition, a null that fails to flag a separable-by-construction design, and the
prefix-vs-stratified sampling bug described in §2.4.

**Note the output directory is refused if it already exists** — pick a fresh one per run rather
than deleting, so a completed result is never silently clobbered.
