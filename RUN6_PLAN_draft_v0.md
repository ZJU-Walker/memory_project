# RUN6 PLAN — v3.5 draft v0: installed side percept + supervised memory chain

**Date:** 2026-08-30 · **Base:** run5 / v3.4.1 ckpt 15750 (`reports/v34_summary_20260830/summary.md`) · **Status:** draft for review (web-chat) — not yet an implementation spec

---

## 0. Scope in one paragraph

Run5 solved stability and produced an in-domain, collection-keyed side code in the writer, but the memory chain (commit → retain → read → use) is still undemonstrated and the aux objective is shortcut-satisfiable. Run6 changes the question from *"does memory emerge under demand?"* to *"does the chain work when each link is supervised and the leaks are closed?"* Two deliberate scoping choices, to be stated in any writeup: (1) the evidence-time side percept is **installed** by direct, non-detached supervision rather than left to emerge; (2) cross-collection generality is a **secondary** metric (June-30 stays eval-only). The headline success criterion is an in-domain, causally verified memory chain on held-out episodes.

---

## 1. Diagnosis run6 is built on

| link | run5 evidence | run6 lever |
|---|---|---|
| write content | OOF side probe 0.98 in-domain, but collection-keyed; no loss ever demanded side at evidence time (emergent/incidental) | supervise directly: `L_write` on evidence-frame values + augmentation |
| commit | per-write state delta ≤0.011; M(k)→v never verified; θ≈0.10 on unit-L2 hidden ⇒ each write moves M(k) ~10% toward v; blank init ⇒ memory near-empty all episode | exact delta-rule commit on the output layer (§5) |
| retain | α≈0.01 unverified at late ckpts; three-way retention control never run | rung 3 (§6) |
| read | retrieval RMS ≈ τ; no evidence the decision-time query lands on the evidence key | supervise the read: `L_read` at leak-free decision frames; query–key alignment as escalation |
| use | causal eval failed at 2500 (writer weak then); never rerun | rerun at 15750 as baseline (P0); run6 headline gate (rung 5) |
| demand | 7-way aux satisfiable by same-frame write; waiting frames leak side via image (image-only swap +17.5); 17.1% endpoints-in-motion | static-wait trim (fix 1, built) + write-ineligibility window (fix 2, build) + leak-audit **launch gate** (P1) |
| gradient reach | if TBPTT 25 < occlusion ≤40 memory steps, the reader's gradient reaches the evidence write in only a fraction of windows — the "writer stores what the reader demands" loop was structurally starved | `L_write` needs no long-range gradient; `L_read` only needs local gradient to `project_q`/read head. Measure the reach fraction anyway (P2) |

Two consequences worth internalizing:
- With writes ending at occlusion onset, **the last committed write is an evidence write**. Even under total key collapse, "memory = last write" carries the side. Key separation is *not* required for this task to pass; it becomes necessary only for multi-item memory later. Do not over-engineer keys in run6.
- With an installed percept, the chain no longer depends on end-to-end credit assignment through 40 memory steps. Each link can be checked in isolation, and a failure localizes.

---

## 2. Changes vs run5

| component | run5 (v3.4.1) | run6 (v3.5) |
|---|---|---|
| data | `bin_memory_0816_subtask`, 60 ep, raw waiting | trimmed (`run6_staticwait`), same 56/4 split (15/29/44/59 held out); fresh 0816-setup episodes → 2nd held-out set when collected; June-30 eval-only (reconvert lerobot to 5-task first) |
| augmentation | none | random-resized-crop (scale 0.85–1.0, translate ±8%) + color jitter (brightness/contrast/saturation 0.2, hue 0.02), all cameras, **one crop per sequence** (consistent framing for writes and reads). No horizontal flip on the main path (actions are robot-frame) |
| write eligibility | every eligible step, all phases | ineligible from occlusion onset through end of decision segment (label-derived at train time); deployable gate = predicted phase from the subtask head |
| commit | θ≈0.10 all layers, one inner step, inner-grad clip 1.0 | output layer: closed-form delta-rule commit (θ_out = 1.0, exact fit); hidden layers θ_hidden = 0 (default) or 0.05 (option B); b3 frozen at 0 |
| retain (α) | 0.01 | unchanged — verify (rung 3) |
| momentum | eta = 0 | unchanged |
| episode init | blank w3/b3 | unchanged |
| read injection | tanh/RMS pin, c = 10.86, τ = 0.0087 | path unchanged; **recalibrate c** after the commit change (read magnitudes grow ~10×) — P4 |
| objectives | flow + subtask CE + 7-way aux (0.1) + detached probes | flow + subtask CE + **`L_write` (0.3)** + **`L_read` (0.3)** + detached probes; **7-way aux dropped** |
| guards | inner write-grad clip 1.0 · state-cotangent 10 · K/V-cotangent 1.0 · memory pre-clip 5 · global 1.0 | unchanged, except the closed-form commit is not a gradient step and bypasses the inner clip (clip stays on hidden if θ_hidden > 0) |
| loader | buckets 14/27/40, TBPTT 25 | + filter: `L_read` only on decision frames whose window contains ≥1 committed evidence write |
| training | 15,750 steps | 10k target; ckpt every 1k; ladder at 1k / 2.5k / 5k / 7.5k / 10k; kill criteria (§7) |

---

## 3. Pre-flight — must pass before any H100 hours

Ordered. P0–P6 are compute-only on existing checkpoints/data; P7 needs the robot and is not a launch blocker.

- **P0 — Baselines on run5@15750** (needed for before/after): rung-2 own-key commit test; causal eval (extend the provenance binding to resume logs — extend, don't weaken); key-collapse check (cosine among committed keys across phases, and cosine of the hidden features h(k)).
- **P0b — Commit-regime diagnostic on run5@15750** (decides whether θ or the inner clip was the binding constraint; ~1 hour, same harness as the own-key test). Log per committed write, on the 4 held-outs: ‖v‖; ‖h‖ (confirm = 1); pre-write residual ‖r‖ = ‖v − M(k)‖; inner-gradient norm before clipping (= ‖r‖·‖h‖) and the clip ratio, plus the numeric threshold behind "severe"; post-write residual ‖r⁺‖; state delta ‖Δw3‖_F. Interpretation:

  | observation | regime | meaning |
  |---|---|---|
  | ‖v‖ ≲ 1, clip ratio ≈ 1, ‖r⁺‖/‖r‖ ≈ 0.90 | unclipped | pure θ under-step: each write closes 10% of the gap |
  | ‖v‖ ≫ 1, clip ratio ≈ ‖v‖, ‖Δw3‖_F ≈ 0.10 fixed | clipped | constant-speed writer: fixed 0.1-norm step toward a far target |
  | ‖r⁺‖/‖r‖ ≈ 0.80 | b3 also updating | bias absorbs half the commit, key-independently |

  Either regime confirms the §5 change (exact step = 1/‖h‖²); the clipped regime additionally means the inner clip must be exempted for the output-layer commit. Also confirm α is applied per memory step (not per frame) and that the values' normalization matches what §5 assumes. Sanity cross-check against v3.3: RMS-normalized h in 1024-d gives ‖h‖ ≈ 32, so a θ = 0.1 step overshot the exact-fit step by ~100× — consistent with the median ~500× clip ratio if ‖r‖ was ≈ 15.
- **P1 — Leak audit v2 on trimmed data, memory reset — LAUNCH GATE.** Decision-frame side decodability from image alone (fresh probe, episode-level OOF) **≤ 0.60**, and the native waiting-shortcut margin ≈ 0. If it fails, `L_read` is satisfiable without memory and the run is pointless: extend the trim, move the decision anchor earlier, or mask the lid region — then re-audit. *Leak measurement before aux loss.*
- **P2 — Window sanity.** Per episode: number of committed evidence writes (≥3, else densify writes inside eligible windows and/or anchor one write at the last visible frame); occlusion length onset→decision in memory steps (≤40, α budget); fraction of `L_read` frames whose TBPTT window reaches an evidence write (report; informs whether `L_read` can shape values at all).
- **P3 — Stage-0 synthetic battery with the new commit** (mandatory, as in V34_PLAN). Random unit keys/values; N writes; ≤40 decay steps; then read. Gates: commit residual ‖v − M⁺(k)‖/‖v‖ ≤ 0.1 immediately post-write; retention cos(M(k), v) ≥ 0.8 after 40 steps; read cos to the correct value ≥ 0.8 with 8 distractor writes at h-cos ≤ 0.3; interference curve vs key similarity. Run with all guards active.
- **P4 — Injection recalibration.** Measure raw-read RMS after realistic writes at 15750 with the new commit; set c so injected RMS matches the v3.4 pre-flight target; confirm RMS/τ sits in a sane tanh regime (≈1–3, not saturated).
- **P5 — Wiring unit tests.** `L_write` gradient reaches the writer/value projection and is exactly zero on ineligible frames; `L_read` gradient reaches `project_q` and (through the commit) the values; eligibility mask verified on 5 labeled episodes with frame grids (same style as `relabel_boundary_check2.png`).
- **P6 — June-30 reconvert** to 5-task labels; point `v34_banana0630_writer_probe_eval.py` at it (eval-only).
- **P7 (parallel, robot) — 8–12 fresh 0816-setup episodes**, sides randomized in collection order, both tasks; label with the auto-labeler + editor. Becomes the second held-out set.

Deliverable: `docs/run6_preflight.md` with P0–P6 numbers before launch.

---

## 4. Frame sets and objectives

Frame sets from subtask labels (7-phase, 0816):
- **E (evidence):** `inspect both bins` (+ `open both lids` frames after both lids are fully open, if the whiteness detector agrees). Side visible.
- **O (occluded):** occlusion onset (both lids closed) through end of `close both lids and reset arms`.
- **D (decision):** trimmed waiting frames + the first W frames of `open {side} bin` before the arm commits. W is set by P1 (image must be side-blind).

Write eligibility: allowed on E and pre-E frames; disallowed on O ∪ D. (Escalation variant: E-only writes.)

**`L_write`** — small linear head `g_w` on the value v_t (post-projection; also log on the raw write token) for t ∈ E: CE(g_w(v_t), side). Non-detached: gradient flows into the writer compressor and value projection, and into the backbone below layer 8 (recommended; the June erosion shows backbone features were what got traded away). Weight 0.3.

**`L_read`** — small linear head `g_r` on the raw read r_t = M_t(q_t), **pre-pin**, for t ∈ D: CE(g_r(r_t), side). Non-detached through `project_q` and the commit (cotangent clips still apply). Only on windows that contain ≥1 committed evidence write (loader filter) — otherwise it trains guessing. Weight 0.3. Under P1, the only way to satisfy it is to read the evidence write from memory. This is the genuine memory demand run5 never had.

Kept: action flow (1.0), subtask CE (run5 weight), detached online ladder heads (telemetry only).

Dropped: 7-way aux. Its leak-free part is subsumed by `L_read`; its leaky part is what we are removing.

Escalation only: **`L_align`** = 1 − cos(q_t, stopgrad(k_ev)) for t ∈ D, weight 0.1 — add only if rung 2 passes and rung 3/4 fail (i.e., the memory holds the value but the query misses it).

---

## 5. Memory-core changes (`write_kv` / `read_key`)

Notation: h(k) = last hidden of the fast MLP on key k (unit-L2 by the v3.4 normalization); M(k) = w3 h(k) + b3.

**Commit (closed form, output layer):**
```
w3 ← (1 − α) · w3                              # decay, as now
h  = hidden(k)                                 # unit-L2
r  = v − (w3 h + b3)                           # residual
w3 ← w3 + θ_out · r hᵀ / ‖h‖²                  # θ_out = 1.0 ⇒ M⁺(k) = v exactly
b3 : frozen at 0                               # an updating bias stores θ·r key-independently: every read returns it
                                               # regardless of query, so rung 4 could not tell a hit from a miss
```
- General form (‖h‖² in the denominator) so it stays exact if the normalization is elsewhere.
- Hidden layers: θ_hidden = 0 (default) → the fast MLP is a *meta-learned feature map + delta-rule associative memory*. Option B: θ_hidden = 0.05 with the existing inner clip.
- This is the same Titans update with the step size chosen as the exact-fit step for a linear output layer, θ* = 1/‖h‖²: v3.3 (RMS-normalized h, ‖h‖² ≈ 1024) overshot it ~100× → divergence, saturated clip, bloat; v3.4 (unit-L2 h) undershoots it 10× → deltas ≤0.011 by arithmetic (Δw3 = θ·r·hᵀ, unit-h entries ~0.03). P0b decides whether the inner clip (1.0 vs ‖r‖ = ‖v‖ at blank init) was binding on top of that.
- Implement as a direct assignment on w3 (or θ_out = 1/‖h‖² with the output layer excluded from the inner write-grad clip). If the exact step goes through the clipped path, an ‖r‖ > 1 update is truncated to norm 1 — better than 0.1, not exact.
- Self-limiting: re-committing an already-stored (k, v) has r ≈ 0 and does nothing, so no constant-speed bloat. Interference: a later write at k′ moves M(k) by (h·h′)·r′ — under θ_out = 1 a collapsed key *fully* overwrites, which is why this change is paired with the write-ineligibility window (§4): the last committed write must be an evidence write.
- Keep the commit inside autograd so `L_read` can shape v and h; cotangent clips (state 10, K/V 1.0) remain and matter more now that commits are large. Fallback if the severe-clip rate climbs: θ_out = 0.5.
- Read path unchanged; c recalibrated (P4).
- Deployable eligibility: gate = 1[predicted phase ∈ allowed set] from the subtask head. Teacher-forced with labels in training; predicted at eval. Log agreement with labels.

**Per-write telemetry (new, same quantities as P0b so before/after is one plot):** ‖v‖, ‖h‖, pre-/post-write residual ratio ‖r⁺‖/‖r‖ (expect ≈0), ‖Δw3‖_F, clip ratio (should read 1.0 on the output layer now), h-cos to the last evidence key, and at D frames: cos(r_t, v_ev) vs cos(r_t, mean of other committed values) — the read SNR.

---

## 6. Eval ladder and pre-registered gates

Every ladder checkpoint, on the 4 held-outs (+ fresh 0816 episodes when available), fresh M0, natural writes, raw + EMA params.

| rung | test | pass | kill / branch |
|---|---|---|---|
| 1 write content | OOF side probe on evidence writes (FIG1 protocol) + approach-frame phase control | ≥ 0.95 by 2.5k; control ≤ 0.6 | < 0.8 at 2.5k ⇒ wiring bug, stop |
| 2 commit | own-key test: ‖v − M(k)‖/‖v‖ pre- vs post-write on evidence keys | post ≤ 0.1 (pre ≥ 0.9) | fails at 1k ⇒ mechanism bug, stop (not more steps) |
| 3 retain | at decision time (≤40 steps later): cos(M(k_ev), v_ev); three-way retention control | cos ≥ 0.8; ordering Frozen < Dynamics-only < Normal on `L_read` accuracy | fails ⇒ α or interference; check key/h collapse before touching α |
| 4 read | `g_r` accuracy on held-out D frames: natural / memory reset at occlusion onset / donor swap; injection ablation | ≥ 0.9 / ≤ 0.6 / follows donor ≥ 0.8; zeroing the injection at D removes the side signal downstream | rung 2–3 pass but 4 fails ⇒ add `L_align` |
| 5 use (headline) | `v34_causal_memory_eval.py` token+action, raw+EMA, provenance-bound | donor swap flips predicted action side on ≥ 0.8 of held-out D frames; reset memory ⇒ ≤ 0.6; native ≥ 0.9 | rung 4 passes but 5 fails ⇒ injection scale/placement (c, layer) — not the memory |

Secondary (report, no gate): June-30 inspect-window transfer probe, fresh + online heads. Expectation: with augmentation + `L_write` it rises above chance; if not, run7 is the generality run (multi-collection data). Also: action loss and subtask CE within ±10% of run5 at matched steps.

---

## 7. Training config, monitoring, kill criteria

- 10k steps, batch 12, 4×H100 FSDP, seed 42, same buckets, EMA kept, ckpt every 1k.
- Loss weights: flow 1.0 · subtask CE (run5) · `L_write` 0.3 · `L_read` 0.3. No warmup on `L_write`; `L_read` from step 0 (option: enable at 500). Sweep `L_read` ∈ {0.1, 0.3, 1.0} only if rung 4 stalls with rungs 2–3 passing.
- Monitors: run5 set + commit residual, retention cos, read SNR, eligibility agreement, severe inner-clip rate.
- Alarms: severe-clip > 10% ⇒ θ_out → 0.5; NaN ⇒ stop; action-loss regression > 20% vs run5 ⇒ halve side-loss weights.
- Early checkpoints matter more than late ones here: rungs 1–2 should pass by 1k–2.5k or the run is misconfigured. Do not "give it more steps."

---

## 8. Decisions needed (recommendation in bold)

1. Commit mechanism: **delta-rule output commit, θ_out = 1** / learnable data-dependent θ (pre-registered v3.4.1 trigger) / multi-step unrolled K
2. θ_hidden: **0** / 0.05
3. Eligibility: **O ∪ D ineligible, writes free elsewhere** / E-only writes
4. `L_read` input: **raw read, pre-pin** / post-pin
5. `L_write` gradient into backbone: **yes** / writer-only
6. 7-way aux: **drop** / keep at 0.05 on eligible frames only
7. Augmentation: **per-sequence crop** / per-frame
8. Fresh 0816 episodes: **eval only** / split into train
9. b3: **frozen at 0** / delta-rule with θ_b = 0.1

---

## 9. What run6 can and cannot claim

- **Can:** with an installed evidence percept and leak-free decision frames, a Titans-style fast-weight memory commits, retains across ≤40 steps of occlusion, is read by a learned query, and causally steers the policy's action.
- **Cannot:** emergence of write content under demand (installed); end-to-end learned write timing (label-gated); cross-collection generality (secondary, eval-only).
- **Run7 (if run6 passes):** remove scaffolds one at a time — learned gate without labels, `L_write` weight → 0, multi-collection data. Each removal is its own ladder run.

---

## 10. Implementation checklist (Claude Code handoff)

- [ ] `run6_staticwait` → `run6` config: eligibility mask, augmentation, losses, commit flags, telemetry
- [ ] memory core: delta-rule path in `write_kv` behind a flag; `theta_out`, `theta_hidden`, `theta_b`; inner-clip bypass for the closed-form update
- [ ] loader: E/O/D frame sets from labels; eligibility; `L_read` window filter; per-sequence crop
- [ ] heads: `g_w`, `g_r`; loss wiring; grad-flow unit tests (P5)
- [ ] scripts: `v35_stage0_synthetic.py` (P3), `v35_commit_test.py` (rungs 2–3), `v35_read_eval.py` (rung 4), leak audit v2 (P1), extend `v34_causal_memory_eval.py` provenance (P0/rung 5), injection recalibration (P4)
- [ ] June-30 lerobot reconvert (P6)
- [ ] `docs/run6_preflight.md` with P0–P6 numbers, then launch
- [ ] cluster: `--gpus-per-task=4` and per-worker `CUDA_VISIBLE_DEVICES` (known `srun --overlap` GPU-0 pitfall)
