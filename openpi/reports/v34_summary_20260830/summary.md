# v3.4.1 Titans-memory status — full summary for next-steps discussion

**Date:** 2026-08-30 · **Repo:** `/iris/u/kewalk/memory_project/openpi` · **Final model:** run5 checkpoint **15750** (training deliberately stopped 2026-08-29; user decision after the writer result landed)

Companion assets in this folder: `figs/` (11 figures) and `videos/` (6 videos). Every claim below names its artifact.

---

## 1. TL;DR

1. **Stability is solved.** After four failed/unstable runs, run5 (blank per-episode output init + eta=0 + four gradient guards) trained 15,750 steps with zero NaN/collapse. The recurrent-backward-explosion problem that killed runs 1–3 is closed.
2. **The writer works — in-domain.** Between steps ~5k and ~10k the model learned to encode "which bin holds the object" into its 16 write tokens: 0.98–1.00 balanced accuracy under episode-level cross-validation on the training collection, with a decisive phase control (same head at chance on pre-evidence frames of the same episodes). `figs/FIG1`, `figs/FIG2`.
3. **But the side signal is collection-specific.** On 30 never-trained June-30 episodes whose inspect frames look nearly identical to training evidence frames (`figs/train_vs_june_evidence.png`), every probe is at chance — even probes fit on the June episodes themselves. The "bins are open" gate generalizes; the "banana is left/right" channel does not. Leading interpretation: the side readout overfits the 0816 collection's exact appearance. `figs/FIG6`, `figs/FIG7`.
4. **The memory itself (commit → read → use) remains undemonstrated.** Per-write state changes are tiny (≤0.011), retrieval is ~tau-scale, and the pre-registered causal evaluations at ckpt 2500 failed all gates (not yet rerun at late checkpoints).
5. **The June-30 data is now relabeled** to the 5-phase training vocabulary with a verified vision-based auto-labeler + browser editor workflow, making it a first-class eval set going forward.

---

## 2. What was trained (system recap)

- **Base:** pi05 (fresh from official `pi05_base`), v3.2 layer-8 dual-query memory interface: top-camera layer-8 patches → task-conditioned writer compressor → **16 write tokens** → unit-norm keys (512-d) / values (2048-d) → per-sample fast-weight MLP `512→1024→1024→1024→2048` (Titans-style, frozen gates θ≈0.10, α≈0.01); reads injected via tanh/RMS pin (c=10.86, τ=0.0087).
- **v3.4.1 = run5 specifics:** blank per-episode output init (`w3/b3` exact zero each episode) + **eta_scale=0** (no carried momentum — the run4 replay isolated momentum as the instability amplifier).
- **Objectives:** action flow + subtask CE + **7-way memory aux CE, weight 0.1** (reads post-write state — satisfiable by a same-frame write; "fix 2" for this is designed but NOT implemented) + two *detached* online probe heads (writer / read ladder metrics; stop-gradient, never train the model).
- **Stability guards (all active):** inner write-grad clip 1.0 · state-cotangent clip 10 · K/V-cotangent clip 1.0 · memory-path pre-clip 5.0 · global clip 1.0. Cross-run comparison proved these do **not** slow learning (run1 with no guards had the same early writer-metric slope; aux hit 0.98 by step ~700 in every run).
- **Data:** `yam/bin_memory_0816_subtask` — 60 episodes (banana / grey-pepper-box × left/right), 7-phase labels, held-out episodes 15/29/44/59 (last of each cell). Sequence buckets 14/27/40, TBPTT 25, batch 12, 4×H100 FSDP, seed 42.
- **Final health at 15750:** loss 0.40, CE 0.027, aux CE 0.024 (≈solved), aux side acc 0.992, memory grad ~0.1, severe inner-clip ~2–3% (warnings only, stable tail).

## 3. Run history in one table

| run | change | outcome |
|---|---|---|
| 1 | baseline v3.4 | grad explosion/collapse limit cycle (~700-step period); killed ~2.8k |
| 2 | +state-cotangent clip, +memory pre-clip | cycle returned via K/V→VLM path; killed ~1.5k |
| 3 | +K/V-cotangent clip, +inner telemetry | learned steepness returned by ~1.3k |
| 4 | fresh base + blank episode output init | much more stable; one marginal severe window at 2400; momentum isolated via fixed-K/V replay |
| **5** | **run4 + eta=0 (= v3.4.1)** | **stable to 15,750; writer takeoff; stopped by choice** |

---

## 4. Results, strongest evidence first

### R1. In-domain writer takeoff (the positive result) — `figs/FIG1_writer_probe_takeoff.png`

Fresh logistic probes refit per checkpoint (episode-level OOF over the 56 non-held-out episodes; one mean feature per episode; evidence frames only; **fresh empty memory, zero writes**):

| ckpt | writer | key | value | state ctrl | shuffle null |
|---|---|---|---|---|---|
| 2500 | .64 | .70 | .64 | .77 | ~.45 |
| 5000 | .70 | .68 | .75 | .77 | ~.47 |
| 9500 | .89 | .86 | .93 | .77 | ~.46 |
| 14750 | **.98** | **.98** | **1.00** (AUC 1.0) | .77 | ~.50 |

- The signal reorganized between ~5k and ~10k and reaches the projected K/V, not just raw tokens.
- The online training metric (`ladder_writer_accuracy` 0.55→0.87, CE 0.69→0.30) tracks the same takeoff, so it was feature change, not just head learning.

### R2. The anti-cheat control — `figs/FIG2_evidence_vs_approach_14750.png`

Same head, same episodes, applied to a matched window immediately **before** the bins open: balanced accuracy **0.50** (paired evidence-minus-approach margin +8.8, 89% positive). A session/episode signature would decode phase-invariantly; this signal appears exactly when the evidence appears. This is the control every earlier "positive" in this project lacked.

### R3. Same-collection held-outs (n=4, weak but positive)

At 14750/15750, on episodes 15/29/44/59 (model never trained on them): checkpoint's own online head **4/4** native-prompt correct (small margins 0.10–1.51); fresh 56-episode head only 2/4 (ep15✓ ep29✓ ep44✗ ep59✗, unchanged 11000→15750). Instruction-switch on identical pixels flips only 1/4 predictions → the encoded variable behaves like scene layout ("banana side"), not prompt-bound target side (caveat: switched prompts are OOD).
Videos: `videos/TRAIN_heldout_ep15…` and `…ep29…` (watch the score commit during the evidence phase); `videos/TRAIN_heldout_ep29_writer_attention15750.mp4` shows where the writer looks.

### R4. June-30 transfer test (the negative result, now airtight)

Data: 30 banana-only episodes collected 2026-06-30 (16L/14R), never seen in training, **relabeled this session** to the training vocabulary (`open both lids / inspect both bins / close both lids and reset arms / open {side} bin`) with a vision-based auto-labeler (both-bins-fully-open whiteness detector; motion cues demonstrably fail) — verified frame grids: `figs/inspect_start_all30.png`, `figs/relabel_boundary_check2.png`.

Findings (probe protocol identical to R1; prompt "find the banana"; fresh M0, zero writes):

1. **Whole observe-window eval:** chance at all 9 checkpoints, both heads (`figs/FIG3`, `figs/FIG4`); mechanism = feature collapse, all 30 episodes score ≈ +8 (`figs/FIG5_transfer_collapse.png`).
2. **Inspect-window-only eval** (the fair version, stride 1): still chance everywhere (`figs/FIG6`, `figs/FIG7`). Notably the **AUC decays from 0.71 (ckpt 500) to ≤0.5 (15750)** — training traded away the pretrained base's small generic signal while specializing.
3. **Within-June probes** (fit on June episodes themselves, balanced pair-holdout): 0.40–0.50 → the side info is absent in **any** linear coordinates, not merely unreadable by the 0816 head.
4. **Feature geometry @15750 (inspect means):** June features are evidence-LIKE (cos 0.80 to training evidence vs 0.64 to approach) but flat along the learned L/R axis (gap −0.03 vs +0.55 in-domain; 4× less spread). **The evidence gate transfers; the side channel does not.**
5. **The scenes are nearly identical** (`figs/train_vs_june_evidence.png`) — same table, baskets, boards, viewpoint — so this is *not* a big domain gap. Leading interpretation: **the side readout is overfit to fine 0816-collection appearance statistics** (calibration/session detail), i.e., the user's "memorization" hypothesis in its precise form. What n=4 cannot settle: memorized-60-episodes vs real-but-collection-keyed percept → decided only by fresh recordings in the exact 0816 setup.
Videos: `videos/JUNE_*` (bars don't react when the inspect window turns green; legend in `figs/video_legend.png`).

### R5. The memory chain (commit → read → retain → use): open, and now the bottleneck

- Per-commit fast-state delta ≤ 0.011; final state max-abs 0.10 after 59 writes; retrieval RMS ≈ τ (0.011) — the memory barely moves.
- Pre-registered causal evals (ckpt 2500, raw+EMA, token+action): **all gates failed** — donor swaps don't steer the output; reset-memory shortcut solves most waiting frames (native margin +8.7 nats; image-only swap follows the donor +17.5).
- Aux is structurally satisfiable by a same-frame write (write happens before aux reads) — designed fix ("waiting write-ineligibility") **not implemented**; note pre-write-timing alone is insufficient (previous waiting frame's write carries the same info).
- Data leak fix 1 (static-waiting trim, endpoints-in-motion 17.1%→0.0%) is implemented and verified in the `run6_staticwait` config (data-only; never trained beyond smoke).
- Not yet run at late checkpoints: causal eval (its provenance check is pinned to the pilot log and must be *extended* for resume checkpoints, not weakened) and the rung-2 own-key commit test.

---

## 5. Evidence-quality cautions (carry these into any discussion)

- All in-domain writer numbers are **one collection, one scene**; "in-domain" must be said explicitly.
- The 4 held-outs give 0/4–4/4 resolution only; they sit at session boundaries (hardest split).
- ep59's pre-registered eval frame is itself in motion (known grid defect, kept for comparability).
- June inspect windows are short (33–67 frames) and the June "inspect" staging (arms place boards aside) differs behaviorally from 0816's lid-opening, even though the resulting view is similar.
- The online ladder metrics never train the model; aux side-acc ≈0.99 proves solvability of a shortcut-riddled objective, not memory use.
- ~2–3% severe inner-clip tail persisted to the end (warnings only; never corroborated by other metrics).

## 6. Candidate next steps (for the web-chat discussion)

Ordered by my assessment of information-per-effort:

1. **Collect ~8–12 fresh episodes in the exact 0816 setup** (same lids/staging/cameras; both tasks × both sides). Decides "collection-keyed percept vs memorized episodes" — the single question tonight's work cannot answer. Cheap; unblocks honest claims about the writer.
2. **Rung-2 own-key commit test at 15750** (pre- vs post-write ‖M(K)−V‖ on matched keys). The state-delta numbers suggest commit is the weak link; this is a small, decisive diagnostic on the existing checkpoint, runnable on the current allocation.
3. **Causal memory eval at 15750** (token+action, raw+EMA) after extending its provenance binding to resume logs. Failed at 2500 when the writer was still weak; the writer now works in-domain, so this is the first real test of the full chain.
4. **Design run6/7 objective changes** only after 2–3 localize the failure: fix-2 waiting **write-ineligibility** (not just pre-write aux timing), possibly an explicit evidence-frame writer objective (non-detached), plus the already-built static-wait data trim. Any training restart should also consider multi-collection data if writer generality is a goal.
5. **Do not resume run5** — 15750 is a clean, well-characterized endpoint; more steps change nothing structural (aux solved, writer plateaued in-domain, memory unused).

## 7. Artifact index (everything referenced above)

- Training/handoff docs: `docs/v3.4.1_claude_handoff.md` (evidence-corrected source of truth; §15 = leak audit + fix 1), `docs/v34_implementation.md`.
- Checkpoints: live root `checkpoints/pi05_yam_mem_v34_run5_eta0/v34_run5_eta0/{5000,15000,15750}` + provenance-bound copies in `diagnostic_checkpoints/v34_run5_eta0_{pilot,resume}_copies/` (500…14750). Raw params = `train_state/params`; standalone `params` = EMA.
- Probes: `scripts/v34_fixed_writer_probe_eval.py` (per-ckpt fresh probes; artifacts `diagnostic_outputs/v34_fixed_writer_probe/full_{17099350,17074121}/`), `scripts/v34_heldout_{fresh_writer_probe,writer_attention,evidence_instruction_probe}…_video.py` (+ dual-head), `scripts/v34_banana0630_writer_probe_eval.py` (observe/inspect/full modes; outputs `diagnostic_outputs/v34_banana0630_writer_probe{,_inspect,_full15750}/`).
- Causal/shortcut: `scripts/v34_causal_memory_eval.py` (+ completed 2500 reports `diagnostic_outputs/v34_run5_causal/`), `scripts/v34_waiting_shortcut_eval.py`, leak audit `diagnostic_outputs/v34_leak_audit/report.md`.
- June-30 data: raw `/iris/u/kewalk/memory_project/data/bin_memory_banana/demo1..30` (5-phase `subtask_labels.json`, 3-task backups), lerobot `~/.cache/huggingface/lerobot/yam/bin_memory_banana_subtask` (**still 3-task — reconvert after the manual label pass**), auto-labeler `examples/yam/autolabel_banana0630_subtasks.py`, editor sync `examples/yam/sync_banana0630_labels_from_editor.py`, browser editor view `/iris/projects/humanoid/ke/relabel_0630_banana` (CFR-30 re-encodes; editor: `Qwen3-VL/qwen-vl-finetune/scripts/subtask_label_editor.py --data_root … --pattern 0630 --port 8898`).
- Cluster: launchers in `cluster_v34/` (note: `srun --overlap` steps all land on GPU 0 unless explicitly pinned — request `--gpus-per-task=4` and export `CUDA_VISIBLE_DEVICES` per worker).

---

## ADDENDUM (2026-08-30 afternoon): the 0830 fresh-setup eval — the decisive test, run

**New data:** `0830_eval` — 10 never-trained episodes recorded in the exact 0816 setup and vocabulary (`open both lids → inspect both bins → close/reset`; no terminal open), banana LEFT in ep0–4 / RIGHT in ep5–9 (`figs/eval0830_inspect_grid.png`). Each episode scored under BOTH instructions ("find the banana" / "find the grey pepper box") → 20 trials, plus matched pre-evidence approach windows (phase control). Checkpoints 500–14750 + **18000** (the run was resumed overnight past 15750, which the checkpoint manager deleted; 18000 is the new latest and got its own fixed-probe artifact).

**Results** (`figs/FIG8_eval0830_curves.png`, `figs/FIG9_eval0830_evidence_gate_18000.png`, `diagnostic_outputs/v34_eval0830_writer_probe{,_full18000}/`):

1. **Pooled accuracy: chance at every checkpoint** (fresh .45–.55, online .45–.55 over 20 trials).
2. **Instruction is ignored:** same prediction under both prompts for 9–10/10 episodes at every checkpoint; mean |Δscore| from switching the prompt ≈ 1.7 against margins of ±10–27.
3. **No banana-side tracking either:** the (prompt-invariant) prediction matches the banana's side 4–7/10 across checkpoints — chance. The apparent ~70% banana-prompt accuracy at ckpts 9500/11000 was noise that inverts by 18000.
4. **The evidence gate fails here too:** unlike June (features collapse, scores frozen) and unlike in-domain (approach ≈ 0 → inspect strongly correct), on 0830 the probe emits LARGE scores in BOTH windows (approach |score| often > inspect), direction unrelated to truth — loud noise, not a percept (FIG9; videos `videos/EVAL0830_*`).

**Triangulated verdict across the three datasets:** in-domain-episodes = near-perfect & evidence-gated · fresh same-setup episodes = large uninformative outputs · different-setup episodes = silent. The writer's side signal therefore does not survive even a *same-setup, new-session* recording — the strongest form of the memorization/session-overfit interpretation. The in-domain 0.98 is real but rests on cues specific to the training episodes/sessions, and instruction-binding was never learned (task identity was session-confounded in training, so it never had to be).

**Implications for next steps:** (a) any writer claim must be "training-collection only"; (b) instruction-binding and session-robustness need to be *forced* by data/objective design (multi-session training data; counterbalanced prompts on identical scenes; possibly an explicit contrastive writer objective); (c) the memory-chain questions (commit/read/use) are still worth testing in-domain, but a positive there would inherit the same generalization caveat. Ops note: the qwen action-expert training on 17130107 was killed for this eval per user authorization — full relaunch cmdline in `cluster_v34/logs/killed_qwen_training_cmdline_20260830.txt`.
