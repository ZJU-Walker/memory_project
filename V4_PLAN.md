# V4 Final Plan — Dual-Bank Visual + Semantic TTT Memory

**Status:** Finalized against the real repo (branch `v4`, worktree `/iris/u/kewalk/memory_project_v4`)
**Supersedes:** the v1.0 architecture draft (`V4_Dual_Bank_Visual_Semantic_TTT_Memory_Plan_EN.md`). All seven design invariants of the draft are preserved; §2 records the places where the implementation deliberately differs from the draft and why.
**Non-interference contract:** `main` stays the v3 line. The running v36 pilot (`v36_pilot_20260831_r6`, iris-hgx-1, artifacts under `memory_project/v35/`) and every older run remain testable and editable in the original tree. Nothing in v4 writes into `memory_project/`, its `.venv`, or its JAX cache.

---

## 1. Repo isolation (done)

| Item | Value |
|---|---|
| Branch | `v4`, branched from `main` @ `eebe2cb` (v3.5/v36 snapshot commit) |
| Worktree | `/iris/u/kewalk/memory_project_v4` (separate checkout; `main` tree untouched) |
| Data | `data -> /iris/u/kewalk/memory_project/data` (symlink, shared read-only) |
| v3 artifacts | `v35_readonly -> /iris/u/kewalk/memory_project/v35` (read-only reference: v36 ckpt-0, calibrations, gate artifacts) |
| v4 artifacts | `v4/{diagnostics,assets,checkpoints}` in the worktree (gitignored, like `v35/`) |
| Python env | own `openpi/.venv` via `uv sync --frozen` (the v3 venv is an *editable install of the v3 tree* — sharing it would silently import v3 code) |
| JAX cache | v4 gets its **own** `OPENPI_JAX_CACHE_DIR` (`v4/cache/jax`). Never share with the running v36: the NFS-shared cache corrupts under concurrent writers (root cause of the train_test SIGABRT flake). |
| GPUs | v4 never runs on GPUs the v36 pilot occupies. Stage 1 is small enough for any single free GPU / the other cluster. |

Workflow: v3 fixes happen in `memory_project/` on `main` as usual; v4 work happens only in `memory_project_v4/` on `v4`. If a shared-file fix (e.g. `memory.py`) lands on `main`, cherry-pick it into `v4`.

---

## 2. Finalization decisions (deltas from the draft)

The draft was architecture-level; these are the binding implementation decisions:

1. **Stage 0 is already done.** The shared TTT core is the v3.5 `TitansMemory` in `delta_output`/`pooled_frame` mode (`openpi/src/openpi/models/memory.py`) — commit→retain→read→use was validated by the v3.5/v36 Gate-C battery (13/13 core checks, exact rank-one commit residuals, analytic decay). Both banks reuse it unchanged. No new memory core is built for v4.
2. **The visual bank IS the v36 pathway, unchanged.** Layer-8 dual-query compressors, pooled-frame delta write, tanh_rms injection, blind memory tokens with zero-init slot embeddings, reseed-CE, the v3.5 write/decision masks — all kept, same module names (`self.memory`, `read_query_compressor`, …) so a v36 checkpoint grafts directly through `AuditedPartialCheckpointWeightLoader`.
3. **The semantic bank does not use the hidden-token projection path at all.** It is a second `TitansMemory` driven purely through the *explicit key-space API* (`delta_write_kv` / `read_key`) that the v3.4 refactor created for exactly this purpose. Keys come from a learned fact-slot table, values from a memory-blind fact head (§4). This makes "no teacher forcing" and "no memory echo" structural properties, not training-time disciplines.
4. **Structured facts are a typed head first, text later.** The draft's `<phase> + <memory_fact>` *text* output is deferred to v4.1. v4-Base supervises facts with a per-slot classification head (the direct generalization of v3.5's `memory_write_side_head`). Reasons: (a) the subtask string is the *only* label channel and its vocabulary is string-matched in three independent places (`label_subtasks._phase_of/_side_of`, `MemoryV34Labels`, converter preflight) plus the frozen manifest `task_vocabulary` — a grammar change forces a full relabel + re-freeze; (b) `causal_token_len=128` leaves little CE budget; (c) the draft itself forbids teacher-forced fact tokens from feeding the write, so text emission is presentation, not mechanism. The typed head satisfies draft invariant 5 (persistent facts ≠ transient phases) by construction.
5. **No new dataset, no new labels, no re-freeze for v4-Base.** Fact labels derive *deterministically* from the frozen v36 manifest (`data/0830_0831_episode_manifest_v36_frozen.json`, sha `9085fe50…`, sides 3×-verified): `target_side=left` ⇒ facts {banana→left_bin, grey_pepper_box→right_bin}. Same 70 episodes, same 54/8/8 seed-36 split, same pinned norm stats. Gate A reuses the identical frozen inputs; the derivation gets its own hashed sidecar + audit script. 0816 stays excluded (Gate-B leak, 0.82 p=.001).
6. **Fusion stays minimal for v4-Base.** Both banks inject as separate token groups with separate calibrated gates and bank type embeddings; the upper blocks select via attention. No learned cross-bank mixer, no bank dropout unless Stage-4 ablations show a permanently ignored bank (draft's contingency, kept as contingency).
7. **Capacity:** V4-Base = both banks at the current geometry (draft's ~9.4M total; in delta mode the per-episode fast state is just `w3` [1024×2048] per bank). V4-Large (asymmetric ~20M) stays gated on *measured* interference, exactly as the draft demands.

---

## 3. Architecture mapped to code

Timestep flow (all inside `_compute_sequence_loss_v32` / `sample_with_memory`):

```
h8 = blocks 0..8 (memory-blind by construction: reads are injected AFTER block 8)
  ├── visual read:   read_query_compressor(h8_top) → project_q → read_key(M_vis)       [16 tokens]
  ├── semantic read: read_key(M_sem, L2Norm(fact_keys))                                 [F tokens]
  ├── inject: per-bank tanh_rms gates + per-bank slot/type embeddings → memory tokens
  ├── blocks 9..17 over [images | context | mem_vis | mem_sem | causal] → subtask CE + flow
  ├── visual write:  write_query_compressor(h8_top) → project_kv → pool → delta_write_kv(M_vis)   (E-steps)
  └── semantic write: fact head(h8) → per-slot (key from table, value = soft target embedding)
                      → delta_write_kv(M_sem)                                            (gated E-steps)
```

Read-before-write ordering per step is already the v3.5 contract (`delta_write_kv` is documented as a transition API: current-step read precedes it). It now covers both banks.

**New modules on `Pi0` (all under `memory_v4_dual_bank=True`, default-off, bit-identity with v36 config when off — repo convention):**

- `self.memory_semantic = TitansMemory(config.memory_semantic, rngs)` — independent `MemoryState`, so one bank can never overwrite the other (draft invariant 1).
- `self.fact_keys` — `nnx.Param[F_max, d_key]`, L2-normalized at use (same pattern as `memory_aux_queries`). One row per (entity, relation) slot. Bin task uses F=2 of a static `F_max=8` (static shapes for jit; unused slots masked).
- `self.fact_head` — memory-blind fact predictor on h8: instruction-row cross-attention (reuse `MemoryQueryConditioner` pattern with `instruction_only` context — the v3.4 lesson: state rows leak phase) → per-slot logits over the fact-target vocab `{left_bin, right_bin, unknown}`.
- `self.fact_value_embed` — `nnx.Param[num_targets, d_value]`; the written value is `L2Norm(softmax(logits_slot) @ fact_value_embed)` — the draft §6.1 "soft embedding of the predicted distribution". Never the ground-truth token, never a post-read representation.
- `self.memory_sem_inject_w`, `self.memory_sem_slot_embedding` (zero-init — v36 lesson: any nonzero init breaks the 1.02× step-0 transparency bound), `self.memory_sem_type_embedding` / visual type embedding — bank identity visible at fusion (draft invariant 2).
- `self.memory_fact_read_head` — read-side supervision on D-valid steps from raw semantic retrieval (generalizes `memory_read_side_head`).

**Config (`Pi0Config`):** `memory_v4_dual_bank: bool = False`, `memory_semantic: MemoryConfig` (v4-Base: identical geometry to `memory`), `memory_fact_slots: int = 8`, `memory_fact_targets: int = 3` (left/right/unknown), `memory_fact_write_conf: float = 0.9`, `memory_fact_write_loss_weight`, `memory_fact_read_loss_weight`, plus a `__post_init__` block enforcing the v4 invariants the way `memory_v35_enabled` does (delta_output both banks, blank_initial_output, tanh_rms, frozen gates at init, zero-init semantic slot embeddings).

**Semantic write policy (draft §7), implemented with existing masks:**
- Eligible only on `seq_write_mask` E-steps (evidence phase, lids open — answer-related writes stop at close, as required for the occlusion task).
- Event-driven: a slot writes only when `max softmax > memory_fact_write_conf` and argmax ≠ `unknown`; otherwise that step is decay-only for the bank (`analytic_decay` — exact under the sparse clock).
- Sparse by construction: F=2 associations per episode vs the visual bank's per-E-step stream.
- Both banks share the v3.5 sparse clock (`alpha_step=0.01` per 15-frame step, analytic gap collapse). A separate, smaller semantic alpha is a recorded knob, not a v4-Base change.

**Inherited landmines (all remain binding):** FP32 matmul pin (TF32 breaks the exact-arithmetic contract — memory.py already pins, keep it for every new einsum on the semantic path), fp32 comparison for `alpha_step`, orbax structural-None merge on load, RMSNorm zero-stream (semantic tokens get the same zero-init slot-embedding treatment under blinding), `state_cotangent_clip`/`kv_cotangent_clip` configured **per bank**, per-bank telemetry in `v35_runtime_identity`/`v35_cumulative_telemetry` (the scan carry now holds two `MemoryState`s — checkpoint & iterator-state schema versions bump).

---

## 4. Write integrity (draft §6) — how each rule is enforced structurally

| Draft rule | Enforcement in code |
|---|---|
| No teacher-forcing leakage | The write value is built only from `fact_head(h8)` logits. Ground-truth fact labels touch only the CE loss on those logits — there is no code path from a label embedding to `delta_write_kv`. |
| No memory echo | `fact_head` consumes h8, which is computed before any injection (block-8 boundary). Retrieved tokens enter only blocks 9..17. A unit test asserts the fact-head output is invariant to arbitrary `M_vis`/`M_sem` contents. |
| Memory-blind writer vs memory-conditioned policy | Already the architecture: CE/flow read the fused stream; both writers read h8. |
| Phase tokens never overwrite facts | Semantic writes exist only for fact slots; phases live in the causal CE and never reach `M_sem`. |
| Unknown ⇒ no write | The `unknown` target and the confidence gate map to decay-only steps. |

---

## 5. Stage plan → concrete runs

**Stage 0 — shared core.** Done (v3.5/v36). One addition: run the synthetic key-space battery (`v35_rung_collect` stage-0 pattern) on the fact-key geometry — F up to `F_max=8` near-orthogonal keys through `delta_write_kv`/`read_key`, measuring cross-slot interference of the rank-one updates. Pure CPU/1-GPU, no training.

**Stage 1 — semantic understanding without memory** (first v4 training run, cheap, single-frame — no sequence scan):
- Config `pi05_yam_v4_stage1`: pi05 + `predict_subtask` path + fact head on h8; no memory modules active.
- Train on the v36 train-54; supervise fact CE only on frames where the fact is *observable* (inspect phase ∩ E-visibility sidecar); label `unknown` elsewhere.
- Pass gates (all machinery exists): fact accuracy on fresh dev episodes; **cross-session transfer** (train 0830→eval 0831 and reverse — the writer-signal-is-session-specific failure of v3.4 is the #1 risk here); prompt swap; state masking (`state_null_embedding`); arm-region patch masking; `unknown` before lids open; class balance. Plus a Gate-B-style probe: episode-OOF side decode from fact-head *features* on pre-evidence frames must be at chance.
- **This is the go/no-go for the whole semantic branch**: if facts aren't visually grounded on fresh data, nothing downstream is authorizable.

**Stage 2 — semantic memory only** (visual injection frozen at zero):
- **2a oracle writes:** write `L2Norm(fact_value_embed[gt_target])` at eligible E-steps (pattern: `_v35_oracle_injected_content`; see also the old `oracle_memory` branch). Trains the read/fuse/use path in isolation. Gates: exact commit residuals (delta rule is analytic), retention over the occlusion horizon, D-step read-head accuracy, reset→chance and donor-swap→donor's answer flips (episode-level, `memory_swap_read_step` extended per bank).
- **2b predicted writes:** swap oracle for the Stage-1 fact head. The 2a↔2b gap *is* the remaining perception error, cleanly measured.

**Stage 3 — visual memory only.** This *is* v36, running now on `main`. Its result slots in unchanged; v4 never reruns it. If v36's pilot fails its gates, the visual-bank fix happens on the v3 line and merges forward — the semantic thread (Stages 1–2) is not blocked.

**Stage 4 — dual-bank fusion.** Both banks on, `pi05_yam_mem_v4` full config, through the cloned gated pipeline (§6). Eval battery per bank and combined: {both, visual-only, semantic-only} × {reset, donor-swap, both-swapped}. Bank dropout only if one bank is provably ignored.

**Stage 5 — generalization/capacity.** Multi-collection training, larger fact vocab, and — required before any "visual bank is necessary" claim (draft §4.2) — a benchmark needing non-semantic visual detail (appearance/pose matching). Task design starts only after Stage 4 passes. V4-Large only on measured interference.

---

## 6. Gates & authorization (clone, don't fork-in-place)

Clone `cluster_v35/` → `cluster_v4/` and the `v35_prepare_pilot.py` orchestrator with:
- New `CONFIG_NAME = "pi05_yam_mem_v4"` — a real new `TrainConfig` (v36 overloaded the v35 name via `project_paths`; don't repeat that), new `V4_*` constants in `openpi/src/openpi/shared/project_paths.py`, artifacts under `v4/diagnostics/runs/<exp>/`, checkpoints under `v4/checkpoints/`.
- Frozen hashes: manifest/dataset/norm hashes stay the v36 values (§2.5); add the fact-label derivation sidecar hash to Gate A.
- Gate B: existing two preregistered probes + a third preregistered probe on fact-head features (pre-evidence frames, episode-OOF, permutation null).
- Stage-06 calibration: per-bank c/tau (the semantic bank's retrieval RMS differs from the visual bank's; one shared calibration would mis-scale one of them).
- Gate C: the 13 core checks + delta-identity checks run per bank; step-0 task health keeps the 1.02× transparency bound with *both* injections armed.
- Two-stage sha-sealed authorization flow unchanged (`v35_training_authorization.py` pattern, new criteria-version strings `openpi.v4.*`). Note: pass `--pilot-authorization` explicitly — the `v35_train.py` default points at a directory real runs never populate.
- Every new script gets its paired `*_test.py`, matching the existing scaffolding.

---

## 7. Success criteria & claim boundary

Unchanged from the draft §13: grounded facts on fresh data (Stage 1), per-bank commit/retain/read (2a, v36), causal reset/swap flips (2, 4), semantic control of fact decisions (2b), visual value beyond facts (Stage 5 benchmark), correct source selection when fused (4). The draft's non-claims also stand — in particular v4-Base does **not** claim unsupervised fact emergence (labels are derived), fully learned write timing (the confidence gate + phase mask are scaffolding), or that the bin task needs the visual bank.

---

## 8. Immediate next actions on `v4`

1. `uv sync` the worktree venv; create `cluster_v4/env.sh` (own `MEMORY_PROJECT_ROOT`, own JAX/UV caches). *(venv build already launched)*
2. Implement chunk A: `memory_v4_dual_bank` config + `memory_semantic` + fact keys/head/value embed + semantic injection, with default-off bit-identity tests vs the v36 config.
3. Implement chunk B: `v4_build_fact_labels.py` (derive from frozen manifest, hashed sidecar) + audit test.
4. Implement chunk C: Stage-1 config + fact leak battery (clone `v35_leakage_*` with the fact-feature probe).
5. Launch Stage 1 on a free GPU. Decision point on its gates before any memory training.
