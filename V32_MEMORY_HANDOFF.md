# V3.2 Memory Architecture — Codex Handoff

## Goal

V3.2 is a clean revision of the VLA–memory interface. It should **not** become a broad redesign of the training objective or fast-weight update rule.

The core change is:

> Use layer-8 top-camera hidden states as the common source for both memory read and memory write, with two separate learned 16-token query banks.

V3.2 should otherwise preserve the existing fast-weight memory mechanics and unrelated V3.1 behavior as much as possible.

---

## 1. Core architecture

For timestep \(t\), first run the normal VLM until the selected layer-8 representation is available.

Let

\[
h^{top}_{8,t} \in \mathbb{R}^{256 \times d}
\]

be the 256 layer-8 hidden tokens corresponding to the top-camera image.

This is the explicit source representation for both memory read and memory write in V3.2.

Other modalities remain part of the normal VLA. V3.2 only constrains the explicit memory interface.

---

## 2. Memory read

Introduce 16 learned read-query tokens

\[
Q^{read} \in \mathbb{R}^{16 \times d}.
\]

They are trainable slow-weight parameters. They are not memory state and are not retrieved memory tokens.

Use them to cross-attend the 256 top-camera layer-8 tokens:

\[
q_t =
\operatorname{CrossAttn}
\left(
Q^{read}, h^{top}_{8,t}
\right),
\]

with

\[
q_t \in \mathbb{R}^{16 \times d}.
\]

Interpretation:

> The read queries learn what the current observation needs to ask the past.

Then query the **pre-write** fast-weight memory:

\[
m_t = M_{t-1}(q_t),
\]

where

\[
m_t \in \mathbb{R}^{16 \times d}.
\]

This replaces direct patch-wise memory reads from all 256 top-camera tokens.

Do not replace the learned-query bottleneck with simple average pooling.

---

## 3. Use retrieved memory for action

After retrieval, inject the 16 memory tokens into the token/residual stream **after layer 8**:

\[
[h_{8,t}, m_t]
\rightarrow
\text{remaining VLM layers}
\rightarrow
\text{action pathway}.
\]

The retrieved memory tokens should remain available throughout all remaining VLM layers so later reasoning and the action expert can use them.

Important choices:

- Do not run a second full VLM pass from layer 0.
- Do not require memory tokens to exist before layer 8.
- The memory tokens are created from layer-8 information and then processed by the rest of the VLM.
- Follow the repository's actual layer indexing; the conceptual boundary is “after the selected layer-8 hidden state,” not a hard-coded guessed final layer number.

The intended causal path is

\[
\text{past}
\rightarrow
M_{t-1}
\rightarrow
m_t
\rightarrow
\text{remaining VLM}
\rightarrow
\text{action}.
\]

---

## 4. Memory write

Introduce a second, independent learned query bank:

\[
Q^{write} \in \mathbb{R}^{16 \times d}.
\]

Important:

\[
Q^{write} \neq Q^{read}.
\]

The read and write query banks should have the same number of slots but should **not share parameters** by default.

Reason:

- read asks: **What do I need to recall now?**
- write asks: **What in the current observation is worth storing for later?**

Use the write queries to cross-attend the **same** layer-8 top-camera source:

\[
z_t =
\operatorname{CrossAttn}
\left(
Q^{write}, h^{top}_{8,t}
\right),
\]

with

\[
z_t \in \mathbb{R}^{16 \times d}.
\]

These 16 writer tokens are then converted into whatever key/value or equivalent signals the existing fast-weight memory update expects. Conceptually:

\[
K_t = P_K(z_t), \qquad
V_t = P_V(z_t),
\]

then

\[
M_t = U(M_{t-1}; K_t, V_t).
\]

The existing fast-weight memory architecture, associative objective, optimizer/update state, momentum, clipping, initialization, etc. should remain unchanged unless a minimal dimensional/interface adaptation is necessary.

V3.2 is **not** intended to redesign the memory update rule.

---

## 5. Do not use the late/final VLM representation as the writer source

This is a major V3.2 choice.

The writer source should be

\[
\boxed{h^{top}_{8,t}}
\]

not \(c_{18}^{top}\), final VLM hidden states, or the retrieved memory tokens themselves.

Desired write path:

\[
\boxed{
h^{top}_{8,t}
\rightarrow
16\ \text{learned write queries}
\rightarrow
\text{memory update}
}
\]

Rationale:

1. A late VLM representation can already mix current vision with retrieved memory, prompt/language, robot state, other modalities, task phase, and policy-relevant information.
2. That makes it harder to know whether memory is storing the visual episodic fact we care about or easy shortcuts such as arm pose/task phase.
3. Using layer-8 top-camera tokens for both read and write gives the memory a more consistent representation space.
4. It removes the direct architectural retrieval-to-late-writer feedback path.

This is a cleaner baseline. It should not be framed as proof that memory echo was the main V3.1 failure mode; the previous diagnostics did not show globally dominant echo.

---

## 6. Causal timestep ordering

The timestep ordering must remain causal.

For timestep \(t\):

1. compute the current layer-8 representation;
2. form the 16 read queries;
3. read from **\(M_{t-1}\)**;
4. use the retrieved memory to compute the current action;
5. form the 16 writer tokens from the current \(h^{top}_{8,t}\);
6. update memory to **\(M_t\)**;
7. carry \(M_t\) to the next timestep.

Conceptually:

\[
M_{t-1}
\xrightarrow{\text{read}}
m_t
\rightarrow
a_t,
\]

while

\[
h^{top}_{8,t}
\xrightarrow{\text{write}}
M_t
\rightarrow
t+1.
\]

Critical invariant:

> The current action must not read a memory state that already contains the current observation's write.

No accidental “write current frame, then immediately read the same frame” behavior.

The read-query and write-query compressors may be computed from the same \(h^{top}_{8,t}\) once it is available, but the memory state transition must still respect the causal ordering above.

---

## 7. High-level data flow

```text
Current observation
        |
        v
  VLM through layer 8
        |
        v
  h8_top: 256 tokens
        |
        +------------------------------+
        |                              |
        | READ                         | WRITE
        |                              |
        v                              v
  16 learned read queries        16 learned write queries
        |                              |
   cross-attend h8_top             cross-attend h8_top
        |                              |
        v                              v
     q_t [16,d]                    z_t [16,d]
        |                              |
        v                              |
   read M_(t-1)                        |
        |                              |
        v                              |
     m_t [16,d]                        |
        |                              |
        v                              |
inject after layer 8                    |
        |                              |
remaining VLM layers                    |
        |                              |
        v                              v
   current action              existing memory update
                                       |
                                       v
                                      M_t
                                       |
                                       v
                                  next timestep
```

---

## 8. Important clarification: “top-camera only”

The explicit memory source is the top-camera token positions at layer 8.

However, do not assume this mathematically guarantees a pure visual representation. Depending on the existing VLM attention structure, those layer-8 top-camera token states may already contain information mixed from other tokens.

Therefore V3.2 should be understood as:

> **top-camera-position layer-8 representation as the controlled memory source**

rather than a guarantee of complete modality isolation.

Do not add new masks solely to force complete modality isolation unless the existing architecture already requires them.

---

## 9. What V3.2 should NOT add yet

To keep this experiment interpretable, do not combine the interface change with additional memory fixes.

Do **not** add:

- a new learned write gate;
- hand-coded “write only when banana is visible” logic;
- freeze-after-visible behavior;
- ROI supervision;
- banana-side auxiliary classification;
- probe loss;
- new memory-use auxiliary losses;
- new forgetting rules;
- new memory regularizers;
- a new action loss;
- a new explicit memory-conditioning gate for the writer;
- other diagnosis-driven changes not required by the 16-token interface.

If V3.1 already contains some unrelated mechanism, preserve it unless it directly conflicts with the new interface.

The purpose is to isolate:

\[
256\text{-token dense memory interface}
\rightarrow
16\text{-token learned read/write bottleneck}.
\]

---

## 10. Implementation invariants

Codex should inspect the repository and decide which modules/files should change. Do not assume repository structure from this handoff.

The implementation must preserve these semantic invariants.

### Read

- source: layer-8 top-camera hidden-state positions;
- source shape conceptually: \([B,256,d]\);
- 16 learned read-query slots;
- read-query output shape: \([B,16,d]\);
- retrieve 16 memory tokens from the pre-write memory state.

### Memory use

- retrieved memory is inserted after the selected layer-8 representation;
- all 16 retrieved memory tokens remain available through the remaining VLM layers;
- they can influence the final action pathway.

### Write

- source: the same layer-8 top-camera hidden-state positions;
- 16 learned write-query slots;
- write-query parameters are separate from read-query parameters;
- no final-layer / \(c_{18}\) writer source;
- no direct use of retrieved memory tokens as writer content;
- update the existing fast-weight memory using the 16 writer tokens.

### Timing

- read \(M_{t-1}\);
- current action uses that read;
- current frame produces \(M_t\);
- \(M_t\) is used only by future timesteps.

### Existing memory mechanics

Do not silently change:

- memory-network architecture;
- fast-weight initialization;
- momentum semantics;
- clipping semantics;
- optimizer/update equations;
- recurrent-state carry behavior;
- training objective;
- unrelated V3.1 training/inference mechanisms.

Only change them if strictly required for compatibility, and document any such change clearly.

---

## 11. Required validation / regression tests

Before judging task performance, verify the architecture mechanically.

### Test 1 — Shape audit

Log/assert at least:

\[
h_8^{top}: [B,256,d]
\]

\[
q_t: [B,16,d]
\]

\[
m_t: [B,16,d]
\]

\[
z_t: [B,16,d].
\]

Confirm the memory update consumes 16 writer tokens rather than the original 256 top-camera tokens.

### Test 2 — Separate read/write parameters

Verify the read-query bank and write-query bank are distinct parameter tensors and receive independent gradients.

### Test 3 — Writer-source isolation

Hold \(h_8^{top}\) fixed while changing retrieved memory and/or later VLM activations.

The writer tokens \(z_t\) should remain identical up to numerical tolerance.

This confirms the old post-attention writer path is actually removed.

### Test 4 — Pre-write causal read

At timestep \(t\), changing the current write while keeping \(M_{t-1}\) fixed must not change the memory used for the current action.

### Test 5 — Retrieved-memory persistence

Confirm all 16 retrieved memory tokens remain in the intended token stream through the remaining VLM layers and are not accidentally dropped after one layer.

Follow the repository's existing masking/position conventions. Do not invent a new position scheme unless necessary.

### Test 6 — Gradient-flow sanity

Confirm the outer task loss can reach the trainable components that are intended to learn:

- read-query bank;
- write-query bank;
- read/write projections;
- the slow memory parameters / initialization that are trainable in the existing design.

Do not accidentally introduce extra detach points beyond whatever recurrent truncation/detachment behavior already exists.

---

## 12. Diagnostics to re-run after V3.2 training

These are evaluation diagnostics, **not new losses**.

Re-run the existing V3.1-style diagnostics so V3.2 can be compared directly.

Important checks:

1. **Read-query attention maps**
   - visualize all 16 query-to-patch maps;
   - check whether different slots specialize;
   - measure entropy/effective-token count.

2. **Write-query attention maps**
   - check whether writing is less spatially diffuse;
   - check whether query slots focus on informative events.

3. **Memory retrieval over time**
   - check whether retrieval still collapses after the banana becomes hidden.

4. **Writer update norm / clipping**
   - check whether writes remain saturated on nearly every frame.

5. **Left-history vs right-history memory swap**
   - hold current observation fixed;
   - swap \(M_L\) and \(M_R\);
   - test whether the predicted action side changes.

6. **Zero-read action test**
   - compare normal memory read with zeroed retrieved memory under the same current observation.

7. **Freeze-after-visible diagnostic**
   - as a diagnostic only, compare normal recurrent writing with freezing memory after the informative visual event;
   - use this to test overwrite/dilution;
   - do not bake this heuristic into V3.2.

These tests should tell us whether V3.2 improves:

- read selectivity;
- write selectivity;
- retention of useful episodic information;
- causal dependence of action on memory.

---

## 13. Success criteria

V3.2 is correctly implemented if:

1. both memory read and write use layer-8 top-camera hidden states;
2. read uses exactly 16 learned query slots;
3. write uses exactly 16 separate learned query slots;
4. read and write query parameters are independent;
5. memory retrieval returns 16 tokens;
6. retrieved memory enters after layer 8 and survives through the remaining VLM;
7. the writer no longer uses \(c_{18}\), final VLM states, or retrieved memory as its primary source;
8. the current action reads \(M_{t-1}\), while the current frame creates \(M_t\) for future timesteps;
9. unrelated V3.1 mechanics remain unchanged.

Task success is an empirical result, not an implementation criterion.

Do not silently add extra fixes just to improve the first V3.2 run. A clean comparison is more important.

---

## One-sentence definition

> **V3.2 uses layer-8 top-camera hidden states as a shared visual source, compresses them with 16 learned read queries to retrieve 16 fast-memory tokens for the remaining VLM/action pathway, and independently compresses the same layer-8 source with 16 learned write queries to update the fast-weight memory for future timesteps.**
