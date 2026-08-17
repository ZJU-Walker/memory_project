# V3.3 Memory Plan — Handoff for Implementation

## 1. Goal

V3.3 should make the memory writer learn **task-relevant information** without introducing task-specific memory mechanisms.

The central objective is:

\[
\boxed{
\text{task instruction}
\rightarrow
\text{task-conditioned write}
\rightarrow
\text{memory}
\rightarrow
\text{future subtask prediction}
}
\]

V3.3 should remain a general memory design that can transfer to tasks other than the current banana / grey-box bin task.

The two main changes from V3.2 are:

1. **Task-conditioned 16-token writer**
2. **Memory-critical trajectory sampling with ordinary subtask supervision**

We do **not** want banana-specific classifiers, side-specific auxiliary heads, object masks, or hard-coded "write during reveal" rules.

---

## 2. Motivation from V3.2

V3.2 introduced separate learned 16-token read and write query banks.

The writer showed some meaningful scene-dependent behavior, but the overall memory path was not causally important:

- the write queries were not explicitly conditioned on the task instruction;
- the 16 write slots were highly redundant;
- zeroing retrieved memory usually did not change the selected side;
- the policy could exploit shortcuts from current observations / robot state;
- after lid closure, side-specific arm movement made the target easy to infer without memory;
- therefore the outer-loop loss had weak pressure to make the writer encode the task-relevant past information.

The main training problem is therefore not simply "bad attention."

The policy needs more examples where:

\[
\boxed{
\text{the answer is no longer visible}
+
\text{the robot has not yet revealed the answer through its motion}
}
\]

and the correct subtask still depends on earlier observations.

---

## 3. New Dataset

There are now 60 newly collected episodes:

- 30 episodes with instruction: `find the banana`
- 30 episodes with instruction: `find the grey pepper box`

For each task:

- 15 target-left episodes
- 15 target-right episodes

So the dataset is balanced across:

\[
\begin{array}{c|cc}
 & left & right\\
\hline
banana & 15 & 15\\
grey\ box & 15 & 15
\end{array}
\]

### Episode structure

Each trajectory follows the same broad sequence:

1. Open both lids
2. Inspect both bins while both objects are visible
3. Close both lids
4. Reset both arms to a neutral pose
5. Wait for several seconds
6. Move toward and open the requested target bin

This new neutral waiting period is especially important because it creates a clean interval where:

- the target object is hidden;
- both arms are reset;
- no side-specific action has started yet;
- the correct target side must come from past information.

---

## 4. Subtask Labels

Use the following subtask vocabulary:

```text
open both lids
inspect both bins
close both lids and reset arms
wait; target bin is left
wait; target bin is right
open left bin
open right bin
```

The critical memory-dependent labels are:

```text
wait; target bin is left
wait; target bin is right
```

These labels should begin only after:

- both lids are closed; and
- both arms have completed the reset to the neutral pose.

They should end before the first side-specific action toward the selected bin.

This waiting phase is the cleanest supervision point for memory.

---

## 5. Architecture Change: Task-Conditioned Write Queries

### V3.2 writer

V3.2 uses a learned write query bank:

\[
Q_0^w \in \mathbb{R}^{16 \times d}.
\]

These learned queries cross-attend to the layer-8 top-camera hidden states:

\[
H_{8,t}^{top}.
\]

### V3.3 writer

In V3.3, the 16 write queries should first be conditioned on the current task instruction.

Conceptually:

\[
Q^w(I)
=
f_{\text{condition}}
\left(
Q_0^w,
H_I
\right),
\]

where:

- \(Q_0^w\): the 16 learned base write queries;
- \(H_I\): hidden representation of the task instruction;
- \(Q^w(I)\): task-conditioned write queries.

Then:

\[
Z_t^w
=
\operatorname{CrossAttn}
\left(
Q^w(I),
H_{8,t}^{top}
\right).
\]

The intended semantics are:

\[
\boxed{
\text{Given my current task, what information in this observation is worth remembering?}
}
\]

For the same visible scene:

\[
\text{banana left,\ grey box right},
\]

the writer should be allowed to produce different write representations for:

```text
find the banana
```

versus:

```text
find the grey pepper box
```

### Important constraints

The writer should be conditioned only on information naturally available at inference.

Do **not** condition the writer on:

- ground-truth target side;
- ground-truth future subtask;
- future actions;
- special banana / grey-box labels;
- manually supplied visibility flags.

The task instruction is the only new semantic conditioning signal.

---

## 6. No Task-Specific Writer Loss in the Core V3.3 Design

Do **not** add a permanent auxiliary objective such as:

\[
Z_t^w \rightarrow \{\text{banana-left}, \text{banana-right}\}.
\]

Do not add:

- banana detector loss;
- grey-box detector loss;
- target-side classifier directly on the writer;
- object segmentation / attention-mask supervision;
- hard-coded rules saying when the writer may update.

The goal is for the existing policy objective to teach the writer what matters.

The V3.3 memory system should remain useful for other tasks without changing its architecture or introducing new task-specific heads.

---

## 7. Main Training Change: Memory-Critical Sampling

### Problem with normal long-trajectory sampling

If training samples include the later action period, the model can learn shortcuts such as:

\[
\text{left arm starts moving}
\rightarrow
\text{target is left}.
\]

Then the subtask/action losses can be solved without retrieving earlier information.

V3.3 should deliberately oversample trajectory segments where memory is required.

---

## 8. Memory-Critical Sample Construction

A memory-critical sample should include the evidence phase and end inside the neutral waiting phase.

Conceptually:

\[
t_s
<
t_{\text{inspect}}
<
t_{\text{close}}
<
t_{\text{reset}}
\leq
t_q
<
t_{\text{action-start}}.
\]

Where:

- \(t_s\): sample start, at or shortly before `inspect both bins`;
- \(t_{\text{inspect}}\): object evidence is clearly visible;
- \(t_{\text{close}}\): lid closing / reset occurs;
- \(t_{\text{reset}}\): both lids closed and both arms neutral;
- \(t_q\): randomly sampled waiting-frame endpoint;
- \(t_{\text{action-start}}\): first side-specific motion toward the chosen bin.

The recurrent memory should be causally unrolled across the complete sample:

\[
\text{inspect}
\rightarrow
\text{close/reset}
\rightarrow
\text{wait}.
\]

At \(t_q\), apply the normal subtask loss to:

```text
wait; target bin is left
```

or:

```text
wait; target bin is right
```

The desired gradient path is:

\[
\boxed{
\mathcal L_{\text{wait-subtask}}
\rightarrow
m_{t_q}
\rightarrow
M_{t_q-1}
\rightarrow
Z_{\text{earlier}}^w
\rightarrow
Q^w(I)
}
\]

The model should therefore learn, through the ordinary subtask objective, which earlier information had to be stored.

---

## 9. Role of Each Phase

### `inspect both bins`

This is the **information-source phase**.

The requested object's identity and spatial relationship are visible here.

These frames should usually appear inside the history of memory-critical samples.

### `close both lids and reset arms`

This is primarily a **retention / interference phase**.

The target side is no longer directly visible, but the subtask label itself does not yet strongly require left/right memory.

These frames should remain inside the history so the memory must survive them.

They should not be the main oversampled endpoint.

### `wait; target bin is left/right`

This is the **memory-required supervision phase**.

The current visual scene should be ambiguous while the subtask target remains side-specific.

This phase should receive substantially increased sampling probability.

### `open left/right bin`

Do not use these frames as the endpoint of the memory-critical branch.

Once side-specific arm motion begins, the target can again be inferred from current observations / robot state.

These frames still belong in normal policy training.

---

## 10. Sampling Strategy

Use two complementary training sample types.

### A. Normal policy samples

Keep the standard trajectory / window sampling used for normal policy learning.

These samples train:

- action flow-matching;
- ordinary subtask prediction;
- behavior across the complete trajectory.

### B. Memory-critical samples

Add an oversampled branch that:

1. starts around or before `inspect both bins`;
2. includes the object-visible evidence;
3. continues through lid closing and arm reset;
4. terminates at a random point inside the neutral waiting phase;
5. applies normal subtask supervision at the waiting endpoint;
6. never crosses into the side-specific action phase.

A reasonable starting sampling mixture is approximately:

```text
50% normal policy samples
50% memory-critical samples
```

This is only an initial value and can be tuned based on diagnostics.

---

## 11. Random Waiting Endpoints

Do not always train at the first waiting frame.

For every memory-critical sample, randomly choose:

\[
t_q
\sim
U
\left(
t_{\text{reset}},
t_{\text{action-start}}
\right).
\]

This provides variable retention horizons.

The model should sometimes be queried:

- immediately after reset;
- one second later;
- several seconds later;
- near the end of the waiting interval.

This encourages persistent memory instead of learning one fixed timing cue.

Avoid simply duplicating every static wait frame because this could overfit the small number of physical episodes.

Prefer:

- sample episodes uniformly;
- sample one random wait endpoint when that episode is selected.

---

## 12. Preserve Dataset Balance

The memory-critical sampler should preserve balance across:

```text
banana-left
banana-right
grey-box-left
grey-box-right
```

Do not let one task or side dominate the oversampled waiting branch.

This is necessary to prevent shortcuts such as:

\[
\texttt{find banana}
\rightarrow
\text{left prior}.
\]

---

## 13. Training Losses

V3.3 should keep the existing generic losses.

Conceptually:

\[
\mathcal L
=
\mathcal L_{\text{flow}}
+
\lambda_s
\mathcal L_{\text{subtask}}.
\]

The important change is **where and how often subtask supervision is sampled**, not a new task-specific objective.

### Normal branch

Use the existing action + subtask losses.

### Memory-critical branch

The most important supervision is the subtask CE at the waiting endpoint.

The action during waiting is intentionally almost identical across left/right cases, so the action loss provides little target-side supervision there.

It is acceptable for the memory-critical branch to emphasize the subtask loss, while normal samples continue training the full action objective.

---

## 14. Generic Interpretation

The V3.3 training rule should not be implemented as:

> "For the banana task, sample these particular frames."

The generic interpretation is:

\[
\boxed{
\text{sample more states where the current observation is insufficient
but the correct high-level decision still depends on previous observations}
}
\]

For the current dataset, the `wait; target bin is left/right` segment naturally provides these states.

For future tasks, the exact labels may differ, but the architecture and learning principle should remain unchanged.

---

## 15. Optional Generic Metadata

If useful for the data pipeline, a generic per-frame field can be stored:

```text
memory_required: true / false
```

For the current dataset:

```text
open both lids                     -> false
inspect both bins                  -> false
close both lids and reset arms     -> false
wait; target bin is left           -> true
wait; target bin is right          -> true
open left bin                      -> false
open right bin                     -> false
```

This allows the sampler to select memory-critical endpoints without hard-coding task-specific object names.

This metadata should affect training sampling only.

It should not be available to the policy at inference.

---

## 16. Important Gradient Check

The memory-critical waiting-frame loss must actually backpropagate through the recurrent memory updates to earlier write tokens.

Explicitly verify:

\[
g_\tau
=
\left\|
\frac{\partial \mathcal L_{\text{wait}}}
{\partial Z_\tau^w}
\right\|.
\]

Plot or log this across the sampled history.

During the earlier `inspect both bins` phase:

\[
g_\tau
\]

must be nonzero.

If the memory state is detached between timesteps, then better sampling alone will not teach the earlier writer what information to preserve.

This check is important before interpreting training results.

---

## 17. Diagnostics for V3.3

Attention heatmaps are useful, but they should not be the primary success metric.

### 17.1 Normal vs zero-read during waiting

At memory-required waiting frames, compare:

```text
normal retrieval
vs.
retrieved tokens = 0
```

V3.3 should show a substantially larger difference than V3.2.

Desired behavior:

\[
\text{normal memory}
\rightarrow
\text{correct target side}
\]

while zero-read should lose confidence or accuracy.

If zero-read still predicts the correct side almost perfectly, the model is still using a shortcut.

### 17.2 Memory swap

Under the same neutral waiting observation:

\[
(O^*,M_L)
\rightarrow
\text{left target}
\]

and

\[
(O^*,M_R)
\rightarrow
\text{right target}.
\]

The prediction should follow memory.

This is the strongest causal test of the memory path.

### 17.3 Instruction-conditioned writer diagnostic

During a visible `inspect both bins` frame containing both objects, run the same image with:

```text
find the banana
```

and:

```text
find the grey pepper box
```

Compare:

\[
Z_t^w(I_{\text{banana}})
\]

against:

\[
Z_t^w(I_{\text{grey}}).
\]

The write representations should be measurably different.

This verifies that instruction conditioning is actually affecting writer behavior.

No special supervision head is required for this diagnostic.

### 17.4 Writer / read attention visualization

Continue producing:

- 16 write-query attention maps;
- 16 read-query attention maps;
- entropy / effective-patch statistics;
- write-token similarity;
- retrieved-token similarity.

However, do not define success as "attention is concentrated exactly on the target object."

For a relational task, useful information may involve:

- target object;
- both bins;
- spatial relation;
- container context.

The primary question is whether the memory causally controls the later decision.

### 17.5 Gradient-over-time diagnostic

For memory-critical samples, log the outer-loss gradient magnitude reaching write tokens at each historical timestep.

We want evidence that the waiting-frame supervision reaches the earlier evidence phase.

---

## 18. Success Criteria

V3.3 should be considered successful if the following pattern emerges.

### Writer

The same visible image with different instructions produces meaningfully different write representations:

\[
Z^w(\text{find banana})
\neq
Z^w(\text{find grey box}).
\]

### Retention

The relevant information survives:

\[
\text{inspect}
\rightarrow
\text{close/reset}
\rightarrow
\text{wait}.
\]

### Read / use

At the neutral waiting point, the correct side depends on retrieved memory.

Zero-read should meaningfully hurt performance.

### Causal memory control

Memory swapping should change the predicted side under a fixed current observation.

The desired final causal structure is:

\[
\boxed{
\text{instruction}
\rightarrow
\text{task-conditioned writer}
\rightarrow
M
\rightarrow
\text{retrieval}
\rightarrow
\text{future subtask / action}
}
\]

---

## 19. Non-Goals for V3.3

Do not add the following unless later diagnostics clearly justify them:

- banana-specific memory head;
- grey-box-specific memory head;
- target-side classifier directly attached to writer tokens;
- object detection / segmentation supervision;
- manually supervised attention masks;
- forced diversity among the 16 write slots;
- hard-coded write gating based on task phase;
- special inference-time rules tied to these particular subtasks.

The purpose of V3.3 is to test whether **better task conditioning and better sampling pressure are sufficient** to make the existing memory architecture learn useful information.

---

## 20. Summary

V3.3 makes two focused changes:

### Architecture

\[
\boxed{
16\text{ learned write queries}
+
\text{task instruction conditioning}
}
\]

so the writer can decide what is relevant for the current task.

### Training

\[
\boxed{
\text{oversample trajectories ending in neutral memory-required waiting states}
}
\]

so the normal subtask loss cannot be cheaply solved from side-specific arm motion.

The core training path is:

\[
\boxed{
\text{inspect both bins}
\rightarrow
\text{write task-relevant information}
\rightarrow
\text{close lids + reset arms}
\rightarrow
\text{wait}
\rightarrow
\text{predict target-side subtask from memory}
}
\]

This keeps the memory mechanism generic while directly addressing the main failure observed in V3.2.
