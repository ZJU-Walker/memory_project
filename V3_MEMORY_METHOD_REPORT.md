# YAM Hidden-Bin Memory Project: Detailed v3/v3.1 Method and Evaluation Report

> **2026-08-12 v3.1 diagnostics/objective addendum (current behavior).** The v3 and
> v3.1 main configs now set `memory_probe_weight=0.0` and
> `memory_probe_diagnostic=False`. Their optimized objective is therefore only
> \(\mathcal{L}_{\mathrm{flow}} + 1.0\,\mathcal{L}_{\mathrm{causal\ CE}}\); the
> binary left/right probe is no longer part of the main loss. Its module and parameter keys remain
> present for strict checkpoint compatibility. If detached probe diagnostics are explicitly enabled,
> probe outputs are stopped from entering the backward graph, probe optimizer/weight-decay updates are
> masked, and both raw and EMA probe parameters remain fixed. With diagnostics disabled, probe labels
> are not materialized and the extra probe read/classifier computation is skipped.
>
> The report below is a 2026-08-09 snapshot. Its claims that the v3/v3.1 probe weight is `0.5`, that
> probe CE trains the memory/VLM/gate, and that probe supervision is part of the current recipe are
> **historical**. This addendum supersedes those claims, including the probe statements in the executive
> summary, Sections 8–9 and 12, the test description, and the compact method statement. It does not
> change the documented read-before-write ordering, RTC training, TBPTT, sequence sampling, or the
> v3.1 `post_attention` writer. An older probe-trained checkpoint remains loadable but is an engineering
> resume/evaluation artifact, not a clean no-probe ablation; the clean comparison must start afresh from
> the same `pi05_base` initialization. Checkpoint parameter trees do not encode static writer/cadence/
> objective provenance, so always supply and record the intended config. The current diagnostic runner,
> schemas, limitations, and safe CLI are documented in
> [`openpi/docs/v31_memory_diagnostics.md`](openpi/docs/v31_memory_diagnostics.md).

**Repository snapshot:** `/iris/u/kewalk/memory_project`
**Report date:** 2026-08-09
**Compared methods:** `pi05_yam_mem_v3` and `pi05_yam_mem_v31`
**Implementation status:** v3 and the controlled v3.1 post-attention writer are present as working-tree changes on top of commit `03d112d` (`update memory ttt v2`). This report describes the current files on disk, not the older root README and not only the committed `HEAD` version.

## 1. Executive summary

The project studies episodic memory for a long-horizon bimanual manipulation task. A YAM robot first inspects two bins, observes which bin contains a banana, loses direct visual access after the bins are closed, and must later open the correct bin. The base policy is Physical Intelligence's π₀.₅ vision-language-action model. The v3 method augments it with a Titans-style neural memory whose **fast weights are updated online within each episode**.

At every policy query, the system follows this order:

1. Encode the current three-camera observation and robot state with π₀.₅.
2. Extract 256 contextualized token vectors corresponding to the top-camera token positions at Gemma layer 8.
3. Query the previous episodic memory with those vectors.
4. Convert the 256 retrieved vectors into 256 memory tokens and append them to the PaliGemma KV cache.
5. Decode a textual subtask such as `observe bins` or `open left bin`.
6. Denoise a 50-step action chunk conditioned on the current observation, retrieved memory, and decoded subtask.
7. Only after making the prediction, update the episode's fast-weight memory. V3 writes the
   layer-8 top-camera hidden vectors; v3.1 instead writes the already-computed post-attention
   memory-token outputs.

The current v3 training recipe is inspired by RoboTTT and includes π₀.₇-style train-time RTC:

- Each sample contains up to 60 sequential policy steps from one episode. Homogeneous
  `T=20/40/60` batches execute the smallest static graph that retains every valid step.
- Consecutive steps are 10 raw frames apart while the action horizon remains 50, so neighboring
  targets overlap by 40 controls exactly like the asynchronous client.
- Every step independently samples a committed-prefix delay from the inclusive range 0–6. Those
  action tokens are kept clean at flow time zero and excluded from the suffix-normalized flow loss.
- The policy is supervised at every step, rather than only at the end of a memory window.
- Backpropagation through the recurrent memory state is truncated in randomly shifted 25-step blocks.
- Half of sequence starts are true episode starts and half are random mid-episode slices. This is meant to prevent the memory state from becoming a simple clock that counts writes since reset.
- A train-only binary probe asks the memory to classify the banana side after the reveal, providing denser retrieval supervision than the rare final decision frames alone.
- Both v3 and v3.1 start directly from the official `pi05_base` parameter checkpoint. Their new
  memory and probe parameters are freshly initialized with the same seed; no v2-trained memory
  weights are inherited.

The total training objective is:

\[
\mathcal{L} = \mathcal{L}_{\text{flow}} + \mathcal{L}_{\text{causal CE}} + 0.5\,\mathcal{L}_{\text{probe}}.
\]

An important implementation detail is that the flow-matching action loss is insulated from the VLM and memory with `stop_gradient` on the prefix KV cache. Therefore:

- the action flow loss trains the action expert and action projections;
- the causal subtask+FAST cross-entropy trains the VLM and memory path;
- the probe loss trains the probe head, VLM/memory path, and learned memory content gate.

The repository contains a completed π₀.₅ baseline, earlier v1/v2 memory experiments, and pre-RTC
v3 checkpoints through step 18,000. Those checkpoints are parameter-tree compatible with the new
RTC code but were not trained with clean committed prefixes or the 10-frame cadence. The saved
step-13,000 offline artifact also used stride one and is historical evidence only. A new checkpoint
must be trained with the current recipe before evaluating RTC behavior. V3.1 is a controlled
write-source ablation: it changes no data, RTC, loss, probe, MLP, read-query, or TBPTT setting.

## 2. Research task

### 2.1 Task definition

The high-level prompt is fixed:

```text
find the bin with banana
```

Each demonstration has two labeled phases:

1. `observe bins`
2. either `open left bin` or `open right bin`

The intended task structure is longer than the two textual labels suggest. During `observe bins`, the robot manipulates/inspects the bins and obtains visual evidence about the banana side. The later left/right choice should depend on information that may no longer be visible in the current image.

The central scientific question is not simply whether the model can imitate the trajectories. It is whether **episode-specific information written during inspection causally affects the later bin choice**.

### 2.2 Training data

The raw training set is in `data/bin_memory_banana/`:

- 30 episodes
- 22,705 total frames
- 638–1,002 frames per episode
- mean length: 756.8 frames
- recorded at approximately 30 Hz
- 16 episodes end with `open left bin`
- 14 episodes end with `open right bin`

Each episode contains:

```text
left_joint_positions.npy     [T, 7]
right_joint_positions.npy    [T, 7]
left_control.npy             [T, 7]
right_control.npy            [T, 7]
top_camera_rgb.mp4           640×480 RGB after decoding
left_camera_rgb.mp4          640×480 RGB after decoding
right_camera_rgb.mp4         640×480 RGB after decoding
subtask_labels.json
metadata.json
```

Each arm contributes six arm joints and one gripper value. The converted data therefore uses:

\[
s_t = [q_t^{\text{left}}, q_t^{\text{right}}] \in \mathbb{R}^{14},
\]

\[
a_t = [u_t^{\text{left}}, u_t^{\text{right}}] \in \mathbb{R}^{14}.
\]

The state is the follower robot state. The action target is the leader/teleoperation command.

### 2.3 Dataset conversion

`openpi/examples/yam/convert_yam_data_to_lerobot.py` converts raw demonstrations into the LeRobot dataset:

```text
yam/bin_memory_banana_subtask
```

The converter:

- concatenates the two arm states into 14 dimensions;
- concatenates the two leader controls into 14-dimensional action targets;
- decodes OpenCV BGR video frames and stores RGB images;
- trims every episode to the shortest available state/action/video stream to handle off-by-one recording lengths;
- expands the contiguous subtask segments into one task string per frame;
- skips demonstrations with missing or incomplete subtask labels;
- stores the per-frame subtask in LeRobot's `task`/`task_index` mechanism;
- does not store the high-level prompt, because the prompt is injected by the training transform.

### 2.4 Held-out data

`data/held_out_eval/` contains two raw demonstrations:

- `demo1`: 793 recorded steps
- `demo2`: 597 recorded steps

These folders do not contain `subtask_labels.json`. Consequently, the current offline evaluator can visualize predictions and compare predicted action chunks to recorded controls, but it cannot automatically compute subtask accuracy or decision accuracy from a ground-truth label file.

## 3. Base π₀.₅ policy

### 3.1 Backbone

The v3 configuration uses the standard OpenPI π₀.₅ components:

| Component | Configuration |
|---|---|
| Vision encoder | SigLIP So400m/14 |
| VLM / prefix expert | Gemma 2B width 2,048 |
| Action expert | Gemma 300M |
| Model dtype | bfloat16 by default |
| Model action dimension | 32 |
| Robot action dimensions used | first 14 |
| Action horizon | 50 |
| Context token buffer | 200 slots |
| Memory causal token buffer | 150 slots |

The three 224×224 camera images each yield 16×16 = 256 SigLIP tokens. The VLM prefix therefore begins with 768 image tokens:

```text
256 top-camera tokens
256 left-wrist-camera tokens
256 right-wrist-camera tokens
```

### 3.2 State and action preprocessing

The raw 14-dimensional state and action arrays are normalized using the saved `pi05_yam` normalization statistics. The memory v3 configuration deliberately reuses the baseline statistics because it uses the same dataset and action representation.

The action transform is mixed delta/absolute:

```text
[6 delta arm joints, 1 absolute gripper,
 6 delta arm joints, 1 absolute gripper]
```

For each action chunk, the current state is subtracted from the 12 masked arm-joint dimensions. The grippers remain absolute. At inference, normalization is reversed and the current state is added back to the arm-joint dimensions.

The 14-dimensional state and action are padded to the model's 32-dimensional action space after tokenization. Thus:

- the discrete state text contains the 14 real YAM state values;
- the FAST branch represents the 14 real action dimensions;
- the flow-matching head is trained on 32 dimensions, with the final 18 dimensions padded to zero;
- the output policy returns only the first 14 dimensions.

### 3.3 Baseline knowledge-insulation objective

Before the memory model, `pi05_yam` was extended to predict a textual subtask and a FAST-tokenized action branch in the VLM, while retaining the continuous flow-matching action head.

The training token string has the conceptual form:

```text
Task: find the bin with banana, State: <14 discretized values>;
<subtask>
Action: <FAST action tokens>|<eos>
```

The prompt/state region is bidirectional context. The subtask and FAST action branch are causal next-token targets. The FAST branch is hidden from the continuous action expert so that the action target cannot leak directly into the flow prediction.

The no-memory baseline config is `pi05_yam`; a completed checkpoint exists at step 29,999 under `openpi/checkpoints/pi05_yam/pi05_yam_KI/29999`.

## 4. Titans-style episodic neural memory

The memory implementation is in `openpi/src/openpi/models/memory.py`.

### 4.1 Slow parameters and fast state

There are two distinct types of learned quantities.

#### Slow/outer parameters

These are normal checkpointed model parameters optimized by AdamW across the training dataset:

- key projection `W_K`;
- value projection `W_V`;
- query projection `W_Q`;
- the gate head that produces inner learning-rate, momentum, and forgetting gates;
- the learned initial fast weights `m0`;
- a separate 2,048-dimensional `memory_gate` that scales retrieved memory content before it becomes VLM tokens;
- the train-only two-class probe head;
- the rest of π₀.₅, except parameters explicitly frozen by the train config.

#### Fast/inner state

Every sequence or live robot episode has its own `MemoryState`:

```text
fast_weights: one batched copy of the memory MLP weights per sample
momentum:     one matching momentum tensor per fast-weight tensor
```

This state is initialized at the learned `m0` at the beginning of each sequence/episode. It is updated by the memory's inner gradient-descent rule, not by AdamW. It is not itself a persistent checkpoint parameter.

With the current default MLP dimensions, the fast-weight MLP has approximately 4.72 million scalar weights per sample. Fast weights plus matching momentum require approximately 37.8 MB per sample in float32, before temporary activations. This is a derived estimate from the current layer dimensions.

The memory-specific slow parameters are approximately 11.03 million scalars when counting the key/value/query projections, inner gate head, learned `m0`, 2,048-dimensional content gate, and two-class probe head. This estimate excludes the much larger π₀.₅ vision, VLM, and action-expert parameter trees.

### 4.2 Memory dimensions

The default `MemoryConfig` is:

```text
d_input     = 2048
d_key       = 512
hidden_dims = (1024, 1024, 1024)
d_value     = 2048
```

The fast MLP therefore has the layer widths:

```text
512 → 1024 → 1024 → 1024 → 2048
```

SiLU is applied after every layer except the output layer.

### 4.3 Key, value, and query construction

Given current hidden tokens

\[
h_t \in \mathbb{R}^{B \times 256 \times 2048},
\]

the model computes:

\[
k_t = \operatorname{L2Norm}(\operatorname{SiLU}(W_K h_t)),
\]

\[
v_t = \operatorname{L2Norm}(\operatorname{SiLU}(W_V h_t)),
\]

\[
q_t = \operatorname{L2Norm}(\operatorname{SiLU}(W_Q h_t)).
\]

Shapes are:

```text
k_t, q_t: [B, 256, 512]
v_t:      [B, 256, 2048]
```

The L2 normalization is implemented with a smooth reciprocal square root so that exact-zero inputs do not produce an undefined norm derivative.

### 4.4 Associative write objective

The current frame is written by fitting the fast MLP `M` to map the frame's keys to its values:

\[
\ell_t(M) = \frac{1}{256}\sum_{j=1}^{256}\left\|M(k_{t,j}) - v_{t,j}\right\|_2^2.
\]

The reported `surprise` is this pre-update associative loss. Because each target value has unit L2 norm, a zero-output fresh memory has surprise approximately 1.0. However, `m0` is trainable. After outer training, a freshly reset checkpoint can have a nonzero and poorly calibrated `m0`, so its first surprise is not guaranteed to remain near 1.0. The existing v3 evaluation artifacts demonstrate this behavior.

### 4.5 Inner update rule

The gate head receives the mean raw hidden vector across the 256 tokens and produces three sigmoid gates per sample:

\[
(\theta_t, \eta_t, \alpha_t) = \sigma(G(\operatorname{mean}_j h_{t,j})).
\]

Their intended initial values are:

```text
theta ≈ 0.10   inner learning rate
eta   ≈ 0.90   momentum retention
alpha ≈ 0.01   forgetting rate
```

The un-clipped Titans-style update is:

\[
S_t = \eta_t S_{t-1} - \theta_t \nabla_M \ell_t(M_{t-1}),
\]

\[
M_t = (1-\alpha_t)M_{t-1} + S_t.
\]

Before this update, the inner gradient is globally clipped to norm 1.0 separately for every sample. The clipping factor is stop-gradient. This prevents unstable second-order derivatives through the norm and bounds the length of each fast-weight update.

The update itself remains differentiable to the outer training loop. When AdamW backpropagates through a sequence, JAX differentiates through the inner `value_and_grad` update, including meta-gradient/second-order paths into the memory projections, `m0`, and upstream hidden representations. These paths extend only within the current TBPTT block. The per-sample clipping scalar is detached, and block boundaries explicitly detach the incoming fast state.

In v3, the gate-head parameters under `memory/gate` are frozen. The intent is to keep the inner-loop write dynamics near the previously measured stable operating point and prevent the outer objective from learning to erase or silence memory before retrieval becomes useful. The separate vector `memory_gate` remains trainable.

### 4.6 Read rule

Reading does not modify the fast state:

\[
r_t = M_{t-1}(q_t),
\]

with

```text
r_t: [B, 256, 2048]
```

Retrieved content is multiplied elementwise by a learned 2,048-dimensional gate:

\[
z_t = g_{\text{content}} \odot r_t.
\]

The content gate is initialized to zero. At initialization, the memory path therefore injects exact zero token embeddings and starts as a content-level no-op. During training, the causal CE and probe objectives can learn both the content gate and the memory read/write projections.

## 5. How v3 obtains and injects memory tokens

### 5.1 Current observation prefix

All three cameras are resized to 224×224. SigLIP produces 256 tokens per view. These 768 image tokens are concatenated in the deterministic dictionary order created by `YamInputs`:

```text
base_0_rgb          top camera        token slots 0–255
left_wrist_0_rgb    left wrist        token slots 256–511
right_wrist_0_rgb   right wrist       token slots 512–767
```

The context tokenizer creates a fixed 200-slot buffer containing:

```text
Task: find the bin with banana, State: <14 discretized normalized values>;
```

Only the actual tokens are valid in the context mask; the remaining slots are padding. The prefix cache reserves all 200 slots, so the static prefix length is:

```text
768 image slots + 200 context slots = 968 slots
```

### 5.2 Hidden-state extraction

The full three-camera/context prefix is run through the Gemma 2B VLM while requesting per-layer hidden states. The v3 config selects layer 8:

```text
memory_layer = 8
```

The code then slices the first 256 token positions:

```python
h_t = hidden[0][memory_layer][:, :256]
```

Thus the positions originate from the top camera. They are not necessarily features of the top camera in isolation: at Gemma layer 8, the prefix uses bidirectional attention, so these positions may already be contextualized by both wrist cameras and the prompt/state text.

### 5.3 One retrieved vector becomes one memory token

The memory is queried once for every one of the 256 current top-camera token positions:

```text
256 current queries → 256 retrieved 2,048-D vectors → 256 memory tokens
```

There is no learned token-count reduction, top-k selection, or external nearest-neighbor store. The token count is fixed at 256, and the historical episode is compressed into the fast MLP weights.

### 5.4 Exact static cache layout

For the current v3 dimensions, the logical layout is:

| Region | Slot range | Length | Contents |
|---|---:|---:|---|
| Images | 0–767 | 768 | three cameras, 256 tokens each |
| Context | 768–967 | 200 | fixed prompt + discretized state + padding |
| Memory | 968–1223 | 256 | gated retrieval vectors |
| Causal | 1224–1373 | 150 | generated subtask at inference; subtask+FAST targets at training |
| Action suffix | 1374–1423 | 50 | noisy action tokens processed by the action expert |

The first generated subtask token is decoded from the output of the last memory token at position 1223. This makes the first causal token directly memory-conditioned.

The full 150-slot causal window is reserved even when the decoded subtask is only a few tokens long. This keeps the action-suffix positions static across samples and training/inference.

### 5.5 Attention geometry and leakage prevention

The attention masks intentionally separate the information sources.

#### Memory-token rows

Each memory token may attend to:

- valid image and prompt/state context tokens whose AR block is zero;
- all 256 memory tokens bidirectionally.

It may not attend to any causal subtask or FAST target token. This prevents the teacher-forced labels from leaking backward into the memory-token representations.

#### Causal-token rows

Each causal token may attend to:

- the valid prefix;
- all memory tokens;
- previous causal tokens and itself through a lower-triangular mask.

#### Action-suffix rows

The continuous action expert may attend to:

- the valid current-observation prefix;
- all memory tokens;
- the subtask tokens;
- its own causal suffix positions as defined by the π₀ action mask.

The action expert may **not** attend to the teacher-forced FAST action tokens. Those tokens encode the same target action and would otherwise cause direct label leakage.

### 5.6 V3.1 post-attention write source

V3.1 keeps the v3 read path unchanged. Let `h_t` denote the 256 layer-8 top-camera-position
vectors, `r_t` the old-memory retrieval, and `m_t = memory_gate ⊙ r_t` the appended memory-token
embeddings. After those tokens attend to the valid current prefix and to one another, the existing
Gemma extension returns 256 final-normalized memory-position outputs `c_t`:

\[
r_t=M_{t-1}(W_Qh_t),
\]

\[
c_t=\operatorname{GemmaAppend}(m_t;KV(\text{images,prompt,state}))_{\text{memory positions}}.
\]

The only v3.1 algorithm change is:

\[
\text{v3: }M_t=\operatorname{Write}(M_{t-1};h_t),
\qquad
\text{v3.1: }M_t=\operatorname{Write}(M_{t-1};c_t).
\]

This is MAC-inspired rather than an exact reproduction of Titans MAC. It implements the paper's
central idea that the long-term memory should receive an attention-contextualized representation,
but it writes only the existing memory-token positions and adds no persistent tokens, sparse
selection gate, projection, or separate attention block. The associative writer still treats all
256 tokens equally. Its key/value projections, fast MLP, momentum, forgetting, clipping, and
inner loss are unchanged.

`c_t` cannot see the teacher-forced causal region because the memory-row mask has zero entries for
all causal columns. It is also produced before the continuous action suffix. Therefore the new
write source contains current observation/context plus retrieved history, but no subtask, FAST, or
continuous-action target leakage. The actual state update remains after prediction, preserving
strict read-before-write behavior.

## 6. v3 sequence data construction

### 6.1 Sequence shape

The v3 config uses:

```text
memory_seq_steps     = 60
memory_stride_frames = 10
action_horizon       = 50
memory_sequence_buckets = (20, 40, 60)
```

For a sampled base frame `f`, the observation steps are:

\[
f, f+10, f+20, \ldots, f+590.
\]

For each step `k`, the action target contains the 50 raw controls beginning at `f + 10k`.
Neighboring chunks therefore overlap by 40 controls. Across all 60 maximum-length steps, the
loader requests `60 × 50` action positions (including the deliberate overlap) and reshapes them
to `[60, 50, 14]` before padding. The observation-start span is nominally 600 frames, and the
final target extends through `f + 639`.

This is 75% of the physical observation-start coverage of the former `16 × 50` recipe, while
exposing the memory to 60 rather than 16 read/write events (3.75 times as many). The reduction
from the superseded T80 RTC recipe deliberately trades late-sequence supervision for lower
training time. Random slices still provide coverage of later portions of long episodes.

### 6.2 Typical collated tensor shapes

With the configured batch size `B=12`, let `T_b` be the selected static bucket length, one of
`20`, `40`, or `60`:

| Tensor | Shape |
|---|---|
| each camera | `[12, T_b, 224, 224, 3]` |
| padded state | `[12, T_b, 32]` |
| action targets | `[12, T_b, 50, 32]` |
| context tokens | `[12, T_b, 200]` |
| causal targets | `[12, T_b, 150]` |
| step-valid mask | `[12, T_b]` |
| TBPTT boundary mask | `[12, T_b]` |
| probe labels/masks | `[12, T_b]` |

LeRobot clamps requested timestamps that cross an episode boundary by repeating the final frame. `seq_step_mask` marks a step valid only when its starting frame is before the episode end. Entire steps whose starts are past the end contribute no loss and perform an exact no-op memory update. A final valid step may still have a tail of its 50-action chunk clamped to the final episode frame.

The LeRobot dataset view fetches the T60 superset. A homogeneous weighted batch sampler first
draws a bucket according to that bucket's total original start-frame probability, then draws all
12 starts inside the bucket with their conditional original weights. This preserves every
start's marginal probability, including the 0.5/0.5 full/slice mixture; it only correlates
sequence lengths within a batch. Worker-side collation crops every temporal tensor before JAX
transfer, so `lax.scan` compiles and executes three static shapes instead of always running T60.
On the current 30-episode dataset, the measured bucket masses are approximately 0.13%, 12.93%,
and 86.95%, giving mean executed `T_b=57.36`. Thus bucketing adds about 4.4% scan-work savings
beyond the much larger 25% maximum-length reduction from T80 to T60. The transform workers still
decode/tokenize the T60 superset; the saving is primarily device transfer and GPU computation.

### 6.3 Future-shifted subtask labels

The label used at a sequence step is taken 15 raw frames in the future:

```text
subtask_lookahead = 15
```

For step `k`, the label index is:

\[
\min(f + 10k + 15,\; \text{episode length}-1).
\]

This is intended to describe the upcoming action chunk rather than only the instantaneous frame.

### 6.4 Full-trajectory and slice sampling

The data loader uses a weighted random sampler over possible sequence start frames:

- total probability 0.5 is distributed uniformly over the 30 true episode-start frames;
- total probability 0.5 is distributed uniformly over allowed nonzero slice starts.

A slice start is allowed only when:

- it is not frame zero;
- at least `memory_min_slice_steps × memory_stride_frames = 20 × 10 = 200` frames remain;
- it is not strictly between the episode's reveal frame and its first task switch.

The reveal-to-decision exclusion prevents an impossible training example: if memory is reset after the banana was revealed but before the left/right decision, the sequence never observed the answer yet would still be graded on the answer.

Random slices intentionally train the learned blank state `m0` to behave competently when a sequence begins mid-episode. This attacks the write-count shortcut, but it also means that a blank-memory ablation can remain surprisingly capable. Therefore, blank-memory performance alone is not a sufficient test that the recurrent writes are unused.

### 6.5 Reveal and close metadata

For each episode, the loader derives:

- episode length;
- the first task-switch frame;
- final banana-side class: left = 0, right = 1;
- reveal frame;
- close frame.

The config points to:

```text
./assets/pi05_yam/reveal_frames.json
```

That file is currently absent. The loader therefore uses its defaults for every episode:

```text
reveal_frame = 300
close_frame  = 450
```

This should be treated as approximate supervision, not manually verified reveal timing.

## 7. Tokenization for v3 training

The memory tokenizer separates the bidirectional context from the causal labels.

### 7.1 Context buffer

For every step:

```text
Task: find the bin with banana, State: <discretized normalized state>;
```

The normalized state is discretized into 256 bins over the assumed interval `[-1, 1]`. The context receives a beginning-of-sequence token, is left-aligned in 200 slots, and has AR mask zero.

### 7.2 Causal buffer

The separate 150-slot causal buffer contains:

```text
<subtask>\nAction: <FAST tokens>|<eos>
```

Every valid causal token is a next-token cross-entropy target. A parallel `causal_fast_mask` marks which causal tokens belong to the FAST action branch, so those positions can be excluded from the continuous action expert's view.

At inference there is no subtask or action label, so only the context buffer is created. The subtask is then generated greedily by `sample_with_memory`.

## 8. v3/v3.1 per-step training algorithm

Every training sequence begins with a fresh batched fast state initialized from `m0`. The
implementation uses a `jax.lax.scan` over the selected 20-, 40-, or 60-step static bucket.

For each step `k`:

### Step 1: apply a TBPTT boundary if requested

If `seq_block_boundary[:, k]` is true for a sample, the incoming fast weights and momentum are wrapped in `stop_gradient`. Their numerical values pass forward unchanged, but later losses cannot backpropagate into earlier blocks.

### Step 2: encode the current observation

The three images and context are prefetched through the VLM. The code saves the KV cache and extracts the 256 layer-8 top-camera-position hidden vectors `h_k`.

### Step 3: read memory before writing

The current `h_k` produces queries. The current fast MLP state returns 256 retrieved vectors, which are multiplied by `memory_gate` and appended as memory tokens.

This matches deployment: the policy cannot use the current frame's write to answer the current frame's question.

### Step 4: teacher-force the causal subtask+FAST segment

The memory tokens and 150 causal-token embeddings are appended to the cache in one extension. The CE prediction states are:

- the last memory-token output for the first causal target;
- each previous causal-token output for subsequent targets.

The CE is averaged over valid causal tokens within each step.

### Step 5: train the action expert with sequence action forcing

For every step and sample, the model independently samples:

\[
\epsilon \sim \mathcal{N}(0,I),
\]

\[
t \sim 0.999\,\operatorname{Beta}(1.5,1) + 0.001.
\]

It constructs:

\[
d \sim \operatorname{UniformInteger}\{0,1,\ldots,6\},
\]

then assigns an action-token-specific flow time:

\[
t_i = \begin{cases}
0 & i < d,\\
t & i \ge d.
\end{cases}
\]

The noisy input is:

\[
x_{t,i} = t_i\epsilon_i + (1-t_i)a_i.
\]

Thus the first `d` actions are an exact clean committed prefix, while the remaining actions are
noised normally. The target flow remains:

\[
u_t = \epsilon-a.
\]

The action expert predicts `v_t`. Prefix positions are excluded from the loss and the suffix is
renormalized so different delays have the same overall scale:

\[
\ell_{\text{flow},k} =
\frac{50}{50-d}\operatorname{mean}_{i,\text{dim}}
\left[\mathbf{1}_{i\ge d}\|v_{t,i}-u_{t,i}\|_2^2\right].
\]

The π₀.₅ action expert receives the token-wise times through token-wise AdaRMS conditioning.
The action expert attends to the current prefix, memory tokens, and ground-truth subtask tokens.
It does not attend to FAST action targets. The entire prefix/causal KV cache is stop-gradient for
this pass, implementing knowledge insulation.

### Step 6: write the current frame

After both predictions, the selected representation is written into the fast memory:

- v3 (`memory_write_source="raw_hidden"`) writes `h_k`;
- v3.1 (`memory_write_source="post_attention"`) writes `c_k`, the 256 memory-token outputs
  already computed in Step 4.

No new attention pass is added. For a padded sequence step, a `where` selects the old state
exactly, preventing even forgetting decay from ticking on padding.

### Step 7: apply the optional post-write quiz

If the step is marked quizzable, the updated memory is read again using the current `h_k`. The gated retrieved vectors are mean-pooled across 256 tokens and passed through a two-class linear head.

The quiz is active only if:

- the step is valid;
- its frame is at or after the reveal;
- the reveal occurred at or after this sequence's start, meaning this sequence had a chance to write the evidence;
- the episode has a recognized left/right final label.

Steps before `close_frame` are marked `visible`; later steps are marked `hidden`. Both contribute to probe training. The visible/hidden flag only splits the logged accuracy metrics.

## 9. Loss aggregation and gradient routing

For each sample, flow and causal CE are averaged over valid sequence steps. At the trainer level:

\[
\mathcal{L}_{\text{flow}} = \operatorname{mean}_{B}(\text{per-sample valid-step flow mean}),
\]

\[
\mathcal{L}_{\text{CE}} = \operatorname{mean}_{B}(\text{per-sample valid-step CE mean}).
\]

The probe loss is aggregated over all active quiz positions in the batch:

\[
\mathcal{L}_{\text{probe}} =
\frac{\sum_{i,k} \mathbf{1}_{\text{quiz}_{i,k}}\,\operatorname{CE}(y_{i,k},\hat y_{i,k})}
{\max(1,\sum_{i,k}\mathbf{1}_{\text{quiz}_{i,k}})}.
\]

Because each bucket batch is homogeneous in sequence length, this batch-level ratio sees
bucket-conditional groups rather than independently mixed lengths. The start-frame marginal and
the flow/CE gradient expectations are preserved, but the nonlinear probe ratio can have different
variance and a slightly different finite-batch expectation. This is an explicit batching
tradeoff of the speed recipe, not a claim of bitwise equivalence to padded mixed-length batches.

The final loss is:

\[
\mathcal{L} = \mathcal{L}_{\text{flow}} + 1.0\,\mathcal{L}_{\text{CE}} + 0.5\,\mathcal{L}_{\text{probe}}.
\]

Verified gradient routing in `scripts/check_memory_train.py` is:

| Objective | Receives gradients | Explicitly insulated |
|---|---|---|
| Flow | action expert, action input/output/time projections | VLM prefix and memory path |
| Causal CE | VLM, memory projections, fast-weight initializer `m0`, learned content gate | action expert |
| Probe CE | probe head, VLM/memory path, `m0`, learned content gate | action expert |

Thus the statement “memory is trained by the action loss” is not accurate for current v3 if “action loss” means the continuous flow loss. Memory is trained mainly by the causal subtask+FAST CE and auxiliary probe objective.

## 10. Truncated backpropagation and rematerialization

### 10.1 Twenty-five-step gradient blocks

The config uses:

```text
memory_block_steps = 25
```

Each sample independently draws a random shift in `{0,…,24}`. Boundaries occur every 25 steps
according to that shift, never at step zero. At the 10-frame cadence this preserves the former
250-frame gradient horizon (`25 × 10 = 5 × 50`). The random shift prevents the same episode phase
from always being separated by a gradient boundary.

At a boundary, only the gradient is cut. The complete fast-weight and momentum content crosses the boundary unchanged. Therefore the forward predictions are bit-identical with or without boundary flags; only credit assignment changes.

### 10.2 Per-step rematerialization

The entire scan step is wrapped with `jax.checkpoint`. During backward, step activations are
recomputed rather than retained across the selected sequence bucket. This keeps activation
memory closer to one step per sample instead of growing linearly with sequence length.

The fast memory content remains recurrent across all steps in the selected bucket, up to 60.
Rematerialization is an activation-memory optimization; it is separate from TBPTT, which
deliberately truncates gradients.

## 11. Image augmentation

Sequence samples use a custom augmentation path that folds batch and time, then augments every frame independently.

For the top camera:

- random crop to 95% of width/height;
- resize back to 224×224;
- random rotation in approximately `[-5°, 5°]`;
- color jitter with brightness 0.3, contrast 0.4, saturation 0.5.

For wrist cameras:

- color jitter only with the same parameters.

Augmentations are independently sampled across frames rather than held constant across an episode sequence.

## 12. Training configuration and checkpoint lineage

The paired configs are `pi05_yam_mem_v3` and `pi05_yam_mem_v31`. They are identical except for
the write-source row:

| Setting | Value |
|---|---:|
| Batch size | 12 |
| Maximum sequence steps | 60 |
| Static sequence buckets | 20 / 40 / 60 |
| Replan/write stride | 10 raw frames |
| Action horizon | 50 controls |
| Neighboring-target overlap | 40 controls |
| RTC simulated delay | inclusive 0–6 controls |
| TBPTT block | 25 policy steps / 250 raw frames |
| Minimum random slice | 20 policy steps / 200 raw frames |
| V3 write source | `raw_hidden` (`h_t`) |
| V3.1 write source | `post_attention` (`c_t`) |
| Full/slice probability | 0.5 / 0.5 |
| Probe weight | 0.5 |
| CE weight | 1.0 |
| Training steps | 20,000 |
| LR warmup | 200 steps |
| Peak LR | 5e-5 |
| Cosine schedule nominal decay length | 30,000 |
| End LR | 5e-5 |
| AdamW β₁ / β₂ | 0.9 / 0.95 |
| AdamW ε | 1e-8 |
| Weight decay | 1e-10 |
| Outer global gradient clipping | 1.0 |
| EMA decay | 0.999 |
| Data workers | 12 |
| FSDP devices | 1 by default |

Because peak and end learning rates are both 5e-5, the configured cosine “decay” is effectively a warmup followed by an approximately constant learning rate.

Both v3 and v3.1 start directly from:

```text
gs://openpi-assets/checkpoints/pi05_base/params
```

Both configs use `PartialCheckpointWeightLoader`: every parameter present in `pi05_base` is loaded
1:1, while the additional Titans memory, content gate, and probe parameters retain their fresh
seed-controlled initialization. With the same seed, v3 and v3.1 therefore begin with identical
parameter values and fresh optimizer states; only the runtime writer selection differs. No v2
memory training is inherited. `--resume` has different semantics: it restores the optimizer and
step of an already existing experiment and should only continue the same config/run.

V3.1 adds no trainable parameter. Its checkpoint tree is therefore shape-compatible with v2/v3,
but a v3 checkpoint does not retrospectively become post-attention-trained merely by loading it
under the v3.1 config.

Checkpoints save every 1,000 steps and keep every 5,000-step checkpoint permanently. With EMA enabled, the exported `params` tree used by serving/evaluation contains EMA parameters; the non-EMA train state is stored separately.

Current local checkpoint state:

```text
baseline pi05_yam:          final checkpoint 29999
retired memory v2 run:      checkpoints through 19000
pre-RTC memory v3 run:      checkpoints through 18000
current RTC v3 config:      runtime-dependent; inspect the selected experiment
post-attention v3.1 config: no trained checkpoint documented in this report
configured RTC final step:  19999
```

## 13. Inference algorithm

`Pi0.sample_with_memory` performs one fused policy call.

### 13.1 Subtask generation

After prefix prefill and memory-token append, the first subtask token is decoded greedily from the last memory-token output. Subsequent tokens are generated autoregressively through the preallocated causal cache.

Generation stops on either:

- the tokenizer token corresponding to the trailing newline used during subtask training;
- the PaliGemma EOS token;
- `max_decode_steps`, default 10.

The generated token and mask buffers are returned in the inference auxiliaries and decoded to a string by the server/evaluator.

### 13.2 Action generation

The action suffix attends to the valid prefix, all memory tokens, and the generated subtask. The model initializes a Gaussian-noise action tensor of shape `[B, 50, 32]` and performs 10 Euler flow-denoising steps by default:

```text
dt = -1 / 10
x ← x + dt × predicted_flow
t ← t + dt
```

The resulting normalized delta action is unnormalized, converted back to absolute arm-joint targets, and truncated to 14 YAM dimensions.

When the asynchronous client supplies an `action_prefix`, it contains the valid remaining
robot-space actions from the old chunk, a `prefix_length`, and the estimated committed `delay`.
The server validates the inclusive 0–6 delay contract and runs those raw absolute actions through
the same delta-action, normalization, and padding transforms as the observation's new target
frame. During every Euler step, the model replaces the first `delay` noisy action tokens with the
transformed old actions and sets their token-wise flow time to zero. It restores them once more
after integration, making the returned committed prefix exactly equal to the still-executing old
chunk.

### 13.3 Read-before-write order

The memory update happens only after subtask and action prediction. V3 writes the same
current-observation hidden vectors captured during prefix prefill; v3.1 writes the post-attention
memory-token outputs that were computed before causal decoding. The call returns:

```text
actions
new_memory_state
generated subtask tokens/mask
surprise
inner gradient norm
theta, eta, alpha
```

## 14. WebSocket serving and robot execution

### 14.1 Server state

`scripts/serve_yam_memory.py` wraps the model in a stateful `MemoryPolicy`:

- a single batch-1 `MemoryState` is stored on the server;
- a lock serializes inference and reset operations;
- every image-bearing inference request performs one read/prediction/write and increments a write counter;
- `{"reset_memory": true}` recreates the fast state from checkpointed `m0` and resets the counter;
- a bare reset request returns without performing inference;
- checkpoint parameters are restored as float32 because the inner update was validated in float32.

Server metadata includes the canonical `config_name` and `memory_write_source` in addition to the
action horizon, RTC delay semantics, and training stride. Startup logs print the same config/write
source next to the memory layer and content-gate norm. The command-line default intentionally
remains `pi05_yam_mem_v3`; a v3.1 checkpoint must be selected explicitly so a nonexistent or
untrained v3.1 checkpoint is never assumed.

Each response includes:

```text
actions
subtask
surprise
writes
gates: theta, eta, alpha
policy_timing.infer_ms
```

The memory state is server-global, not keyed by client or episode ID. Multiple concurrent clients would share/corrupt one episode state even though calls are locked. Current deployment assumes one client and explicit reset at every episode boundary.

### 14.2 Robot client

`examples/yam/client_memory.py`:

- connects to the WebSocket server;
- reads both YAM arms over CAN;
- reads the top and two wrist RealSense cameras;
- uses `RealtimeActionChunkBroker` to start the next request after 10 controls while continuing
  to execute the previous 50-action chunk;
- sends the old chunk's right-padded remaining 40-action overlap as the RTC prefix;
- measures executed-step delay, skips the corresponding returned prefix positions on swap, and
  blocks at six steps rather than exceeding the checkpoint's trained delay range;
- displays the subtask, surprise, and write count;
- records an overlaid H.264 video when enabled;
- supports keyboard reset (`r`) and quit (`q`);
- clamps the maximum per-control-step joint delta for safety;
- resets both the broker's cached chunk and server memory at episode start.

One server call equals one memory write. Requests therefore occur at logical frames
`0, 10, 20, …`, matching `memory_stride_frames=10`; the 50-action horizon is an overlap/safety
buffer rather than the write interval.

## 15. Evaluation methodology

### 15.1 Structural and equivalence checks

`scripts/check_memory_train.py` contains two levels of validation.

#### CPU-safe checks

The default checks validate:

- training-time teacher forcing reproduces inference-time generated tokens step by step;
- CE from the sequence loss matches an independently constructed per-step oracle;
- losses are finite and have expected shapes;
- padded steps do not affect the forward result or memory state;
- gradient routing matches the intended knowledge-insulation design;
- TBPTT fences block gradients at the correct per-sample steps;
- TBPTT fences do not change forward values;
- quiz scheduling, visible/hidden accounting, and gradient routing;
- sequence shapes, randomized fences, reveal/slice rules, and inference passthrough;
- separate context/causal tokenization and causal FAST masks.

#### Real-data GPU check

With `--real`, the script loads a real v3 batch, grafts a checkpoint into the model, compiles loss and gradients, checks all losses/gradients for finiteness, and confirms nonzero gradients in:

- the memory subsystem;
- probe head;
- VLM attention;
- action expert.

On 2026-08-09, the now-superseded `T=80` graph passed a real batch-4 single-H200 diagnostic: steady loss
and gradient calls took 7.68 and 33.21 seconds, all losses were finite, and gradients were finite
and nonzero in memory, probe head, VLM attention, and action expert. A separate two-H200 FSDP smoke
used the full global batch of 12 and completed two optimizer updates plus an Orbax checkpoint save;
the second train step took approximately 49 seconds. That smoke predated the final initialization
lineage correction and used a parameter-compatible v2 checkpoint, so it validates graph execution,
FSDP, memory capacity, and checkpointing rather than the new default initializer. The current
configs now load `pi05_base` directly; production runs must start fresh rather than resume that
discarded smoke. Those timings are historical T80 evidence and must not be presented as measured
T60 bucket timings; the new recipe requires a separate GPU benchmark.

### 15.2 Offline raw-episode replay

`scripts/eval_yam_mem_subtask_raw.py` evaluates a raw held-out demonstration without converting it to LeRobot.

For every selected raw frame:

1. Load the 14-dimensional follower state.
2. Load the top and two wrist RGB frames.
3. Apply the config's YAM input, normalization, resize, prompt, tokenization, and padding transforms.
4. Call `sample_with_memory` with deterministic per-frame folded RNG.
5. Optionally keep or discard the returned memory state according to the ablation.
6. Decode the predicted subtask.
7. Unnormalize the action chunk and convert delta arm actions back to absolute targets.
8. Record surprise and latency.

Outputs under `openpi/scripts/eval_results/` are:

- MP4: every raw top-camera frame with held subtask prediction, surprise, timeline cursor, and memory-write marker;
- PNG: all 14 predicted action trajectories overlaid on recorded teleoperation controls;
- NPZ: prediction frames, surprise curve, and call latency;
- TXT: frame, surprise, and predicted subtask per policy call;
- stdout summary: run-length-compressed subtask timeline, content-gate norm, inner gates, and latency statistics.

The action overlay is an open-loop behavior-cloning comparison. It does not demonstrate closed-loop task success because later recorded observations come from the demonstrator's actions, not from the policy's predicted actions.

### 15.3 Offline ablations

The evaluator currently implements two controls.

#### `--ablate-memory`

The model computes each write, but the returned fast state is discarded. Every next prediction reads the learned blank state `m0`.

This isolates the contribution of recurrent within-episode writes, but it does **not** remove:

- the learned blank prior `m0`;
- the memory-token positions and attention computation;
- current-observation information that memory-token rows can receive by attending to the prefix;
- competence explicitly trained by random blank-memory slices.

#### `--zero-gate`

Writes are threaded normally, but the 2,048-dimensional content gate is forced to zero. The memory read contributes zero input embeddings.

This removes retrieved memory content while preserving the memory-token slots and their attention to the current prefix. It is therefore a strong content ablation, but not identical to removing the entire memory-token block from the architecture.

### 15.4 Closed-loop robot evaluation

The robot client executes each returned chunk against the real YAM arms. This is the final task-level evaluation because policy actions determine future observations. A valid closed-loop comparison should reset the server memory before every trial and keep constant:

- initial robot pose;
- banana side;
- camera placement;
- chunk length/write cadence;
- control rate;
- checkpoint and randomization policy;
- number of trials per side.

The principal success metric should be whether the final opened bin is correct. Secondary metrics can include stage completion, time to decision, safety interventions, subtask timing, and surprise dynamics.

## 16. Existing evaluation artifacts and interpretation

The repository contains v3 step-13,000 evaluation artifacts for `held_out_eval/demo1`, with and without recurrent writes.

The normal and `--ablate-memory` predictions agree:

```text
82.6% over all 793 prediction frames
85.8% from frame 400 onward
84.3% from frame 450 onward
```

These agreement values were computed directly from the two saved TXT prediction files.

The saved normal run has:

```text
793 predictions, frames 0–792
steady mean call latency ≈ 82.0 ms
p50 ≈ 84.0 ms
p95 ≈ 91.3 ms
```

The ablated run has a similar latency profile. The normal surprise falls from a very large first-write value to below 0.001, while the blank-state ablation repeatedly reports very large surprise because each observation is compared against the same learned blank state.

These results do not yet prove or disprove useful memory because:

1. The evaluation used `stride=1`, producing 793 writes; its pre-RTC checkpoint had been trained
   with `stride=50`, while the current RTC config uses `stride=10`. It matches neither cadence.
2. The held-out demo lacks machine-readable decision labels.
3. Random-slice training deliberately makes the learned blank state capable mid-episode.
4. Overall framewise subtask agreement is dominated by long `observe bins` and post-decision regions; the scientifically important measurement is the correct left/right choice after the evidence becomes hidden.
5. Offline replay is not closed-loop execution.

The latest local v3 checkpoint is step 18,000, but no corresponding offline result artifact is present.

## 17. Current implementation mismatches and risks

### 17.1 Clean base initialization versus historical checkpoints

The current v3 and v3.1 configs start directly from official `pi05_base` and initialize all added
memory/probe parameters fresh. Historical v2 and pre-RTC v3 checkpoints remain shape-compatible,
but they use different writer/cadence objectives and must not initialize the clean base experiment
or be presented as RTC-trained checkpoints.

### 17.2 Aggressive replanning cadence and latency budget

The current runtime and training contracts are aligned at:

```text
action_horizon            = 50
steps_between_inference   = 10
memory_stride_frames      = 10
simulated_delay           = 0..6 inclusive
```

At 30 Hz, requests and memory writes occur every `10/30 ≈ 0.33` seconds. The asynchronous client
allows at most six old actions (`0.20` seconds) to execute while inference is outstanding and
blocks if the response has not arrived. Real closed-loop evaluation must measure p50/p95/p99
server latency; otherwise this cadence may introduce control stalls even though action continuity
remains safe.

### 17.3 Historical offline evaluator mismatch

The saved offline artifact used stride one and therefore does not evaluate the current cadence.
The evaluator now defaults to `stride=0`, which resolves to the config's 10-frame training stride.
Any explicit `--stride` should normally remain 10 for cadence-matched evaluation.

### 17.4 Missing reveal annotations

`reveal_frames.json` is referenced but absent, so every episode uses frames 300/450 as reveal/close defaults. Incorrect reveal timing can:

- quiz the memory before the banana is actually visible;
- mark visually hidden probes as visible or vice versa;
- exclude or include the wrong random slice starts.

### 17.5 Learned blank-state semantics

`m0` is trainable and random slices always reset memory to `m0`. This makes `m0` both an initial prior and a general mid-episode fallback state. It may encode dataset-level timing or side biases and can reduce the contrast between normal and blank-write ablations.

### 17.6 Current-query dependence of the probe

The binary probe queries memory using the current `h_k`, then mean-pools the retrieved vectors. Probe accuracy therefore reflects both stored content and the current query representation. It is not a query-independent readout of the fast weights.

### 17.7 Training/evaluation lineage

V3 and v3.1 now share the same official `pi05_base` initialization and optimizer budget. A fair
writer ablation must use matched seeds and must compare fresh runs; historical v2/v3 checkpoints
belong to a different lineage. Comparisons against the 30,000-step no-memory YAM baseline still
need to report the different training objective and optimization budget.

### 17.8 One global server memory

The WebSocket server keeps one shared memory state. It does not associate state with an episode or client identifier. Evaluation automation must serialize trials and issue an explicit reset before each one.

### 17.9 Documentation drift

The root `README.md` describes an older `pi0_memory.py`/`train_memory.py` architecture that is no longer the active implementation. The present v3 path lives inside `models/pi0.py`, the standard `scripts/train.py`, and the sequence-aware loader/transforms.

## 18. Recommended rigorous v3 evaluation protocol

### 18.1 First, run cadence-matched offline replay

For both held-out demos and the latest RTC-trained checkpoint, run at `stride=10`:

1. normal recurrent memory;
2. blank-state/recurrent-write ablation;
3. zero-content-gate ablation.

Manually annotate the held-out reveal, close, and correct decision side before interpreting accuracy.

Report at minimum:

- correct left/right choice in a fixed decision window;
- first stable decision frame;
- subtask prediction timeline;
- normal-versus-ablation agreement specifically in the decision window;
- predicted action error in the decision-relevant arm joints;
- content-gate norm;
- surprise curve without assuming surprise is calibrated across checkpoints;
- latency and number of writes.

### 18.2 Add controls that target causal memory use

Useful additional controls would be:

- swap the fast state between a left-banana and right-banana episode at the hidden decision frame;
- shuffle write observations within an episode while preserving write count;
- feed the same current decision observation with memory states from opposite-side episodes;
- reset memory just before reveal versus just after reveal;
- compare a memory state after true reveal frames with one after visually similar non-reveal frames.

The strongest test is a counterfactual state swap: if the current image is held fixed and only the episode memory changes, the predicted left/right decision should follow the memory's episode.

### 18.3 Closed-loop real-robot trials

Use a balanced randomized trial table with equal left/right banana placements. For every trial:

1. place/reset the robot and scene;
2. send a bare memory reset and verify `writes=0`;
3. use a 50-action broker horizon;
4. execute the complete task;
5. record success, opened side, subtask timeline, writes, and surprise;
6. repeat under the selected ablation with matched conditions.

A binary success table over multiple trials per side is more informative than overall offline framewise subtask agreement.

## 19. Corrected commands for the current v3/v3.1 code

Run commands from `openpi/` unless stated otherwise.

### 19.1 Structural checks

```bash
JAX_PLATFORMS=cpu uv run python scripts/check_memory_train.py
```

Real-data GPU check:

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
uv run python scripts/check_memory_train.py \
  --real \
  --config pi05_yam_mem_v3 \
  --ckpt gs://openpi-assets/checkpoints/pi05_base/params
```

### 19.2 Train v3.1 or resume an existing matched run

Start a fresh v3.1 experiment from the configured official `pi05_base` partial loader and a fresh
optimizer/memory initialization:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_yam_mem_v31 \
  --exp-name=pi05_yam_mem_v31_mac_writer
```

Resume an existing v3 run:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_yam_mem_v3 \
  --exp-name=pi05_yam_mem_v3_0730 \
  --resume
```

Do not use `--resume` for a new base-initialized run, and never combine `--resume` with
`--overwrite`.

### 19.3 Cadence-matched offline evaluation

Normal:

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
uv run python scripts/eval_yam_mem_subtask_raw.py \
  --config pi05_yam_mem_v3 \
  --ckpt-dir checkpoints/pi05_yam_mem_v3/pi05_yam_mem_v3_0730/18000 \
  --raw-demo ../data/held_out_eval/demo1 \
  --stride 10
```

Blank recurrent-write ablation:

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
uv run python scripts/eval_yam_mem_subtask_raw.py \
  --config pi05_yam_mem_v3 \
  --ckpt-dir checkpoints/pi05_yam_mem_v3/pi05_yam_mem_v3_0730/18000 \
  --raw-demo ../data/held_out_eval/demo1 \
  --stride 10 \
  --ablate-memory
```

Zero retrieved-content ablation:

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
uv run python scripts/eval_yam_mem_subtask_raw.py \
  --config pi05_yam_mem_v3 \
  --ckpt-dir checkpoints/pi05_yam_mem_v3/pi05_yam_mem_v3_0730/18000 \
  --raw-demo ../data/held_out_eval/demo1 \
  --stride 10 \
  --zero-gate
```

### 19.4 Serve v3

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
uv run scripts/serve_yam_memory.py \
  --config pi05_yam_mem_v3 \
  --dir checkpoints/pi05_yam_mem_v3/pi05_yam_mem_v3_0730/18000 \
  --port 8000
```

Serve a trained v3.1 checkpoint by selecting both its matching config and directory explicitly:

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
uv run scripts/serve_yam_memory.py \
  --config pi05_yam_mem_v31 \
  --dir checkpoints/pi05_yam_mem_v31/<experiment>/<step> \
  --port 8000
```

### 19.5 Cadence-matched robot client

Run from the environment where `gello_software`, `openpi_client`, cameras, and CAN devices are available:

```bash
python examples/yam/client_memory.py \
  --host <gpu-host> \
  --port 8000 \
  --action-horizon 50 \
  --hz 30
```

Use the actual safe control rate supported by the hardware, but record any deviation because it changes wall-clock time between memory writes.

## 20. Source-file map

| Concern | Current source |
|---|---|
| Memory equations and fast state | `openpi/src/openpi/models/memory.py` |
| Memory integration, inference, sequence loss | `openpi/src/openpi/models/pi0.py` |
| Model dimensions and memory flags | `openpi/src/openpi/models/pi0_config.py` |
| Observation/sequence fields | `openpi/src/openpi/models/model.py` |
| Context and causal tokenization | `openpi/src/openpi/models/tokenizer.py` |
| YAM input/output mapping | `openpi/src/openpi/policies/yam_policy.py` |
| Sequence construction and token transforms | `openpi/src/openpi/transforms.py` |
| Episode tables, sequence timestamp fetch, sampler | `openpi/src/openpi/training/data_loader.py` |
| V3/v3.1 configs and optimizer settings | `openpi/src/openpi/training/config.py` |
| Outer train step, loss combination, EMA/checkpoints | `openpi/scripts/train.py` |
| Structural/equivalence tests | `openpi/scripts/check_memory_train.py` |
| Offline raw replay and ablations | `openpi/scripts/eval_yam_mem_subtask_raw.py` |
| Stateful WebSocket serving | `openpi/scripts/serve_yam_memory.py` |
| Closed-loop YAM client | `openpi/examples/yam/client_memory.py` |
| Raw-to-LeRobot conversion | `openpi/examples/yam/convert_yam_data_to_lerobot.py` |

## 21. Compact method statement

The v3/v3.1 system augments π₀.₅ with a per-episode Titans-style fast-weight MLP and train-time
real-time chunking. At each policy step, the model encodes three images and robot state, extracts
the 256 Gemma layer-8 hidden vectors located at the top-camera token positions, queries the
previous fast memory, and inserts 256 gated retrieved vectors as memory tokens between the
current-observation context and a causal subtask segment. The model then decodes a subtask and
predicts a 50-step continuous action chunk before updating the fast memory on an associative
key/value reconstruction loss. V3 constructs the writer keys/values from the raw layer-8 vectors;
v3.1 constructs them from the existing memory-token block's 256 post-attention outputs while
leaving the read queries and every other component unchanged. Training uses homogeneous
20/40/60-step buckets at a 10-frame replan/write cadence, producing 40-action overlap between
neighboring targets. Each step
independently samples an inclusive 0–6 committed prefix, keeps it clean at token-wise flow time
zero, and trains only the renormalized noisy suffix. Recurrent gradients use randomly shifted
25-step TBPTT, preserving a 250-frame physical credit horizon. The recipe retains the 50/50
mixture of true episode starts and random slices and the post-reveal left/right memory probe. The
continuous flow loss is knowledge-insulated from the VLM/memory; the causal CE and probe
objectives train the memory path. Evaluation threads the fast state through raw held-out episodes
or real closed-loop robot trials and compares normal memory against discarded-write and
zero-content-gate controls. The key unresolved question is whether recurrent episode writes
causally change the hidden left/right decision under cadence-matched, labeled, closed-loop
evaluation.
