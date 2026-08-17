# π0.5 Memory v3.1 + Training-Time RTC：当前项目完整交接文档

> 更新日期：2026-08-11（America/Los_Angeles）  
> 代码仓库：`/iris/u/kewalk/memory_project/openpi`  
> 用途：把当前 v3.1 的研究问题、实现、训练方法、RTC、checkpoint、验证结果、真实机器人接口和未解决问题完整交接给另一个 ChatGPT/研究者。  
> 代码基准 commit：`5d83123 feat: implement v3.1 memory training with RTC`

---

## 0. 给接手者的阅读规则

本文刻意区分三种状态：

1. **已经实现并验证**：当前代码已经包含，而且至少有结构测试、单元测试或真实 GPU smoke test 支持。
2. **当前实验事实**：来自正式训练日志和实际 checkpoint，不能把它误写成方案建议。
3. **尚未完成/建议下一步**：还没有 commit，不能声称已经修复。

最重要的当前状态：

- v3.1 已实现、commit，并在 2×H100 上训练到 **step 2500**。
- step 2500 是完整 checkpoint，包含 `params`、`train_state`、`assets` 和 metadata，可继续训练或用于 inference。
- 正式 v3.1 **从官方 `pi05_base` 初始化**，没有从 v2 或旧 memory checkpoint 初始化。
- sequence 最大长度已从 T80 改成 T60，并加入 20/40/60 静态 bucket。
- training-time RTC 已加入：delay 在 `0..6` 个 control steps 内均匀采样；真实运行时用异步 replan 和 hard action prefix。
- 真实机器人第一次连接当前 server 时，在第一次 cold inference/JIT 阶段遇到 WebSocket ping timeout。该问题**已定位，但还未修复并 commit**。
- `origin/main` 仍落后于本地 commit；当前实现尚未 push 到远端。
- 本文不包含集群密码、token 或其他秘密信息。

---

## 1. 我们究竟在研究什么

### 1.1 任务

机器人执行一个需要 episode memory 的双臂 YAM 任务：

1. banana 的位置在 episode 前期可见；
2. 随后左右 bin 关闭，banana 被遮挡；
3. 后期机器人必须根据早期看到的信息选择并打开正确一侧的 bin。

默认 prompt：

```text
find the bin with banana
```

### 1.2 真正的科学问题

我们不是只想证明“sequence loss 下降”或“memory probe accuracy 很高”。真正需要回答的是：

> episode 前期 observation 写入的 fast-weight memory，是否在 banana 被遮挡后，**因果性地改变**机器人后期的 left/right decision 和最终 task success？

因此训练 loss 和 probe accuracy 只能说明系统在拟合某些信号，不能单独证明 memory 被机器人策略真正使用。最终必须有 matched baseline 和 memory intervention。

### 1.3 v3 与 v3.1 的核心对照

v3 和 v3.1 的 dataset、sequence sampling、RTC、loss、probe、TBPTT、optimizer、base initialization 都应保持一致；唯一想研究的变量是 memory 写入表示：

| 版本 | read/query 输入 | write 输入 |
|---|---|---|
| v3 | layer-8 top-camera hidden `h_t` | 同一个 raw hidden `h_t` |
| v3.1 | layer-8 top-camera hidden `h_t` | post-attention contextual representation `c_t` |

v3.1 是一个 **MAC-inspired contextual writer**，但不是 Titans MAC 的逐字复现。我们复用模型本来已经计算出来的 memory-token post-attention 输出，不增加新的 Transformer pass，也不增加新的 writer attention 参数。

---

## 2. 当前正式实验和 checkpoint

### 2.1 配置与实验名

```text
config: pi05_yam_mem_v31
experiment: attnwrite_base_s10_d6_t60_b20-40-60_tb25_bs12_seed42
seed: 42
```

完整 checkpoint 路径：

```text
/iris/u/kewalk/memory_project/openpi/checkpoints/pi05_yam_mem_v31/attnwrite_base_s10_d6_t60_b20-40-60_tb25_bs12_seed42/2500
```

step 2500 下已确认有：

- `params/`
- `train_state/`
- `assets/`
- `_CHECKPOINT_METADATA`

训练为了做 real-robot evaluation 暂停在 step 2500。它不是 crash 后的不完整目录，可以用 `--resume` 继续。

### 2.2 初始化来源：必须说清楚

v3.1 通过下面的 partial loader 初始化：

```python
PartialCheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")
```

语义：

- π0.5 原有参数从官方 `pi05_base` 加载；
- v3.1 新加入的 memory、probe 参数按 seed 42 fresh initialization；
- optimizer state fresh；
- **不加载 v2 memory**；
- **不从旧 v3/v3.1 optimizer state 开始**，除非明确在同一个 v3.1 experiment 中使用 `--resume`。

因此不要把当前实验描述成“v2 fine-tune”。它是“official π0.5 base + fresh v3.1 memory”。

### 2.3 当前日志与数值

日志：

```text
/iris/u/kewalk/memory_project/openpi/training_logs/v31_t60_buckets_seed42.log
```

| step | total loss | CE | flow | probe loss | probe acc | hidden acc | visible acc | pre-clip grad norm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 0.6529 | 0.4755 | 0.0055 | 0.3437 | 0.8978 | 0.9321 | 0.8538 | 2708.8582 |
| 2100 | 0.4383 | 0.1840 | 0.0041 | 0.5003 | 0.7445 | 0.7951 | 0.6796 | 532.3862 |
| 2200 | 0.3436 | 0.1497 | 0.0043 | 0.3793 | 0.8167 | 0.8687 | 0.7501 | 686.0812 |
| 2300 | 0.2861 | 0.1340 | 0.0040 | 0.2961 | 0.9054 | 0.9188 | 0.8881 | 431.1685 |
| 2400 | 0.2557 | 0.1311 | 0.0041 | 0.2411 | 0.8977 | 0.9148 | 0.8760 | 72.3805 |
| 2500 | 0.1981 | 0.1102 | 0.0034 | 0.1689 | 0.9298 | 0.9279 | 0.9321 | 54.0251 |

step 2500 还有：

```text
param_norm=1946.9696
probe_count=374.1600
sequence_bucket_steps=58.0000
sequence_valid_fraction=0.9411
```

日志中的 `grad_norm` 是 clip 前 global norm；optimizer 使用 `clip_gradient_norm=1.0`。step 2000 附近的大 norm 表示 clipping 很活跃，虽然到 2500 已显著下降。不能只看训练 loss 判断真实机器人成功。

---

## 3. 基础 π0.5 模型和输入

### 3.1 Observation

每个 policy observation 使用：

- top/base camera
- left wrist camera
- right wrist camera
- bimanual robot state：原始 14 DoF（左臂 7 + 右臂 7）

图像 resize 到 `224×224`。每个 camera 经 SigLIP 产生 256 tokens，三路共 768 image tokens。

模型内部将 state/action pad 到 32 维：

- raw state/action：14
- model state/action：32
- action 输出经过 output transform 后回到 robot 的 14 维 absolute target

YAM dataset 保存 absolute joint-position targets。输入 transform 对左右臂各 6 个 arm joints 转成相对当前 state 的 delta，两个 gripper 维保持 absolute；随后使用 `pi05_yam` norm stats 做 normalization 并 pad 到 32 维。推理输出依次 inverse-normalize、把 arm deltas 加回当前 state，并只保留前 14 维。RTC committed prefix 也必须走完全相同的 `DeltaActions → Normalize → Pad` 输入管线，不能把 robot-unit absolute actions 直接送进 model-space prefix。

### 3.2 Backbone

- VLM：π0.5 PaliGemma/Gemma 2B 路径
- Gemma hidden width：2048
- Gemma depth：18
- memory hidden capture layer：8
- action expert：Gemma 300M 路径，width 1024
- max prompt/context slots：200
- causal text/FAST buffer：150
- action horizon：50

### 3.3 一个 policy step 的高层数据流

```text
三路 image + state + prompt
          │
          ▼
SigLIP + Gemma prefix
          │
          ├── layer-8 top-camera rows ──> h_t [B, 256, 2048]
          │                                  │
          │                                  └─ query old fast memory M_{t-1}
          │                                                   │
          │                                                   ▼
          │                                       retrieved r_t [B,256,2048]
          │                                                   │
          │                                      trainable content gate
          │                                                   │
          ▼                                                   ▼
current prefix tokens  +  256 gated memory tokens  +  causal tokens
                              │
                              ▼
                      Gemma extension attention
                              │
                  ┌───────────┴───────────┐
                  ▼                       ▼
    memory-position output c_t       subtask/FAST CE
       [B,256,2048]
                  │
                  └── v3.1 writer input; prediction 后 commit M_t

action expert 使用 prefix/extension KV denoise 50-step action chunk
```

---

## 4. Fast-weight memory 的精确定义

### 4.1 Slow parameters 与 fast state

**Slow/outer-learned parameters** 随 AdamW 更新并保存到 checkpoint：

- `W_K`, `W_V`, `W_Q`
- learned initial fast weights `m0`
- memory content gate `memory_gate`
- probe head
- 产生 `theta/eta/alpha` 的 inner gate 参数（当前 frozen）

**Fast/per-episode state** 只在一个 episode 内在线更新：

- fast MLP weights `M_t`
- 与每个 fast weight 同 shape 的 momentum `S_t`

reset 后从 learned `m0` 和 zero momentum 创建 fast state。output layer 在 fresh initialization 时是 zero，但训练后的 `m0` 是 learnable 的，不能假设 trained checkpoint reset 后所有 fast weights/read 都严格为 0。

### 4.2 Fast MLP shape

```text
d_input = 2048
d_key = 512
hidden_sizes = (1024, 1024, 1024)
d_value = 2048

512 -> 1024 -> 1024 -> 1024 -> 2048
```

中间层使用 SiLU，最后一层线性输出。每个 batch sample 有独立 fast weights；fast weights 加 momentum 约为 37.8 MB/sample 的 float32 state。memory inner update/read 全部保持 float32。

### 4.3 K/V/Q 投影

给定 writer input `x_t`：

```math
K_t = \operatorname{L2Norm}(\operatorname{SiLU}(W_K x_t)),
```

```math
V_t = \operatorname{L2Norm}(\operatorname{SiLU}(W_V x_t)).
```

给定 raw query representation `h_t`：

```math
Q_t = \operatorname{L2Norm}(\operatorname{SiLU}(W_Q h_t)).
```

shape：

- `K_t`, `Q_t`: `[B, 256, 512]`
- `V_t`: `[B, 256, 2048]`
- `M(Q_t)`: `[B, 256, 2048]`

read 始终使用 raw layer-8 `h_t` 形成 query。v3.1 只改变 write source。

### 4.4 Associative write objective

```math
\mathcal{L}_{write,t}
= \frac{1}{256}\sum_{i=1}^{256}
\left\|M_{t-1}(K_{t,i})-V_{t,i}\right\|_2^2.
```

实现中对 256 tokens 做 mean、feature 维做 squared-error sum。每 sample 分别求 inner gradient：

```math
G_t = \nabla_{M_{t-1}}\mathcal{L}_{write,t}.
```

inner gradient norm 以 1.0 为上限做 per-sample clipping，clip scalar stop-gradient。

### 4.5 Titans-style momentum + forgetting

gate 输出：

- `theta`：inner learning rate
- `eta`：momentum retention
- `alpha`：forgetting rate

```math
S_t = \eta_t S_{t-1} - \theta_t G_t,
```

```math
M_t = (1-\alpha_t)M_{t-1} + S_t.
```

初始化 bias 对应：

```text
theta ≈ 0.10
eta   ≈ 0.90
alpha ≈ 0.01
```

gate kernel 初始为 0，当前 config 冻结整个 `.*memory/gate.*`，因此主实验中这三个 operating point 基本为常数。这是为了避免早期 CE 通过快速擦掉 memory 得到便宜的优化路径。

### 4.6 Read 与 content gate

```math
r_t=M_{t-1}(Q(h_t)),
```

然后：

```math
\tilde r_t = g_{content}\odot r_t.
```

`memory_gate` 是 trainable 2048 维向量，从 0 初始化。`\tilde r_t` 作为 256 个 memory tokens 加入当前 Gemma extension。

---

## 5. v3.1 contextual writer 到底做了什么

### 5.1 `h_t` 与 `c_t`

Gemma prefix forward 从第 8 层 top-camera token 的 256 个位置取得：

```text
h_t ∈ R^[B,256,2048]
```

系统用 `h_t` 从旧 memory `M_{t-1}` 读取 `r_t`，把 gated retrieval 作为 256 memory tokens 送入 Gemma extension。memory rows 经过 attention 后的最终输出：

```text
c_t = mem_out ∈ R^[B,256,2048]
```

`c_t` 已融合当前 observation prefix、旧 memory retrieval 和 memory-token positions 间的 contextualization。v3.1 在动作和 subtask 预测完成后执行：

```python
write_source = c_t.astype(float32)
M_t = memory.write(M_{t-1}, write_source)
```

### 5.2 参数与 compute

`mem_out` 本来就会计算，因此：

- v3/v3.1 parameter tree 的 key、shape、dtype 完全相同；
- 不增加 Transformer pass；
- 不增加 writer attention layer；
- checkpoint tensor shape 兼容；
- 静态 `memory_write_source` 决定语义。

枚举：

```text
raw_hidden      # v3/default
post_attention # v3.1
```

### 5.3 与 Titans MAC 的关系

v3.1 借用“把 attention contextualized representation 写入 long-term memory”的思想，但不是完全相同的 MAC：

- 没有专门 writer attention；
- 没有 top-k/attention-score write mask；
- 仍对全部 256 positions 做等权 associative objective；
- 复用 π0.5 当前 Gemma extension 的 `mem_out`；
- read query 仍是 raw `h_t`。

最准确名称是 **MAC-inspired post-attention writer**。

### 5.4 Read-before-write

每个 policy step 严格按：

1. 用 `M_{t-1}` 和当前 `h_t` read；
2. 生成 contextual `c_t`；
3. 预测 subtask/action chunk；
4. 完成当前输出后才写入得到 `M_t`。

当前 frame 不能先写自己再立即读自己。

### 5.5 防止 teacher-forced label leakage

训练 causal block 包含 teacher-forced subtask/FAST tokens。memory-row attention mask 限制为：

- 可看当前有效 prefix；
- 可看 memory-token block；
- 不可看后面的 teacher-forced causal/FAST tokens；
- continuous action suffix 在更晚的 action-expert pass 计算，也不能进入 `c_t`。

测试已验证：固定 prefix/state，只改变 teacher-forced causal labels，`h_t`、`c_t`、new memory state 和 write aux 保持相同。

### 5.6 Surprise 语义变化

v3 surprise 是 raw layer-8 representation 的 associative error；v3.1 是 contextual `c_t` 的 associative error。因此不能跨版本直接解释成同一种视觉 novelty，也不能要求同一个 observation 第二次 end-to-end write 的 surprise 必然降低，因为第二次 old-memory retrieval 会改变 `c_t`。

---

## 6. Sequence 数据如何采样

### 6.1 时间单位和 overlap

必须区分：

- raw control frame
- policy sequence step：相隔 10 raw frames
- action horizon：每 observation 监督 50 raw-frame actions

```text
stride = 10 raw frames
action horizon = 50 raw frames
max sequence length = 60 policy steps
```

sequence 从 raw frame `f` 开始时，observation 起点：

```text
f, f+10, f+20, ..., f+590
```

最后一个 50-step target 到 `f+639`。相邻 action chunks 重叠 40 steps。训练 cadence 与真实“执行约 10 步就 replan/write”一致。

换句话说，当前 recipe **不是执行完整 50 actions 后才写一次 memory**。50 是预测/监督 horizon；10 是 observation、replan 和 write cadence。训练 sequence 中每隔 10 raw frames 取一个新 observation 并做一次 read/predict/write，而每个 observation 都监督未来 50 actions，所以相邻监督窗口有 40-step overlap。

### 6.2 Full/slice mixture

总 sampling mass：

- 50% episode-start/full sequence
- 50% 允许的 random slice starts

slice 最短 20 policy steps，即约 200 raw frames。reveal 到 decision 的 dead-zone starts 被排除，避免 sampled sequence 没见过 reveal 却要求 memory 回答后期 label。

每个 policy step 的 subtask label 来自该 episode 的 per-frame LeRobot task table。对 sequence 第 `k` 步，lookup frame 是 `f + 10k + 15`（超出 episode 时 clip 到末帧）；也就是使用 15 raw-frame lookahead 让 textual subtask transition 稍早地预示即将发生的控制阶段。高层 prompt 始终保持 `find the bin with banana`，subtask 则进入 teacher-forced causal target。

### 6.3 Reveal annotation caveat

配置指定 `./assets/pi05_yam/reveal_frames.json`，但当前 repo 中该文件不存在。transform 使用默认边界（现有报告记录为 reveal 约 300、close/decision 约 450）。所以 visible/hidden probe 划分与 dead-zone filtering 目前是近似值，不是逐 episode 人工 annotation。

### 6.4 T80 改成 T60

v3/v3.1 都改成 T60 以保持 writer-only 对照：

- 最大 scan compute 理论减少约 25%；
- context 从最多约 800 raw frames 降为约 600；
- 相同 20k optimizer steps 下，真实数据统计显示有效 transitions 约少 14.2%；
- 默认 reveal≈300、decision≈458 下通常仍覆盖 reveal→decision，但 decision 后监督变少。

若要匹配旧 T80 的有效 transition 总量，估算约需 23.3k T60 updates；不过 LR/EMA 都按 optimizer step 计数，这也是实验选择。

### 6.5 20/40/60 static buckets

dataset fetch T60 superset；homogeneous bucket batch 在进入 JAX/GPU 前把尾部 time axis 裁为 20、40 或 60。一个 `jax.jit` 最多缓存三个静态 executable，`lax.scan` 真正执行对应 T。

为了保持原 sequence-start 边际概率：

1. 选 bucket 的概率等于该 bucket 所有原始 start weights 总和；
2. bucket 内按条件化原 weights、有放回抽完整 batch；
3. 不做均匀 bucket balancing。

30 episodes 的实测 mass：

```text
T20:  0.13%
T40: 12.93%
T60: 86.95%
```

平均执行 T≈57.36，所以 bucket 相对固定 T60 只额外节省约 4.4% scan compute；主要加速来自 T80→T60。

语义 caveat：homogeneous batch 保持单样本边际分布，但 probe loss 是 batch-level `sum CE / active count` ratio，分长度装 batch 会改变其 finite-batch 方差，严格说期望也不保证与 mixed padded batch 完全相同。

### 6.6 Padding

末尾不足 bucket length 的位置 `seq_step_mask=False`：

- 不计 flow/CE/probe loss；
- memory state exact no-op；
- 不执行 forgetting tick；
- padding 不改变后续有效 memory。

---

## 7. TBPTT 如何做

```text
memory_block_steps = 25
```

每个 sample 使用随机 shift 的 block boundary。到 boundary：

- fast memory 数值内容继续传入下一 block；
- incoming fast-state PyTree 经过 `stop_gradient`；
- 前向 episode memory 连续，但 meta-gradient 不跨超过约 25 policy steps。

这不是每 25 步 reset memory：

```text
M_24 数值 ─────────────> step 25 forward 正常使用
     梯度路径 ── stop ──X
```

T60 通常有 2–3 个 blocks。每个 sequence step 还被 rematerialize/checkpoint，以较多 backward recompute 换低 activation memory。

---

## 8. Training-time RTC 的精确实现

### 8.1 目的

真实部署时机器人在 server inference 期间继续执行旧 chunk。新 chunk 返回时，开头若干 action 对应时刻已经过去。RTC 告诉模型：旧计划中这些动作已承诺/在执行，请生成与它们连续的新 50-step chunk。

### 8.2 Delay sampling

```text
simulated_delay = 6
d ~ UniformInteger{0,1,2,3,4,5,6}
```

上限为 inclusive；30 Hz 下 6 steps≈200 ms。每个 `(sequence step, batch sample)` 独立采样。

### 8.3 Flow noise construction

对 50-step target chunk `a`：

```math
t \sim 0.001 + 0.999\,\operatorname{Beta}(1.5,1).
```

前 `d` 个 committed actions：

- time 强制 0；
- 输入 clean target；
- 不加 flow noise；
- 不计 flow loss。

后 `50-d` suffix 正常加噪、训练 velocity。suffix loss 乘：

```math
\frac{50}{50-d}
```

以维持不同 delay 下总 loss scale。

### 8.4 Token-wise conditioning 与 hard prefix

同一 chunk 内 prefix time=0、suffix 有随机 time，所以 AdaRMS 支持 per-action-token time embedding。该路径已通过 tiny π0.5 JIT test。

推理 `ActionPrefix` 包含：

```text
actions
delay
prefix_length
```

每个 denoise step 后都把 prefix 位置精确恢复为 committed action，因此是 hard constraint。

### 8.5 RTC 不改变 memory write

`c_t` 在 action suffix denoising 前由 current observation + old memory 得到。RTC prefix 只作用于 action expert。因此同 observation/old memory 下，不同 `action_prefix/delay/noise` 得到相同 new memory state/write aux。该 invariant 已由测试验证。

---

## 9. 三个 loss 和梯度流

总 loss：

```math
\mathcal L
= \mathcal L_{flow}
+ 1.0\,\mathcal L_{CE}
+ 0.5\,\mathcal L_{probe}.
```

### 9.1 Action flow loss

- 连续 50-step action chunk 的 flow-matching objective；
- RTC committed prefix 不计 loss，只训练 suffix；
- 每 sample 先在 valid sequence steps 上归一化，再做 batch mean；
- action expert 的 prefix/memory KV 在 flow path 上 stop-gradient。

所以当前实现中，**continuous action flow loss 只训练 action expert，不训练 VLM/memory**。memory 是否被 action 使用，主要通过 subtask/FAST teacher-forced CE 建立表征与条件路径，而不是 flow loss 直接给 memory 动作梯度。

### 9.2 Causal CE loss

causal target 主要包括：

- textual subtask
- newline/separators
- FAST action tokens
- EOS

固定 causal buffer 150，loss mask 排除 padding。CE 可以训练 VLM、memory slow parameters、learned `m0`、content gate；不训练独立 action expert。

### 9.3 Memory probe loss

probe 是 left/right 二分类诊断头。当前 step write 完后，用 raw `h_t` query new memory `M_t`，pool 后预测类别。

只有 banana reveal 已经发生在同一个 sampled sequence 内时，该 step 才可 probe，避免从后期 slice 开始却要求 memory 回答从未见过的信息。

```math
\mathcal L_{probe}
= \frac{\sum \text{active quiz CE}}{\max(\sum \text{active quiz count},1)}.
```

visible/hidden 只用于 metric 拆分：visible 是 banana 仍可见，hidden 是 bin close 后不可见；二者没有不同 loss weight。

probe accuracy 不能自动证明 policy 使用 memory：probe 可能读出信息但 action/subtask path 忽略它，也可能学到 dataset shortcut。

### 9.4 梯度路径总结

| loss | 主要训练模块 | 被 detach/不训练的关键模块 |
|---|---|---|
| flow | action expert | VLM prefix KV、memory path |
| causal CE | VLM、memory slow params、m0、content gate | action expert |
| probe | probe、VLM、memory | action expert |

v3.1 相比 v3 新增了更强的 `M_{t-1} → read → c_t → write → M_t` recurrent/meta-gradient 路径；TBPTT boundary 仍在 block 起点截断 incoming fast-state gradient。

---

## 10. Gradient accumulation 和 H100/H200

### 10.1 为什么新增 accumulation

原 global batch=12。2×H200 可直接处理更大 local batch，但 2×H100 80GB 不能容纳相同 direct microbatch。因此新增可选：

```text
gradient_accumulation_steps
```

默认值为 1，所以原 H200 路径没有被删除或改变。

### 10.2 H100 正式设置

2×H100 80GB 正式运行：

```text
global batch = 12
global microbatch = 2
accumulation steps = 6
effective batch = 2 × 6 = 12
```

global microbatch 2 在两张 GPU 上约等于每卡 local microbatch 1。先前尝试 `microbatch4 × accumulation3` 在真实大图 OOM，最终可行的是 `2 × 6`。

### 10.3 正确 accumulation 的语义

它不是每 microbatch 各做 optimizer update，而是：

1. global batch 12 reshape 成 6 个 global microbatches；
2. 用 `jax.lax.fori_loop` 在 device 内累积 gradients/statistics；
3. probe 使用整个 effective batch 的 numerator/denominator，避免平均 microbatch ratios 的 bias；
4. 最后只做一次 global clip、AdamW、EMA、LR step 和 optimizer step increment。

toy tests 已验证 accumulation 3 和 6 与 direct full-batch update 等价（允许正常浮点误差）。

### 10.4 实测硬件

- 2×H200：旧 T80 direct B12 大约 50 s/update；T60/bucket 正式 H200 throughput 尚需单独测，不能把旧数字当新数字。
- 2×H100 80GB：T60+bucket+microbatch2×accum6，steady 约 90–93 s/update。
- H100 real accumulated T60 update 已完成，loss/grad finite。
- XLA memory estimate 约 53.25 GiB，但 JAX preallocation 可接近每卡 80GB，必须避免同卡其他 job。

两卡时 `--fsdp-devices 2` 是单个 2-way FSDP group，没有额外 DP replica group。它降低 parameter/optimizer state 显存，但 repeated collectives 会变慢。

---

## 11. 当前完整训练配置

| 项目 | 当前值 |
|---|---|
| config | `pi05_yam_mem_v31` |
| base initialization | official `pi05_base` |
| dataset | `yam/bin_memory_banana_subtask` |
| prompt | `find the bin with banana` |
| writer | `post_attention` |
| memory layer | 8 |
| action horizon | 50 |
| replan/data stride | 10 |
| RTC delay | uniform integer `0..6`, inclusive |
| max sequence T | 60 |
| buckets | 20/40/60 |
| TBPTT block | 25 |
| full/slice mass | 0.5/0.5 |
| min slice | 20 policy steps |
| subtask lookahead | 15 raw frames |
| batch size | 12 |
| H100 accumulation | 6，microbatch 2 |
| H200 accumulation | 1 by default |
| optimizer | AdamW |
| betas | 0.9 / 0.95 |
| epsilon | 1e-8 |
| weight decay | 1e-10 |
| global grad clip | 1.0 |
| LR warmup | 200 steps |
| peak/post-warmup LR | 5e-5（当前 schedule 实际保持该值） |
| EMA | 0.999 |
| planned updates | 20,000 |
| log interval | 100 |
| checkpoint interval | 500（初期先在 1000 保存，之后改为 500） |
| workers | 12 |
| seed | 42 |

checkpoint 导出的 `params` 使用 EMA params；`train_state` 含 optimizer/current state，用于 resume。

---

## 12. Training commands

从以下目录运行：

```bash
cd /iris/u/kewalk/memory_project/openpi
```

### 12.1 2×H100 80GB：从 step 2500 继续

```bash
CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
HF_HOME=/iris/u/kewalk/.cache/huggingface \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
OPENPI_DATA_HOME=/iris/u/kewalk/.cache/openpi \
.venv/bin/python -u scripts/train.py pi05_yam_mem_v31 \
  --exp-name attnwrite_base_s10_d6_t60_b20-40-60_tb25_bs12_seed42 \
  --batch-size 12 \
  --gradient-accumulation-steps 6 \
  --fsdp-devices 2 \
  --seed 42 \
  --resume
```

不要同时加 `--overwrite`。若用本地 watchdog，先检查 training log 目录中人为创建的 `.stop`；直接运行 `train.py` 不读取它。

### 12.2 2×H200：direct path

```bash
CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
HF_HOME=/iris/u/kewalk/.cache/huggingface \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
OPENPI_DATA_HOME=/iris/u/kewalk/.cache/openpi \
.venv/bin/python -u scripts/train.py pi05_yam_mem_v31 \
  --exp-name attnwrite_base_s10_d6_t60_b20-40-60_tb25_bs12_seed42 \
  --batch-size 12 \
  --gradient-accumulation-steps 1 \
  --fsdp-devices 2 \
  --seed 42 \
  --resume
```

Orbax 可按目标 sharding restore；params/optimizer shape 与 microbatch/T 无关。换硬件会重新 compile，floating-point reduction order 也可能改变，不应要求 bitwise 相同。

---

## 13. Stateful server 如何管理 memory

### 13.1 一次 request 的语义

`scripts/serve_yam_memory.py` 持有一个 server-side `MemoryState`。每次 request：

1. transform raw observation；
2. 用当前 memory read；
3. 预测 subtask + 50 actions；
4. 用 v3.1 `c_t` write；
5. 在 lock 内 commit new state；
6. 返回 actions、subtask、write count、surprise、gates 等 aux。

reset 会重新创建 episode memory，并把 write count 归零。

### 13.2 并发限制

当前只有一个全局 recurrent memory state，并用 lock 保证 RNG/state 原子更新。它不是 per-client memory store：

- real evaluation 建议只连接一个 control client；
- 第二个 client 会污染同一 episode memory；
- 每个新 episode 必须 reset。

### 13.3 Metadata guard

server metadata 包括 config、writer、horizon、RTC max/semantics 和 stride。real client 严格要求：

```text
config_name = pi05_yam_mem_v31
memory_write_source = post_attention
action_horizon = 50
rtc_enabled = true
rtc_max_delay >= 6
rtc_delay_semantics = inclusive_max
memory_stride_frames = 10
```

checkpoint tensor 本身不编码 `memory_write_source`。错误用 v3 config load v3.1 params 会 shape-compatible 并启动，却执行 raw writer，所以 metadata guard 很重要。

### 13.4 当前 server 命令

```bash
cd /iris/u/kewalk/memory_project/openpi

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
HF_HOME=/iris/u/kewalk/.cache/huggingface \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
OPENPI_DATA_HOME=/iris/u/kewalk/.cache/openpi \
.venv/bin/python -u scripts/serve_yam_memory.py \
  --dir checkpoints/pi05_yam_mem_v31/attnwrite_base_s10_d6_t60_b20-40-60_tb25_bs12_seed42/2500 \
  --config pi05_yam_mem_v31 \
  --port 8000
```

`hostname -I` 曾返回 `10.79.12.149 172.17.0.1`。robot client 应连接 `10.79.12.149`；`172.17.0.1` 是 Docker bridge/internal address。

---

## 14. Real-time client 和异步 replan

### 14.1 默认参数

```text
control rate = 30 Hz
action horizon = 50
steps_between_inference = 10
initial_delay_steps = 6
max_async_delay_steps = 6
delay_tolerance_steps = 0
delay_buffer_size = 8
```

nominally 每 10 controls 发起一次 inference/write，即约每 0.333 秒一次。

### 14.2 Broker 工作流程

假设正在执行 old chunk：

1. 执行到第 10 个 action 后，在当前 `infer(obs)` 创建后台 request；
2. 前台继续执行 old chunk；
3. request 携带 old chunk 尚未执行的 suffix，shift 到 prefix index 0，右侧补 0；
4. 用最近 8 次 inference delay 的最大值保守估计 delay；
5. server 生成新 chunk，同时 hard-restore committed prefix；
6. 新 chunk 到达时，按 inference 期间已执行 steps 取正确 offset；
7. 切换到新 chunk。

observation timing race 已修复：request 使用真正达到 replan stride 时的 `obs`，不再由 worker 竞争读取共享 `_latest_obs`。

### 14.3 Backstop

训练只见过最大 delay 6，因此 runtime 不允许执行超过 6 个 unconfirmed steps。到 backstop 若新 chunk 未返回，broker 阻塞等待，不继续执行训练范围外 old actions。

这是安全选择，但若 steady inference 常超过 200 ms，控制会反复 pause。必须实测 `infer_ms`。

### 14.4 Reset、ramp、recording

client 启动：

1. reset server memory；
2. 获取第一个 target 并 smooth ramp；
3. ramp 后再 reset broker+server，正式 episode 从 clean memory 开始；
4. `r` reset，`q` stop；
5. top-camera overlay 可录 H.264。

每步有 `max_joint_delta` clamp。若 clamp 经常生效，发送的 committed prefix 是原计划 action，而 robot 实际执行 clamped action，会造成 RTC mismatch；应记录 clamp rate。

### 14.5 Dry run

至少 25 steps 才确保覆盖一次 async replan：

```bash
cd /home/david/memory_project/openpi
python -u examples/yam/client_memory.py \
  --host 10.79.12.149 \
  --port 8000 \
  --dry-run \
  --dry-run-steps 25
```

早期 dry run 已验证 write count 从 1→2、返回 14 维 finite action；当前 timeout 修复后仍需重跑。

### 14.6 Real client

```bash
cd /home/david/memory_project/openpi
python -u examples/yam/client_memory.py \
  --host 10.79.12.149 \
  --port 8000 \
  --action-horizon 50 \
  --steps-between-inference 10 \
  --initial-delay-steps 6 \
  --max-async-delay-steps 6 \
  --delay-tolerance-steps 0 \
  --hz 30
```

---

## 15. 当前真实机器人 WebSocket 故障：已定位，未修复

### 15.1 现场错误

real client 在第一次 `first_target = policy.infer(...)` 等 server 时退出：

```text
websockets.exceptions.ConnectionClosedError:
sent 1011 (internal error) keepalive ping timeout;
no close frame received
```

同时有 CAN/gravity-comp frequency warning，但 client 尚未进入正式 control/ramp loop。因此直接故障是 WebSocket inference timeout，不是 action broker 或 CAN 算法。

### 15.2 根因

WebSocket async server handler 同步调用：

```python
self._policy.infer(obs)
```

第一次 inference 触发重型 JAX cold compile，阻塞 asyncio event loop，server 无法回复 ping/pong。`websockets` 默认 ping interval/timeout 大约 20 秒；cold compile 更长，client 于是关闭连接。

### 15.3 临时 workaround（尚未实现）

robot-side `websocket_client_policy.py` 的 connect 可加：

```python
ping_interval=20,
ping_timeout=300,
```

它允许更长 cold compile，但不是完整 server-side event-loop 修复。

### 15.4 推荐永久修复（尚未实现）

server handler 应把阻塞 inference 放到 worker thread，例如：

```python
response = await asyncio.to_thread(self._policy.infer, obs)
```

同时应：

- client 设置明确、较长 cold-start timeout；
- server 启动后 synthetic warmup，再允许 robot motion；
- pre-loop first inference/ramp 外层也有完整 cleanup，异常时关闭 camera/CAN/gravity-comp threads；
- 记录 server inference timing 和 broker observed delay；
- warm 后确认 steady inference ≤6 control steps（200 ms）。

### 15.5 再次 real motion 前的顺序

1. 实现 non-blocking handler + cold timeout；
2. dry-run 25 steps，确认 async replan；
3. 检查 write count 1→2、metadata、finite actions；
4. 测 cold compile 和 steady `infer_ms`；
5. 若 steady inference >200 ms，不能直接把 runtime delay 提到 >6，因为训练只覆盖 0..6；需优化 inference 或重新训练更大 delay range；
6. 最后再做 low-risk real trial。

---

## 16. 已完成的验证

### 16.1 Core/model

- RTC/config bounds 与 inclusive `0..D` semantics
- clean-prefix/noisy-suffix construction
- suffix-only loss renormalization
- ActionPrefix validation 和 exact hard restore
- token-wise time embedding/AdaRMS conditioning
- v3/v3.1 parameter tree key/shape/dtype 相同
- writer source selection 和 float32 cast
- teacher-forced label insulation
- memory state/action-prefix independence
- tiny π0.5 `sample_actions` JIT
- tiny `sample_with_memory` JIT
- tiny sequence loss with RTC

### 16.2 Train/infer 与 sequence

- 多步 inference 与 training oracle 的 full fast weights + momentum 对齐
- padded invalid step state exact no-op
- TBPTT boundary 前向 state 连续、跨界 gradient 截断
- variable T 的 probe log 使用 fixed-maxT numerator/count padding，避免不同 bucket 在 host stack 时 crash
- homogeneous bucket batch/crop/sampler correctness

### 16.3 Gradient accumulation

- accumulation 3 与 full batch toy update 对齐
- accumulation 6 与 full batch toy update 对齐
- metrics/probe denominator 跨整个 effective batch
- real 2×H100 microbatch2×6 T60 update 完成且 finite

### 16.4 Server/client

- server RTC prefix transform、shape/range/finite validation
- server metadata
- broker async observation timing、delay estimation、swap offset、reset/error handling
- dry-run replan contract（早期 server/node 成功）

已记录结果：

```text
rtc_test.py: 12 passed
serve_yam_memory_test.py: 12 passed
action_chunk_broker focused suite: passed
config focused tests: passed
data loader local tests: 19 passed
ruff / format focused checks: passed
```

有一次 real-dataset/local diagnostic 在 Hugging Face cache lock/token path 的只读权限处提前失败；它不是模型 assertion failure。环境层问题应与代码测试区分。

---

## 17. 还不能下的结论

当前不应声称：

- v3.1 已在真实机器人上成功完成 task；
- post-attention writer 一定优于 v3 raw writer；
- probe 93% 说明 memory 对 action 有因果作用；
- step 2500 已收敛；
- RTC steady-state latency 一定在 6 steps 内；
- surprise 可跨 v3/v3.1 直接比较；
- reveal/hidden boundary 是每 episode 的精确 annotation。

只能说：实现和训练路径工作、loss/probe 在训练上改善、checkpoint 完整、RTC contract 结构测试通过；最终科学结论仍需 matched evaluation。

---

## 18. 推荐 evaluation 设计

### 18.1 公平 writer ablation

训练 matched v3 baseline：

```text
v3:   memory_write_source=raw_hidden
v3.1: memory_write_source=post_attention
```

两者必须：

- 从相同 official `pi05_base` 初始化；
- memory/probe 使用相同 seed fresh init；
- 相同 T60、stride10、buckets、TBPTT25、RTC D6；
- 相同 dataset starts 和 optimizer schedule；
- 不从旧 v2 memory 继承。

最好至少 seeds 42/43/44，不只比较一个 seed。

### 18.2 Memory interventions

建议至少：

1. `online`：正常 read/write
2. `no_write`：episode 中不 commit new writes
3. `reset_at_hide`：banana 遮挡时 reset memory
4. `shuffle_memory`：episodes/sides 之间交换 state
5. `zero_read`：写仍发生，但送入 policy 的 retrieval 置零

若 memory 真有因果作用，hidden phase 下 `online` 应明显优于 `no_write/reset_at_hide/shuffle/zero_read`，且正确 opened side/task success 一致下降，不仅是 probe accuracy 改变。

### 18.3 每个 trial 应记录

- ground-truth banana side
- opened side
- full task success
- visible/hidden subtask timeline
- server memory write count
- surprise / theta / eta / alpha
- content gate norm
- cold/steady inference latency
- broker estimated/actual delay
- backstop blocking 次数和持续时间
- safety clamp activation rate
- reset 时刻
- video

不要只报 framewise text accuracy；最终打开正确 bin 才是主要行为指标。

---

## 19. 最重要的风险清单

1. **WebSocket cold compile timeout**：已定位但尚未修复。
2. **steady latency vs RTC range**：训练最大 6 steps；超过时 broker 安全阻塞，但控制 pause。
3. **缺失 per-episode reveal annotation**：当前依赖默认约 300/450 frame。
4. **static config/checkpoint mismatch**：params 不编码 writer type，必须使用 v3.1 config + metadata guard。
5. **单全局 server memory**：多 client 会互相污染。
6. **probe shortcut**：高 probe accuracy 不等于 action 因果使用 memory。
7. **frozen inner gate**：当前是稳定常数 operating point，不是 learned data-dependent gate。
8. **v3.1 surprise 语义变化**：现在是 contextual representation fitting error。
9. **bucket objective nuance**：保持 sample marginal，但 homogeneous batch 改变 finite-batch probe ratio 方差/权重。
10. **resume 不 data-exact**：checkpoint 不保存 sampler/iterator cursor。
11. **large pre-clip gradients**：step 2000 左右很大，虽有 clip 且后续下降，仍应监控 NaN/loss spike。
12. **action safety clamp mismatch**：clamp 常触发时，RTC prefix 与实际轨迹可能不一致。
13. **代码尚未 push**：本地 commit `5d83123` 比 `origin/main` 新。

---

## 20. 代码文件地图

| 功能 | 文件 |
|---|---|
| π0.5 config、writer enum、RTC/TBPTT config | `openpi/src/openpi/models/pi0_config.py` |
| model read/context/write、sequence loss、RTC | `openpi/src/openpi/models/pi0.py` |
| fast-weight memory equations | `openpi/src/openpi/models/memory.py` |
| ActionPrefix validation/restore | `openpi/src/openpi/models/rtc.py` |
| Gemma token-wise AdaRMS boundary | `openpi/src/openpi/models/gemma.py` |
| v3/v3.1 training config | `openpi/src/openpi/training/config.py` |
| sequence transform/building | `openpi/src/openpi/transforms.py` |
| weighted starts + homogeneous buckets | `openpi/src/openpi/training/data_loader.py` |
| training loop + gradient accumulation | `openpi/scripts/train.py` |
| stateful v3.1 policy | `openpi/scripts/serve_yam_memory.py` |
| robot client | `openpi/examples/yam/client_memory.py` |
| async RTC broker | `openpi/packages/openpi-client/src/openpi_client/action_chunk_broker.py` |
| WebSocket client（当前 timeout 修复目标） | `openpi/packages/openpi-client/src/openpi_client/websocket_client_policy.py` |
| model/RTC tests | `openpi/src/openpi/models/rtc_test.py` |
| server tests | `openpi/scripts/serve_yam_memory_test.py` |
| broker tests | `openpi/packages/openpi-client/src/openpi_client/action_chunk_broker_test.py` |
| bucket tests | `openpi/src/openpi/training/data_loader_test.py` |
| structural diagnostics | `openpi/scripts/check_memory_train.py` |
| 更长的 v3 历史报告 | `V3_MEMORY_METHOD_REPORT.md` |

---

## 21. Git/worktree 状态

实现 commit：

```text
5d83123 feat: implement v3.1 memory training with RTC
```

上一个远端基准：

```text
03d112d update memory ttt v2
```

当前有未 tracked 的本地 artifacts，包括 training logs、watch script、论文 PDF 和 local known-host file。它们没有进入实现 commit。后续 commit 不要把 secrets、cache、logs 或大 PDF 无意加入。

---

## 22. 接手后的第一批任务

按优先级：

1. 修复 WebSocket server event-loop blocking 和 client cold-start timeout，并加测试。
2. server warmup，dry-run 25 steps，确认 write count、RTC prefix、reset、metadata。
3. 测 cold/steady inference latency，确认 steady delay ≤6 steps。
4. 做第一组低风险真实 robot trials，记录 latency/clamp/backstop/video。
5. 为 30 episodes 添加真实 reveal/close/decision annotation。
6. 实现 evaluation-time interventions，而不是只看 online policy。
7. 训练 matched v3 raw-writer baseline 和多个 seeds。
8. 决定 step 2500 后立即继续到 20k，还是做 checkpoint curve evaluation。
9. push/备份 commit 和实验配置，确保 checkpoint 与 code revision 可复现。

---

## 23. 可直接复制给 ChatGPT 的任务说明

```text
请把这份文档当作当前代码和实验的 handoff context，不要假设尚未完成的部分已经修复。

系统是 official π0.5 base + fresh fast-weight memory 的 v3.1。read 用 Gemma layer-8 top-camera raw hidden h_t；writer 用 old-memory retrieval 经当前 Gemma extension 后的 256 个 post-attention memory-token outputs c_t。read-before-write。训练 stride10、horizon50、T60、bucket20/40/60、TBPTT25，RTC delay uniform 0..6，loss=flow+CE+0.5 probe。正式 seed42 checkpoint 在 step2500。

当前最优先的问题不是重新设计 memory，而是：修复 cold JAX compile 导致 async WebSocket event loop 无法回复 ping、为 client 设置 cold timeout、warmup 并测 steady inference 是否落在训练过的6-step/200ms RTC范围。完成后才继续真实机器人动作测试。提出修改时请严格区分：已实现事实、待实现修复、会改变科学实验定义的新方案。
```

---

## 24. 一句话总结

当前 v3.1 已把 π0.5 episodic fast-weight memory 从“写 raw layer-8 observation hidden”改为“写由旧 memory 和当前 observation 共同 contextualize 的 post-attention memory-token representation”，同时完成 stride10/T60/bucket/TBPTT、training-time RTC、异步 action chunk broker、H100 正确 gradient accumulation 和 stateful server；正式训练到 step 2500，但在真实机器人因果评估前，仍必须解决 cold-start WebSocket timeout、实测 RTC latency，并建立 matched v3/intervention evaluation。
