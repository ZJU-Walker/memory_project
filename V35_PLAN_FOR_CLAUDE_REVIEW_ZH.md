# v3.5 实施方案（中文版参考稿）

版本：Revision 5.1，implementation audit 后补充 portable fresh-base bootstrap 说明。

状态：**已获准按本冻结方案实施，代码实现正在进行；所有放行 gates 通过前不能启动训练。** 另一个会话负责 0816/0830 的标注、人工检查和转换；本工作不会修改那些文件。

## 1. 目标与结论边界

v3.5 将建立一条可以直接验证的记忆链：

- 只在 `inspect both bins`（E，evidence）阶段写入受 prompt 控制的目标侧信息。
- 让记忆经过 `close both lids and reset arms`（O，occlusion）阶段继续保留。
- 在严格静止的 `wait {side}`（D，decision）区间监督读取。
- 通过 causal intervention 分别验证 memory write、retention、read 和下游 use。

存储值的语义明确为：**受 prompt 控制的 target side**，而不是与 prompt 无关的 object layout。因此，在同一图像上切换 banana/gray-box prompt 时，writer 的目标侧表示也应随之改变。

最终结论限定为：**0816+0830 collection 内部 held-out 的 memory mechanism result**。不声称跨 session 泛化，也不能把 open-loop arm-side steering 描述成闭环任务成功。

## 2. 冻结的数据协议

### 2.1 Collections

- 0816：60 个 episodes。
- 0830：8 月 30 日在相同装置、不同日期额外采集的 30 个 episodes：
  - part1：16 个有效完整链；
  - part2：14 个有效完整链；
  - 排除没有 terminal execute 阶段的 `0830_bin_part2/demo14`。
- 合并后的 LeRobot 数据集预期为 **90 episodes**，使用新的版本化名称。
- June-30 collection 保持为独立的 eval-only transfer probe，不合并进训练集。
- 每个 episode 必须记录 stable ID、source path、collection/session ID、prompt、target side、label hash、include/exclude reason 和 split。
- 启动 manifest 使用 schema version 2。每条 included record 还必须包含规范化的 `collection`（`0816` 或 `0830`）、0830 `part`、规范化 object、转换后的 episode index/frame count、人工 E-visibility/contact-sheet provenance，以及独立并带 hash 的 `D_valid` detector sidecar。Manifest 还绑定 block-confound audit 的 hash。训练会重新核验 prompt、label bytes、完整五阶段、task vocabulary/order 和 side consistency，但绝不会根据已封存 final-test state 重新计算 `D_valid`。

### 2.2 标签与边界规则

- E 只能在两个 bin 内的两个物体都已经可见后开始。人工 contact-sheet 审核是最终标准；针对 collection 校准的 detector 只用于初筛。
- 将最后 5 个 raw E frames 排除在 write eligibility 之外。将 **final eligible E anchor** 精确定义为 raw index `<= semantic_E_end - 5`、并且 contact sheet 仍能确认两个物体完整可见的最后一个 sampled frame。这样最后一次 direct commit 来自干净可见帧，而不是 E 到 O 的过渡帧。
- D 从双臂回到 neutral、且目标侧动作尚未开始时起算。
- Memory eligibility 使用完整 **14 维 robot state** 的严格 static core。
- semantic subtask labels 与严格的 `D_valid` sidecar 分开保存。自动 static detector 不能改写相邻语义阶段。
- 已确认的语义错误，包括 0816 episode 26，由标注会话修正；其他边界必须人工确认。
- 转换必须 fail closed：不允许 silent skip；必须是完整五阶段覆盖、固定七项 task vocabulary、prompt/side 一致，episode end 必须匹配最短有效 stream。
- 冻结 manifest 前，检查 0830 每个 part 内的 object 和 side 是否随机交错，而不是按 side 分时段连续采集。
- 分别报告 0816、0830 part1 和 part2 的 E-window 长度与 E-to-D raw-frame gap 分布。

### 2.3 Split

- Train：74 个 episodes，即 0816 的 56 个加 0830 的 18 个。
- Development：8 个 episodes，包括 0816 已有的 4 个 stable-ID held-out episodes，以及 4 个 0830 episodes。新增的 4 条使用 `split_seed=35`，只根据 manifest fields 选择；先满足 object/side coverage，并保证每个 `part × object × side` cell 至少保留 1 条 training episode。Model outputs、images 和 probe results 不能影响选择。
- Final test：0830 中每个 `part × object × target-side` cell 保留 1 条，共 8 条。
- 使用 `openpi.v35.sha256-ranked-manifest-fields.v1` 冻结精确 assignment：稳定 hash ranking 只使用 seed 35、stable ID、part、object 和 target side；先从 0830 每个 cell 选 1 条 final-test，再为每个 object-side pair 选 1 条 development，同时保证每个 0830 cell 至少保留 1 条 train。0816 的 4 个 development stable IDs 保持固定。Manifest 记录 algorithm-spec hash，所有 consumer 都重新计算并核验 assignment。
- Final-test episodes 不能参与训练、normalization、阈值选择、branch 选择或探索性分析。
- Split 封存前可以对 final-test labels 做结构和完整性 QA；封存后，在最终评估前不能查看其 observations 或 derived features。
- Normalization statistics 只能使用 74 个 training episodes 计算。

如果把新增的 30 个 0830 episodes 全部用于训练，就必须明确放弃 fresh full-chain final-test 结论。

### 2.4 非 Gate 的 Fresh-Base Baseline

标注和 split 冻结后，只在 **74 个 training episodes** 上记录一次 FIG1 风格的 step-0 writer baseline，并分别报告 0816 和 0830 strata。共享的 pretrained parameters 来自官方 fresh Pi0.5 base；v3.5 writer/value path 及其 side head 全部新初始化。同时评估初始化后的 online head 和 fresh episode-level out-of-fold probe，不能查看 development 或 final-test observations。

- 该 baseline 只记录初始条件；不设 accuracy threshold，也不能当作 writer 已经迁移或已经工作的证据。
- 它不能改变 augmentation、thresholds、branch selection 或 launch eligibility。
- 它不是旧 runs 的 sweep。必须记录官方 base source、精确 object ID 或 hash、初始化 seed 和生成的 parameter-tree hash。

## 3. Memory Transition Clock 与 Decay

v3.5 的标准 memory clock 为：每 **15 个 raw dataset frames** 执行一次 sampled memory step，与 `memory_stride_frames=15` 一致。下文的“frame transition”都指 sampled memory step，而不是每个 raw video frame。

- 第一个 pilot 使用每个 15-frame memory step 的 `alpha_step = 0.01`。
- 绝不能在每个 raw frame 上应用 `alpha=0.01`。
- 等价的 per-raw-frame rate 是 `1 - 0.99 ** (1/15) = 0.0006698`。
- 如果在线执行以不同的 raw-frame 间隔 `delta_f` 更新 memory，则用以下公式保持相同的物理 decay 速度：

  `alpha(delta_f) = 1 - (1 - alpha_step) ** (delta_f / 15)`

- Sequence buckets、TBPTT boundaries、final-E anchoring 和 reachability 使用 sampled steps 定义，但所有报告也必须给出 raw-frame 时长。
- 将 delay 精确定义为：final eligible E commit 之后、第一次 `D_valid` read 之前发生的 valid sampled transitions 数量。
- 使用 training split 冻结 `n_delay` 的 p50、p90 和 maximum，其中 `n_delay` 的单位是 valid sampled transitions。Retention evaluation 重放真实 transition 数；主要 gate 在冻结的 p90 `n_delay` 上评估，对应 raw-frame gaps 和完整 per-episode 分布只作描述性报告。

### 3.1 Long-Delay Training Windows

独立采样的 training windows **不会**隐式继承前一个 window 的 memory state。为防止 long-delay episodes 静默失去全部 D supervision，训练中明确使用两类 windows：

1. **Natural windows：**普通连续 sampled sequences。它们保留真实 E/O dynamics，并确保 O-phase observations 在 memory 存在时得到训练。只有 final eligible E anchor 与 D steps 能放进 configured bucket 时，natural window 才能监督 D。
2. **Skip-O analytic windows：**运行 1 或 2 个 eligible E anchor steps，得到 post-commit `W_E`，解析跳过无写入区间，再运行 D steps：

   `W_D = (1 - alpha_step) ** n_delay * W_E`

   hidden fast leaves 保持不变，`b3` 与 momentum 保持 0。在 v3.5 output-only、fixed-alpha 规则下，它与逐步重放被省略的 valid non-write transitions 精确等价。

- Analytic decay 使用 FP32。
- 每个 sparse jump 用 `seq_decay_gap_before[t]` 明确表示，即读取 step `t` 前被省略的 valid non-E transitions 数。Scan 顺序固定为 `analytic_decay -> read -> current-step transition`；dense steps 的 gap 为 0，从而避免第一次 D read 的 off-by-one decay。
- 被跳过区间只能包含 valid、non-E、non-writing steps，且不能包含 memory-reset event；违反时 fail closed。
- Skip-O windows 天然 state-valid 且 credit-reachable。
- 每个 training episode 必须至少生成一个有效 skip-O D candidate；失败属于 data/sampling gate，不能 silent drop。
- 在 memory-critical sampling 内，natural 与 skip-O families 目标比例为 50/50，并按 collection/object/side 平衡。Long-delay D supervision 来自 skip-O；natural windows 继续覆盖真实 O frames。
- 如果未来 branch 允许 hidden/bias/momentum 改变或 O 内写入，configuration validation 必须拒绝 analytic skipping。

当前 labels 的初步只读 audit 得到 `n_delay` p50/p90/max 约为 `14/16/20`，所以目前没有 episode 看起来超过 40-step bucket。一个已知 short-D episode 没有合法 same-residue grid。最终 split artifact 才是权威结果：该 episode 必须使用明确的 clock-aware sparse D step、重新标注，或在记录原因后排除 memory supervision；不能静默消失。

## 4. Memory Core 设计

生产路径的 `write_kv` 当前接收 16 对 key/value tokens，因此单个 rank-one 更新不能同时精确拟合全部 16 对。v3.5 pilot 明确将每一帧简化为**一个 pooled association vector**：

1. 分别平均 16 个 projected keys 和 values，再做 L2 normalization，得到 `k_bar` 和 `v_bar`。
2. 只更新 fast memory 的输出矩阵 `w3`。
3. hidden fast leaves 不更新也不 decay；slow/base hidden 参数仍然可以训练。
4. fast `b3` 与全部 momentum leaves 强制保持精确 0。
5. 每个有效 sampled step 都先 read 再做 state transition。因此第一个 E step 读取的是空 memory，不可能发生 same-frame write bypass。
6. Transition 规则：
   - 有效 E step：先 decay `w3`，再执行 direct delta commit；
   - 非 E 有效 step：只 decay `w3`；
   - padding step：严格 no-op。
7. 对 row-vector 实现 `h @ W`：

   `W_dec = (1 - alpha_step) W`

   `r = v_bar - h @ W_dec`

   `W_new = W_dec + delta_rate * outer(h, r) / (h dot h)`

8. Pilot 使用 `delta_rate = 1.0`。这是 direct delta assignment，不是现有 `theta` loss 下的 exact gradient step。
9. delta 模式禁用 drift trust region；保留 state 和 key/value cotangent guards。
10. pooled norm 或 `h dot h` 低于预注册数值下限时必须 fail closed 并生成 telemetry。
11. 该分支的 alpha 固定并 stop-gradient，同时直接测试 read-before-write 的 decay 顺序。
12. 无论周围 model dtype 如何，pooled keys/values、memory 使用的 hidden activations、`h dot h`、fast `w3`、analytic powers/decay、commit residual/outer-product update、16-slot raw reads、它们的 mean、`L_read` 和 calibration 全部保持 FP32。只有最终 pinned memory tokens 进入 Transformer stream 时才允许 cast，并同时报告 pre-cast 与 post-cast injected RMS。

Own-key exact commit 不代表自然的 16 个 query tokens 一定能够成功读取。Query/key alignment、production commit residual 和 real-gap retention 是独立 gates。

如果每帧一个 vector 的容量不足，后续 capacity branch 可以使用 regularized 16-pair least-squares update：

`W += H.T @ inverse(H @ H.T + lambda * I) @ (V - H @ W)`

它不属于第一个 pilot。

## 5. Write 与 Read 训练

- `seq_write_mask`：当前帧、无 lookahead、带 E-tail guard 的 eligibility；只有这些 steps 可以 commit。
- `seq_decision_mask`：严格静止的 D eligibility。
- `L_write`：从未 detach 的 `v_bar` 预测 target side；初始权重 `0.3`。
- `L_read`：先平均 16 个 injection scaling 之前的 raw retrieved vectors，再预测 target side；初始权重 `0.3`。
- 关闭旧的七分类 memory auxiliary loss。
- 保留 detached diagnostic ladder heads，但不向 backbone 传递梯度。
- 精确定义 `L_read_mask = D_valid & valid_frame & read_state_valid`。
- `read_state_valid` 表示当前 read 前至少发生过一次成功、非退化的 eligible E commit。
- **不能**用 `read_credit_reachable` mask `L_read`。它只作为 credit-assignment diagnostic 记录，因为即使较早的 commit 位于 TBPTT 边界之前，`L_read` 仍能训练 `project_q` 和 read head。
- 定义 `reachable_fraction = sum(L_read_mask & read_credit_reachable) / sum(L_read_mask)`，并按 collection、object、side 和 E-to-D gap bucket 报告。
- Final-E anchoring 作为 sampling target。每个受监督的 D sequence 必须 state-valid；条件允许时争取至少一个 credit-reachable E write。
- 对每个包含 `D_valid` 的 training window，将 final-eligible-E anchoring 设为 sampler 硬规则。无效 window 必须 resample；作为额外防线，对任何仍进入 preprocessing 的 state-invalid D step，mask `L_read`、side-bearing subtask CE 和 flow/action loss，并要求其训练计数精确为 0。
- 将 **use-pressure step** 定义为：state-valid D observation 的 transformed action-target chunk 与第一次 side-specific execute motion 重叠。按 episode 和 collection 报告数量。所有 state-valid D steps 都可以训练 flow，但 open-loop side-steering evaluation 只在 use-pressure steps 上定义。
- 使用第 3.1 节定义的 natural 与 skip-O windows。Skip-O path 必须使用 manifest-derived 精确 `n_delay`，不能使用近似 bucket length。
- side losses 先按 episode 聚合，再在 `(collection, object, side)` cells 上做 macro/balanced aggregation。
- gradient accumulation 必须使用精确 numerator/count denominators。
- direct heads 绕过了原 memory key/value clip，因此需要 branch-local feature cotangent cap。

## 6. 时间一致的 Sequence Augmentation

保留现有初始强度和 camera policy，但让随机参数在一个 sequence 内保持时间一致：

- Top camera：95% crop+resize、最多 5 度 rotation 和 color jitter。
- Wrist cameras：不做 spatial transform，只做 color jitter。
- 不做 horizontal flip。
- 同一 sample、同一 camera 的所有帧复用同一组 transform 参数。
- 不同 samples 独立采样 transform。

测试必须验证时间一致性和不同 sample 间的随机性。该 augmentation recipe 必须在 step 0 前冻结。非 gate 的 fresh-base writer baseline 不能改变它；如果一次失败的 pilot 之后确实需要更强 recipe，必须建立单独命名、预注册的新 branch，不能在当前 run 内调整。

## 7. Fresh-Base 初始化与 Injection Calibration

当前代码中的实际公式是：

`memory_tokens = tanh(w) * retrieved * c / max(rms(retrieved), tau)`

实现注意：`tau` 是 RMS floor，不是 tanh gate 的输入。在 v3.5 中，`tanh(w)` 是另一个独立且固定的 per-channel injection gate，不是 pilot 中可学习的参数。初始化和校准规则如下：

1. 共享 pretrained parameters 只能从官方 fresh Pi0.5 base `gs://openpi-assets/checkpoints/pi05_base/params` 加载。run5 或 v3.4 checkpoint 均不参与，并记录解析后的 source object ID 或 hash。每个 cluster 都可以把该官方 asset 独立下载到本机 project-local `v35/cache`；这个 cache 不是必须同步的 transfer artifact。
2. 使用记录过的 seed，全新初始化 memory/query compressors、conditioner、slot embedding、state-null parameter、injection projection、detached ladder heads 和 write/read side heads。Optimizer、EMA 和 global step 也从头开始。
3. 将 `memory_inject_w` 的每个 channel 设置为 `w = atanh(0.5) = 0.549306...`，验证 `tanh(w) = 0.5` 满足 FP32 tolerance，然后在 calibration 前以及整个 pilot 中冻结。任何 missing、overwritten、closed 或 sign-flipped channel 都是初始化失败，不是 conditional reset 情形。
4. Direct commit 后，只重放 **74 个 training episodes** 的真实 E-to-D sequences。Development 和 final-test episodes 不能用于调节 `c`、`tau` 或任何 calibration statistic。
5. 测量单次干净 evidence commit 的 raw-read RMS。Reset memory 只作为 exact-zero sanity check，不能充当 noise distribution。
6. 先取 `tau = median(signal_RMS) / 0.75`，再验证 `a(r) = min(RMS(r)/tau, 1)` 对干净 committed read 的 median 位于 `0.7-0.8`。
7. 使用以下两种真实 noise：（a）query hidden vector 与 stored `h(k_bar)` 的 cosine `<= 0.1` 的 reads；（b）mixed-precision post-commit residual read。如果真实 queries 中 low-cosine 样本不足，则增加预注册的 synthetic orthogonal-query controls，并报告样本不足。
8. 定义 open channels `O = {j: abs(tanh(w_j)) >= 0.1}`。按要求初始化时，`O` 必须包含全部 channels。在 `u = tanh(w) * retrieved / max(RMS_all(retrieved), tau)` 中保留 production 的 all-channel denominator，然后设置 `c = median_train74(RMS_O(layer8_residual)) / median_signal_train74(RMS_O(u))`，并使用相同 indices。若任何 channel 意外关闭或 denominator 接近 0，则 fail closed。
9. 要求真实 noise 的 injected-RMS 与 layer-8-residual-RMS 比值 p95 `<= 0.10`。
10. 在冻结的 p90 `n_delay` 上，要求 episode-median retained signal amplitude `a(r) >= 0.40`；episode p10 只作描述性报告。如果单个 `tau` 不能同时满足 clean-signal、decayed-signal 和 real-noise 条件，则 injection calibration 失败。
11. 为 pilot 冻结校准后的 `c` 和 `tau`。每个 rung 记录 signal/noise raw RMS、full/per-channel injected-to-layer-8 ratios、固定的 `tanh(w)`，以及 pre/post-cast injected RMS。

该初始化方案有意保留官方 Pi0.5 的共享 backbone 和 action-policy parameters，同时让所有 v3.5-specific memory、query 和 diagnostic path 从头开始。因此，不假定 step 0 已经具备任何 memory 能力。在 step 0 和每个 rung 同时运行 writer-dependent direct-carry 与 train-side-prototype oracle controls。它们绕过 memory state/query retrieval，用于判断下游 consumer 能否使用一致、正确 pin 的 side code。必须应用 Gate D 的数值退出规则后才能把 consumer 判定为瓶颈。Consumer-recovery branch 只能重新初始化可分离的 v3.5-specific consumption parameters，或引入小型 memory-specific adapter。**不能**盲目重置全部共享的 layers 9-18，从而破坏 official-base policy。

使用标准 official checkpoint-loading path，并设置显式的 shared-versus-fresh allowlist。任何 shared leaf 缺失或 shape 不匹配，以及任何 v3.5-specific leaf 被意外加载，都必须 fail closed。Initialization manifest 记录 loaded shared leaves、freshly initialized leaves、initialization seed、source hash 和最终 parameter-tree hash。自定义 raw-checkpoint transplant 不属于 v3.5 launch scope。

Calibration 与 step-0 release gates 都依赖完全相同的 fresh parameter tree，因此初始化采用 two-phase、zero-update bootstrap，而不能放松 training gate。Initialize phase 在不构造 data loader、不读取 batch 的情况下创建 audited raw step-0 state。完成 train-only calibration 后，finalize phase 验证同一个 parameter-tree hash，绑定校准后的 `c`、`tau`、`alpha_step`、固定 gate 以及 data/norm identities，并以未更新的 optimizer 和精确初始 sampler/RNG state 保存 checkpoint 0。随后 Gate A、Gate B 和 step-0 Gate C/task health 使用该 finalized checkpoint 生成 pilot authorization。`train.py` 不允许 fresh 启动 v3.5；它必须先验证外部 pilot authorization，再从 finalized checkpoint 0 resume，之后才能读取第一个 batch 或执行第一次 optimizer update。Update 250 及之后的 checkpoints 必须逐字节内嵌该 authorization。

## 8. 放行 Gates

### Gate A：Data

- 人工审核完成。
- 90-episode 计数、source manifest、labels、stable split 和 hashes 全部冻结。
- task vocabulary 和 stable-ID mapping 精确一致。
- 0830 每个 part 内的 object/side ordering 通过 block-confound audit。
- 按 collection 报告 E-window 和 E-to-D gap 分布。
- normalization 只使用 train split。
- 每个 D loss step 都满足 `read_state_valid`；`read_credit_reachable` 只报告比例，不作为 gate。
- 每个包含 `D_valid` 的 training window 都有 final eligible E anchor；state-invalid D side-loss/action-loss 计数精确为 0。
- 按 episode、collection、object 和 side 报告 use-pressure step 数量。
- 经过 5-raw-frame tail guard 和 stride alignment 后，每个 training episode 至少有 1 个 eligible sampled E step，目标至少 2 个。按 collection 报告 min/p10/p50/p90/max。0 eligibility 直接导致 Gate A 失败，不能通过 silent drop 处理。
- 按 episode 和 cell 报告 eligible raw-E frames、eligible sampled-E steps 和 successful commit counts；明确列出只有一个 sampled E step 的 episodes。
- 每个 training episode 至少有一个 skip-O D candidate，并报告 natural/skip-O sample counts 与有效 `n_delay` 分布。

### Gate B：Leakage

在最终 D frames 和最终 preprocessing 上运行 episode-grouped out-of-fold probes。Primary gates 只使用 **74 个 training episodes**；development 结果只作描述，final-test observations 保持封存。先聚合一个 episode 内全部 `D_valid` logits，再得到一个 episode prediction；绝不能把 D frames 当成独立样本。

只预注册两个 primary gating probes：

1. 最终 joint images 加 robot state；
2. step 0 fresh-v3.5-initialization 的 concatenated `k_bar` 加 `v_bar` features。

两个 probes 乘两个 collections 组成一个四项检验 family：

- 在每个 collection 内，先在每个 object 内计算 left/right balanced accuracy，再对 objects 等权 macro-average；
- 在每个 `collection × object` stratum 内打乱 side labels，执行至少 1,000 次 episode-level permutations；
- 每次 permutation 都重新运行完整的 episode-grouped OOF pipeline，不能只打乱固定 predictions；
- 计算 `p_perm = (1 + count(T_perm >= T_observed)) / (B + 1)`；
- 对四项 primary tests 做 Bonferroni correction：每个 observed statistic 都不能高于 null 的第 98.75 百分位，等价于 `p_perm > 0.0125`；
- 在 pooled 74 个 training episodes 上，两个 primary probes 的 episode-macro balanced-accuracy point estimate 都必须 `<= 0.62`；
- confidence intervals 只报告，不能根据 CI bounds gate。

`0.62` pooled point threshold 有意比 `0.60` 更不脆弱：chance、n=74 时，每个 primary probe 的 false stop 约从 4% 降至约 2%。Family-corrected permutation tests 仍是主要 collection-level statistical gate。

Top-only、每个 wrist、all-images-only、state-only、prompt-only、layer-8-only、`k_bar`-only 和 `v_bar`-only probes 仍是必须报告的 descriptive diagnostics，并给出 CIs。单独报告 per-side recall 和 per-cell 结果；单一 side subset 无法定义 balanced accuracy。

如果任何 primary probe 的 family-corrected collection test 或 pooled point-estimate gate 失败，则停止 natural branch。之后只能重新采集，或建立单独命名的 neutralized/scaffold branch。Descriptive probe 的异常需要调查，但不会自动停止。

### Gate C：Mechanism 与 Step 0

- Synthetic FP32 pooled own-key post-commit residual：`<= 1e-5`。
- Production mixed-precision relative commit residual `||r_plus|| / (||v_bar|| + 1e-8)` p95：`<= 1e-2`。
- `delta_rate = 0.5` 时精确闭合一半 residual。
- alpha ordering 和 15-frame clock conversion 正确。
- hidden、bias 和 momentum invariants 成立；padding 严格 no-op；legacy mode 保持 bit-exact。
- Dtype tests 必须证明：在 BF16/mixed-precision model execution 下，fast state、pooling/normalization、memory hidden computation、direct commit、analytic decay 和 raw retrieval 仍保持 FP32。
- 对 `n_delay={0,1,40,100}`，dense repeated decay 与 analytic skip 的 FP32 forward state/read 和 gradients 必须在 relative error `<= 1e-6` 内一致。测试还要验证第一次 D read 发生在当前 D decay 之前，并验证包含非法 E/write/reset/invalid transition 的 skip interval 会 fail closed。
- `L_write` 梯度能到达 value projection/backbone。
- 在 reachable case，`L_read` 梯度能到达 query projection/head 和更早的 E commit。
- 在 unreachable case，`L_read` 仍然训练 query projection/head，但不能跨过 TBPTT boundary。
- State-invalid 与 padding cases 被完全 mask，loss gradient 为 0。
- 第 7 节定义的真实 sequence injection calibration 通过。
- 在 E-only output decay 下，own-key retention 是闭式的：expected cosine 为 1，expected norm ratio 为 `rho_expected = (1-alpha_step) ** n_delay`。在冻结的 p90 `n_delay` 上，要求 cosine `>= 0.9999`、相对 `rho_expected` 的 norm-ratio error `<= 1e-3`，并且 absolute norm ratio `>= 0.55`。如果冻结的 p90 delay 导致 `rho_expected < 0.55`，则必须在 launch 前修改 alpha，不能等待一个数学上不可能的 gate 失败。报告 p50、maximum-delay、对应 raw-frame gaps 和 per-episode 结果。
- 在冻结的 p90 `n_delay` 上，episode-median retained injection amplitude 保持 `>= 0.40`；真实 query-misalignment/mixed-precision noise 低于第 7 节限制。
- Calibration 前验证每个 channel 都满足 `w=atanh(0.5)`，且 `tanh(w)=0.5` 在 FP32 tolerance 内。冻结 `memory_inject_w`；任何 closed 或 sign-flipped channel 都直接导致 Gate C 失败，而不是触发 conditional checkpoint-leaf reset。
- 每个 rung 按 collection/object/side 和 delay bucket 报告真实 D-query geometry。对 16 个 queries 分别计算与 stored `h(k_bar_E)` 的 hidden-space cosine，以及 `beta_j = dot(h(q_j), h_E) / dot(h_E, h_E)`；报告 mean/max cosine、low-alignment fraction、`beta_mean`、`mean(abs(beta))`、cancellation ratio、sign consistency、`cos(mean_raw_read, v_bar)`，以及 anchor-predicted 与 actual raw-read residual。Own-key retention 验证的是实现；learned writer quality 和 D-query alignment 才是学习得到的 links。
- step 0 记录 oracle injection 与 action-to-memory attention，作为 consumer-path diagnostics；不能因此查看 final-test 数据。

### Gate D：1k Pilot

- 只运行一个 1,000-update main pilot，不在 run 内做参数 sweep。
- 固定保存 0、250、500、1,000 completed updates 的 checkpoints。
- 不能通过最大化 8 个 development episodes 的 accuracy 选择 checkpoint。Rungs 使用预注册的 episode/cell-macro mechanism metrics 判断；1k endpoint 是固定的 pilot 决策点。
- 每个 rung 都记录 production commit residual、real-gap retention cosine/norm ratio、raw-read RMS、injected-token RMS 和 reachable fraction。
- 在每个 episode 内先聚合 eligible E、D 或 use-pressure frames，然后为每个 episode 和 condition 生成一个 hard outcome。下面所有 development thresholds 都使用 episode counts。
- 使用八个 matched conditions/measurements 分离 read 与 use：
  - natural memory；
  - reset memory；
  - opposite-side donor memory；
  - zero injection；
  - oracle direct-carry injection：取同一个 episode 的 final eligible E `v_bar`，使用冻结的 pin 后直接注入，绕过 memory state 和 query retrieval；
  - train-side-prototype oracle：注入 requested side 对应的冻结 mean training `v_bar`；training-set diagnostic 使用 leave-one-episode-out，从而分离 consumer 能力与单个 episode 的 writer 质量；
  - opposite-side prototype oracle-donor：注入另一侧 prototype，并测量 consumer 是否跟随 donor；
  - D 阶段 action-expert/frame-token queries 对 memory-token keys 的 attention mass。

每个 side prototype 先在 episode 内平均 eligible `v_bar`，再在 side 内平均 episodes 并做 L2 normalization。Training diagnostics 使用 leave-one-episode-out prototypes，每个 rung 记录 prototype artifact hash。Correct-side 与 opposite-side prototype directions 都 pin 到校准后的 median clean-read **injected** RMS，绝不能使用 raw `v_bar` magnitude 注入。

1k 时的 development 硬阈值：

- prompt-bound writer claim：在 74 个 training episodes 的 FIG1 episode-level OOF protocol 下，natural-prompt 和 counterfactual-prompt writer side accuracy 分别 `>= 0.90`；development writer 至少正确 `7/8`。它只 gate writer claims，不阻断 natural-prompt mechanism chain；
- read head：natural 至少正确 `7/8`，reset target-side 最多正确 `4/8`，opposite-memory donor 至少跟随 `7/8`；
- open-loop action use：native 至少正确 `7/8`，reset target-side 最多正确 `4/8`，opposite-memory donor 至少跟随 `7/8`；
- zero injection：target-side 最多正确 `4/8`，并且 paired predicted-side verdict 与 reset 最多在 `1/8` episode 上不同。另外报告但不单独硬 gate `macro(s_zero - s_reset)`；其中 `s = (RMS(delta_right_6) - RMS(delta_left_6)) / (RMS(delta_right_6) + RMS(delta_left_6) + 1e-8)`，只使用 use-pressure steps 上每只手臂的 6 个非 gripper joints，并先在 episode 内平均；
- correct-side train-prototype oracle 至少成功 `7/8`，opposite-side prototype oracle-donor 至少跟随 `7/8`，paired action-side prediction 至少在 `7/8` 上翻转；
- writer-dependent direct-carry oracle 单独报告；writer gate 通过后至少要求 `7/8` 一致；
- Gate C 的 production commit 和 retention 阈值继续通过。

Attention mass 是 read/use attribution diagnostic。报告它相对于 uniform token baseline 的 enrichment，以及相对 reset/zero injection 的 paired change；它不是因果证据，不设置硬 gate。

在 training episodes 上运行相同 causal battery，但只作为 supporting evidence。Reset/donor effect 可以证明这些 training samples 上存在 causal use；无论 effect 存在或缺失，都不能证明 held-out generalization。

在 74 个 training episodes 上使用 leave-one-episode-out prototypes 评估 correct-side 与 opposite-side prototype oracles。Opposite-side oracle-donor follow rate 是样本量充分的主要 consumer diagnostic，因为 memorized actions 会抵抗、而不是制造 intervention-driven flip。

在固定、无 augmentation 的 calibration suite 上定义 task health：

- 评估前冻结 suite 的 stable IDs、frame indices、preprocessing/norm hash、flow timestep、action noise 和 RNG；
- 只实例化一次 v3.5：共享 parameters 来自官方 Pi0.5 base，新增 leaves 使用固定 seed 初始化。在启用 memory transition/injection 前，用完全相同的 suite 记录 **fresh official-base path** 的 flow loss 与 subtask CE。校准后的 v3.5 step-0 evaluation 必须复用同一个 parameter tree，使两个 reference 使用完全相同的 fresh heads，唯一差异是是否启用 v3.5 path；
- v3.5 step 0 必须满足 flow loss `<= 1.10x` source reference，且 subtask CE `<= source + 0.05`；
- losses、gradients、parameters 和 memory state 中没有 NaN/Inf；
- 每个 rung 的 flow loss 必须同时 `<= 1.10x` source reference 和 `<= 1.10x` v3.5 step-0 值；
- 每个 rung 的 subtask CE 必须同时 `<= source + 0.05` 和 `<= v3.5_step0 + 0.05`；
- severe-clip rate 定义为 pre-clip global grad norm 大于 configured clip threshold 的 `10x`，要求不超过 optimizer steps 的 `1%`；
- branch-local feature-cotangent-cap bind rate 不超过 eligible E/D loss terms 的 `5%`。

Pilot exit rule：

- **Pass：**全部硬阈值通过。继续训练到冻结的 full budget：**10,000 completed updates**。
- **Inconclusive：**core、injection、retention、reset 和 task health 通过，但 learned natural/donor count 是 `5/8` 或 `6/8`；或者每个 prototype-oracle count 至少 `6/8`，相对 step 0 至少增加 1 个成功 episode，但仍低于 `7/8`。同一 run 只允许延长一次到 2,500 updates。
- **Fail：**numerical/invariant/retention gate 失败；reset 至少正确 `5/8`；natural 或 donor 最多成功 `4/8`；task health 失败；或者任意 prototype-oracle count 最多 `5/8`（或保持 `6/8` 但相对 step 0 没有至少增加 1 个 episode）。停止该 branch，并根据诊断进入 core、writer 或 consumer fallback。
- 2,500 updates 时必须通过相同阈值，否则停止；不允许第二次延长。

1k 通过后，在 2,500、5,000 和 10,000 updates 保存固定 post-pilot rungs。如果全部 mechanism 与 task-health gates 持续有效，则将 raw 10,000-update checkpoint 作为预注册 primary endpoint；不能通过最大化 development accuracy 选择 checkpoint。

Pilot 和 branch selection 期间保持 June-30 封存。如果需要 transfer result，只能在冻结的最终 endpoint 上运行一次；它绝不作为 gate，也不参与 normalization/calibration。

Counterfactual prompt binding 是在 development episodes 上声明 **prompt-bound writer** 的硬 gate，但它不阻断 mechanism rungs，因为 natural-prompt memory chain 仍可独立评估。Final-test episodes 在 branch 和 checkpoint policy 冻结前保持完全不可见。

## 9. 代码范围

- `models/memory.py`：pooled association、output-only direct delta、FP32 fast state/read/commit、analytic skip-O decay、invariants 和 telemetry。
- `models/pi0.py` 与 `pi0_config.py`：E-only transition、`L_write`、`L_read`、state-valid/reachable tracking、state-invalid side-loss defense、injection calibration support、两个 oracle interventions 和 time-consistent augmentation。
- `training/data_loader.py` 与 `transforms.py`：E/O/D sidecar、5-frame E tail guard、strict D mask、hard final-E anchoring、natural/skip-O sampling、use-pressure mask 和 seeded stable-ID split。
- `training/config.py`：独立的 v3.5 config 和 dataset version。
- 使用标准 official-base checkpoint loading 和显式 shared/fresh initialization allowlist；`scripts/v35_step0_bootstrap.py` 负责 zero-update initialize/finalize 边界，`scripts/v35_train.py` 从 sealed artifact 安装 calibration values，`scripts/train.py` 只接受已授权的 resume，并负责精确 loss denominators 与 completed-update checkpoint 语义。Launch 不需要自定义 raw-checkpoint transplant。
- `scripts/v35_prepare_pilot.py` 与 `cluster_v35/prepare_pilot.sh` 提供从 fresh step 0 到 calibration、Gate A/B/C 和 pilot authorization 的 create-only、resume-validating 路径；它们绝不运行 optimizer update。
- Pilot authorization 绑定完整的 production Python source tree、v3.5 scripts、dependency lockfiles 和 runtime package versions。每次 verify 或 train 入口都会重新加载当前 Gate A/B/C evidence 并验证 live checkpoint 内容；任何 source、environment、parameter、optimizer、iterator 或 telemetry 变化都会使 authorization 失效。只允许从 authorization 明确指向的外部 sealed source rung resume；中途 crash checkpoint 必须拥有独立 sealed rung binding，否则只能从前一个 authorized source 重新开始。
- Pre-pilot 入口会依据冻结的 dataset inventory 验证 train/development parquet 与 metadata 字节；对 sealed final-test 文件只检查 path 与 size，绝不打开其 payload；replay 和 leakage batch size 固定为 8，避免 resume 后混用不同采集协议的 evidence。
- 新增 manifest-driven leakage、writer、retention、attention 和 causal evaluators。
- 新增 unit 和 integration tests。
- 不修改独立数据准备会话负责的 labeling/conversion 文件。

## 10. 执行顺序

1. 用户和 Claude 审批本 revision，包括 10,000-update full budget。
2. 完成人工标签、side/block audit、manifest、split、conversion 和 train-only normalization。
3. 从官方 Pi0.5 base 运行 zero-update initialize phase，全新初始化每个 v3.5-specific leaf，并冻结精确的 step-0 tree identity。官方 base 可以在每个 cluster 的 project-local cache 中独立下载。
4. 实现 memory clock、pooled output-only delta core 和数值测试。
5. 实现 masks、final-E anchoring、natural/skip-O sampling、state-valid/reachable tracking、losses 和 sequence-consistent augmentation。
6. 设置并冻结 `tanh(memory_inject_w)=0.5`，只在 74 个 training episodes 上校准 `c` 与 `tau`，然后冻结它们。
7. 使用校准后的 injection values 和精确初始 sampler/RNG state finalize 同一个 zero-update state；不能读取 data batch。
8. 使用 create-only、resume-validating 的 pre-pilot orchestrator 验证冻结的 dataset inventory，运行 Data、Leakage 和 Step-0 gates，然后为该精确 run identity 封存并验证 pilot authorization 与 live checkpoint；整个过程不运行 optimizer update。
9. 只有所有 launch gates 通过后，才允许从 finalized checkpoint 0 进行 authorized resume 并启动 1k pilot。
10. 应用固定的 pass/inconclusive/fail exit rule；branch 与 reporting policy 冻结后只运行一次 final test。

## 11. Claude Review 已解决的决定

1. **Split：**采用 `74 train / 8 development / 8 final test`；0830 是同一装置、不同日期的新数据，June-30 保持 eval-only。
2. **Leak gate：**每个 collection 使用两个 primary probes，对四个 primary tests 做 Bonferroni-corrected permutation tests，并增加 pooled 74-episode point-estimate gate；其他 modalities 只作描述。
3. **Memory core：**pilot 每帧使用一个 pooled vector 和 output-only direct delta；删除 exact-16-pair claim。
4. **主要参数：**以 raw parameters 为 primary。EMA 只作一致性检查，因为 reset EMA 在 1k updates 时没有明确意义。
5. **Prompt binding：**它是 development 数据上 prompt-bound writer claim 的硬 gate，但不阻断 natural-prompt mechanism rungs。
6. **Read supervision：**使用 `read_state_valid` 而不是 `read_credit_reachable` mask loss；reachability 单独报告。
7. **Fresh-base initialization：**共享 pretrained parameters 从官方 Pi0.5 base 加载；memory/query compressors、conditioner、slot embedding、state-null、injection projection、ladder heads 和 side heads 全新初始化。Oracle injection 用于诊断新的 consumer path，不假定 step 0 已具备能力。
8. **Fixed injection gate：**每个 channel 以 `w=atanh(0.5)` 初始化，验证 `tanh(w)=0.5`，并在 calibration 前及整个 pilot 中冻结；不存在 conditional inherited-leaf reset。
9. **Long delays：**在 natural windows 之外使用明确的 skip-O analytic decay；不能假设独立 samples 之间隐式传递 state。
10. **Development gates：**8 个 development episodes 使用 episode counts，不能使用不可实现的小数阈值。
11. **Pooled leakage point gate：**使用 `0.62` 并接受预注册的 false-stop tradeoff；family-corrected permutation tests 仍是主要 gate。
12. **Injection scale：**要求固定 gate 的全部 channels 都为 open；只使用 74 个 training episodes 校准 `c` 和 `tau`，并在 pilot 中冻结 `memory_inject_w`、`c` 和 `tau`。
13. **Source reference：**v3.5 step 0 和每个 rung 都与同一无 augmentation suite 上的 fresh official-base path 比较；previous-run checkpoint 或自定义 raw-checkpoint transplant 都不属于 launch。
14. **Portable bootstrap：**官方 Pi0.5 weights 在每个 cluster 的 `memory_project/v35/cache` 内独立下载，不需要同步。所有不可重新下载的 data、norm assets、manifests、gate evidence、checkpoints 与 provenance 都使用相对 `memory_project` 的 identity。Optimizer training 只能从 finalized zero-update checkpoint 进行 authorized resume 后开始。

本 revision 立即用于代码实现。只有全部 launch gates 通过后才能开始训练。
