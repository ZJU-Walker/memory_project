# v3.5 当前结构、状态与 Claude 交接说明

更新时间：2026-08-31（America/Los_Angeles）。

用途：这是一份当前 v3.5 实现的操作性交接文档，说明已经完成的内容、验证证据、最新真实 H100 运行结果、当前 blocker，以及继续工作时不能破坏的科学审计规则。完整设计合同仍以 `V35_PLAN_FOR_CLAUDE_REVIEW.md` 为准。

## 1. 当前结论

v3.5 的主体实现已经基本完成，但**还没有获得 optimizer training 授权**。

已实现部分包括：90 集转换数据、冻结的 74/8/8 split、仅 train-74 的 normalization、可移植目录、memory core、E/O/D mask 与 sampler、calibration collector、Gate A/B/C/D producer 与 reducer、两阶段 step-0 bootstrap、source identity freeze、live checkpoint authorization，以及完整 pre-pilot orchestrator。

真实 4×H100 pre-pilot 实验 `v35_fresh_pilot_20260831_r2` 已运行。它成功完成 fresh Pi0.5 bootstrap，并生成 provisional step-0 identity；随后在 calibration 之前停止。原因是 `initialization_graft_manifest.json` 使用 pretty JSON，而 immutable-stage validator 只接受 compact、排序、单行并且只有一个末尾换行的 canonical JSON。

失败是 fail-closed，发生在以下步骤之前：

- calibration replay；
- Gate A/B/C；
- pilot authorization；
- 任何 optimizer update；
- 任何 final-test 数据读取。

因此科学状态仍然干净，但 pre-pilot qualification 尚未完成。

立即要做：

1. 让 graft manifest writer 输出 validator 要求的 canonical bytes。
2. 添加 producer 写出后直接由 production consumer 读取的回归测试。
3. 运行 focused suites 和完整 v3.5 regression。
4. 只同步明确修改的文件到主项目。
5. 使用新实验名，例如 `v35_fresh_pilot_20260831_r3`。

修源码后不能继续 `r2 --resume`，因为 `r2` 的 source snapshot 绑定的是旧源码，正确行为就是拒绝 source drift。

## 2. E、O、D、X 的含义

- **E（evidence）**：`inspect both bins`。只有满足可见性和 tail guard 的 E 帧可以写入 prompt-conditioned target-side evidence。
- **O（occlusion）**：`close both lids and reset arms`。O 内禁止写入，只允许 decay。
- **D（decision）**：严格静止的 `wait; target bin is left/right`。策略应在目标侧动作开始前读取 E 中的记忆。
- **X（execute）**：`open left bin` 或 `open right bin`。

memory value 的语义是“根据当前 prompt 应该选择哪一侧”，不是与 prompt 无关的 object-location 表征。

最终 claim 只允许描述 0816+0830 collection 内 held-out memory mechanism；不能声称跨 session generalization，也不能把 open-loop arm-side steering 描述为 closed-loop task success。

## 3. 可移植目录

完整传输单元是 `memory_project`：

```text
memory_project/
  data/
    lerobot/yam/bin_memory_0816_0830_v35_subtask/
    0816_0830_episode_manifest_v35_frozen.json
    0816_0830_episode_manifest_v35_frozen_*.json
  openpi/
    cluster_v35/
    scripts/v35_*.py
    src/openpi/
  v35/
    assets/pi05_yam_0816_0830_v35/
    cache/
    checkpoints/
    diagnostics/
    tmp/
    wandb/
```

所有生产路径都通过 `MEMORY_PROJECT_ROOT` 解析。LeRobot data、model/tokenizer cache、JAX cache、assets、diagnostics、checkpoints、tmp 和 W&B 都在项目目录中。

official Pi0.5 base identity 是：

```text
gs://openpi-assets/checkpoints/pi05_base/params
```

新 cluster 可自行下载，不需要同步 official-base cache。v3.5 不使用 run5 或 v3.4 checkpoint。

Python 环境可由每个 cluster 独立建立，但 interpreter 必须从当前复制的 `memory_project/openpi/src` 导入 `openpi`；foreign editable checkout 会 fail closed。

## 4. 冻结数据状态

数据位置：

```text
data/lerobot/yam/bin_memory_0816_0830_v35_subtask
```

数据组成：

- 0816：60 集；
- 0830 part1：16 集；
- 0830 part2：14 集；
- 排除 `0830_bin_part2/demo14`，因为没有 terminal execute；
- 转换后总数：90；
- split：74 train / 8 dev / 8 sealed final；
- split seed：35；
- normalization：仅 74 train，共 64,618 rows。

冻结哈希：

| Artifact | SHA-256 |
| --- | --- |
| Dataset tree | `21b02ea4752280abee252535f1c519f611b7308a6540c73a699ee2bcbd47ed5f` |
| Relocation inventory | `6856b49a6910c9449c983af582a922ce037300bc51c01eea1ed3317f1f409b73` |
| Frozen manifest | `c3b41d3247204aee4b7428ffdcc80a21d699a62aadec478cc499b52b6881dd57` |
| Norm stats | `b8ed515a495b04b7a58cdc6d18e18aeec888e6f6133dccf35635eb497dc9e3d7` |
| Norm provenance | `c46bb0e714b8657f08bc73f0b252d8f9b79f623fba9829623c249b9eb89511bf` |
| Train storage | `2eb4eefccd1a03ab05f304a1add74382f2f4067aea0164ccff6e7a41dc6c9074` |
| Dataset/frame protocol | `c8f4ef48a4717e45c992ee456bdf7ec4220bf1db11d203a48ce71c1f3a1c96db` |

项目内数据副本共 202 files、58,534,560,294 logical bytes。

旧 Hugging Face cache 下的原始上传源仍在上传，不能删除或修改：

```text
/iris/u/kewalk/.cache/huggingface/lerobot/yam/bin_memory_0816_0830_v35_subtask
```

manifest 已绑定 stable ID、raw source、converted index/frame count、prompt、object、side、collection、0830 part、split、label hash、E visibility manual review、独立 `D_valid` sidecar 和 block-confound audit。

## 5. Memory Core

原始“一个 rank-one update 精确拟合 16 token”在数学上不成立。因此 v3.5 每个 E frame 先将 16 个 key/value pool 成一个 association：`k_bar`、`v_bar`。

当前规则：

1. 只更新 output fast matrix `W`；
2. hidden fast leaves 不变；
3. fast bias 与 momentum 保持精确零；
4. 每步先 read，再执行当前 transition；
5. eligible E：decay 后 direct delta commit；
6. valid non-E：仅 decay；
7. invalid/padding：严格 no-op。

```text
W_dec = (1 - alpha_step) * W
r = v_bar - h @ W_dec
W_new = W_dec + delta_rate * outer(h, r) / (h dot h)
```

pilot 固定：

- `delta_rate = 1.0`
- `alpha_step = 0.01`
- 每 15 raw frames 一个 memory clock step
- delta mode 禁止旧 drift trust region

fast W、commit、analytic decay、raw read 与 calibration 全部使用 FP32。退化的 pooled norm 或 hidden norm 会 fail closed。

own-key commit exactness 与 natural D-query retrieval 分开验证，因为写入准确不代表 D query 与 E key 自动对齐。

## 6. Data Mask 与 Sampler

主要字段：

- `seq_write_mask`：当前帧 E write；
- `seq_decision_mask`：strict static D；
- `seq_occlusion_mask`：O；
- `seq_read_state_valid`：当前 read 前已有成功 E commit；
- `seq_read_credit_reachable`：前一个 E commit 是否仍在当前 TBPTT differentiable window；
- `seq_decay_gap_before`：当前 read 前省略的 valid non-write sampled transitions；
- `seq_use_pressure_mask`：action chunk 已覆盖 first execute motion 的 D step；
- `seq_sparse_skip_o`：analytic skip-O family；
- episode、collection、object、side、memory-cell IDs。

两种 memory-critical window：

- natural contiguous E/O/D；
- E commit + FP32 analytic O decay + D 的 skip-O window。

skip-O 只允许省略 semantic O 中 valid、non-writing、non-reset sampled steps。tail-E 或 O 外 gap 会在 sampler 分配概率前被拒绝。

state-invalid D 的 loss 不仅 numerator 为零，denominator 也会正确排除。

## 7. Model 与诊断

已实现：

- write-side loss；
- read-side loss；
- detached ladder heads；
- branch-local feature cotangent cap；
- 每 update severe global clip 统计；
- finite update 统计；
- query/key cosine telemetry；
- direct-carry oracle；
- correct/opposite prototype oracle；
- opposite-side donor；
- reset/zero-read controls；
- action-expert-to-memory attention；
- same-tree no-memory 与 calibrated-memory task-health comparison。

augmentation 在同一 sample/camera 的时间维度复用参数，不同 sample 独立随机。

## 8. Fresh Base、Calibration 与 Authorization

v3.5 只加载 official Pi0.5 shared leaves。所有 v3.5-specific memory/query/consumer/diagnostic/side heads 都 fresh init；optimizer、EMA、global step 也从零开始。

`memory_inject_w` 每个 channel 初始化为 `atanh(0.5)` 并冻结。`c` 与 `tau` 只从 train-74、使用真实 16-slot FP32 production pinning 计算。

两阶段 step-0：

1. initialize exact fresh tree 和 raw checkpoint 0；
2. train-only replay + calibration；
3. finalize 相同 tree，写入 calibration 与 exact initial iterator/RNG；
4. Gate A/B、rung0 Gate C/task health；
5. seal pilot authorization；
6. 只有此后才允许 resume checkpoint 0 开始训练。

pre-pilot 一开始会冻结完整 source/runtime snapshot，包括 production OpenPI Python、v35 scripts、dependency locks 与 runtime package versions。每个 stage execute/skip、authorization、verify 和 training 都重验。

authorization v2 绑定 source、semantic config、manifest、norm、storage、calibration、official base、actual params、train-state/optimizer、iterator/RNG、telemetry 和外部 sealed source rung。

`verify-only` 与真实 train 都会实际 restore Orbax live state，而不是只相信 checkpoint 自己携带的 JSON。

## 9. Pre-Pilot 15 个阶段

入口：

```text
openpi/cluster_v35/prepare_pilot.sh
```

阶段：

1. bootstrap initialize；
2. calibration preflight；
3. train-74 replay selection；
4. 4 个并行 replay collectors；
5. replay seal；
6. injection calibration；
7. bootstrap finalize；
8. Gate A；
9. Gate B features；
10. Gate B decision；
11. rung selection；
12. rung0 collect/seal；
13. semantic config SHA；
14. pilot authorization；
15. final verify-only。

整个流程不执行 optimizer update。Gate B fail 会在读取 dev/rung 数据前停止。生产 replay 与 leakage batch size 固定为 8。

## 10. 当前 H100/Slurm 状态

- allocation：`17130107`
- node：`iris-hgx-1`
- GPU：4×H100
- allocation keeper：`cluster_scripts/train_hs.py`，每卡约 1 GB

用户授权后，只停止了独立 Qwen step `17130107.87`；父 allocation 保留。`iris-hgx-2` 上的 job `17126912` 没有被触碰。

4-GPU JAX smoke 已确认可见全部设备。

最新运行：

```bash
openpi/cluster_v35/prepare_pilot.sh \
  --experiment-name v35_fresh_pilot_20260831_r2 \
  --gpus 0,1,2,3
```

`r2` 已写出：

```text
v35/checkpoints/pi05_yam_mem_v35/v35_fresh_pilot_20260831_r2/
  initialization_graft_manifest.json
  step0_bootstrap_provisional.json
  prepilot_source_identity.json
```

bootstrap 成功后，orchestrator exit 2：

```text
immutable JSON is not canonical with one trailing newline:
.../initialization_graft_manifest.json
```

## 11. 当前直接 Blocker

生产者/消费者的 byte contract 不一致：

- `openpi/src/openpi/training/weight_loaders.py::_write_manifest()` 使用 `json.dumps(..., indent=2, sort_keys=True)`；
- `openpi/scripts/v35_prepare_pilot.py::_load_immutable_json()` 只接受 compact sorted canonical JSON + 一个末尾换行。

建议：

1. 复用一个 canonical JSON byte helper；
2. 固定 sorted keys、compact separators、`allow_nan=False`、明确 `ensure_ascii`、一个 trailing newline；
3. writer 保持 create-only；
4. 不改变 `manifest_sha256` 语义，除非有明确 schema migration；
5. 添加 `_write_manifest()` 写出后由 production immutable loader 立即读取的测试。

不要放宽 orchestrator 去接受任意 pretty JSON。应修正 immutable producer。

修复后必须用新实验名，不能 resume `r2`。

## 12. 已发现并修复的重要问题

已解决的问题包括：

1. 16-token rank-one exact-fit 数学错误；
2. old theta gradient-step factor 表述错误；
3. hidden fast-map 在 delta mode 漂移；
4. BF16 commit/read 精度风险；
5. E/O/D clock 定义不清；
6. long-delay episode 丢失 D supervision；
7. wait motion leakage；
8. normalization 使用 dev/final 的泄漏风险；
9. manifest side/object/prompt 不一致未 fail；
10. episode 从 sampler 静默消失；
11. Gate B fail 后仍继续读取 dev；
12. inference 没有 explicit mask 时意外写入或完全 no-op；
13. norm producer/consumer 路径不一致；
14. partial pre-pilot 后源码漂移；
15. foreign editable checkout；
16. symlink 逃逸 project root；
17. 只相信 checkpoint-owned provenance；
18. bootstrap checkpoint0 首次 W&B resume 失败；
19. resume 时 replay batch protocol 改变；
20. Gate-A Torch/config import 顺序导致 cold subprocess segfault；
21. frozen historical manifest pretty-print 与 canonical envelope 混用；
22. bias-free NNX Linear 的 structural `None` leaf 被错误当数组。

## 13. Structural-None 真实故障

第一个真实 fresh bootstrap 在以下 leaf 失败：

```text
read_query_compressor/key_proj/bias = None
```

修复后，schema/hash 显式记录 `dtype=none`、`shape=()`、`structural-none`，同时仍拒绝任意 object。

验证：

- structural-None focused suite：40 passed；
- exact weight-loader suite：19 passed；
- `v35_diag_structnone3` 真实 4×H100 bootstrap 成功。

保留的诊断实验：

- `v35_fresh_pilot_20260830`：旧 structural-None failure；
- `v35_diag_none_leaf2`：定位 leaf path；
- `v35_diag_structnone3`：修复后成功 bootstrap；
- `v35_fresh_pilot_20260831_r2`：bootstrap 成功，随后 canonical graft-manifest failure。

这些目录是 provenance/debug evidence，不应删除或静默复用。

## 14. 测试证据

- 完整 v3.5-focused suite：285 passed，5 deselected；
- orchestration/authorization/Gate-A：49 passed；
- iris-ws-18 单 GPU memory/gradient contracts：17 passed；
- structural-None focused：40 passed；
- exact weight-loader：19 passed；
- H100 Gate-C gradient contracts：5 passed；
- 多轮 authorization/exact-resume focused suites 已通过；
- Ruff、format、`git diff --check`、shell syntax 在最新 blocker 前均通过。

重要限制：structural-None patch 后还没有重新运行完整 285 tests，只运行了 focused suites 和真实 H100 bootstrap。修 graft encoding 后应补跑完整 suite。

## 15. Worktree 保护规则

worktree 是 dirty 的，包含用户自己的 labeling、conversion、report、v3.4 和 v3.5 改动。不能 reset、clean、checkout 或覆盖无关文件。

同步只能使用明确 file whitelist，禁止 `--delete`。用户的 label/conversion 文件在前后 hash 检查中保持不变。

当某个 pre-pilot experiment 正在运行时，不要修改 production Python，因为 source snapshot 会让 lineage 失效。根目录下非 production source 的交接文档可以安全更新。

## 16. Claude 接手顺序

1. 阅读本文件、英文 handoff、`V35_PLAN_FOR_CLAUDE_REVIEW.md`、`v35/README.md`、`openpi/docs/v35_training_authorization.md`。
2. 确认 `r2` 已停止。
3. 保留所有历史实验目录和旧 HF upload source。
4. 只修 graft manifest canonical byte encoding，并加 producer-consumer regression。
5. 跑 weight loader、bootstrap、prepare pilot、calibration、authorization focused tests。
6. 跑 Ruff、format、diff check、shell syntax。
7. 跑完整 v3.5 suite。
8. 明确 whitelist 同步到主项目，再次检查用户 label/conversion hash。
9. 使用新实验名，例如 `v35_fresh_pilot_20260831_r3`。
10. 先 `--plan`，再跑真实 4×H100 prepare。
11. 15 个阶段和 final verify-only 没有全部通过前，不能运行 training。
12. 如果新 stage failure 需要改源码，保留该实验，修复后换新实验名。

建议命令：

```bash
openpi/cluster_v35/prepare_pilot.sh \
  --experiment-name v35_fresh_pilot_20260831_r3 \
  --gpus 0,1,2,3 \
  --plan

openpi/cluster_v35/prepare_pilot.sh \
  --experiment-name v35_fresh_pilot_20260831_r3 \
  --gpus 0,1,2,3
```

只有 authorization 与 verify 完成后：

```bash
openpi/cluster_v35/train.sh \
  --experiment-name v35_fresh_pilot_20260831_r3 \
  --calibration v35/diagnostics/runs/v35_fresh_pilot_20260831_r3/calibration/injection_calibration.json \
  --pilot-authorization v35/diagnostics/runs/v35_fresh_pilot_20260831_r3/pilot_authorization.json \
  --target 1000 \
  --fsdp-devices 4
```

## 17. 尚未关闭的风险

1. graft manifest canonical encoding 阻塞 stage 2；
2. 真实 train-74 calibration replay 尚未完成；
3. Gate A/B/C 和 rung0 尚未在 authorized experiment 中封存；
4. pilot authorization 尚不存在；
5. 真正的 1,000-update pilot 尚未开始；
6. structural-None patch 后完整 regression 尚未补跑；
7. 真实 replay/rung 仍可能暴露 unit test 未覆盖的 runtime 或显存问题；
8. final test 必须继续 sealed，直到预注册 endpoint。

## 18. Training Ready 定义

只有以下全部成立，才可启动 1,000-update pilot：

- 新实验 source identity 冻结且未变化；
- official-base graft 与 actual step0 tree 通过认证；
- train-74 replay 与 injection calibration 通过；
- finalized checkpoint0 与 calibrated tree 精确一致；
- Gate A pass；
- Gate B pass；
- rung0 Gate C/task health 满足要求；
- pilot authorization canonical 且有外部 rung binding；
- `v35_train.py --verify-only` 实际 restore live checkpoint 并通过；
- 所有 gate 前 optimizer update 数为零；
- launch 使用完全相同的 authorized source、data、norm、calibration、config、iterator 和 checkpoint。

在这之前，准确状态是：**实现已存在，pre-pilot qualification 被 blocker 阻止，training 尚未授权。**

