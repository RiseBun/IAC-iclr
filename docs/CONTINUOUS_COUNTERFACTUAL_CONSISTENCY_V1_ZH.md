# Continuous Counterfactual Foresight Consistency（CCFC）V1

## 1. 指标要回答什么

CCFC 是 IAC 面向 WAM 的主反事实指标。它不问“未来视频看起来是否逼真”，也不问“动作最后是否成功”，而问：

> 在同一历史、同一模型和同一随机扰动下，风险条件改变所引起的想象未来变化，是否与 action head 实际改变的自车运动一致？

因此它对应一条可检验的因果链：

```text
风险干预 R：改变未来场景
       ↓
WAM 想象未来 F：图像中的自车运动响应 ΔP_F(t)
       ↔  CCFC  ↔
实际动作 A：action head 的自车运动响应 ΔP_A(t)
```

CCFC 不是精确轨迹恢复器。图像侧只需在可观测区间提供时间对齐的平面 `SE(2)` 量：

```text
P(t) = [x(t), y(t), heading(t)]
```

其中 `x` 是前向位移，`y` 是横向位移，`heading` 是运动方向。它们用于比较“响应变化”，而不是声称从单目视频恢复了全局真值轨迹。

## 2. 严格的反事实配对

一个正式样本必须同时有 `clear` 和 `risk` 两个分支，并通过以下审计：

- `history_fingerprint` 相同：历史图像、ego state、历史 waypoint 和相机标定一致；
- `wam_model_id` 相同；
- `nuisance_seed` 相同；
- 两个分支均为 `future_images_source=wam_generated`；
- action trajectory 来自原生 action head，不能是 logged/oracle/proxy/candidate；
- 图像解码器 `candidate_bank_used_by_decoder=false`；
- 未来时间戳逐项相同；
- action head 在两个分支之间确实产生了非零干预。

任一审计失败，样本只能作为诊断输出，不能进入正式因果均值。没有可观测区间或没有材料动作响应时，指标返回 `abstain`，不把拒答当成错误。

## 3. 计算定义

先把图像和动作两条分支分别转换成统一的平面 pose：

```text
P_F,clear(t), P_F,risk(t)
P_A,clear(t), P_A,risk(t)
```

反事实响应定义为：

```text
ΔP_F(t) = P_F,risk(t) - P_F,clear(t)
ΔP_A(t) = P_A,risk(t) - P_A,clear(t)
```

角度差使用 wrap 到 `[-π, π]`。只在图像两分支均 `usable` 且 action 响应超过预注册 deadband 的时间点计分。默认 deadband 为平移 `0.05 m` 或 heading `0.01 rad`。

CCFC 保留三个必要子项：

### 3.1 Response direction

平移响应使用 `Δx, Δy` 向量余弦，heading 响应使用符号一致率。它回答“风险改变后，未来和动作是否朝同一个方向改变”。

### 3.2 Response magnitude

逐时间点计算响应差异。米制模式使用预注册容差：平移 `0.50 m`、heading `0.05 rad`，并把超出容差的误差截断为 0 分。该项保留动作响应的绝对幅度。

### 3.3 Response temporal alignment

在相同时间轴上对 `[Δx, Δy, Δheading]` 做 observability 加权余弦，回答“响应是否在相同的时间发生”。它能区分“最后都刹车了”和“风险出现后是否在正确时刻开始刹车”。

三个子项合成唯一主分数：

```text
CCFC = (direction × magnitude × temporal)^(1/3)
```

采用几何平均是因为因果一致性要求三个条件同时成立：方向错、幅度错或时序错任一项接近 0，都不应被其他项的高分掩盖。报告仍必须保留三个子项、coverage、有效区间数和拒答原因。

## 4. 两种报告口径

### `metric`（正式主口径）

保留米和弧度的响应幅度，使用固定容差。该口径回答 WAM 的想象变化与实际动作变化在物理量上是否一致，是正式论文表格的默认 CCFC。

### `scale_free`（形状诊断）

分别按每个分支的最大响应幅度归一化，只衡量方向和时间形状，不用于替代米制结果。它可诊断单目尺度误差是否掩盖了正确的反事实响应。

### `arc_relative`（长时域形状诊断）

8 帧/4 秒协议额外提供 `arc_relative`：clear/risk 的图像与 action pose 各自按二维
轨迹弧长归一，再比较风险干预引起的响应。它与 Level-1 的长时域主归一保持一致，
可避免终点前向位移接近零时的病态分母；但它会弱化反事实响应的绝对幅度，因此只作为
敏感性诊断，不能替代正式米制 CCFC。

## 5. 与其他指标的关系

| 层级 | 指标 | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| Level 0 | logged future probe | 图像运动测量器是否有信号 | 不能证明 WAM 因果性 |
| Level 1 | Continuous Foresight–Action Alignment | 单个未来分支与 action head 是否对齐 | 不能证明风险改变引起了该动作 |
| Level 2 | **CCFC** | 想象未来响应与动作响应的因果一致性 | 没有独立执行就不能证明闭环成功 |
| Level 3 | Foresight‑Conditioned Success | CCFC 高低是否对应独立执行成功 | 需要真实闭环 rollout 和 task success |

CCFC 是本项目当前要提交的 WAM 性能指标；Event‑CC、Event‑FCS 和四类事件链只作为分层、案例解释或 Level‑3 成功条件，不能替代连续主分数。

## 6. 代码和复现

核心函数：

```text
src/iac_new/continuous_motion.py
  compare_counterfactual_se2_consistency(...)
```

反事实评测器：

```bash
PYTHONPATH=src:. python scripts/evaluate_counterfactual_continuous_alignment.py \
  --records wam_counterfactual_branches.jsonl \
  --require-eight-frame-four-second \
  --require-ready \
  --output counterfactual_continuous_report.json
```

输出中的 `continuous_cfc.metric.score` 是正式米制 CCFC，`continuous_cfc.scale_free.score`
和 `continuous_cfc.arc_relative.score` 是形状诊断；顶层 `summary.metric_score_mean` 只对
通过 readiness audit 的配对求均值。

当 WAM backend 完成生成后，先用 `scripts/build_wam_level1_continuous_manifest.py` 保留
8 帧图像、原生 action-head 和双分支 lineage，再运行候选盲 Level-1 decoder。随后使用：

```bash
PYTHONPATH=src:. python scripts/build_counterfactual_continuous_records.py \
  --manifest results/wam_level1_manifest.jsonl \
  --scores results/wam_level1_scores.jsonl \
  --output results/wam_counterfactual_continuous_records.jsonl

PYTHONPATH=src:. python scripts/evaluate_counterfactual_continuous_alignment.py \
  --records results/wam_counterfactual_continuous_records.jsonl \
  --require-eight-frame-four-second --require-ready \
  --output results/wam_ccfc_report.json
```

记录组装器会拒绝缺少 `clear/risk`、不匹配 history/model/seed/time axis、非 candidate-blind
decoder 或含 realized future state 的输入；因此没有完整双分支时，流程会 fail-closed。

当前仓库已完成公式、审计门槛和合成对照测试，但服务器上尚未获得正式的 WAM `clear/risk` 原生 action-head 配对。因此当前不能报告 WAM 的真实 CCFC 排名，也不能把 NAVSIM logged future 的结果当作 WAM 因果证据。
