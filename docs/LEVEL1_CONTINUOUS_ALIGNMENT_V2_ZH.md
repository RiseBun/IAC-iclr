# Level 1 连续前视—动作对齐 V2

> 纵向残差增强及其场景隔离负结果见
> [Level 1 纵向残差增强 V3](LEVEL1_LONGITUDINAL_RESIDUAL_V3_ZH.md)。V3 显著降低
> 原始 RAFT 速度误差，但没有超过强 history null，因此纵向门槛仍未通过。

## 1. Level 1 回答什么

Level 1 不使用事件标签，只回答：

> WAM 未来图像中的连续自车运动，是否包含超出历史状态外推、且与原生 action waypoint 对应的增量信息？

直接的 `m_F(t) ≈ m_A(t)` 不够，因为未来图像和 action 可能只是共同复制历史中的匀速直行趋势。V2 因此把 history-only null 和 future specificity controls 提升为必要条件。

![Level 1 连续对齐 V2](figures/level1_continuous_alignment_v2.svg)

可编辑源文件：[`level1_continuous_alignment_v2.mmd`](figures/level1_continuous_alignment_v2.mmd)。

## 2. 连续表示

图像分支和 action 分支分别产生同一时间轴上的：

```text
m(t) = [v(t), a(t), v_lat(t), yaw_rate(t), curvature(t)]
```

图像分支必须在读取 action waypoint 前完成，输出 `m_F(t)`、observability、abstention 和速度 `q05/q50/q95`。action waypoint 随后通过确定性运动学变换得到 `m_A(t)`。

## 3. 三个必要证据

### 3.1 Continuous Alignment

```text
D_F = d(m_F(t), m_A(t))
```

逐分量报告 MAE、RMSE、容差内比例及 coverage。任何单一总分目前都只允许作为 calibration diagnostic。

### 3.2 Foresight Gain

强 history-only null 使用所有允许的 `t ≤ 0` 状态，以历史速度线性回归得到 constant acceleration，并保持最后 yaw-rate：

```text
m_H(t) = CA-CYR(history only)
G_H = D_H - D_F
D_H = d(m_H(t), m_A(t))
```

`G_H > 0` 才表示 future 比历史外推更接近 action。CV-CYR 同时保留为较弱的审计基线，但不作为正式门槛。

### 3.3 Future Specificity

```text
L_shuffle = D(m_F,matched-shuffle, m_A) - D_F
L_reverse = D(m_F,time-reversed, m_A) - D_F
```

matched-shuffle 只使用历史速度和未来区间数量匹配，速度 caliper 为 `0.5 m/s`，禁止使用 action 或 future reference 选 donor。正 lift 表示真实 future 具有样本特异性和正确的时间结构。

## 4. 不确定性指标

速度后验不再只报告区间命中率。V2 同时报：

- 90% empirical coverage 与 absolute calibration error；
- mean interval width；
- 90% interval score；
- WIS；
- coverage-risk curve。

这能同时惩罚过窄、过宽和大量拒答。

## 5. 逐分量通过规则

对每个运动分量分别进行样本级 bootstrap。只有以下三项 95% CI 下界均大于 0，才标记 `incremental_signal_resolved=true`：

1. `G_H`：胜过强 history-only null；
2. `L_shuffle`：胜过 matched shuffled future；
3. `L_reverse`：依赖正确时间顺序。

速度还必须通过后验校准门槛。数据必须通过 fail-closed provenance audit：
`future_images_source=wam_generated`、action 来源不是 logged/oracle/proxy/candidate、
且存在 `wam_model_id`。同时满足真实 WAM action head 和 `8 帧/4 秒` 协议，
才有正式 Level 1 资格。

正式 WAM 图像评测使用 `scripts/build_wam_level1_continuous_manifest.py` 将固定
NAVSIM base manifest 与已完成的 WAM branch 输出合并。合并后的 manifest 使用
WAM 生成的 8 张未来图像，保留历史状态、时间戳和相机标定，新增
`wam_action_head` 作为最后比较阶段的 action reference，并删除
`realized_future_ego_state`，防止未来真值进入图像分支。没有完成生成的 branch
直接失败。NAVSIM realized future 只用于 Level 0 measurement validation。

## 6. NAVSIM 78 样本代理结果

当前数据是 logged future、`4 帧/2 秒`，因此只能验证指标行为，不能作为正式 WAM Level 1 结论。

| 分量 | 图像 MAE | 相对强历史基线 Gain，95% CI | Shuffle lift，95% CI | Reverse lift，95% CI | 增量信号 |
|---|---:|---:|---:|---:|---|
| 速度 | 1.140 m/s | -0.865 [-1.156, -0.621] | 0.013 [-0.161, 0.195] | 0.029 [-0.109, 0.178] | 失败 |
| 加速度 | 1.441 m/s² | -1.158 [-1.474, -0.866] | -0.169 [-0.535, 0.202] | 0.170 [-0.205, 0.669] | 失败 |
| 横向速度 | 0.048 m/s | 0.050 [0.018, 0.086] | 0.164 [0.107, 0.228] | 0.051 [0.032, 0.072] | 通过 |
| Yaw-rate | 0.0129 rad/s | 0.0716 [0.0513, 0.0928] | 0.2296 [0.1609, 0.3022] | 0.0624 [0.0433, 0.0824] | 通过 |
| 曲率 | 0.0230 1/m | 0.0031 [-0.0034, 0.0093] | 0.0596 [0.0397, 0.0799] | 0.0033 [0.00007, 0.00664] | 失败 |

速度 90% 后验在 135 个 usable 区间上的经验覆盖为 `51.4%`，平均宽度 `0.896 m/s`，calibration error 为 `0.386`，WIS 为 `0.685 m/s`。当前速度后验明显过窄。

结论：图像 future 对横向运动具有可辨识的增量信息；纵向速度和加速度没有胜过只看历史的 CA-CYR 外推。不能通过把历史速度与图像速度简单融合来制造正 gain，那只会让预测退回 history null。

## 7. 下一步能力目标

1. V3 已验证后处理式相对速度残差不能跨场景稳定超过 CA-CYR。
2. 下一版把残差变量放入光流重投影优化器，而不是继续调融合权重。
3. 使用新的纵向激励样本和真实 `8 帧/4 秒` WAM future/action 重新运行三项门槛。
4. 纵向分量未通过前，不构造 Level 1 总分，也不进入事件或 Level 2。

## 8. 复现

```bash
PYTHONPATH=src:. python scripts/evaluate_continuous_motion_alignment.py \
  --manifest navsim_event_balanced_78_iac_manifest.jsonl \
  --scores results/flow_ab_event78_20260826/raft_large/scores.jsonl \
  --output results/continuous_level1_v2_event78/raft_strict.json
```

实现位于 `src/iac_new/continuous_motion.py` 和 `scripts/evaluate_continuous_motion_alignment.py`。
