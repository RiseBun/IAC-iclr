# Level 1 纵向能力：优化器内残差 V4

## 1. 结论

V4 把纵向量从“后处理融合”移入连续轨迹优化器。速度只承担评测所需的
行为量，不承担精确里程计职责：主要看 `delta-speed` 的大小、方向、响应时序和
不确定性，而不是要求单目图像恢复绝对米制速度。

在服务器上对现有 78 个样本完成 `78/78` 解码、`0` 个错误。当前清单本身是
`4` 个未来帧、约 `2` 秒，故正式 `8` 帧/`4` 秒 Level-1 资格仍为 false。

V4 的结果应拆成两个结论：

1. **探针可用性：** 解码器能稳定输出连续运动后验，平均覆盖率约 `0.767`，
   绝对速度 MAE `0.418 m/s`，可用于后续行为级诊断。
2. **增量因果证据：** 尚未建立。纵向结果没有超过强历史状态空模型，匹配
   shuffle 也没有显著优势；因此不能说图像未来已经提供了可归因的纵向 foresight。

## 2. 为什么改成优化器内残差

历史状态给出可靠的当前速度和短期运动趋势，前视图像则提供未来变化的视觉证据。
如果先从图像估计绝对速度、再和历史速度融合，深度/地面尺度误差会被直接当成
未来信息。V4 固定历史曲线 `v_H(t)`，优化的是有界残差 `r(t)`：

```text
v_F(t) = clip(v_H(t) + r(t), 0.05, 30.0)
|r(t)| <= R,       R = 3.0 m/s
E = E_flow + lambda_r ||r/R||^2
          + lambda_s ||Delta r/R||^2
```

光流重投影误差决定残差方向，历史曲线只提供零点、边界和正则。候选 action
waypoint 不进入该优化器；它只能在最后一步被转换为 `m_A(t)` 与 `m_F(t)` 比较。
这样不会把动作答案泄漏给图像测量器。

## 3. 实现链路

```text
历史 ego state
  -> CA-CYR history prior v_H(t)
想象前视帧
  -> RAFT-Large forward/backward consistency
  -> ground-plane ego geometry
  -> optimize [bounded speed residual, curvature]
  -> continuous motion posterior + observability/abstention
  -> compare delta-speed / acceleration / lateral / yaw / curvature
  -> action waypoint comparison (only at the end)
```

`history_anchored_optimizer_residual_decoder` 是 V4 的唯一纵向模型协议。残差
后验的局部 profile 也在残差坐标中扰动，避免 profile 重新引入绝对速度缩放。

## 4. 78 样本服务器结果

结果文件：
`results/continuous_level1_optimizer_residual_v4_20260827/level1_strict_v4_final.json`

| 指标 | V4 结果 |
|---|---:|
| 解码成功 | 78 / 78 |
| 严格可评估样本 | 44 |
| 平均 interval coverage | 0.767 |
| speed MAE | 0.418 m/s |
| acceleration MAE | 0.660 m/s² |
| lateral speed MAE | 0.0466 m/s |
| yaw-rate MAE | 0.00746 rad/s |
| curvature MAE | 0.0128 1/m |
| 90% speed posterior coverage | 0.771 |
| 平均 posterior 宽度 | 1.034 m/s |

### 4.1 速度“够不够用”门

当前 provisional proxy gate 采用：`delta-speed MAE <= 0.5 m/s`、显著变化方向
准确率 `>= 0.70`、平均覆盖率 `>= 0.50`。V4 的 `delta-speed MAE` 为 `0.445 m/s`，
显著变化方向准确率为 `0.697`，所以方向门尚差约 `0.003`，整体 provisional gate
为 false。这说明速度已接近可用范围，但不应通过放宽阈值来宣布成功。

### 4.2 增量性与时间特异性

| 对照 | 纵向结果 | 判定 |
|---|---:|---|
| 相对强历史空模型的 speed gain | `-0.143`，95% CI `[-0.236, -0.054]` | 未通过 |
| matched-shuffle speed lift | `0.044`，95% CI `[-0.049, 0.144]` | 未通过 |
| time-reverse speed lift | `0.089`，95% CI `[0.020, 0.158]` | 通过时间顺序检查 |
| acceleration matched-shuffle lift | `-0.004`，95% CI `[-0.189, 0.183]` | 未通过 |

横向速度、yaw rate、curvature 的增量证据已通过对应的历史、shuffle 和时间顺序
检查；纵向速度和加速度没有通过。因此当前最佳结论是“横向连续对齐已有信号，
纵向仍被历史先验解释”，而不是“整体 WAM 因果评测已成立”。

## 5. 当前数据契约问题

现有 `navsim_event_balanced_78_iac_manifest.jsonl` 的协议是
`navsim-state-aware-history4-future4-v1`：4 个历史帧、4 个未来帧、约 2 秒。
这与最终目标 `8` 个未来帧、`4` 秒不一致。必须先构建并审计新的
`history4-future8-4s` 清单，再重复同一 V4 流程。不能用插值帧把 2 秒伪装成 4 秒，
也不能把当前结果写成 8 帧实验。

## 6. 复现

```bash
PYTHONPATH=src:. python scripts/evaluate_continuous_decoder.py \
  --manifest navsim_event_balanced_78_iac_manifest.jsonl \
  --config configs/navsim_continuous_decoder_longitudinal_residual.json \
  --output results/continuous_level1_optimizer_residual_v4_20260827/scores.jsonl \
  --device cuda

PYTHONPATH=src:. python scripts/evaluate_continuous_motion_alignment.py \
  --manifest navsim_event_balanced_78_iac_manifest.jsonl \
  --scores results/continuous_level1_optimizer_residual_v4_20260827/scores.jsonl \
  --reference-source logged_gt \
  --output results/continuous_level1_optimizer_residual_v4_20260827/level1_strict.json
```

## 7. 下一步

先冻结 `8` 帧/`4` 秒的数据契约，并在 NavSim 与 Waymo 各抽取场景不重叠的纵向
激励样本（加速、减速、停止后重启、稳定巡航）。然后保持 V4 超参数不变，重新跑：

1. 速度行为门：`delta-speed` MAE、变化方向、响应延迟、posterior coverage-risk；
2. 三个增量门：强历史空模型、history-speed matched shuffle、time reversal；
3. 只有纵向增量门通过后，才把相对速度升级为正式因果指标，并进入事件级链条。
