# Level 1 纵向残差增强 V3

## 1. 结论先行

V3 将保留测试集上的纵向速度 MAE 从原始 RAFT 解码器的 `1.396 m/s`
降到 `0.357 m/s`，但仍差于只使用历史状态的 CA-CYR 强基线，后者约为
`0.310 m/s`。因此这是有效的误差压缩，不是通过 Level 1 的纵向增量能力。

禁止根据测试结果继续选择融合权重。下一步必须改变图像测量本身。

## 2. 为什么不能直接融合绝对速度

单目前视图像的绝对尺度不稳定，而历史状态已经提供准确的 `t=0` 米制速度。
让 RAFT 再估计一次绝对速度会把深度尺度误差直接带入评测。V3 只保留未来图像
速度序列的相对变化率：

```text
r_image = slope(v_raw) / scale(v_raw)
delta_v_image(t) = v0_history * r_image * t
innovation(t) = delta_v_image(t) - delta_v_history(t)
v_pred(t) = v_history(t) + beta * innovation(t)
```
`beta=0` 时预测严格等于 CA-CYR history null。绝对图像速度、未来 action waypoint
和候选轨迹均不进入该预测器。

## 3. 防泄漏校准协议

78 个样本按 `scene_id` 和固定 SHA-256 顺序拆分：

| 子集 | 用途 | 样本数 |
|---|---|---:|
| fit | 只拟合单一 `beta` | 36 |
| calibration | 只确定 90% conformal 区间 | 26 |
| evaluation | 最终能力判断，前两步完全不可见 | 16 |

拟合集得到 `beta=0.12693`。校准集得到速度区间半径 `2.118 m/s`。
校准目标仅为 logged trajectory proxy，不使用 WAM action head。

## 4. 严格测试结果

16 个 evaluation 样本中有 10 个满足严格 observability，共 30 个 usable interval。

| 指标 | 结果 | 是否通过 |
|---|---:|---|
| 原始 RAFT speed MAE | 1.396 m/s | 否 |
| V3 speed MAE | 0.357 m/s | 改善但未过 history null |
| V3 acceleration MAE | 0.428 m/s² | 未过 history null |
| speed gain over CA-CYR | -0.047，95% CI [-0.109, 0.011] | 否 |
| acceleration gain over CA-CYR | -0.046，95% CI [-0.097, 0.006] | 否 |
| matched-shuffle speed lift | 0.013，95% CI [-0.034, 0.068] | 否 |
| time-reverse speed lift | 0.071，95% CI [-0.004, 0.191] | 否 |
| 90% speed interval coverage | 100%，平均宽度 3.837 m/s | 覆盖但过宽 |

fit 上的 speed gain 为 `+0.052`，95% CI `[0.017, 0.095]`；到 calibration
变为 `-0.192`，到 evaluation 为 `-0.047`。这证明全局线性 `beta` 不能跨场景泛化，
不能把拟合集结果当作能力结论。

当前数据仍是 logged future、`4 帧/2 秒`，正式 WAM Level 1 资格为 false。

## 5. 下一步模型改造

停止继续调后处理融合权重。下一版应在连续轨迹优化器内部直接估计图像纵向残差：

1. CA-CYR 生成冻结的历史速度先验 `v_history(t)`；
2. 优化变量改为有界的逐时刻残差或加速度残差，而不是绝对速度；
3. 光流重投影误差负责决定残差，历史先验只提供零点与正则；
4. 使用 8 帧时序对应约束残差平滑，并保留 forward/backward consistency；
5. 在新的、未用于 V3 诊断的纵向激励样本上重新验证三道门。

如果直接残差优化仍不能跨场景超过 CA-CYR，再引入 SEA-RAFT/AllTracker 或米制深度
challenger。当前证据不支持先增加事件层。

## 6. 复现

```bash
PYTHONPATH=src:. python scripts/calibrate_longitudinal_residual.py \
  --manifest navsim_event_balanced_78_iac_manifest.jsonl \
  --scores results/flow_ab_event78_20260826/raft_large/scores.jsonl \
  --output results/continuous_level1_longitudinal_relative_v3_20260827/calibration.json

PYTHONPATH=src:. python scripts/evaluate_continuous_motion_alignment.py \
  --manifest navsim_event_balanced_78_iac_manifest.jsonl \
  --scores results/flow_ab_event78_20260826/raft_large/scores.jsonl \
  --reference-source logged_gt \
  --longitudinal-calibration \
    results/continuous_level1_longitudinal_relative_v3_20260827/calibration.json \
  --output results/continuous_level1_longitudinal_relative_v3_20260827/test_strict.json
```
