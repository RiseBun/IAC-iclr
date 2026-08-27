# Level-1 纵向能力：独立 Holdout 复核（2026-08-27）

## 目的

验证 V5 的低频速度残差约束是否只改善了原实验 split，还是能在完全相同的
25 条 NAVSIM 场景内不重叠窗口上泛化。V4 与 V5 使用同一 manifest、同一
RAFT-Large 后端、同一 observability 和同一评估脚本；本次 holdout 只用于报告，
不再调参。

## 结果

| 指标 | V4 baseline | V5 curvature residual | V5 变化 |
|---|---:|---:|---:|
| speed MAE（13 个严格可评估样本） | 0.5921 m/s | **0.5285 m/s** | -10.7% |
| acceleration MAE | 0.5534 m/s² | **0.5239 m/s²** | -5.3% |
| delta-speed MAE | 0.6897 m/s | **0.6253 m/s** | -9.3% |
| significant change direction accuracy | 0.6426 | **0.7537** | +0.1111 |
| interval coverage | 0.8173 | 0.8173 | unchanged |
| speed posterior coverage | 0.6556 | 0.6556 | unchanged |

只比较两种方法都标记为 `evaluable` 的相同区间时，配对结果为：

| 分量 | V4 MAE | V5 MAE | V4 - V5 | paired sample bootstrap 95% CI |
|---|---:|---:|---:|---:|
| speed | 0.6123 | **0.5859** | +0.0264 | [0.0031, 0.1615] |
| acceleration | 0.5220 | **0.5003** | +0.0218 | [0.0062, 0.0620] |

共同可评估区间为 85 个，来自 13 条样本。横向速度、yaw-rate 和曲率没有
统计显著变化，说明本次改动主要作用在纵向残差，而不是整体指标漂移。

## 解释边界

V5 的 holdout 泛化增益成立，但它仍不是“速度能力已解决”：

- delta-speed MAE 仍高于当前 `0.5 m/s` provisional gate；
- 普通 change-direction accuracy 仍未达到门槛；
- 速度后验经验覆盖率只有 `0.6556`，低于名义 `0.90`；
- history-only gain、matched-shuffle lift 和 time-order lift 的速度置信区间仍跨过 0；
- 数据是 NAVSIM realized future，只能验证图像测量能力，不能产生 WAM 因果结论。

因此当前决策是：**V5 冻结为正式 WAM Level-1 的图像解码候选，holdout 不再用于
调参；纵向能力继续作为显式失败项报告。** 等真实 WAM 生成 future 到位后，
同一 V5 解码器直接恢复 `m_F(t)`，再与 WAM action head 的 `m_A(t)` 做对齐。

## 复现

```bash
PYTHONPATH=src:. python scripts/evaluate_continuous_decoder.py \
  --manifest datasets/navsim_level1_v5/navsim_level1_v5_eval_nonoverlap.jsonl \
  --config configs/navsim_continuous_decoder_longitudinal_residual_v5.json \
  --output results/level1_v5_holdout_fresh_20260827/scores.jsonl \
  --device cuda

PYTHONPATH=src:. python scripts/evaluate_continuous_motion_alignment.py \
  --manifest datasets/navsim_level1_v5/navsim_level1_v5_eval_nonoverlap.jsonl \
  --scores results/level1_v5_holdout_fresh_20260827/scores.jsonl \
  --reference-source logged_gt \
  --output results/level1_v5_holdout_fresh_20260827/level1_strict.json
```
