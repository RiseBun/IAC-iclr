# Level 1 纵向增强 V5：低频残差约束

## 1. 改动

V5 在 V4 的 history-anchored optimizer residual 上增加：

```text
lambda_curvature * mean((Delta^2 r_k / r_scale)^2)
```

其中 `r_k = v_k - v_history,k`。它不禁止真实的线性制动/加速趋势，只抑制没有连续几何证据支持的曲率型漂移。V4 配置保持不变；V5 配置为
`configs/navsim_continuous_decoder_longitudinal_residual_v5.json`，权重为 `0.08`。

## 2. 同集配对实验

实验集固定为 `datasets/navsim_level1_v5/navsim_level1_v5_eval_nonoverlap.jsonl` 的 25 条场景内不重叠窗口。两组都完成 25/25 解码，严格可评估样本都是 13 条，平均 interval coverage 都为 `0.817`。

| 指标 | V4 | V5 | 变化 |
|---|---:|---:|---:|
| speed MAE | 0.592 m/s | 0.528 m/s | -0.064 |
| acceleration MAE | 0.553 m/s² | 0.524 m/s² | -0.030 |
| delta-speed MAE | 0.690 m/s | 0.625 m/s | -0.064 |
| significant direction accuracy | 0.643 | 0.754 | +0.111 |
| speed posterior coverage | 0.656 | 0.656 | unchanged |
| speed abstain fraction | 0.175 | 0.175 | unchanged |

V5 的相对 speed MAE 改善约为 `10.7%`。这证明二阶约束减少了长时域残差漂移，但还没有证明未来图像包含超出历史先验的因果信息：

- speed 相对 history null 的 95% CI 仍跨过 0；
- matched shuffle 与 time reversal 仍未同时通过；
- 速度 posterior 覆盖率 `0.656`，低于名义 `0.90`；
- native realized future 仍然只是图像测量验证，不是 WAM 生成未来。

因此 V5 是“能力增强成功、正式纵向因果门仍未通过”。

## 3. 复现

```bash
PYTHONPATH=src:. python scripts/evaluate_continuous_decoder.py \
  --manifest datasets/navsim_level1_v5/navsim_level1_v5_eval_nonoverlap.jsonl \
  --config configs/navsim_continuous_decoder_longitudinal_residual_v5.json \
  --output results/level1_v5_curvature_20260827_eval25/scores.jsonl \
  --device cuda

PYTHONPATH=src:. python scripts/evaluate_continuous_motion_alignment.py \
  --manifest datasets/navsim_level1_v5/navsim_level1_v5_eval_nonoverlap.jsonl \
  --scores results/level1_v5_curvature_20260827_eval25/scores.jsonl \
  --reference-source logged_gt \
  --output results/level1_v5_curvature_20260827_eval25/level1_strict.json
```

下一步仍需接入持久米制尺度 posterior；二阶残差约束只是 V5 的第一块，不能替代 PTC-Depth 风格的跨未来尺度校准。
