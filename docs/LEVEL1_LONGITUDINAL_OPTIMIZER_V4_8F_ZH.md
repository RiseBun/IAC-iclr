# V4 长时域验证：8 帧 / 4 秒

## 1. 数据契约已完成

基于旧 78 个 NavSim 锚点，脚本从原始连续 NAVSIM pickle 回查真实时间序列，
重建了 4 个历史帧和 8 个未来帧。没有插值或复制帧：

```text
navsim_level1_history4_future8_4s_78_manifest.jsonl
```

审计结果：

| 项目 | 结果 |
|---|---:|
| 输入锚点 | 78 |
| 成功转换 | 78 |
| 失败 | 0 |
| 历史帧 | 4 |
| 未来帧 | 8 |
| horizon | 3.999687 - 4.000281 s |
| 缺失图像 | 0 |
| 唯一样本 | 78 |

协议从旧的 `navsim-state-aware-history4-future4-v1` 变为
`navsim-state-aware-history4-future8-v1`，验证器现在同时支持显式的
`history4_future4` 和 `history4_future8`。

## 2. V4 长时域结果

结果目录：
`results/continuous_level1_optimizer_residual_v4_20260827_8f/`

| 指标 | 结果 |
|---|---:|
| 解码成功 | 78 / 78 |
| 严格可评估样本 | 47 |
| 平均 interval coverage | 0.566 |
| speed MAE | 0.740 m/s |
| acceleration MAE | 0.721 m/s² |
| lateral speed MAE | 0.0540 m/s |
| yaw-rate MAE | 0.00751 rad/s |
| curvature MAE | 0.0143 1/m |
| 90% speed posterior coverage | 0.690 |
| 平均 posterior 宽度 | 1.003 m/s |

纵向行为门的结果为：

| 指标 | 结果 | 判定 |
|---|---:|---|
| delta-speed MAE | 0.779 m/s | 未通过 0.5 m/s |
| 显著变化方向准确率 | 0.616 | 未通过 0.70 |
| 平均 coverage | 0.566 | 通过 0.50 |
| provisional capability gate | false | 不能作为正式纵向指标 |

## 3. 增量证据

长时域纵向结果没有超过强历史空模型：speed gain 为 `-0.155`，95% CI
`[-0.263, -0.047]`。matched-shuffle speed lift 为 `0.091`，95% CI
`[-0.003, 0.189]`，没有达到预先规定的显著性门槛。time-reversal speed lift
为 `0.173`，95% CI `[0.102, 0.248]`，说明解码器使用了时间顺序，但不能证明它
恢复了正确的未来纵向变化。

横向速度、yaw rate 和 curvature 仍然同时通过历史空模型、matched-shuffle 和
time-reversal 检查；因此当前最可靠的结论是：**长时域连续横向对齐有增量信号，
纵向速度仍然主要被历史先验解释。**

## 4. 方法学含义

2 秒窗口中的纵向 `0.418 m/s` MAE 不能外推到 4 秒。真实 8 帧实验暴露了残差
在长时域上的漂移和 posterior 欠覆盖（0.690，而名义目标是 0.90）。这验证了
我们的 fail-closed 设计：速度可以作为诊断量输出，但暂时不能升级为正式因果
评测指标，也不能通过扩大容差或后处理校准掩盖长时域失真。

## 5. 复现

```bash
PYTHONPATH=src:. python scripts/extend_navsim_manifest_to_8f.py \
  --input navsim_event_balanced_78_iac_manifest.jsonl \
  --pkl-root /path/to/navsim_logs/mini \
  --sensor-root /path/to/sensor_blobs/mini \
  --output navsim_level1_history4_future8_4s_78_manifest.jsonl \
  --audit-output navsim_level1_history4_future8_4s_78_audit.json

PYTHONPATH=src:. python scripts/evaluate_continuous_decoder.py \
  --manifest navsim_level1_history4_future8_4s_78_manifest.jsonl \
  --config configs/navsim_continuous_decoder_longitudinal_residual.json \
  --output results/continuous_level1_optimizer_residual_v4_20260827_8f/scores.jsonl \
  --device cuda

PYTHONPATH=src:. python scripts/evaluate_continuous_motion_alignment.py \
  --manifest navsim_level1_history4_future8_4s_78_manifest.jsonl \
  --scores results/continuous_level1_optimizer_residual_v4_20260827_8f/scores.jsonl \
  --reference-source logged_gt \
  --output results/continuous_level1_optimizer_residual_v4_20260827_8f/level1_strict.json
```

## 6. 下一步

不要继续调纵向阈值。下一步应固定 V4 参数，增加真正的长时域纵向激励样本，
并引入一个独立的米制深度或多视角 challenger，检验问题来自光流地面尺度还是
来自未来图像本身。只有在 8 帧/4 秒上重复通过三道增量门，纵向相对速度才可
进入正式 WAM 因果指标。
