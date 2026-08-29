# Pure-speed reciprocal shadow residual v1

## 目的与边界

这个分支只回答一个窄问题：在历史状态相同、快慢候选互换的视频 twin 中，图像侧测量是否能判断哪一段视频代表更快的前向进度。它是 Level-1 的候选盲辅助残差，不是 IAC 正式因果指标，也不证明 WAM 的动作由未来图像驱动。

NeuFlow 提供时序光流，UniDepth-L 提供单帧米制深度辅助。深度只能用于尺度/前向进度残差，不能作为真值；相机标定和地面几何仍是主尺度约束。

## 输入协议

manifest 与 scores 必须分离：

- manifest：`video_id`、`twin_id`、`condition`（`fast`/`slow`）、`future_times_s`，以及可选 `control_type`；
- scores：只允许候选盲图像测量，例如 `predicted_progress_curve`、`predicted_trajectory` 或 `predicted_motion.forward_rate_mps`；不能读取 action head、候选 ID 或 waypoint。

每个 `(twin_id, control_type)` 必须恰好有一条 fast 和一条 slow。统计在 twin 级别完成，避免 reciprocal 两行造成伪增大样本量。

## 指标与控制

- `condition_accuracy`：预测 fast 进度严格大于 slow 的 twin 比例，并给出 bootstrap 95% CI；
- `margin_fast_minus_slow`：快慢进度率差；
- `time_reversal_drop`：clean 准确率减去时间倒序准确率；
- `wrong_identity_rejection`：错误身份控制下的拒绝率，即 `1 - condition_accuracy`。

时间倒序检验时间节奏，wrong-identity 检验身份绑定。常量速度和 history-only 结果应作为外部基线写入报告，不进入候选盲 scores。

## 可复现实验

```bash
python scripts/evaluate_pure_speed_reciprocal.py \
  --manifest <manifest.jsonl> \
  --scores <candidate_blind_scores.jsonl> \
  --output <pure_speed_report.json>
```

只有在独立、scene-disjoint 的更大 twin 池上同时满足以下条件，才考虑把该残差升级为正式指标组成部分：clean-case 增益的 bootstrap 区间不跨 0；时间倒序显著下降；wrong-identity 接近随机水平；加入后不损害 lateral/yaw/turn-shape 主分；并且覆盖率和 abstention 风险可校准。当前服务器没有原始 11-twin pilot 的 manifest/scores，因此不能重报那组数字。

## NAVSIM 78 样本能力试验

在服务器上用 NAVSIM Level-1 的 78 个 8 帧、4 秒样本，NeuFlow v2 `neuflow_mixed.pth` 提取相邻帧对应，UniDepth v2 ViT-L 提供米制深度；随后在道路下半区用 `solvePnPRansac` 从 3D-2D 对应估计自车前向位移。没有把 action waypoint 输入该分支。

结果和前端拆分如下：

| 前端 | 平均进度 MAE | 中位数 | P90 | 相对 history-only | 逐样本胜出 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RAFT-Large + UniDepth-L + PnP | `0.347 m` | `0.270 m` | `0.785 m` | `-26.5%` | `40/78` |
| NeuFlow v2 + UniDepth-L + PnP | `0.327 m` | `0.252 m` | `0.689 m` | `-30.8%` | `38/78` |
| history-only | `0.472 m` | - | - | - | - |

两种前端的区间符号准确率均为 `93.59%`。因此主要增益来自 UniDepth-L 的米制深度和 PnP 相机几何，而不是 NeuFlow 本身；NeuFlow 相对 RAFT 的平均改善只有 `0.020 m`，且逐样本胜出更少。当前推荐保留 RAFT 作为 lateral/yaw/turn-shape 前端，仅把 UniDepth-L + PnP 吸收到纵向尺度分支。该分支可以作为 Level-1 的 shadow residual / 辅助证据；它尚未通过 reciprocal identity/order 控制，也未取得正式因果指标资格。

运行脚本为 `scripts/run_neuflow_unidepth_probe.py`。该试验还暴露出一个重要实现边界：仅用竖直光流乘深度的简化公式 MAE 为 `2.080 m`，加入相机外参和 PnP 后才降至 `0.327 m`，说明几何约束是必要条件。
