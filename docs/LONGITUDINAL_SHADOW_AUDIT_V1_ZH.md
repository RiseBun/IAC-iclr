# 纵向 shadow residual 审计 v1

## 目的

本审计验证 `UniDepth-L + PnP` 纵向尺度源是否能在不读取 action waypoint 的条件下，稳定优于正式的 `constant_acceleration_yaw_rate` history null。审计只使用已有 probe 输出，不重新训练模型。

## 非重叠集结果

`eval_nonoverlap` 含 25 个窗口、19 个 scene、175 个 interval。结果为：

| 方法 | MAE | 相对强 null 的平均增益 | interval 配对 95% CI | scene-cluster 95% CI | 胜出率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RAFT-Large + UniDepth-L + PnP | `0.343 m` | `0.127 m` | `[0.029, 0.221]` | `[-0.072, 0.336]` | `46.9%` |
| NeuFlow v2 + UniDepth-L + PnP | `0.323 m` | `0.147 m` | `[-0.051, 0.284]` | `[-0.127, 0.374]` | `53.1%` |
| constant-acceleration-yaw-rate null | `0.470 m` | - | - | - | - |

interval 级 bootstrap 会把同一 scene 的相关样本当成独立样本，因此正式判断看 scene-cluster CI。两种方法的 scene CI 均跨 0，尚不能声称对独立场景有稳定增益。

## 运动分桶

RAFT 与 NeuFlow 的趋势一致：

- acceleration：平均增益约 `0.306-0.307 m`，95% CI 约 `[0.153, 0.465]`；
- braking：RAFT `0.489 m`、NeuFlow `0.524 m`，95% CI 均为正；
- cruise：RAFT `-0.114 m`、NeuFlow `-0.080 m`，CI 跨 0，方法不优于强 null。

这说明纵向图像证据主要在加速和制动变化中有用，匀速场景不应强行计入正式幅度分。

## `0.3393201223` 伪影

该值只出现在 NeuFlow 输出，不出现在 RAFT 输出。`all_78` 中出现 5 次，但 5 次都对应完全相同的帧对：

`352b2504322b522a.jpg -> 2879c99b2db55234.jpg`

这是重叠窗口重复使用同一帧对造成的 5 次记录；去重后只有 1 个 unique frame pair。`eval_nonoverlap` 中 NeuFlow 也只出现 1 次。因此不能把它解释成 5 个独立失败，但必须在实现中对该帧对触发异常值/不确定性处理。

## 当前决策

1. `UniDepth-L + PnP` 吸收为纵向 shadow residual；
2. RAFT 继续负责 lateral/yaw/curvature；NeuFlow 不替代 RAFT；
3. cruise 场景采用 abstain/低权重，不作为纵向幅度证据；
4. scene-cluster CI 跨 0，因此纵向残差暂不升级为正式 IAC 指标；
5. 下一步不是继续换 flow，而是扩大 scene-disjoint 数据并加入 reciprocal identity/order 控制。
