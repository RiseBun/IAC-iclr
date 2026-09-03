# CCFC 子分解与真实视频退化审计（2026-09-03）

## 目的

当前成对干预 CCFC 的几何平均分为 0.1235。几何平均会被任一子项拖低，因此先拆分 `response_direction`、`response_magnitude` 和 `response_temporal_alignment`，再判断是否需要修改光流探针。与此同时，在同一批真实未来视频上加入受控模糊、JPEG 压缩和亮度闪烁（B 组），用于区分“探针域偏差”和“模型生成视频本身的问题”。本实验不修改正式阈值、不改变主榜分数。

## 实验协议

- 数据：benchmark-v1 中按清单顺序固定的 78 条样本；A/B 使用完全相同的 sample ID、时间戳、相机标定和 logged GT。
- A 组：原始 logged 真实视频。
- B 组：仅替换图像，参数为 Gaussian blur σ=1.5、JPEG quality=50、逐帧亮度闪烁幅度 10%；历史帧和未来帧均处理，状态与标签不变。
- 探针：冻结 `configs/plane.json` 与冻结 RAFT-Large；`--include-uncertain --disable-shape-fallback`。
- 结果位置（服务器）：
  - A：`/mnt/slurmfs-4090node3/iac_b_controls/pilot78/alignment_A.json`
  - B：`/mnt/slurmfs-4090node3/iac_b_controls/pilot78/alignment.json`

## A/B 结果（78 条）

| 指标 | A 原始真实视频 | B 中等退化 | B−A |
|---|---:|---:|---:|
| 可评估样本 | 60/78 | 60/78 | 0 |
| 平均 interval coverage | 0.7354 | 0.7333 | −0.0021 |
| 横向速度 MAE | 0.1578 m/s | 0.0247 m/s | −0.1331 |
| 横向速度容差内比例 | 97.65% | 100.00% | +2.35 pp |
| yaw-rate MAE | 0.0406 rad/s | 0.0059 rad/s | −0.0347 |
| yaw-rate 容差内比例 | 96.75% | 99.09% | +2.34 pp |
| 曲率 MAE | 0.0179 1/m | 0.0168 1/m | −0.0011 |
| 曲率容差内比例 | 88.49% | 91.15% | +2.66 pp |
| 归一化相对轨迹 profile MAE | 0.0837 | 0.0733 | −0.0104 |
| 弧长归一 path cosine | 0.9699 | 0.9762 | +0.0063 |
| 可用 interval（624 个） | 353 | 352 | −1 |
| abstain interval（624 个） | 271 | 272 | +1 |

## 解释

1. **轻度模糊不是当前低 CCFC 的主因。** B 组没有出现 coverage 崩溃或 abstain 激增，横向/yaw/曲率和归一化轨迹反而略有改善，说明原始高频噪声被平滑后读数更稳定。
2. **“能读出就计分、读不出就 abstain”的原则在当前门控下成立。** B 组 624 个 interval 中只发生 1 个可用/abstain 变化；探针没有把退化统一打成错误分。
3. **该对照不能替 WAM 生成域背书。** A/B 都是真实 logged 视频，不能排除 WAM 生成视频的闪烁、结构漂移或转弯运动不真实。它只排除了“普通图像退化会让冻结探针整体失效”这一解释。
4. **正式分数暂不调整。** 仍需找到并审计产生 357/580 的原始 CCFC 行级报告；在原始报告缺失前，不从 pilot 子项均值推断正式 CCFC。

## 下一步

- 对原始 357 条 CCFC 报告输出三项子分布及配对 bootstrap CI，确认 0.1235 主要由哪个子项拖低。
- 对 WAM 真生成视频做同样的身份错配、时间倒放和最差转弯可视化；这一步才区分“模型没有生成对应运动”和“探针读不动生成域”。
- 在 dev 集上再评估信号专属 observability（shape 与 speed 分开）；benchmark 继续冻结，不在同一批 580 条上调阈值。

本文件是诊断审计，不是新的主榜结果。

## 当前可追溯的 CCFC pilot 子分（非 357 条正式表）

目前服务器上能找到的原始 CCFC 行级文件是 5/25 条 pilot，而不是记分板所引用的 357/580 全量文件。对可追溯 pilot 的均值如下：

| 文件 | 几何分数（metric） | direction | magnitude | temporal |
|---|---:|---:|---:|---:|
| command 0→2（正确配对，30 行） | 0.654 | 0.849 | 0.571 | 0.802 |
| command 0→2（图像错配，30 行） | 0.124 | 0.151 | 0.341 | 0.198 |
| 5-sample command（正确配对，15 行） | 0.614 | 0.998 | 0.528 | 0.994 |
| 5-sample command（图像错配，15 行） | 0.008 | 0.002 | 0.168 | 0.006 |

这组 pilot 说明探针确实读到了干预内容（错配时 direction/temporal 接近零），同时也说明 metric 几何平均容易被 magnitude 拖低。它不能替代 357 条正式分布；正式报告找到后必须按同样脚本重算并附配对 bootstrap CI。
