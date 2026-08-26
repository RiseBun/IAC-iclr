# IAC 方法更新（2026-08-24）

## 本轮确定的评测边界

IAC 的主任务是从历史图像和 WAM 未来图像恢复“可行轨迹”，不是估计精确车速。主轨迹分由三个分量组成：横向位置、航向变化、曲率。速度仍然输出，但单独报告不确定度和 `usable / uncertain / abstain` 状态，不进入主联合分，除非显式设置 `continuous_decoder.score_speed=true`。

SegFormer 和 temporal-road consensus 只修改像素权重；默认不删除非道路像素，也不把 actor mask 作为硬几何约束。硬约束只用于单独的消融实验。

每个未来时间区间都输出：有效静态像素比例、流幅值、前后向一致性、方向可观测状态、速度状态和速度不确定度。方向区间不可观测时才影响 IAC 主结果；速度区间 abstain 不会污染方向/曲率评分。

## 上限实验

`work_dirs/iac_synthetic_upper_bound_20260824_v3.json` 使用已知相机模型、已知平面道路和已知 SE(2) 运动合成光流，再运行完整 candidate-blind decoder：

- 无噪声：主联合误差 `0.000134`，平均横向误差 `4.5e-5 m`，平均航向误差 `1.2e-5 rad`。
- 1 px 高斯光流噪声：主联合误差约 `0.00503`，平均横向误差 `0.00133 m`，航向余弦 `0.999995`。

这证明优化器和轨迹参数化本身可以达到很低误差；真实数据上的误差主要来自光流、深度/平面假设、动态目标和长时累计，而不是“轨迹点形式”本身。

现有 NAVSIM pose+LiDAR oracle：
`artifacts/navsim_oracle_flow_50_summary.json`

- 50 条记录中 45 条可评估，5 条因缺 LiDAR 无法评估。
- 未来地面区域中位 EPE `1.18 px`。
- 中位方向余弦 `0.9984`。
- 中位观测/预测流尺度比 `0.619`，说明尺度仍明显不稳定，不能把速度当作可靠主分量。

## WAM 接入诊断

`work_dirs/wam_iac_diagnosis_20260824.json` 将两个门分开：

1. IAC recovery gate：candidate-blind 解码与独立 realized ego state/日志轨迹比较。
2. WAM future-response gate：动作干预是否改变未来图像。

当前弯道小批量中，IAC 方向门通过（航向余弦约 `0.99995`、横向误差 `0.459 m`），WAM 响应门不通过（动作-图像距离相关性 `0.077`、响应比 `0.097`）。因此此批结果应标为 `wam_future_response_weak`，不能判为 IAC 恢复失败。

正式 Counterfactual Consistency / Foresight-Conditioned Success 仍要求有成对 action、未来图像、独立 realized state，以及 task_success；缺少这些字段时必须保持 `formal_benchmark_ready=false`。
