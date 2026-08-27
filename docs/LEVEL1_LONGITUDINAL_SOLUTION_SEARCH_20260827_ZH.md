# Level 1 纵向能力：全网解法收敛与 V5 方案

## 1. 先给结论

当前 8 帧/4 秒结果说明，问题不是继续调 V4 的残差权重，而是未来段没有一个跨时间持续的米制尺度和深度约束。V4 对每个未来间隔单独修正速度，短时可降低误差，长时却会暴露尺度漂移、动态区域污染和误差累积。

因此下一版采用 **history-calibrated persistent scale + global temporal residual**：

```text
历史自车状态 + 历史/未来图像
  -> RAFT-Large F/B consistency
  -> 静态地面/平面几何与动态抑制
  -> 历史段米制尺度校准（只用 t <= 0 的状态）
  -> 未来段共享尺度与时序深度更新
  -> 全局速度/加速度残差优化（8 个间隔联合）
  -> motion posterior + observability/abstention
  -> 与 action waypoint 的连续运动学直接比较
```

这条主线比“直接换一个 monocular depth 网络”更适合 IAC：深度网络只能提供辅助几何证据，不能成为评测真值；动作 waypoint 仍然只在最后的比较阶段进入。

## 2. 当前失败结果告诉我们什么

在 78 个样本的真实 8-frame/4-second manifest 上，V4 解码完整率为 78/78，但严格可评估样本只有 47 个。speed MAE 为 `0.740 m/s`，acceleration MAE 为 `0.721 m/s^2`；相对于 history-only null 的 speed gain 为 `-0.155 m/s`，95% CI `[-0.263, -0.047]`。因此纵向增量门仍为 false。

同时，lateral speed、yaw rate 和 curvature 仍有可分辨信号。这说明 RAFT 的基本时间匹配没有失效，主要失效点是：

1. 单目光流的像素运动到米制纵向速度的尺度在未来段没有持续约束。
2. 每个间隔独立修正，误差会在 4 秒窗口内漂移。
3. 未来帧中的动态物体、遮挡和低纹理区域会污染纵向几何。
4. 绝对速度比 `delta-v` 更容易被历史速度先验“解释掉”，所以必须同时报告变化量、响应时延和不确定性。

## 3. 搜索到的方法如何取舍

### 3.1 主线：PTC-Depth 风格的持久尺度更新

[PTC-Depth](https://openaccess.thecvf.com/content/CVPR2026/html/Han_PTC-Depth_Pose-Refined_Monocular_Depth_Estimation_with_Temporal_Consistency_CVPR_2026_paper.html) 用光流三角化产生稀疏米制深度，再利用外部位姿/里程计进行递归的贝叶斯尺度更新，最后把相对深度校准为时间一致的 metric depth。官方实现还允许输入外部 `R/t` 和预计算光流，推理不依赖 LiDAR。

这与我们的接口最匹配：历史 `pose/speed/yaw_rate` 已经存在，RAFT 也已经存在。V5 应将其改造成：

- 用 3 个历史间隔估计静态 ROI 的 robust scale posterior；
- 对未来 8 个间隔共享一个尺度状态，而不是每帧重新估计；
- 只用图像几何和历史状态更新尺度，禁止读取未来 action/waypoint；
- 维护 `scale_mean/scale_var`，将尺度不确定性传播到速度 posterior；
- 在遮挡或静态点不足时 abstain，而不是硬输出速度。

### 3.2 几何增强：MonoPP / 地面平面

[MonoPP](https://mono-pp.github.io/) 用已知相机安装位置和 planar-parallax geometry，从单目视频蒸馏 metric-scaled depth、可行驶区域和动态物体掩码，并辅助 metric pose。它验证了“相机高度 + 地面平面 + 时序几何”比裸单帧深度更能提供汽车场景的尺度。

在 IAC 中，它不应替代 RAFT，而应作为静态地面 ROI 与动态抑制的几何 teacher：

- 地面平面只用于尺度/自车运动估计；
- 非地面区域只作为一致性检查，不直接决定 ego speed；
- 采用 Huber/RANSAC，避免车辆、行人和路边物体进入尺度拟合。

### 3.3 辅助先验：UniDepthV2 / Metric3Dv2

[UniDepthV2](https://arxiv.org/abs/2502.20110) 直接从单帧预测 camera-conditioned metric 3D points，并输出不确定性；[Metric3Dv2](https://arxiv.org/abs/2404.15506) 通过 canonical camera transform 处理焦距/相机差异，并证明 metric depth 能降低 SLAM 平移漂移。

它们适合做三件事：

- 未来帧的深度先验和异常检测；
- PTC/地面几何失败时的 challenger；
- 按场景分层的诊断，不直接充当正式标签。

原因是 foundation depth 在跨域、动态场景和相机内参变化下仍可能有系统性尺度偏差；若把它直接当真值，会把模型偏差写进评测器。

### 3.4 learned challenger：SpatialTrackerV2

[SpatialTrackerV2](https://spatialtracker.github.io/) 是前馈式单目 3D 点跟踪器，联合估计相机运动、一致场景几何和像素级世界坐标轨迹，并分解自车运动与物体运动。它很适合作为独立 learned challenger：如果它和可解释的 RAFT+地面几何在静态场景上都给出相同纵向趋势，证据更强；但它不应替代正式基线，因为其自身也是被评估的学习系统。

### 3.5 暂不作为本版修复：WAM 训练侧方法

[4D-WAM](https://arxiv.org/abs/2608.10107) 和基于 4D trajectory fields 的 WAM 方法把几何/轨迹一致性加入 WAM 训练，可作为以后提升被评 WAM 的方案。但它们不能解决当前评测器的尺度漂移，也不能在评测时读取 action 作为图像分支输入，所以不放入 V5 的主修复。

## 4. V5 的具体方法结构

### 4.1 输入契约

- `history_images`: 4 个历史前视帧；
- `future_images`: 8 个未来前视帧，`0.5 ... 4.0 s`；
- `history_ego_state`: pose、speed、acceleration、yaw rate；
- `history_waypoints`: 仅允许 `t <= 0`；
- `camera`: intrinsics、extrinsics、distortion；
- 可选 `metric_depth_prior`: UniDepthV2/Metric3Dv2/MonoPP；
- 可选 `spatial_tracker_prior`: SpatialTrackerV2 输出，仅作 challenger。

未来 action waypoint 不得进入图像解码器、尺度校准器、动态掩码或任何 hyper-parameter 选择过程。

### 4.2 共享尺度状态

对历史相邻帧，在静态地面 ROI 上计算：

```text
s_i ~= metric_displacement(history_state_i, history_state_{i+1})
      / image_geometry_displacement_i
```

用 Huber/RANSAC 得到历史尺度 posterior `q(s_0)`。未来每个间隔的三角化深度、地面平面残差和 F/B consistency 只更新 `q(s_t)`；采用随机游走或一阶平滑先验，禁止独立重置尺度。

### 4.3 全局纵向优化

将 V4 的独立 interval residual 改为 8 个间隔联合优化：

```text
v_k = v_history_prior(k) + r_k
v_{k+1} - v_k = dt * a_k
L = L_flow_reprojection
  + lambda_scale * L_scale_posterior
  + lambda_smooth * ||Delta a||_1
  + lambda_prior * L_history_boundary
```

`r_k` 用低频 spline 或二阶平滑参数化，允许真实制动/加速但抑制逐帧抖动。输出同时包含 `v`, `delta_v`, `a`, `scale_posterior`, `coverage` 和 `abstain_reason`。

### 4.4 与 waypoint 的比较

最后才从 action waypoint 计算 `m_A(t)=[speed, acceleration, lateral speed, yaw rate, curvature]`，和图像侧 posterior `m_F(t)` 做：

- speed / `delta-v` MAE；
- acceleration MAE；
- response-delay error；
- sign accuracy（仅作诊断，不替代连续量）；
- interval score、90% coverage、coverage-risk；
- 相对于 history null、matched shuffle、time reversal 的 paired gain。

## 5. 实验顺序与通过门槛

固定同一 78-sample split，按以下顺序做消融：

1. V4 独立残差（现有复现基线）；
2. V5 共享尺度 + 全局残差；
3. V5 + MonoPP/地面平面动态抑制；
4. V5 + UniDepthV2 或 Metric3Dv2 先验；
5. SpatialTrackerV2 learned challenger。

每组都必须报告 2 秒、3 秒、4 秒分桶结果，而不是只报一个平均值。正式纵向指标只有在以下条件同时满足后才启用：

- speed 和 `delta-v` 相对 history null 的 paired gain 95% CI 下界大于 0；
- matched-shuffle lift 与 time-reversal lift 通过方向性检验；
- 90% posterior coverage 接近名义覆盖，且 coverage-risk 曲线单调；
- 4 秒误差不出现相对 2 秒的失控增长；
- 主线与至少一个独立 challenger 在静态/制动/加速分层上趋势一致。

在门槛通过前，速度仍是 Level 1 的 diagnostic probe，不能被写成已验证的正式因果指标。

## 6. 这次搜索后的决策

**立即实现：** PTC-Depth 风格的历史米制尺度 posterior、共享尺度状态和 8 间隔全局纵向优化；保留 RAFT-Large F/B consistency 与地面平面。

**并行接入：** MonoPP 风格地面/动态 mask；UniDepthV2 或 Metric3Dv2 作为辅助先验和失败诊断。

**独立验证：** SpatialTrackerV2 challenger。

**暂缓：** 事件级标签、WAM 训练侧 4D-WAM 改造，直到 Level 1 纵向连续量先通过增量门。

## 参考

- Han et al., *PTC-Depth: Pose-Refined Monocular Depth Estimation with Temporal Consistency*, CVPR 2026: https://openaccess.thecvf.com/content/CVPR2026/html/Han_PTC-Depth_Pose-Refined_Monocular_Depth_Estimation_with_Temporal_Consistency_CVPR_2026_paper.html
- Elazab et al., *MonoPP: Metric-Scaled Self-Supervised Monocular Depth Estimation by Planar-Parallax Geometry*, WACV 2025: https://mono-pp.github.io/
- Piccinelli et al., *UniDepthV2*, 2025: https://arxiv.org/abs/2502.20110
- Yin et al., *Metric3Dv2*, 2024: https://arxiv.org/abs/2404.15506
- Tusk et al., *SpatialTrackerV2*, 2025: https://spatialtracker.github.io/
- Chen et al., *Video Depth Anything*, CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Video_Depth_Anything_Consistent_Depth_Estimation_for_Super-Long_Videos_CVPR_2025_paper.html
- ACT-Bench / ACT-ESTIMATOR, ICLR 2025: https://openreview.net/pdf/1619cbb8136e375835171538185debedcc483a99.pdf
- *4D-WAM*, 2026: https://arxiv.org/abs/2608.10107
