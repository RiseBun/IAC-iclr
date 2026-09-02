# IAC：Level-1 到 Level-3 进展与结论

更新时间：2026-09-02  
项目：Imagined-future and Action Consistency（IAC）

## 一句话介绍

IAC 评测的不是“视频像不像”或“任务成功没有”，而是检查 WAM 的**想象未来、动作输出和实际执行**是否形成一致的链条：

```text
history + condition
      ↓
WAM imagined future images
      ↓
candidate-blind image measurement
      ↓
continuous motion profile
      ↓
native action / counterfactual intervention
      ↓
independent closed-loop realization
      ↓
CCFC / FCS
```

核心原则是：动作不能泄漏进图像解码器；图像侧只看历史和 WAM 生成的未来图像，动作只在最后比较阶段出现；闭环结果必须来自独立模拟器，而不是把动作轨迹当成“实际状态”。

## 1. 为什么需要三层

单独的视频质量分数只能说明画面逼真，单独的轨迹或任务成功率只能说明动作结果好，二者都不能证明“模型是因为想象了这个未来才采取这个动作”。因此 IAC 分成三个递进层次：

| 层级 | 要回答的问题 | 输入/输出 | 当前状态 |
|---|---|---|---|
| Level-1 | 图像侧能否稳定读出未来运动？ | future images → 横向/航向/曲率运动量 | **已基本收敛，已形成 benchmark_v1 主表** |
| Level-2 | 想象未来的变化是否对应动作的反事实变化？ | action intervention + future images + native action | **机制验证已跑通，正式 CCFC 尚未成立** |
| Level-3 | 想象、动作和真实执行是否共同预测任务结果？ | CCFC + 独立 rollout + task success | **闭环执行链已跑通，正式 FCS 尚未成立** |

## 2. Level-1：连续图像侧测量

### 2.1 冻结方法

Level-1 使用 4 帧历史图像和 8 帧未来图像，未来时间为 `0.5, 1.0, ..., 4.0 s`：

```text
RAFT-Large forward/backward flow
  → consistency / dynamic suppression
  → ground-plane ego geometry + camera calibration
  → candidate-blind continuous decoder
  → observability / abstention
  → lateral motion, yaw rate, curvature posterior
  → final comparison with WAM action / logged reference
```

速度、加速度和绝对米制前向距离不进入正式主分。相对前向形状可作为辅助诊断；停车由独立 stop 层处理，不能用零光流伪造精确速度。

### 2.2 新 `benchmark_v1` 正式实验

数据为 scene-disjoint 的 580 条记录（NAVSIM 500 + Waymo 80），严格关闭 shape fallback。logged future ego state 只作为测量参考，因此这是图像测量验证，不是 WAM 因果结果。

| 指标 | 结果 | 结论 |
|---|---:|---|
| 非停车形状覆盖 | **440/468 = 94.0%** | 大多数运动样本至少有一个形状区间可观测 |
| 停车识别 | **92/112 = 82.1%** | 停车可单独识别，但不声称精确速度 |
| lateral-speed MAE / 容差内 | **0.095 m/s / 98.4%** | 横向运动幅度稳定 |
| yaw-rate MAE / 容差内 | **0.029 rad/s / 97.0%** | 航向变化稳定 |
| curvature MAE / 容差内 | **0.022 1/m / 86.1%** | 曲率可用，但长尾误差更重 |
| 转弯层 yaw 增量 | **三门通过**（106/114） | 正确 future 优于 history、错未来和倒序对照 |
| 转弯层 curvature 增量 | **三门通过**（106/114） | 使用了 future 内容及其时间顺序 |
| 全池 curvature 增量 | **通过** | 混合 benchmark 上仍有增量特异性 |

三门是 history null、matched-shuffle 和 time-reversal；每个配对差的 bootstrap 95% CI 下界必须大于 0。

### 2.3 可靠性边界

开发审计用于分析长尾，不替代正式主表：

- 78 条开发集：平均 interval observability **77.2%**，全区间可观测样本率 **61.5%**，总体核心通过率 **82.1%**；
- 强转弯区间覆盖 **100%**，但核心通过率只有 **40.0%**，主要问题是横向误差累积而不是不可观测；
- scene-aware 非重叠 25 条：核心通过率 **88.0%**，区间覆盖率 **58.5%**，但制动只有 1 条，统计功效不足；
- 未校准的 joint-pose 后验覆盖率很低，经过独立 conformal 校准后约 **95.6%–97.3%**，但校准样本仍小。

**Level-1 结论：**方向、横向运动、yaw 和相对轨迹形状已经足以作为统一图像侧测量尺；绝对速度、加速度和绝对距离仍只能作诊断。它证明“测量器读到了未来运动”，不证明 WAM 的 F→A 因果性。

## 3. Level-2：Counterfactual Consistency（CCFC）

### 3.1 正式目标

对同一历史构造成对干预，例如 `clear/risk` 或 `left/right`，比较：

```text
Δ imagined motion profile  ↔  Δ native action profile
```

正式 CCFC 需要同时具备：

1. WAM 能生成可解码的 future images；
2. 有独立的 native action head；
3. 干预确实进入模型并有可审计 lineage；
4. 两条分支使用相同 history、seed、采样设置和相机预处理；
5. 反事实差异在 holdout 上通过 action-response 和 specificity gate。

### 3.2 当前实验进展

#### DriveWAM：内部 F→A 机制正例

- future cache 置零 gate：**25/25** 样本 action 发生响应；平均最大 waypoint 差 **4.970**，bootstrap 95% CI **[4.203, 5.756]**；
- wrong-identity future K/V transplant：**22/25 = 88%** 更接近源 B 的 native action；平均轨迹距离减少 **25.2%**，95% CI **[17.7%, 33.5%]**；
- 终点三分量方向一致 **18/25 = 72%**；
- 当前判定：`f2a=content_sensitive_mechanism`。

这证明 action head 读取了 future representation 的内容，而不只是读取“cache 是否存在”；但还没有 clear/risk 语义干预，因此不能称为 semantic CCFC。

#### WorldDrive：原生规划链与内部未来中介

- 原生 action-conditioned future：5/5 有效；lateral MAE **0.3155 m**，yaw MAE **0.0391 rad**，curvature MAE **0.00816 1/m**；
- 正常顺序与倒序比较：误差增量 **+3.0623**，bootstrap 95% CI **[1.6785, 4.4460]**；
- 错身份 action 控制：identity Top-1 **4/5 = 80%**；
- 25 条非重叠历史的内部 future-latent 干预中，仅 **1/25** 达到冻结的 material 横向/航向门槛。

这说明 imagined future 确实能进入 action head，但 material 响应稀疏，不能升级为稳健的模型级 CCFC。

#### Epona 与 DrivingWorld

- Epona 已能运行外部 pose/yaw 控制并生成 4 秒未来，但没有独立 native action head，属于 `externally_controlled_video`；
- DrivingWorld 的旧 reciprocal 实验验证了图像响应链，但缺少正式 native action-lineage，不能直接作为本轮 CCFC 分数。

**Level-2 结论：**我们已经验证了“未来表征 → 动作”的内部机制，并建立了 wrong-identity、time-order 和 action-swap 控制；但公开驾驶 WAM 尚未同时满足 native action、像素 future 和语义 clear/risk intervention 三个条件，因此目前只能报告机制级或受限反事实结果，不能报告正式 semantic CCFC 排行。

## 4. Level-3：Foresight-Conditioned Success（FCS）

### 4.1 正式目标

Level-3 在 Level-2 之上接入独立闭环执行：

```text
imagined future → native action → simulator rollout
                                      ↓
                         realized ego state + task success
```

`realized_future_ego_state`、`task_score/pdm_score` 和 `task_success` 必须由独立模拟器产生，不能直接复用 WAM 输出的 waypoint，也不能把缺失标签记为 0 分。

### 4.2 当前闭环成果

- 已从原生 NAVSIM 数据导出 4 history + 8 future 的对齐记录；
- 5 个 twin、`logged/left/right` 三分支，共 **15 branches**；
- 独立 PDM rollout 成功 **15/15**；
- 临时 task-success **14/15 = 93.3%**；
- action injection lineage **15/15**。

这证明执行、分支标识、realized state 和任务标签的流水线可以工作。但该批记录中的 future image 仍是 `wam_pending`，action 仍属于 staging/native branch condition，不是某个 WAM 的完整原生 action head 输出，因此不能把 93.3% 写成 WAM 的正式 FCS。

### 4.3 当前 Level-3 结论

Level-3 的**执行基础设施已经跑通**，但正式 FCS 还缺：

- 同一 WAM 的真实 native action 分支；
- 每个反事实分支对应的独立 realized state；
- 每个分支的可靠 task-success 标签；
- 至少 25 个独立窗口的统计验证。

## 5. 当前能力分层

| WAM 能力 | 可报告内容 | 当前代表 |
|---|---|---|
| `native_action_conditioned` | Level-1、受控 CCFC、最终 FCS（需通过全部 gate） | DriveWAM、WorldDrive（目前仍是 pilot） |
| `externally_controlled_video` | 受限 action→image 反事实响应 | Epona |
| `video_only` | Level-1 图像测量 | DrivingWorld |
| `action_only` | 动作侧规划诊断 | SimWAM |

因此，IAC 不是为某一个 WAM 定制的评分器，而是通过 adapter 接收：

```text
history + condition
  → future_images + native_action + lineage
```

缺少字段时 fail-closed，只报告该模型实际具备的层级，不用候选轨迹、历史先验或文字事件补齐。

## 6. 总体结论

1. **Level-1 已形成可复现的统一图像侧基准。** `benchmark_v1` 上横向、yaw、曲率的点误差和未来增量均有明确结果；速度与绝对距离被正确降级为诊断。
2. **Level-2 已证明机制存在，但尚未证明语义因果。** DriveWAM 的 wrong-identity transplant 和 WorldDrive 的内部 latent 干预说明未来内容可以影响 action，但 material、semantic、holdout 三个条件尚未同时满足。
3. **Level-3 已完成闭环流水线验证，但没有正式 FCS。** 当前 15 条 rollout 是接口和执行链证据，不是某个 WAM 的最终任务成功分数。
4. **项目已经从“做一个图像质量指标”进入“分层验证 WAM 因果一致性”的阶段。** 下一步不是继续更换光流网络，而是为具备 native action + pixel future 的模型补齐语义干预、独立 realized state 和足够规模的 holdout。

## 7. 推荐引用的现有文档

- 正式发布包：[`benchmark_release_v1/README_zh.md`](../benchmark_release_v1/README_zh.md)
- Level-1 主表：[`benchmark_release_v1/docs/LEVEL1_MAIN_TABLE_BENCHMARK_V1_ZH.md`](../benchmark_release_v1/docs/LEVEL1_MAIN_TABLE_BENCHMARK_V1_ZH.md)
- 三层架构协议：[`docs/IAC_EVENT_CAUSAL_ARCHITECTURE.md`](IAC_EVENT_CAUSAL_ARCHITECTURE.md)
- 旧总状态档案：[`docs/PROJECT_STATUS_20260831_ZH.md`](PROJECT_STATUS_20260831_ZH.md)
