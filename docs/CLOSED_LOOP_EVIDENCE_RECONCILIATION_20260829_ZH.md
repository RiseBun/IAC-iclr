# 闭环证据对账（2026-08-29）

## 结论先行

项目之前确实做过闭环/干预实验。此前“没有闭环实验”的说法不准确；准确说法是：

> 已完成闭环干预诊断，但旧结果尚未按当前 Level-2/Level-3 数据契约接入正式评分器。

因此，旧实验可以作为模型响应失败或 IAC 上界的证据，不能直接充当最终联合 WAM 指标。

## 已完成的实验

| 实验 | 规模 | 已证实内容 | 证据边界 |
|---|---:|---|---|
| DriveWAM action-image intervention | 20 scenes × 3 branches = 60 | 同一历史输入下注入 `logged/left/right` 动作，且 `action_injection_verified=true` | 该 checkpoint/configuration 的 action-to-image 响应很弱 |
| Native real future-image control | 60 branches | 同一 IAC 能从真实 NAVSIM future image 提取可用轨迹支持 | IAC 能力上界，不是 WAM 因果证据 |
| NuPlan closed-loop simulation | 多次 `closed_loop_nonreactive_agents` / `closed_loop_reactive_agents` runs | 规划器在仿真中的执行、碰撞、舒适性、进度等指标 | aggregate simulation metrics，没有 WAM imagined-future/action 的逐分支配对 |

## 其他 WAM 的历史实验清单

| WAM / 模型 | 做过的实验 | 历史结果或规模 | 当前证据等级 |
|---|---|---:|---|
| DrivingWorld | reciprocal paired motion probe；同 history 的候选分支 | calibration `11/12`，holdout `15/16`；action-image diagonal Top-1 `0.8125`；CC lift `0.3612` | 已完成的 WAM 因果响应审计；没有独立 rollout/FCS |
| Epona | native NAVSIM 生成与真实 future 对照；单样本 smoke/action-control；归档的 4 条 native batch | generated compatibility `0.3279`；control compatibility `0.6352`；每组 4 条；无 task-success | 已完成图像测量/模型对照；不是正式闭环成功率 |
| DriveWAM | 20-scene action-image intervention；另有归档的 4 条 native generated 解码 | intervention 为 60 branches，generated compatibility `0.0876`；归档 batch compatibility `0.2970`；无 task-success | 已完成干预诊断；当前 checkpoint 的 action-to-image 响应弱 |
| DiffusionDriveV2 | 4 个离散 action profile 的视频/action smoke probe | 每个 profile 约 10 帧，并有 RGB 差异统计 | 仅 smoke/响应探针；缺少 branch lineage、native action-head 对齐和独立 rollout |
| WorldDrive | checkpoint 已注册并可用 | 没有找到已完成的 WAM-IAC 输出或闭环报告 | 尚未做正式评测 |
| Vista / CAMI2V | checkpoint/代码登记 | 没有 native action-head 闭环结果 | 不属于当前 native action-WAM 主比较 |

说明：Epona 的 `native_iac_control` / `native_iac_generated` 结果使用了独立的 NAVSIM realized state 作为图像测量参考，但没有显式 `task_success`，也不是同一 history 下的 risk/clear action-head 反事实闭环。因此它们可以比较图像测量和生成质量，不能直接给出 FCS。

DriveWAM 旧诊断的关键数值：

- native real future-image control：realized-state compatibility `0.5454`，`21/60` 达到 FCS 阈值；
- DriveWAM generated 4/4 chunk 0：compatibility `0.0876`，FCS-compatible `0/60`；
- DriveWAM generated 1/1 chunk 0：compatibility `0.0891`，FCS-compatible `0/60`；
- 4/4 run 的跨动作分支图像 MAE 为 `0.000831`，说明生成视频近似 branch-invariant。

原始诊断报告见 [DRIVEWAM_DIAGNOSTIC_20260825.md](DRIVEWAM_DIAGNOSTIC_20260825.md)。

## 为什么还不能叫正式 Level-2/3

正式评分要求每个 twin 同时具备：

1. 同一 history 下的 clear/risk WAM-generated future；
2. WAM native action-head 的实际输出，而不是由候选轨迹或动作条件反推；
3. candidate-blind 图像解码结果；
4. 独立 closed-loop rollout 导出的 `realized_future_ego_state`；
5. 独立的 `task_success` / task score；
6. branch、model、checkpoint、seed、时间轴和 action-injection lineage。

旧 DriveWAM 实验已经覆盖第 1、部分第 2 和第 3 项，并验证了动作注入；但生成分支响应几乎为零，且旧结果没有以当前 `counterfactual_continuous_records.jsonl` 契约落盘。NuPlan aggregate parquet 也不能替代逐 branch 的 realized state 和 task-success join。

因此当前可报告的是：

- `closed_loop_intervention_attempted: true`；
- `wam_action_to_image_response: weak`；
- `formal_level2_ccfc_ready: false`；
- `formal_level3_fcs_ready: false`。

## 最小复用/补跑路径

1. 保留旧 DriveWAM 报告作为 `diagnostic_only` 基线，不把它混入正式排行榜。
2. 若能找回旧 JSON，先用当前审计器检查 branch/twin、action lineage 和时间轴，再转换为现行 JSONL。
3. 找不回原始 JSON 时，不重算旧数字；直接使用具备公开 action-conditioning 接口的 WAM 重跑同一 20-scene intervention。
4. 对通过 action-response gate 的 checkpoint，再补独立 simulator rollout，生成 `realized_future_ego_state` 和 `task_success`，最后计算 Level-2 CCFC 与 Level-3 FCS。

当前正式评分器继续 fail-closed：没有逐分支独立 realized state 和任务成功标签时，不输出联合 WAM 分数。
