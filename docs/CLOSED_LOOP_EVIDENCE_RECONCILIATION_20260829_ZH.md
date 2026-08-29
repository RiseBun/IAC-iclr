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

