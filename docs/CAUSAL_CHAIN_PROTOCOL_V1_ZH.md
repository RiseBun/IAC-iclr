# IAC 四类交互因果链协议 V1

## 目标

本协议把四类交互场景实现为成对的风险/安全反事实，而不是四个互斥分类标签：

```text
cut-in / lead brake -> imagined conflict -> decelerate / emergency brake
pedestrian crossing -> imagined occupancy conflict -> yield / stop -> restart
blocked lane -> imagined obstruction -> lane change / avoidance / stop
unprotected turn / merge -> imagined gap evolution -> yield / gap acceptance
```

单独收集风险场景只能证明模型在相关数据上会刹车。每个 `counterfactual_pair_id` 必须包含同一 history 下的 `risk` 与 `clear` 两条记录，并固定 planner、候选动作和 nuisance seed。

## 证据链

每条记录包含四种彼此独立的证据：

1. `trigger`：来自数据标注、人工共识或 simulator/logged state。
2. `imagined_consequence`：来自冻结的 image/latent interaction probe 或盲人工共识。
3. `selected_response`：来自 WAM action trajectory、planner output 或 controller telemetry。
4. `realized_outcome`：来自独立 logged/simulator telemetry，并提供显式 `task_success`。

自动 VLM tag 可以用于候选样本挖掘，但不属于正式 imagined evidence source。

## 核心对比量

对同一反事实 pair：

```text
imagined_risk_contrast
  = P(conflict | risk future) - P(conflict | clear future)

protective_action_contrast
  = P(protective action | risk future) - P(protective action | clear future)
```

两者均达到预注册阈值，才记为 `directionally_aligned`。V1 默认阈值为 `0.20`。

保守的 pair score 定义为：

```text
min(
  positive imagined-risk contrast,
  positive protective-action contrast,
  risk/clear 两侧期望想象与期望动作的支持度,
  independent safe-outcome support
)
```

如果未来确实显示风险但 planner 在 risk/clear 两种未来下选择相同动作，则 action contrast 为零，整条因果链得分为零。

结果不是单一 `no_collision`。cut-in 还要求安全 TTC/headway；行人链要求安全停车/间距并在清空后恢复通行；阻塞车道要求可行驶区域合规和安全进展；无保护转向/汇入要求安全 TTC 和进展。行人 `risk` 记录的动作 horizon 必须同时覆盖 yield/stop 与之后的 restart；无法观察完整消解过程的片段不作为完整链计分。

最终主数采用四类等权宏平均，避免样本最多的 cut-in 主导总分。缺少任意一类时 `macro_mean_causal_chain_score` 为 `null`。

## 必需标识

每条记录必须保留：

```text
chain_episode_id
counterfactual_pair_id
scene_id
history_id
chain_type
world_state: risk | clear
world_intervention_id
generated_future_id
planner_id
planner_run_id
planner_nuisance_seed
```

pair 内必须固定 `chain_type / scene_id / history_id / planner_id / planner_nuisance_seed`，并确保 intervention、future 和 planner run 标识不同。

## CLI

先从 nuPlan scenario tag 挖掘高召回候选。这里的 tag 只能用于抽样，不能作为正式 trigger 真值：

```bash
PYTHONPATH=src:. python scripts/mine_nuplan_causal_candidates.py \
  --db-root /path/to/nuplan-v1.1/splits/mini \
  --sensor-root /path/to/sensor_blobs \
  --output nuplan_causal_candidates.jsonl \
  --max-per-chain 40
```

候选经盲人工确认、构造 risk/clear future 并完成固定 planner 重跑后，再执行正式评分：

```bash
PYTHONPATH=src:. python scripts/evaluate_causal_chains.py \
  --records causal_chain_records.jsonl \
  --output causal_chain_report.json \
  --minimum-contrast 0.20 \
  --require-ready
```

`--require-ready` 在任何 pair 缺失 clear/risk 对照、证据来源不独立、标识不完整或四类覆盖不全时返回非零退出码。

## 当前能力边界

本次实现完成了 ontology、数据契约、反事实 readiness audit 和评分器。它没有宣称现有 RAFT lateral probe 已经能够识别 cut-in、行人、障碍和 gap state。正式数据进入评分前，interaction probe 必须在独立盲标 WAM 视频上完成校准；在此之前可以使用 `blinded_human_consensus` 验证协议和构建首批 gold set。
