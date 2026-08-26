# 四类交互因果链候选挖掘记录（2026-08-26）

## 结论

服务器 `iac` 上已经生成首批 31 条可读真实候选，但尚未产生四类模型准确率。这 31 条只允许进入盲人工确认，不能直接进入正式 causal-chain score。

| 因果链 | 可读候选数 |
|---|---:|
| cut-in / lead brake | 7 |
| pedestrian crossing | 5 |
| blocked lane | 9 |
| unprotected turn / merge | 10 |

## 数据来源与运行结果

- 数据：nuPlan v1.1 mini，64 个 SQLite 日志库。
- scenario tag 命中：9,289 条。
- 固定 seed、按日志优先去重后抽样：每类 40 条，共 160 条。
- 完整前视窗口：2 秒 history + 6 秒 future，`CAM_F0`。
- 通过图片存在性和时间窗检查：31 条。
- 3 条缺完整时间窗，126 条对应 sensor blob 在当前服务器不可用。
- 服务器产物：`~/iac_new/work_dirs/causal_chain_candidates_v1_20260826/candidates.jsonl`。

复现命令：

```bash
/mnt/slurmfs-4090node1/homes/zchen897/miniforge3/envs/nuplan/bin/python \
  ~/iac_new/iac_raft_event_causal_v1/scripts/mine_nuplan_causal_candidates.py \
  --db-root ~/nuplan/dataset/nuplan-v1.1/splits/mini \
  --sensor-root ~/navsim_workspace/dataset/sensor_blobs \
  --output ~/iac_new/work_dirs/causal_chain_candidates_v1_20260826/candidates.jsonl \
  --max-per-chain 40
```

## 不能越过的证据边界

nuPlan scenario tag 只用于高召回抽样。`near_*`、`stopping_*` 或 `starting_unprotected_*` 不等价于图像中一定存在目标因果链，因此记录显式保留：

```text
trigger_label_status = candidate_only_requires_blind_confirmation
counterfactual_pair_status = not_constructed
```

当前 tag 对 lead vehicle、pedestrian、road obstruction 和 unprotected turn 有覆盖，但没有直接的 `vehicle_cut_in` 与 `merge_gap` 标签。这两类必须通过额外轨迹规则或人工检索补齐。

## 下一门槛

1. 对 31 条候选做盲人工确认，标出 trigger、conflict onset、可观察性和完整消解时刻。
2. 每类至少保留 risk seed，并构造同 history 的 clear future；固定 planner、candidate bank 和 nuisance seed 重跑 action head。
3. 用冻结 interaction probe 对人工 gold set 做校准；在校准完成前，只能使用 `blinded_human_consensus` 作为 imagined consequence 的正式来源。
4. 独立 simulator/logged telemetry 补齐 collision、TTC/clearance、drivable-area compliance 和 progress，之后才运行四类宏平均分。
