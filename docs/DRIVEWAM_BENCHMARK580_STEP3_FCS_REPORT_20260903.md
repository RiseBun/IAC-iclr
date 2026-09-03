# DriveWAM benchmark-v1：Step 3 FCS 实验报告

日期：2026-09-03  
模型：DriveWAM（benchmark-v1 native action run）  
基准池：580 条（NAVSIM 500，Waymo 80）

## 结论

Step 3 已在可运行的 NAVSIM 子集完成。我们把 DriveWAM 原生 4 秒动作轨迹注入独立的 NAVSIM PDM 闭环模拟器，得到模拟器产生的 realized ego state 和任务分数；模拟器不读取 WAM 生成图像，也不把计划轨迹冒充 realized state。

| 项目 | 结果 |
|---|---:|
| NAVSIM benchmark rows | 500 |
| 精确 metric-cache rows | 491 |
| 独立 rollout rows | 491 |
| rollout errors | 0 |
| PDM task-score 均值 | 0.7312 |
| FCS（score ≥ 0.5） | 397 / 491 = **0.8086** |
| FCS 95% Wilson CI | [0.7714, 0.8409] |
| Waymo rows | 80，NAVSIM PDM **N/A** |

因此当前可报告的是 **DriveWAM-FCS-NAVSIM = 0.8086（491 条）**，不是把 580 条强行填成一个分数。这里的 FCS 是 **native action 的独立 PDM 执行结果**：rollout 不读取 WAM 生成图像，不能单独证明动作由 imagined future 导致。Waymo 没有对应的 NAVSIM PDM metric cache/地图接口，标记 N/A 而不是 0。

## 分层结果

| 场景分层 | n | 成功 | FCS | PDM 均值 |
|---|---:|---:|---:|---:|
| acceleration | 100 | 94 | 0.9400 | 0.8695 |
| braking | 87 | 63 | 0.7241 | 0.6469 |
| lateral_turn | 102 | 76 | 0.7451 | 0.6703 |
| stop | 102 | 76 | 0.7451 | 0.6265 |
| straight_cruise | 100 | 88 | 0.8800 | 0.8350 |

分层 FCS 仅用于诊断，不替代总体分数。当前闭环交通策略是 `static_cached_objects_compat`，因此结果应称为 NAVSIM-PDM realized-task FCS；要升级为更强结论，需要在动态 agent 轨迹和完整交通交互可用时复跑。

## 可复现协议

1. 从私有 benchmark manifest 读取 500 条 NAVSIM 行。
2. 用当前 ego timestamp 将行与精确 metric cache 对齐；缓存按 NAVSIM 当前场景 token 查找。
3. 从四个 DriveWAM shard manifest 按全局顺序读取 native action，统一为 `[8,3]`、0.5–4.0 s。
4. 使用 `PDMSimulator/BatchKinematicBicycleModel`，4.0 s horizon、0.1 s interval。
5. 由 PDM scorer 计算 task score；`task_success = (score >= 0.5)`。
6. 通过 `attach_counterfactual_rollouts.py` 的独立状态和模拟器 action-injection 校验后，生成可供 FAU/FCS 汇总的分支记录；该 injection 仅用于执行 native action，不是 CCFC 干预。

## 缺失项

9 条 NAVSIM 行没有通过当前 metric-cache 的完整路线/未来窗口过滤，已写入 staging diagnostics；不会静默删除或记为失败。80 条 Waymo 行因没有 NAVSIM PDM 对应缓存，保留在 benchmark 总池中但对本次 FCS 标记 N/A。

## 服务器产物

- 精确缓存：`/mnt/slurmfs-4090node3/user_data/zchen897/fcs_metric_cache_exact/metric_cache`
- 分支 staging：`/mnt/slurmfs-4090node3/user_data/zchen897/fcs_metric_cache_exact/branches580_drivewam.jsonl`
- rollout：`/mnt/slurmfs-4090node3/user_data/zchen897/fcs_metric_cache_exact/rollouts_drivewam.jsonl`
- 已附加 FCS：`/mnt/slurmfs-4090node3/user_data/zchen897/fcs_metric_cache_exact/branches580_drivewam_fcs.jsonl`
- 汇总：`/mnt/slurmfs-4090node3/user_data/zchen897/fcs_metric_cache_exact/fcs_summary.json`
- staging diagnostics：`/mnt/slurmfs-4090node3/user_data/zchen897/fcs_metric_cache_exact/branch_staging_summary.json`

## 与 CFAC/FAU 的关系

本步骤只回答“native action 在独立执行中是否产生可行的 realized task outcome”。它不把 FCS 反过来当作 CFAC，也不使用 WAM 的 imagined profile 作为 realized state。CFAC/FAU 仍需使用冻结 Level-1 探针从 WAM future visual state 重算；三者应并列报告。
