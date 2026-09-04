# WAM 评测运行产物索引（2026-09-04）

本文档对应服务器上的内部运行包：

```text
/mnt/slurmfs-4090node3/user_data/zchen897/wam_repro/artifact_bundle_20260904/
```

该目录包含原始逐样本 JSON、统一索引、冻结配置和 pilot CCFC 原始报告。它是复现实验交接包，不是公开发布包；其中的 manifest 含私有 GT 状态和服务器源路径，不应上传到公开仓库。

## 1. 结果总览

| 结果 | 数值 | 分母 | 产物 |
|---|---:|---:|---|
| Level-1 图像侧 coverage | 0.9966 | 578/580 | `level1_580/alignment.json` |
| 形状 CFAC（正式 pilot） | 0.7610 | 564/580 | native4 `native_action_alignment_shape_v1.json` |
| 旧 CFAC（dir×mag×temp 公式，仅诊断） | 0.4825 | 580 | `drivewam_cfac_fau_native4_report.json` |
| CCFC metric / arc_relative | 0.1235 / 0.2234 | 357/580 | **全量** `.../ccfc_eval_native4/ccfc_command_report.json` |
| FAU（三形状量 vs 私有 GT） | 0.6509 | 562/580 | 同上 CFAC/FAU report；待与新 CFAC 聚合口径对齐后冻结 |
| FCS（NAVSIM-PDM） | 0.8086 | 397/491 | 见 `docs/DRIVEWAM_BENCHMARK580_STEP3_FCS_REPORT_20260903.md` |

注意：357/580 全量 CCFC 行级文件**已找到**（`ccfc_command_report.json`，含 metric /
scale_free / arc_relative）。`ccfc_pilot/` 仍只是小样本对照，不能替代全量表。

### 形状重算审计

基于已有 `logged_gt` Level-1 alignment 的形状候选重算结果为：`shape_cfac=0.8494`、
`arc_relative_path_cosine=0.9760`、`relative_observable_curve_cosine=0.9820`，可评
457/580。它验证的是图像测量层的上界，不是正式 WAM CFAC；正式 CFAC 必须使用同一
WAM 的 native action 作为 `reference_source=action` 重新计算。结果文件为
`reports/diagnostics/shape_relative_cfac.json`。

随后已复用服务器上完整的 DriveWAM native4 导出，按新协议完成正式形状口径重算：
`primary_shape_composite=0.7610`，564/580 条可评估，interval coverage=0.6981。
详见 `docs/DRIVEWAM_NATIVE4_SHAPE_ALIGNMENT_20260904_ZH.md`。  
旧 CFAC=0.4825 是同三字段的更严公式（dir×mag×temp），不是纵向米制分；含纵向的诊断列是
`experimental_composite`。FAU=0.6509 的 components 已是形状量，缺的是与新聚合口径对齐冻结。

## 2. 逐样本文件

### Level-1 / CFAC / FAU

- `level1_580/scores.jsonl`：580 行，冻结 RAFT-Large continuous decoder 的逐样本输出。
- `level1_580/alignment.json`：580 条对齐记录；包括 `image_motion_profile`、`raw_image_motion_profile`、`comparison.per_interval`、`distance_alignment_*`、`pose_alignment_*`、`foresight_gain`。
- `level1_580/manifest.jsonl`：580 行输入协议；包括 4 个历史帧、8 个未来帧、图片路径、`history_times_s`、`future_times_s`、相机内参和参考轨迹。该文件是内部文件，含 GT 和源路径。
- `level1_580/per_sample_artifacts.json`：由 `scripts/package_wam_run_artifacts.py` 生成的统一可读索引，每条记录同时给出图像/动作时间戳、运动剖面、逐 interval 有效性和对齐结果。

每条统一记录的关键字段：

```json
{
  "sample_id": "benchmark_v1-00000",
  "image_timestamps_s": {"history": [-1.5, -1.0, -0.5, 0.0], "future": [0.499898, 1.000079, "...", 4.001018]},
  "action_timestamps_s": [0.499898, 1.000079, "...", 4.001018],
  "image_motion_profile": {"rows": [{"time_s": 0.499898, "lateral_speed_mps": "...", "yaw_rate_radps": "...", "curvature_1pm": "...", "observability": "...", "status": "usable"}]},
  "interval_observability": ["..."],
  "interval_status": ["usable", "abstain", "..."],
  "comparison": {"...": "逐样本比较与分数"}
}
```

`action_timestamps_s` 在 Level-1 文件中表示同轴参考动作/轨迹时间点；对真实 WAM 提交，应替换为其 native action trajectory 的时间轴。

### CCFC pilot

`ccfc_pilot/` 保留完整原始报告，不做错误的 sample-id join：

- `command_eval25_ccfc.json` / `command_eval25_ccfc_swap.json`
- `command_5sample_ccfc.json` / `command_5sample_ccfc_swap.json`
- `future_latent_ccfc.json` / `future_latent_ccfc_swap.json`

每个 `reports[*]` 包含：

- `comparison.metrics`：speed、acceleration、lateral speed、yaw rate、curvature 的图像响应/动作响应数组；
- `continuous_cfc.metric.subscores.response_direction`；
- `continuous_cfc.metric.subscores.response_magnitude`；
- `continuous_cfc.metric.subscores.response_temporal_alignment`；
- `continuous_cfc.metric.coverage`、`evaluable_intervals`、`total_intervals`；
- `counterfactual_group_id`、`intervention_types`（在文件顶层）和泄漏审计字段。

CCFC pilot 报告当前没有显式保存每个 interval 的绝对时间戳，只有按时间顺序排列的响应数组和 interval 数量；不能事后编造时间戳。下一次完整导出必须在每个分支中写入 `future_timestamps` 和 `action_timestamps`。

## 3. 图像侧运动与有效区间

冻结探针输出的是连续运动量，而不是文字事件：

```text
P_F(t) = progress / lateral_offset / heading / speed / lateral_speed / yaw_rate / curvature
```

每个 interval 同时保存：

- `observability`、`speed_observability`、`shape_observability`；
- `status`、`speed_status`、`shape_status`、`flow_status`；
- `pose_intervals` 的 q05/q50/q95；
- `speed_interval_mps` 的 q05/q50/q95。

因此“读不出”是 `abstain`，不是把不确定样本强行记为错误。

## 4. 冻结配置

文件：

```text
level1_580/plane.json
```

关键冻结项：RAFT-Large、32 updates、前后向一致性、地面平面 ego geometry、candidate-blind continuous decoder；主形状量为 lateral/yaw/curvature，速度质量门独立，不污染形状 eligibility。

## 5. 运行命令

### Level-1 解码

```bash
cd /mnt/slurmfs-4090node1/homes/zchen897/iac_iclr_repo
PYTHONPATH=src ~/miniforge3/envs/drivingworld/bin/python scripts/evaluate_continuous_decoder.py \
  --manifest /mnt/slurmfs-4090node3/user_data/zchen897/iac_waymo/reports/benchmark_v1_raft_plane/decoder_manifest.jsonl \
  --config configs/plane.json \
  --output /mnt/slurmfs-4090node1/homes/zchen897/iac_new/datasets/waymo_external/reports/benchmark_v1_raft_plane/scores.jsonl
```

### Level-1 对齐

```bash
PYTHONPATH=src ~/miniforge3/envs/drivingworld/bin/python scripts/evaluate_continuous_motion_alignment.py \
  --manifest /mnt/slurmfs-4090node3/user_data/zchen897/iac_waymo/reports/benchmark_v1_raft_plane/decoder_manifest.jsonl \
  --scores /mnt/slurmfs-4090node1/homes/zchen897/iac_new/datasets/waymo_external/reports/benchmark_v1_raft_plane/scores.jsonl \
  --reference-source logged_gt \
  --disable-shape-fallback \
  --output /mnt/slurmfs-4090node1/homes/zchen897/iac_new/datasets/waymo_external/reports/benchmark_v1_raft_plane/level1_strict.json
```

### 真实视频退化 pilot（B 组）

```bash
PYTHONPATH=src ~/miniforge3/envs/drivingworld/bin/python scripts/build_logged_degradation_probe.py \
  --manifest /mnt/slurmfs-4090node3/user_data/zchen897/iac_waymo/reports/benchmark_v1_raft_plane/decoder_manifest.jsonl \
  --output-root /mnt/slurmfs-4090node3/user_data/zchen897/iac_b_controls/medium \
  --output-manifest /mnt/slurmfs-4090node3/user_data/zchen897/iac_b_controls/manifest_medium_normalized.jsonl \
  --name medium --sigma 1.5 --jpeg-quality 50 --flicker 0.10
```

### CCFC 子分数审计

```bash
PYTHONPATH=src ~/miniforge3/envs/drivingworld/bin/python scripts/analyze_ccfc_subscores.py \
  /mnt/slurmfs-4090node3/user_data/zchen897/wam_repro/WorldDrive/results/command_counterfactual_eval25/generated_selected_20step/ccfc.json \
  /mnt/slurmfs-4090node3/user_data/zchen897/wam_repro/WorldDrive/results/command_counterfactual_eval25/generated_selected_20step/ccfc_swap.json
```

WorldDrive 生成 pilot CCFC 报告的原始启动命令没有保存在当前仓库/服务器日志中；上面给出的是可复现的报告审计命令和所有原始输入文件路径。完整 benchmark 导出时，必须把生成命令、模型 revision、seed、lineage 与时间戳一并写入提交包。

## 6. 下载内部运行包

在本地 PowerShell 执行：

```powershell
scp -r iac:/mnt/slurmfs-4090node3/user_data/zchen897/wam_repro/artifact_bundle_20260904 C:\Users\LPN19\Desktop\iac\artifact_bundle_20260904
```
