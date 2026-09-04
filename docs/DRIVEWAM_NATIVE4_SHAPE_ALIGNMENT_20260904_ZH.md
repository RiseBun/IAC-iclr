# DriveWAM native4 形状主分重算

日期：2026-09-04  
模型：`drivewam_navsim_diffusion_v1`；580 条 benchmark-v1 主池

## 结论

服务器上已有完整的 580 条 native4 导出，并非只有旧的聚合报告。每条记录均包含
WAM 生成未来图像、原生 action trajectory、4 秒时间轴、模型版本、seed 和 lineage。
本次用冻结 RAFT-Large 探针对这 580 条逐样本重跑，再以 `reference_source=action` 对齐；
因此这是符合新协议的单次 CFAC（形状口径），而不是旧的 `logged_gt` 图像上界。

纵向 `speed_mps`、`acceleration_mps2` 和米制前向位移仍然只作诊断，未进入主分。

## 主结果

| 项目 | 结果 | 分母/说明 |
|---|---:|---|
| 形状 CFAC（`primary_shape_composite`） | **0.7610** | 564/580 可评估；16 条整体 abstain |
| interval coverage | **0.6981** | 按 2320 个未来区间聚合 |
| lateral speed MAE | 0.0823 m/s | 564 条；容差内 96.53% |
| yaw rate MAE | 0.0395 rad/s | 564 条；容差内 90.47% |
| curvature MAE | 0.0324 1/m | 564 条；容差内 79.79% |
| arc-relative pose path cosine | 0.9580 | 505 条；尺度归一化 |
| relative-observable curve cosine | 0.9566 | 536 条；尺度归一化 |

`primary_shape_composite` 的定义为配置的三个主形状量归一化 MAE 的指数平均：

```text
exp(-mean(normalized MAE of lateral_speed, yaw_rate, curvature))
```

它不含米制速度、加速度或绝对前向距离，不能与旧报告中的 `CFAC=0.4825` 直接比较。
旧值保留为 `legacy_diagnostic_score`。

## 质量与失败边界

- 580/580 条输入审计通过，580/580 条 decoder 运行成功，错误数为 0。
- 564 条满足主形状比较；16 条因形状观测不足而 `abstain`，不是被填成 0 分。
- 生成视频的形状信号可被探针读取，但不是所有区间都可观测；coverage 必须与分数一起报告。
- 该结果是单分支图像—原生动作一致性，不声称反事实因果；成对干预仍单列为 CCFC。

## 可追溯产物（服务器）

```text
/mnt/slurmfs-4090node1/homes/zchen897/work_dirs/benchmark_v1_drivewam_eval_native4/native_action_manifest_v1.jsonl
/mnt/slurmfs-4090node1/homes/zchen897/work_dirs/benchmark_v1_drivewam_eval_native4/native_action_decoder_scores_v1.jsonl
/mnt/slurmfs-4090node1/homes/zchen897/work_dirs/benchmark_v1_drivewam_eval_native4/native_action_alignment_shape_v1.json
```

逐样本 alignment JSON 含 `image_motion_profile`、`raw_image_motion_profile`、
`reference_motion_profile`、三个主形状量的 MAE/容差、每个 interval 的
`shape_status`/`observability`、以及图像和 action 的时间轴。

## 运行命令

```bash
PYTHONPATH=/tmp python3 /tmp/iac_eval_continuous_decoder.py \
  --manifest /mnt/slurmfs-4090node1/homes/zchen897/work_dirs/benchmark_v1_drivewam_eval_native4/native_action_manifest_v1.jsonl \
  --config /tmp/iac_eval_plane.json \
  --output /mnt/slurmfs-4090node1/homes/zchen897/work_dirs/benchmark_v1_drivewam_eval_native4/native_action_decoder_scores_v1.jsonl \
  --device cuda

PYTHONPATH=/tmp python3 /tmp/iac_eval_continuous_alignment.py \
  --manifest /mnt/slurmfs-4090node1/homes/zchen897/work_dirs/benchmark_v1_drivewam_eval_native4/native_action_manifest_v1.jsonl \
  --scores /mnt/slurmfs-4090node1/homes/zchen897/work_dirs/benchmark_v1_drivewam_eval_native4/native_action_decoder_scores_v1.jsonl \
  --output /mnt/slurmfs-4090node1/homes/zchen897/work_dirs/benchmark_v1_drivewam_eval_native4/native_action_alignment_shape_v1.json \
  --reference-source action --config /tmp/iac_eval_plane.json --include-uncertain
```

native4 使用 WAM 原生的 `[1, 2, 3, 4]` 秒时间轴；协议同时接受公共包的 8 点
`[0.5, ..., 4.0]` 轴，不要求为了记分而插值。
