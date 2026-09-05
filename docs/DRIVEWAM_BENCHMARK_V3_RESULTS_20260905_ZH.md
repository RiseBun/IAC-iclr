# DriveWAM Benchmark v3 实验结果

## 范围

- 数据集：NAVSIM v3，1000 个非重叠样本；直行上限 30%，包含转弯、加速、制动和停车分层。
- 模型：`drivewam_navsim_checkpoint_20260824`，沿用服务器已有 checkpoint 与 LingBot-VA base，不重复下载。
- 输出位置：`/mnt/slurmfs-4090node3/user_data/zchen897/benchmark_v3_runs/`。
- 图像探针：冻结 RAFT-Large、前后向一致性、地面平面自车几何、candidate-blind continuous decoder；纵向米制速度仅作诊断。

## Step 1 / CFAC 与 FAU

官方路径从 WAM 生成图像重新运行探针，未读取作者提交的运动剖面；GT 只在私有评测端 join。

| 指标 | 结果 |
|---|---:|
| CFAC primary shape composite | 0.7638 |
| CFAC 有效样本 | 823/1000 |
| CFAC bootstrap 95% CI | 0.7490--0.7785 |
| FAU-F（想象图像 vs 私有 GT） | 0.5449 |
| FAU-A（native action vs 私有 GT） | 0.4904 |
| FAU 几何均值 | 0.5169 |
| FAU/CFAC 有效样本 | 823/1000 |

停车样本不进入运动均值；4 条样本因私有 GT 没有与公共 `[1,2,3,4] s` 完全一致的时间轴而标记 `unavailable`，没有插值或近邻替代。

逐样本结果：`drivewam_cfac_fau_v3.json`。

## Step 2 / CCFC

对每个样本使用相同 history、seed 和 nuisance，仅改变导航命令为 left/right，各运行一次；两支均同时保留 WAM 生成 future images 与 native action。两支图像都通过冻结 Level-1 探针后再评分。

| 指标 | 结果 |
|---|---:|
| 成对组数 | 1000 |
| 结构有效组 | 1000/1000 |
| metric CCFC（诊断） | 0.1190，453 对 |
| scale-free CCFC | 0.1594，453 对 |
| arc-relative CCFC（主报告） | 0.2178，453 对 |
| claim scope | `command_conditioned_action_image_consistency` |

这不是 semantic clear/risk 干预，因此不宣称语义危险因果；它是统一协议允许的 command-conditioned CCFC。

逐样本记录与评分：`ccfc_v3_full_records.jsonl`、`ccfc_v3_full_report.json`。

## Step 3 / FCS

native action 进入独立 NAVSIM PDM kinematic-bicycle closed-loop rollout。该步骤测量可执行动作的实现效果，不读取 WAM 生成图像，也不把 waypoint 当作 realized state。

| 指标 | 结果 |
|---|---:|
| 输入/成功 | 978 / 503 |
| FCS task success rate | 0.5143 |
| state reference | `navsim_pdm_kinematic_bicycle_closed_loop` |
| traffic policy | `static_cached_objects_compat` |

## 可复现实验命令

服务器上的完整命令和输入、输出路径保存在各脚本的 argparse 帮助与 shell 日志中。核心命令如下：

```bash
PYTHONPATH=/mnt/slurmfs-4090node1/homes/zchen897/iac_new/src \
python /mnt/slurmfs-4090node1/homes/zchen897/iac_new/scripts/evaluate_v3_drivewam_cfac_fau.py \
  --level1-input /mnt/slurmfs-4090node3/user_data/zchen897/benchmark_v3_drivewam_level1_input.jsonl \
  --level1-scores /mnt/slurmfs-4090node3/user_data/zchen897/benchmark_v3_runs/level1_v3_drivewam.jsonl \
  --private-manifest /mnt/slurmfs-4090node3/user_data/zchen897/iac_navsim_benchmark_v3/benchmark_v3_navsim_private.jsonl \
  --output /mnt/slurmfs-4090node3/user_data/zchen897/benchmark_v3_runs/drivewam_cfac_fau_v3.json
```

所有评分状态采用 fail-closed：缺少图像、native action、私有 GT 或时间轴时报告 `unavailable`，不填零分。
