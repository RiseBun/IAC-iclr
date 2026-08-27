# WAM Level-1 生成输出就绪审计（2026-08-27）

## 结论

当前服务器还没有可以支撑正式 Level-1 因果结论的 WAM 生成分支输出。
评测器、8 帧/4 秒协议、候选盲图像解码器和 action-head 对齐已就绪；正式结果
仍等待 WAM backend 产生带完整 lineage 的完成态 JSONL。

## 已核查目录

| 目录 | 发现 | 能否进入正式 Level-1 |
|---|---|---|
| `DiffusionDriveV2/selector_outputs/...` | 4 个离散动作 profile，各 10 帧，有 mp4 和 RGB 差异统计，但没有 `branch_id/source_key`、WAM action-head trajectory 或 8 帧/4 秒协议 | 否 |
| `wam_repro/DrivingWorld/outputs/iac_smoke/...` | 20 张 smoke png，没有生成元数据和 action 对齐关系 | 否 |
| `iac_new/results` 与 `iac_new/datasets` | 只有 NAVSIM 原生 future、pending branch 和历史评测结果，没有 `future_images_source=wam_generated` 的完成态分支 | 否 |

把这些图像直接当成 WAM 证据会混淆“模型生成过视频”和“未来图像与动作头具有可测的因果一致性”。

## 必须提交的生成输出

WAM backend 每个 branch 输出一行 JSONL，至少包含：

```json
{
  "branch_id": "<stable lineage id>",
  "source_key": "<fixed NAVSIM source key>",
  "wam_model_id": "<checkpoint or model revision>",
  "future_images_source": "wam_generated",
  "wam_generation_status": "complete",
  "future_images": [".../future_000.png", ".../future_001.png"],
  "future_times_s": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
  "action_condition": {
    "trajectory": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
  }
}
```

其中 `future_images` 必须正好 8 张，`action_condition.trajectory` 必须是 `[8,3]` 的
独立 action-head 输出。输出行不能带 `realized_future_ego_state`，也不能把
`logged/oracle/proxy/candidate` 标成 action source。

## 接入顺序

```bash
PYTHONPATH=src:. python scripts/audit_wam_level1_outputs.py \
  --generated results/<wam_output>.jsonl \
  --output results/<wam_output>_audit.json \
  --check-files
```

只有审计结果中的 `formal_level1_input_ready=true` 才能继续：

```bash
PYTHONPATH=src:. python scripts/build_wam_level1_continuous_manifest.py \
  --base datasets/navsim_level1_v5/navsim_level1_v5_all_78.jsonl \
  --generated results/<wam_output>.jsonl \
  --output results/wam_level1_v5_manifest.jsonl \
  --check-files
```

随后使用冻结的 RAFT-Large/SEA-RAFT 解码器从生成 future image 恢复
`m_F(t)=[v,a,v_lat,yaw_rate,curvature]`，再与该行的 `wam_action_head` 得到的
`m_A(t)` 比较。NAVSIM 原生 future 只验证测量能力；它不能替代上述 WAM 输出。
