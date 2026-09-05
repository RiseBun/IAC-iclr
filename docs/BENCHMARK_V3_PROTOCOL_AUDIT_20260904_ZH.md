# benchmark-v3 协议审计与实验状态

## 冻结协议

`benchmark-v3` 的主榜固定为 1000 条 NAVSIM 窗口。Waymo 不再进入主榜分母，
仅作为同一探针和同一提交协议下的外部泛化集。

| 项目 | 冻结值 |
|---|---:|
| 主榜样本 | 1000 NAVSIM |
| 历史图像 | 4 帧，时间不晚于锚点 |
| 未来轴 | 4 秒，最多 8 帧（0.5--4.0 s） |
| 直行巡航 | 300（30%，硬上限） |
| 横向转弯 | 503 |
| 加速 | 82 |
| 制动 | 65 |
| 停车 | 50（5%） |
| scene group | 675 |
| 同 scene 窗口间隔 | 至少 12 帧 |

选集审计通过：无重复 `sample_id`，窗口不重叠，动态特殊运动（转弯/加速/制动）
占 65%。每条记录含历史/未来时间戳、相机内外参、私有真实状态和稳定的
`source_key`。公开发布时移除图像路径和私有 GT。

## Step 3/FCS 可用性

NAVSIM PDM metric cache 当前覆盖 978/1000 条。缺失的 22 条不是图像缺失，而是
NAVSIM 路线或完整未来条件未通过 PDM cache 构建。它们保留在 v3 主榜，FCS 对应
单元按协议记为 `unavailable`，不能填零，也不能从分母中静默删除。

因此：

- Step 1、CFAC、FAU 的主榜分母仍是 1000，按各自 coverage 报告；
- Step 3/FCS 先在 978 条 cache-valid 样本上运行，最终报告 `n/1000` 与可执行
  coverage；
- 22 条缺失样本单独列出 `source_key`、stratum 和失败原因。

## Waymo 外推集

Waymo 只回答“同一图像侧探针能否跨数据域工作”，不参与 v3 主榜排序。外推集
使用同一 4 历史 + 4 秒未来协议、相机标定和相同 RAFT-Large 配置；若某模型没有
可重建的 future visual state，则只能报告 action-only 结果，不能伪造 CFAC/FAU。

## 实验路径

服务器上的 v3 私有选集与审计：

```text
`<private_run_root>/iac_navsim_benchmark_v3/`
  benchmark_v3_navsim_private.jsonl
  benchmark_v3_navsim_audit.json
```

DriveWAM 输入样本在有空间的存储节点构建，避免 4090 节点主盘（已满）继续写入：

```text
`<private_run_root>/benchmark_v3_drivewam_inputs/`
```

生成图像和 native action 完成后，严格按以下顺序运行：

```text
WAM 输出审计
  -> 冻结 RAFT-Large Step 1
  -> CFAC / FAU（私有 GT join）
  -> native action 独立 NAVSIM-PDM rollout
  -> FCS（仅 cache-valid 行）
```

旧的 benchmark-v1/580 结果只作为历史 pilot，不得写入 v3 主表。
