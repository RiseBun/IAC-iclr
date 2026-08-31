# WorldDrive 原生 action–future 一致性 pilot

日期：2026-09-01
结论等级：`native_action_future_pair_pilot`
正式 CCFC：否

## 1. 要回答的问题

本实验不是问 WorldDrive 生成的视频是否“好看”，而是依次排除三个更强的替代解释：

1. 像素 future 是否真的随轨迹条件变化，而不是忽略 action；
2. IAC 是否能从这些 future 中候选盲地读出横向、yaw、curvature；
3. 原生 stage-2 action 是否与该模型生成的 future 一致，并且这种一致性是否依赖正确身份和正确时间顺序。

完整链路：

```text
4 history frames + ego status
        ↓
WorldDrive stage-1 planner + stage-2 future-aware refiner
        ↓ native final 8×(x,y,yaw)
TA-DWM trajectory-conditioned pixel generator
        ↓ 8 future frames / 4 s
candidate-blind RAFT-Large + ground-plane decoder
        ↓ lateral / yaw / curvature posterior
compare only after image decoding
```

action waypoint、候选 bank 和 logged future state 均未输入图像解码器。

## 2. 冻结资产

| 资产 | SHA256 |
|---|---|
| stage-2 planner/refiner | `a9facb540cc8a76641629dc768e99464961d4964e377841e227db78f534ee158` |
| TA-DWM 1024 world model | `e8a10c3f9285562a43197ca8390438e7170186a58e1aae6d1af77053937d5e03` |
| 256 trajectory anchors | `44f64a763473c3a80482aaa3f78669445f56af40a1c00741a351c6c0650e758b` |
| CogVideoX VAE | `a410e48d988c8224cef392b68db0654485cfd41f345f4a3a81d3e6b765bb995e` |

图像协议为 4 帧历史、8 帧未来、`0.5…4.0 s`。生成使用 bfloat16、20 diffusion steps；同组分支固定历史和 diffusion noise。Level-1 使用 `configs/navsim_continuous_decoder_plane.json`，速度不进入主分。

## 3. 外部动作响应门

5 个 strict NAVSIM 样本各生成 `left/logged/right`，共 15 条 future、120 张 PNG。

- left/right future 像素 L1 均值：`0.087085`
- bootstrap 95% 下界：`0.073625`
- 预注册门槛：`0.005`
- 完全相同 seed 重跑：24/24 PNG 逐字节相同，重复 L1 为 `0`

因此像素变化由 action 条件引起，不是随机采样噪声。候选盲图像解码结果：

| 指标 | 结果 |
|---|---:|
| 精确 branch Top-1 | `7/15 = 46.7%`，chance `33.3%` |
| 终点 lateral 排序 | `12/15 = 80.0%` |
| 终点 yaw 排序 | `14/15 = 93.3%` |

解释：Level-1 已能稳定读出转向方向与排序，但不应被描述为精确恢复相邻轨迹幅值。

## 4. 原生 action–future 配对

官方 stage-2 planner/refiner 从每个历史独立输出最终动作。两次完整导出 JSONL 逐字节一致；5 个样本中 4 个的 refiner 选择不是 stage-1 top-1，说明完整 stage-2 路径实际生效。

用各自原生最终动作驱动同一 TA-DWM 后，Level-1 结果为：

| 指标 | 正常顺序 |
|---|---:|
| valid | `5/5` |
| weighted joint error | `0.767560` |
| soft compatibility | `0.535167` |
| joint coverage | `0.744892` |
| heading cosine | `0.998488` |
| lateral MAE | `0.315471 m` |
| yaw MAE | `0.039103 rad` |
| curvature MAE | `0.008160 1/m` |

## 5. 特异性控制

### 5.1 错身份 action

将每个图像解码结果与 5 个样本的原生 action 全部比较：

- 正确 identity Top-1：`4/5 = 80%`
- `mean(wrong action error) - correct action error`：`+3.891957`
- paired bootstrap 95% CI：`[2.915499, 5.042000]`

### 5.2 future 时间倒序

只倒转 8 张 future 帧的顺序，时间戳、历史、action、配置均不变：

| 指标 | 正常 | 倒序 |
|---|---:|---:|
| weighted joint error | `0.767560` | `4.034721` |
| soft compatibility | `0.535167` | `0.106970` |
| lateral MAE | `0.315471 m` | `1.631644 m` |
| yaw MAE | `0.039103 rad` | `0.281828 rad` |
| curvature MAE | `0.008160 1/m` | `0.079489 1/m` |

正常顺序在 5/5 样本上优于倒序；误差增量均值为 `+3.062275`，bootstrap 95% CI `[1.678503, 4.446048]`。

## 6. 当前能说与不能说

能说：

- WorldDrive 确实同时具备可执行的 native action head 和随轨迹变化的像素 future；
- IAC 图像侧能读出与原生 action 对应的横向/转向信息；
- 结果依赖正确样本身份和正确时间顺序，不是静态外观或 history-only 先验即可解释。

不能说：

- 不能把 `n=5` pilot 当成稳定总体估计；
- 不能把 planner top-k proposal 当成两条原生最终动作；
- 不能把 action-conditioned `A→F` 自动解释为“想象未来因果驱动动作”的 `F→A` 证据；
- 没有同场景 clear/risk 两支原生最终动作，不能输出正式 CCFC；
- 没有独立 rollout/task success，不能输出 FCS。

## 7. 下一步唯一高价值实验

构造同一语义场景的 `clear/risk` 输入干预，由同一个 stage-2 planner 在两支上独立产生最终 native action；随后固定生成 seed，分别得到 8 帧 future。只有当 action delta 与图像解码 delta 在 direction、magnitude、temporal 三项上一致，并通过 readiness audit，才升级为正式 CCFC。

优先扩到至少 25 个 scene-aware non-overlap 窗口；在此之前不再投入 flow/depth 前端替换，也不再搜索另一篇 joint DiT 来绕开当前因果缺口。

## 8. 服务器产物

```text
/mnt/slurmfs-4090node3/user_data/zchen897/wam_repro/WorldDrive/results/action_response_5sample_20step/
/mnt/slurmfs-4090node3/user_data/zchen897/wam_repro/WorldDrive/results/native_planner_5sample/actions.jsonl
/mnt/slurmfs-4090node3/user_data/zchen897/wam_repro/WorldDrive/results/native_future_5sample_20step/manifest.jsonl
/mnt/slurmfs-4090node3/user_data/zchen897/wam_repro/WorldDrive/results/native_future_5sample_20step/level1_scores.jsonl
/mnt/slurmfs-4090node3/user_data/zchen897/wam_repro/WorldDrive/results/native_future_5sample_20step/level1_scores_time_reversed.jsonl
/mnt/slurmfs-4090node3/user_data/zchen897/wam_repro/WorldDrive/results/native_future_5sample_20step/native_specificity_report.json
```
