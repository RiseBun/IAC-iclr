# CFAC / CCFC 低分原因诊断（2026-09-04）

## 结论

探针侧：横向 / yaw 可靠，曲率基本可用；**纵向米制（speed / accel / 前向米）未过误差预算**，
不得进主分。但要把三件事分开：

1. **Level-1 / experimental_composite**：把 speed/accel 折进复合分会明显拖分；
2. **旧正式 CFAC=0.4825**：components 已是 lateral/yaw/curvature，低分来自
   `dir×mag×temp` 几何平均，不是“混进了速度”；新形状 CFAC=0.7610 是换公式+更严 gate；
3. **CCFC**：全量 357 的 metric=0.1235 确被前向米制拖累；但 arc_relative 也只有 0.2234，
   direction/temporal≈0.25——**不能**只用小 pilot 的高 direction 外推。

本诊断不修改观测阈值。主分政策：只保留形状/相对量；metric 纵向与旧公式分仅作
`legacy_diagnostic`。FAU 报告本身已是三形状量，待聚合口径对齐后冻结。

## 1. CFAC 底层误差分解

来源：benchmark-v1 580 条 Level-1 alignment，服务器文件 `level1_strict.json`。这里的 `logged_gt` 是图像侧量尺验证，不是 native WAM action head 的正式因果证据。

| 分量 | 样本数 | MAE 均值 | MAE 中位数 | 容差内比例 | 判断 |
|---|---:|---:|---:|---:|---|
| lateral speed | 457 | 0.0945 m/s | 0.0232 | 98.42% | 可靠，少量长尾 |
| yaw rate | 457 | 0.0290 rad/s | 0.0033 | 96.99% | 可靠，少量长尾 |
| curvature | 457 | 0.0224 1/m | 0.0047 | 86.11% | 基本可用，但长尾明显 |
| speed | 333（汇总口径） | 1.6031 m/s | 1.3265 | 57.78% | 主要问题 |
| acceleration | 333（汇总口径） | 1.6264 m/s² | 1.2565 | 66.42% | 主要问题 |

### 轨迹层证据

米制前向距离的误差很大：

- metric profile MAE：4.3004 m；
- metric endpoint absolute error：6.6954 m；
- scale-free profile MAE：0.0813；
- scale-free curve cosine：0.9820；
- arc-relative pose path cosine：0.9760。

这说明轨迹**形状和方向大体正确，但米制前向幅度不稳定**。因此 CFAC 低分不能解释成“图像完全没有表达动作”；更准确的说法是“形状读得出来，纵向幅度没有稳定对齐”。

## 2. 可观测性不是主要错误来源，但会降低有效分母

580 条样本共 4640 个 interval：

- `usable`：2912；
- `abstain`：1727；
- `uncertain`：1。

样本级平均 interval coverage 为 0.7968，但覆盖率分布很不均匀：

| 样本级 coverage | 样本数 |
|---|---:|
| [0, 0.25) | 152 |
| [0.25, 0.55) | 74 |
| [0.55, 0.75) | 23 |
| [0.75, 1.0] | 331 |

这些 `abstain` 不是低分，而是“当前 interval 没有足够证据”。正式汇总必须同时报告 score 和 coverage，不能把 coverage 低的样本填成 0。

## 3. CCFC：全量 357 vs 小 pilot

**全量行级文件已找到**：
`/mnt/slurmfs-4090node1/homes/zchen897/work_dirs/benchmark_v1_drivewam_ccfc_eval_native4/ccfc_command_report.json`

| 模式 | n | mean score | 说明 |
|---|---:|---:|---|
| metric | 357 | **0.1235** | 当前默认/旧主分；forward_mae ≫ lateral_mae |
| scale_free | 357 | 0.1861 | 去尺度后仍低 |
| arc_relative | 357 | **0.2234** | direction/temporal 均值约 0.25，magnitude 约 0.78 |

小 pilot（约 16 组）曾出现 metric magnitude≈0、scale-free≈0.85 的对照，**只说明米制幅度敏感**，
不能代表 357 全量——全量上即使 arc_relative，方向/时间对齐仍然弱。

子分必须分报；不得把 pilot 均值写成正式 CCFC。

## 4. 当前能排除和不能排除的解释

### 已基本排除

- RAFT 在真实视频上完全失效：lateral/yaw 的容差内比例超过 96%；
- 普通模糊、JPEG 压缩和 10% 闪烁导致整体 coverage 崩溃：A/B pilot coverage 几乎不变；
- CCFC 完全随机：正确配对相对于图像错配，direction/temporal 明显更高。

### 尚未排除

- WAM 生成视频是否存在生成域特有的几何闪烁或转弯幅度失真；
- 当前 metric 幅度标定是否适合 WAM 生成视频；
- 全量 357 上 direction/temporal 偏低的根因（模型成对响应弱 vs 探针在生成域读偏）。

## 5. 对低分样本的正确判读

| 观察到的组合 | 结论 |
|---|---|
| direction 高、temporal 高、magnitude 低 | WAM 可能表达了正确动作，但幅度/尺度不匹配 |
| direction 高、temporal 低 | 方向正确，但动作和图像时间轴错位 |
| direction 低、magnitude 低 | 图像运动与动作不一致，或探针受生成伪影影响 |
| coverage 低、`abstain` 多 | 证据不足，不能判错 |
| path cosine 高、metric MAE 高 | 形状正确、米制幅度错误 |

## 6. 下一步

1. 以全量 357 的 arc_relative/shape 子分（含 bootstrap CI）作为 CCFC 候选主分，metric 留诊断；
2. 对 WAM 生成视频做身份错配、倒放和最差转弯人工审计；
3. 在独立 dev 集校准任何阈值，不在 benchmark-v1 上调参；
4. 主榜同时报告分数、三项子分和 coverage；FAU 按与新 CFAC 一致的聚合口径冻结；
5. 不把 `abstain` 改成 0，也不因为人工能看出动作就跳过量尺校验。
## 7. 本轮纵向尺度降权诊断

为回答“纵向识别较弱时是否把 CFAC/CCFC 压得过低”，新增了显式审计脚本 scripts/diagnose_cfac_ccfc_failures.py。它不覆盖冻结主榜，而是并列计算：

- shape_score：只使用 lateral speed、yaw rate、curvature；
- longitudinal_score：只使用 speed、acceleration，作为诊断；
- shape_priority_score：形状归一化误差 + 0.25 × 纵向归一化误差。

580 条已有结果的审计为：完整可评估 457/580；将 123 条 abstain 按零覆盖率计入时，平均 coverage 为 0.6278（仅在完整可评估样本上则为 0.7968）。形状诊断分数 0.8753，纵向诊断分数 0.6432，形状优先综合诊断分数 0.7968。主导失败为纵向尺度 177 条、coverage 51 条、形状残差 49 条，无明显失败 303 条。

按形状/相对量重算的 Level-1 候选为：`shape_cfac=0.8494`、`arc_relative_path_cosine=0.9760`、
`relative_observable_curve_cosine=0.9820`（457/580）。由于该 alignment 的参考源是
`logged_gt`，它只能作为图像测量上界，不能冒充 DriveWAM 的正式 CFAC。

小 pilot（16 组）曾出现 metric 0.24 / scale_free 0.85 / arc_relative 0.83；**全量 357** 则为 metric 0.1235 / scale_free 0.1861 / arc_relative 0.2234。pilot 只能说明米制幅度敏感，不能外推全量。

`shape_priority`（纵向权重 0.25）只是审计模式，不是正式主分。正式主分应使用形状字段或 arc_relative，而不是用降权掩盖不可靠米制量。

复现：

```bash
python scripts/diagnose_cfac_ccfc_failures.py \
  --alignment <level1_strict.json> \
  --ccfc '<ccfc.json glob>' \
  --longitudinal-weight 0.25 \
  --output diagnostics/cfac_ccfc_failure_diagnosis.json
```

下一步：冻结全量 357 的 arc_relative/shape 子分与 CI；不把 `shape_priority` 升主分；不改观测门槛。
