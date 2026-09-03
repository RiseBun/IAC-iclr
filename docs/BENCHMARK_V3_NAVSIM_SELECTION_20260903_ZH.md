# benchmark-v3：全 NAVSIM 主榜选集

## 决策

benchmark-v1 的 580 条由 NAVSIM 与 Waymo 混合组成，无法让所有样本完整执行 Step 3。benchmark-v3 将主榜改为 **全 NAVSIM、1000 条、4 历史帧 + 8 未来帧、4 秒 horizon**。

Waymo 不删除，但降为外部泛化集，不进入 v3 主榜分母。

## 为什么是 1000 条

500 条已经可以做 pilot，但对 CFAC/FAU/FCS 的分层置信区间仍偏宽。1000 条在计算成本和统计稳定性之间更合适；相较 500 条，二项比例的标准误约降低 29%。

## 分层配额

| 分层 | 数量 | 占比 | 作用 |
|---|---:|---:|---|
| lateral_turn | 503 | 50.3% | 覆盖横向、航向和曲率变化 |
| straight_cruise | 300 | 30.0% | 作为直行基线，上限固定为 30% |
| acceleration | 82 | 8.2% | 保留全部可用强加速样本 |
| braking | 65 | 6.5% | 保留全部可用强制动样本 |
| stop | 50 | 5.0% | 作为停车边界案例，避免停车主导分数 |
| **合计** | **1000** | **100%** | 动态特殊运动（转弯/加速/制动）占 65% |

当前 NAVSIM inventory 有 3860 个候选窗口。选择器允许同一场景包含多个窗口，但窗口起点至少相隔 12 帧，因此时间窗口不重叠；v3 选出 675 个场景组、1000 个不重叠窗口。

## 特殊场景定义

v3 不再把“直行”当作默认主体，而是显式优先选择：

- lateral/heading 明显变化的转弯窗口；
- 未来 4 秒速度变化大于阈值的加速窗口；
- 未来 4 秒速度变化小于阈值的制动窗口；
- 少量停车窗口作为边界和 abstention 校验。

当前 inventory 没有可靠的高层语义标签，因此“特殊场景”先以可复现的运动学分层定义。后续可在不改变主榜分母的情况下附加路口、并线、遮挡等语义标签。

## 审计结果

- 输入候选：3860；
- 选中：1000；
- 数据集：NAVSIM 1000；
- 直行比例：30.0%；
- 停车比例：5.0%；
- 转弯/加速/制动比例：65.0%；
- 窗口内不重叠：通过；
- 每条记录：4 history + 8 future、0.5–4.0 秒、相机标定、真实未来状态（私有参考）。

## 下一步

1. 对 v3 清单生成完整 NAVSIM metric cache；
2. 用同一清单跑 DriveWAM native action 和 future image；
3. 重新计算 Step 1、CFAC、FAU、FCS；
4. 将公开提交格式剥离未来图像路径和私有 GT，只保留协议元数据。

服务器产物：

`/mnt/slurmfs-4090node3/user_data/zchen897/iac_navsim_benchmark_v3/benchmark_v3_navsim_private.jsonl`

审计：

`/mnt/slurmfs-4090node3/user_data/zchen897/iac_navsim_benchmark_v3/benchmark_v3_navsim_audit.json`
