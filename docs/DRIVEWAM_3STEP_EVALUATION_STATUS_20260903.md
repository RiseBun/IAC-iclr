# DriveWAM 三步评测链路：冻结结果与边界

日期：2026-09-03  
模型：DriveWAM，benchmark-v1 固定 580 条主池

## 一句话结论

DriveWAM 已经完成我们的三步评测链路：

1. **Step 1：Visual Motion Measurement**——从 WAM 生成的未来图像重算运动剖面；
2. **Step 2：CCFC**——比较成对干预下 `Δ imagined motion` 与 `Δ native action`；
3. **Step 3：FCS**——把 native action 放入独立 PDM 模拟器，测 realized task outcome。

“完成”指协议、代码、数据联结和结果均已跑通；并不意味着 580 条每一项都具有相同的可用分母。评测严格使用 coverage / N/A 标记，不用 0 填充缺失能力。

## 主结果

| Step | 测量对象 | 结果 | 有效分母 | 解释 |
|---|---|---:|---:|---|
| Step 1 | WAM future image → 冻结 RAFT-Large 探针 → `P_F` | 图像测量 coverage **0.9966** | 580 | 生成图像可被统一探针处理；不把作者提交的 imagined profile 当输入 |
| Step 2 | paired command branches：`ΔP_F` ↔ `ΔP_A` | **CCFC metric 0.1235（诊断）**；arc_relative **0.2234** | 357 / 580 | 正式主分尚未改默认；全量 arc_relative 仍偏低，主因是 direction/temporal≈0.25，不只是米制幅度 |
| Step 3 | native action → NAVSIM PDM 独立闭环 | FCS **0.8086**（397/491） | 491 / 500 NAVSIM | realized state 来自独立模拟器，不读取 WAM future image；FCS 不使用光流探针 |

单次 native4：**形状 CFAC（正式 pilot）`primary_shape_composite`=0.7610（564/580）**，
interval coverage **0.6981**，字段仅 lateral / yaw / curvature。  
旧 **CFAC=0.4825** 也是这三字段，但是
`(direction × magnitude × temporal)^(1/3)` 几何平均，不是“混进了速度”；含纵向米制的是
`experimental_composite≈0.60`，仅诊断。新旧 CFAC **不可直接比高低**。  
**FAU=0.6509（562/580）** 的 components 同样已是三形状量（相对私有 GT / history residual）；
当前 `missing` 是因为尚未按新 CFAC 聚合口径重新冻结，不是因为 FAU 含 speed。FAU 不能替代 FCS。

### 各主榜列的独立 coverage

coverage 不是一个可以跨指标复用的分母，必须随指标单独报告：

| 主榜列 | 有效样本 | 总样本 | coverage |
|---|---:|---:|---:|
| CFAC | 564 | 580 | 0.9724 |
| CCFC | 357 | 580 | 0.6155 |
| FAU | 562 | 580 | 0.9689 |
| FCS | 491 | 500 NAVSIM | 0.9820 |

FCS 的 491/500 不能被写成整个 benchmark 的 coverage；Waymo 80 条在该 PDM
环境中是 `unavailable`。

## Step 1：图像侧测量

评测端不信任模型自行填写的 `imagined_motion_profile`，而是对 `future_images_source=wam_generated` 的图像重新运行冻结探针：

`future images → RAFT-Large + 前后向一致性 → 动态抑制/地面几何 → candidate-blind continuous decoder → lateral / yaw / curvature profile`

输出包括 `P_F(t)`、可观测性、coverage-risk 和质量门状态。Step 1 的结论是“图像侧量尺可以运行”，不是“WAM 一定生成了正确未来”。

## Step 2：CCFC

CCFC 是成对干预条件下 `Δ imagined motion` 与 `Δ native action` 的一致性，不要求
clear/risk，也不声称语义因果。DriveWAM command-paired 全量报告在
`.../benchmark_v1_drivewam_ccfc_eval_native4/ccfc_command_report.json`（580 组，
可评分 357）。三种已存模式均值为：

| scale_mode | mean score | n |
|---|---:|---:|
| metric（当前默认/旧主分） | 0.1235 | 357 |
| scale_free | 0.1861 | 357 |
| arc_relative | 0.2234 | 357 |

metric 下 forward_mae ≫ lateral_mae，米制前向确实拖分；但 **arc_relative 的
direction/temporal 均值仍约 0.25**，说明全量成对一致性即使去掉整体尺度也不高。
小 pilot（约 16 组）上看到的高 direction **不能外推到 357**。  
单次形状 CFAC=0.7610 与 CCFC 不可混称。

## Step 3：FCS

Step 3 只使用 DriveWAM native action 做独立执行检查，不把计划轨迹冒充 realized
state，也不读取 WAM 生成图像：

- 491 条 NAVSIM 样本：`PDMSimulator/BatchKinematicBicycleModel`，4.0 s horizon、0.1 s interval；
- PDM task-score 均值：**0.7312**；
- 成功阈值：`score >= 0.5`；
- FCS：**397/491 = 0.8086**，95% Wilson CI `[0.7714, 0.8409]`；
- rollout errors：0；模拟器 action injection（仅用于执行 native action）verified：491/491；
  这不是 CCFC 的干预注入，也不改变 CCFC 分数定义；
- 9 条 NAVSIM 行因无可用路线/完整未来窗口未进入 PDM cache；Waymo 80 条没有 NAVSIM PDM 对应缓存，标记 `N/A`，不是失败。

当前交通策略为 `static_cached_objects_compat`，所以结果命名为
`NAVSIM-PDM native-action realized-task FCS`。它是独立执行结果，但本身不证明
动作由 imagined future 导致；这需要 CFAC/CCFC 的视觉—动作证据。它还不应被描述
成完整动态交通闭环的最终结论。

## 是否可以说“DriveWAM 完成三步”

可以，准确表述是：

> DriveWAM 在 benchmark-v1 上完成了 Step 1、Step 2 和 Step 3 的可复现实验链路；Step 1/2 使用 580 条主池并报告各自 coverage，Step 3 在 491 条可运行 NAVSIM 样本上完成，Waymo 和 cache 缺失样本按协议 N/A。

不应表述为“580 条全部获得 FCS”，也不应把 CFAC/FCS 合并成一个未经校准的单分数。

## 复现实验产物

- Step 3 报告：`docs/DRIVEWAM_BENCHMARK580_STEP3_FCS_REPORT_20260903.md`
- staging 构建脚本：`scripts/build_drivewam_fcs_staging.py`
- Step 3 分支及 rollout 文件保存在服务器的 `fcs_metric_cache_exact` 目录；公开仓库不包含私有图像、私有 GT 或大体积 metric cache。
