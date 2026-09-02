# Level-1 主表（benchmark_v1）

更新：2026-09-02  
探针：冻结 RAFT-Large + 地面平面（`configs/plane.json`），形状/速度分门，**关闭** shape fallback  
数据：`benchmark_v1` 580 条（NAVSIM 500 + Waymo 80），scene-disjoint，无窗重叠  
边界：验证**图像测量器**。未来是 logged 真值，不证明 WAM 因果。

三门定义：配对差（对照 MAE − 图像 MAE）的 bootstrap 95% CI **下界 > 0**。对照为 history 外推、速度匹配的错未来、时间倒序。

---

## 主表

| # | 指标 | 结果 | 口径 | 证明什么 | 不证明什么 |
|---|---|---|---|---|---|
| 1 | 形状覆盖（非 stop） | **440/468 = 94.0%** | 至少 1 个形状 interval 可评 | 运动场景几乎都能读横向/航向/曲率 | 停车也必须出连续轨迹 |
| 2 | 停车识别 | **92/112 = 82.1%** 标为 `stopped` | 仅 stop 层；类别不是速度 | 零流+低历史速度能判停 | 精确 m/s |
| 3 | `lateral_speed` 点误差 | MAE **0.095 m/s**，容差内 **98.4%** | 形状可评 457 条；容差 0.50 m/s | 横向速度读得准 | 比历史外推**多**读到横向（三门不过，见下） |
| 4 | `yaw_rate` 点误差 | MAE **0.029 rad/s**，容差内 **97.0%** | 同上；容差 0.15 rad/s | 航向变化读得准 | — |
| 5 | `curvature` 点误差 | MAE **0.022 /m**，容差内 **86.1%** | 同上；容差 0.06 /m | 弯形可读 | 容差内略低于前两项 |
| 6 | 航向增量（转弯层） | **yaw 三门过** | `lateral_turn` 106/114 可评 | 未来图像比历史 yaw 外推、错未来、倒放都更好 | 全池混进直行/停车后三门会稀释 |
| 7 | 曲率增量（转弯层） | **curvature 三门过** | 同上 | 弯形用了正确未来与时间顺序 | 同上 |
| 8 | 曲率增量（全池） | **三门过** | 形状可评 457 | 曲率增量在诚实混合集上仍成立 | 不推广到 lateral |

`lateral_turn` 上 lateral 本身：容差内 98.6%，但 history/shuffle 的 CI 跨 0，**只过 reversal**。因此主表不把「横向增量」写成过。

---

## 怎么读：点误差 ≠ 增量

- **点误差（3–5 行）**：探针输出和真值有多近。准，就可以拿去比较两条 WAM 视频。
- **增量（6–8 行）**：不看未来、或看错未来，误差会不会显著变差。过了才能说「读的是这段未来，不是历史惯性」。

转弯时横向大致是 `v × 转角`。历史里已有速度和 yaw-rate，外推已经能把横向定住，所以 lateral **准但没有增量**。航向和曲率会在 4 秒里加紧/放松，历史瞬时 yaw-rate 不够用，所以 yaw/curvature **既准又有增量**。

全池 lateral/yaw 三门不过，是因为 112 停 + 153 巡航没有横向增量可测，不是尺子坏了。增量只在 `lateral_turn` 上报。

---

## 有效性（给评审的四句话）

1. **能评**：运动样本 94% 有形状读数；停车 82% 能标停，不拿死流去估 m/s。  
2. **读得准**：lateral/yaw 容差内 97%+；这是后续 A→F 门和 CCFC 图像侧的尺子。  
3. **用了未来**：转弯层 yaw、曲率三门过；曲率全池也过。  
4. **不夸大**：绝对速度不进主表；lateral 不声称增量；本表不是 WAM 因果。

---

## 不进主表

绝对速度、加速度、米制前向距离、全池 lateral/yaw 三门、shape fallback、旧 78 条事件均衡集（只作开发 smoke）。

复现配置：[`configs/plane.json`](../configs/plane.json)。运行入口为
[`scripts/evaluate_continuous_decoder.py`](../scripts/evaluate_continuous_decoder.py)
和 [`scripts/evaluate_continuous_motion_alignment.py`](../scripts/evaluate_continuous_motion_alignment.py)，
并使用 `--disable-shape-fallback` 生成严格主表结果。原始报告目录不随公开包分发。
