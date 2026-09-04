# CFAC / CCFC 低分原因诊断（2026-09-04）

## 结论

当前低分不是“横向运动完全读错”，而是三个问题叠加：

1. **CFAC 的纵向米制幅度误差大**：速度、加速度和前向位移是主要拖分项；
2. **部分 interval 不可观测**：这些应保持 `abstain`，不能当作模型错误；
3. **CCFC 的 metric 幅度项过于脆弱**：pilot 中方向和时间通常较高，但 metric 幅度接近零；换成 scale-free 后大幅恢复。

本诊断不修改正式评分阈值，也不把 scale-free 诊断分数替换成主榜 metric 分数。

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

## 3. CCFC pilot 的拖分项

当前可追溯 CCFC 原始报告是 pilot（16 个 report，含 metric/scale-free/arc-relative 三种模式），不是记分板声称的 357 条全量行级文件。

| 模式 | direction | magnitude | temporal | 几何平均 |
|---|---:|---:|---:|---:|
| metric | 0.8849 | **0.0805** | 0.7679 | 0.2376 |
| scale-free | 0.8849 | 0.8073 | 0.8711 | 0.8496 |
| arc-relative | 0.8437 | 0.8011 | 0.8675 | 0.8293 |

因此 pilot 的主要矛盾非常清楚：

```text
direction 基本能读到
temporal 大体能对齐
metric magnitude 被米制幅度误差拖低
```

例如一个真实 report 中：

```text
direction = 0.9749
magnitude = 0.0000
temporal  = 0.9036
metric score = 0.0000
scale-free score = 0.8351
```

这不是证明 metric 幅度项应该删除，而是证明它当前对生成视频/动作之间的尺度不匹配非常敏感。它必须作为独立子分报告，不能只发布几何平均总分。

## 4. 当前能排除和不能排除的解释

### 已基本排除

- RAFT 在真实视频上完全失效：lateral/yaw 的容差内比例超过 96%；
- 普通模糊、JPEG 压缩和 10% 闪烁导致整体 coverage 崩溃：A/B pilot coverage 几乎不变；
- CCFC 完全随机：正确配对相对于图像错配，direction/temporal 明显更高。

### 尚未排除

- WAM 生成视频是否存在生成域特有的几何闪烁或转弯幅度失真；
- 当前 metric 幅度标定是否适合 WAM 生成视频；
- 记分板 `357/580` 对应的完整 CCFC 子分布，因为原始 357 条 JSON 尚未找回。

## 5. 对低分样本的正确判读

| 观察到的组合 | 结论 |
|---|---|
| direction 高、temporal 高、magnitude 低 | WAM 可能表达了正确动作，但幅度/尺度不匹配 |
| direction 高、temporal 低 | 方向正确，但动作和图像时间轴错位 |
| direction 低、magnitude 低 | 图像运动与动作不一致，或探针受生成伪影影响 |
| coverage 低、`abstain` 多 | 证据不足，不能判错 |
| path cosine 高、metric MAE 高 | 形状正确、米制幅度错误 |

## 6. 下一步

1. 找回并复现 357 条正式 CCFC 原始 report，输出三项子分布和配对 bootstrap CI；
2. 对 WAM 生成视频做身份错配、倒放和最差转弯人工审计；
3. 在 dev 集校准 metric 幅度，不在 benchmark-v1 上调参；
4. 主榜继续同时报告 `CFAC/CCFC`、三项子分数和 coverage；
5. 不把 `abstain` 改成 0，也不因为人工能看出动作就跳过量尺校验。

