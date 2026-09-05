# Level-1 测量复核：RAFT 推理分辨率与区域航向 shadow（2026-09-05）

本轮实验回答两个独立问题：

1. 在 WAM 输出保持 `448x256` 不变时，把 RAFT 推理画布从 `512x288` 提高到 `768x432` 是否改善测量？
2. 保留道路估计，增加候选盲的背景航向可靠性融合，是否改善航向测量？

主评分公式、阈值、双向一致性规则和解码坐标均未改变。区域航向只作为 shadow，不进入冻结主分。

## 1. 先修复并核对的 K 问题

v3 WAM 图像已经是 `448x256`，但原清单中的 K 仍为原始 `1920x1080` 坐标：

```text
[[1545, 0, 960], [0, 1545, 560], [0, 0, 1]]
```

直接在 `448x256` 上使用该 K 是标定错位。正确的 resize-only K 为：

```text
[[360.5, 0, 224.0],
 [0, 366.2222222222, 132.7407407407],
 [0, 0, 1]]
```

本轮 WAM 数字全部使用修正后的 K。旧的未缩放 K 结果不具有可比性，不作为结论。

修复点：

- `scripts/build_drivewam_v3_level1_input.py` 要求显式的 K 源坐标，默认 `1920x1080`；
- `scripts/calibrate_level1_manifest_intrinsics.py` 要求显式 `--intrinsics-source-width/height`，拒绝对已校准清单重复校准；
- 所有清单记录 `intrinsics_source_size`、`intrinsics_image_size` 和 `intrinsics_calibrated`。

## 2. A 组：固定解码坐标的 RAFT 分辨率对照

### 协议

- 数据：同一 `benchmark_v3` 的 1,000 条 NAVSIM 样本；真实参考和 DriveWAM 生成视频分别统计。
- WAM 图像内容：固定为生成的 `448x256`，不重新生成视频。
- baseline：RAFT-Large 在 `512x288` 推理，流和前后向 mask 映射回 `448x256`。
- candidate：RAFT-Large 在 `768x432` 推理，流和前后向 mask 映射回同一 `448x256`。
- 两者使用同一 checkpoint、更新次数、地面几何、候选盲连续解码器和评分公式。
- 配对统计只在同一 `sample_id` 上进行；负的 candidate-minus-baseline MAE 表示 candidate 较好。

### 真实参考视频

| 配置 | 可评估样本 | interval coverage | lateral MAE | yaw MAE | curvature MAE | arc translation MAE | arc heading MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512 | 946 | 0.9438 | 0.12135 m/s | 0.03219 rad/s | 0.01265 1/m | 0.07817 | 0.04784 rad |
| 768 | 948 | 0.9598 | 0.11351 m/s | 0.03140 rad/s | 0.01246 1/m | 0.07872 | 0.05537 rad |

768 有 1 条输入在解码阶段没有有限运动像素，按协议记为 decoder abstain（不是填零）；因此逐样本对照保留 `1000` 个 sample_id，但有效误差只在共同可评估样本上计算。

配对差值（768 - 512）：

- lateral：`-0.00535`，95% bootstrap CI `[-0.04393, 0.03427]`；
- yaw：`-0.00083`，CI `[-0.01107, 0.00967]`；
- curvature：`-0.00021`，CI `[-0.00109, 0.00070]`；
- matched-shuffle 错轨迹拒绝率：`88.29% → 88.11%`。

结论：真实视频上没有显著收益。

### DriveWAM 生成视频（修正 K）

| 配置 | 可评估样本 | interval coverage | lateral MAE | yaw MAE | curvature MAE | arc translation MAE | arc heading MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512 | 934 | 0.7291 | 0.54217 m/s | 0.13779 rad/s | 0.03886 1/m | 0.46617 | 0.32597 rad |
| 768 | 920 | 0.7193 | 0.49427 m/s | 0.12808 rad/s | 0.03837 1/m | 0.46953 | 0.32278 rad |

配对差值（768 - 512）：

- lateral：`-0.01880`，CI `[-0.09069, 0.04733]`；
- yaw：`-0.00350`，CI `[-0.01753, 0.01026]`；
- curvature：`+0.00022`，CI `[-0.00200, 0.00243]`；
- matched-shuffle 错轨迹拒绝率：`65.35% → 65.46%`。

结论：方向上有小幅改善迹象，但所有核心 MAE 的 CI 都跨 0，coverage 下降约 `0.98` 个百分点；不足以接入冻结主线。`512x288` 保持为正式配置，`768x432` 仅保留为可复现实验选项。

## 3. B 组：区域航向可靠性融合 shadow

实现位于 `scripts/evaluate_regional_heading_shadow.py`，汇总位于 `scripts/summarize_regional_heading_shadow.py`。测量阶段只使用图像、流、双向一致性、相机几何和道路/区域特征；参考轨迹在所有预测完成后才加入，因此 `candidate_bank_used_by_measurement=false`。

### 真实参考视频

| 方法 | interval coverage | 航向增量 MAE | 中位绝对误差 | 反向航向拒绝率 |
|---|---:|---:|---:|---:|
| 道路单源 | 1.000 | 0.3083° | 0.1500° | 99.03% |
| 31 特征区域融合 | 1.000 | 0.1496° | 0.0935° | 99.76% |

### DriveWAM 生成视频

| 方法 | interval coverage | 航向增量 MAE | 中位绝对误差 | 反向航向拒绝率 |
|---|---:|---:|---:|---:|
| 道路单源 | 0.869 | 2.8750° | 0.6082° | 87.64% |
| 31 特征区域融合 | 0.869 | 2.4029° | 0.3055° | 90.77% |

在 3,476 个共同可观测 interval 上，融合相对道路单源的 MAE 差为 `-0.4722°`，95% bootstrap CI `[-0.5569°, -0.3900°]`，融合更优比例 `70.5%`。

结论：区域融合在真实和 WAM 视频上都有独立测量增益，但 WAM 绝对误差仍显著大于真实参考。因此本轮只记录为 shadow；不把它与现有 lateral/yaw/curvature 主分相加，也不重新调阈值。下一步若要接入，必须先在冻结 dev 集上预注册融合权重和 abstention 规则，再在 benchmark v3 上一次性重跑。

## 4. 运行产物与复现

私有评测结果目录（发布包不包含原始路径）：

```text
`<private_run_root>/benchmark_v3_runs/calibrated_resize_ablation/`
```

关键文件：

- `wam_kfix_base512_alignment.json`
- `wam_kfix_highres768_alignment.json`
- `wam_kfix_resolution_ablation.json`
- `real_base512_alignment.json`
- `real_highres768_alignment.json`
- `real_resolution_ablation.json`
- `real_heading_shadow_{0,1,2,3}.jsonl`
- `real_heading_shadow_summary.json`
- `wam_heading_shadow_{0,1,2,3}.jsonl`
- `wam_heading_shadow_summary.json`
- `wam_input_k1920_to_448.jsonl`

核心命令（在服务器 `iac_new` 根目录）：

```bash
PYTHONPATH=src:. $PY scripts/evaluate_continuous_motion_alignment.py \
  --manifest $RUN/wam_input_k1920_to_448.jsonl \
  --scores $RUN/wam_kfix_base512_merged.jsonl \
  --reference-source action --include-uncertain \
  --config configs/plane_native4.json \
  --output $RUN/wam_kfix_base512_alignment.json

PYTHONPATH=src:. $PY scripts/analyze_measurement_resolution_ablation.py \
  --baseline $RUN/wam_kfix_base512_alignment.json \
  --candidate $RUN/wam_kfix_highres768_alignment.json \
  --output $RUN/wam_kfix_resolution_ablation.json

PYTHONPATH=src:. $PY scripts/summarize_regional_heading_shadow.py \
  --inputs $RUN/wam_heading_shadow_{0,1,2,3}.jsonl \
  --reference-source action \
  --output $RUN/wam_heading_shadow_summary.json
```

本轮不改变 Level-1 主指标定义：仍以 lateral / yaw / curvature 及 relative/arc shape 为主；纵向米制量只作诊断。
