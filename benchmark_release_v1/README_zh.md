# IAC Benchmark · Level-1 发布版 v1

English version: [README.md](README.md)

IAC（Imagined-future and Action Consistency）是一个用于评测世界动作模型
（WAM）的可复现基准，目标是检查模型想象的未来是否与其输出动作保持一致。

本发布版提供 **Level-1 连续图像侧测量层**：使用候选盲的几何探针，从图像序列
中提取前向/横向运动、航向和曲率，再与 WAM 的未来动作或轨迹进行比较。它是
可靠的测量层，但单独不能宣称已经证明因果一致性。

## 发布包结构

```text
benchmark_release_v1/
├── configs/         冻结的 RAFT-Large 地面平面配置
├── datasets/       脱敏后的 benchmark/dev manifest 与审计文件
├── docs/            冻结的 benchmark 协议与 Level-1 主表
├── scripts/         数据集构建、审计和 scorer
├── src/iac_new/     Level-1 几何、光流与评分库
├── tests/           确定性的单元测试与协议测试
├── weights/         冻结的 RAFT-Large 权重及校验和
├── pyproject.toml
├── VERSION
├── README.md        English documentation
└── README_zh.md     中文文档
```

原始 NAVSIM/Waymo 图像、私有绝对路径、WAM checkpoint 和实验日志均未放入
发布包。它们仍保存在数据存储区，并通过 manifest 接口在运行时挂载，以避免
数据泄漏、版权问题和不必要的仓库膨胀。

## 固定评测协议

- 评测服务器上的私有输入包含 4 帧历史图像（`t <= 0`）和 8 帧未来图像。
  本公开包中的 manifest **不包含未来图像**，只包含协议元数据和脱敏后的样本身份。
- 未来帧时间为 `0.5, 1.0, ..., 4.0 s`，必须保留精确时间戳。
- 必须提供相机内参、外参和畸变参数。
- benchmark 与 dev 按 scene/log group 隔离，构建过程确定性且有审计文件。
- 当前冻结版本包含：benchmark 500 条 NAVSIM + 80 条 Waymo；dev 250 条 NAVSIM
  + 20 条 Waymo，共 580/270 条。
- 样本分层包含停车、制动、加速、横向转弯和直行巡航，避免被直行样本主导。

公开 manifest 保留样本身份、split、stratum、时间戳、标定和历史状态，但移除
原始图像路径及未来真值。运行时需要在本地环境中配置私有数据根目录。

## 安装与验证

```bash
python -m pip install -e .
PYTHONPATH=src:. python -m pytest tests -q
sha256sum -c weights/SHA256SUMS.txt
```

测试覆盖标定、时序几何、光流可靠性、轨迹解码、split 隔离以及 Level-1 连续
评分器。发布分支会在服务器环境中运行完整测试套件并记录结果。

## 运行 Level-1 评测

评测服务器上，先使用 `configs/plane.json` 对 WAM 生成的未来图像记录运行冻结的
图像解码器，得到 decoder-score JSONL。公开 manifest 没有未来图像和未来参考状态，
不能单独完成评分。动作或轨迹字段只能在最后的比较阶段读取，不能输入图像侧解码器。

```bash
python scripts/audit_wam_level1_outputs.py --generated <wam_generated_records.jsonl> \
  --output <wam_output_audit.json>
python scripts/evaluate_continuous_decoder.py \
  --manifest <private_wam_generated_records.jsonl> \
  --config configs/plane.json \
  --output <decoder_scores.jsonl>
python scripts/evaluate_continuous_motion_alignment.py \
  --manifest <private_evaluation_manifest.jsonl> \
  --scores <decoder_scores.jsonl> \
  --reference-source action \
  --require-eight-frame-four-second \
  --output <out_dir>
```

报告包括路径归一化后的横向、航向和曲率误差，候选盲可观测性以及 coverage-risk
曲线。速度仅作诊断，在 v1 中不属于正式 Level-1 主指标。本发布版不宣称 CCFC 或
FCS；它们需要单独的、包含成对 WAM 干预和生成未来图像的评测包，不属于公开 v1 协议。

## 数据准备

使用 `scripts/build_level1_benchmark_v1.py` 从私有 NAVSIM/Waymo 记录构建确定性
划分；使用 `scripts/prepare_waymo_level1_samples.py` 将 Waymo Perception v2
shard 转换为 4+8 帧接口；使用 `scripts/build_public_benchmark_manifest.py`
生成无泄漏的公开 manifest。

数据协议见 `docs/BENCHMARK_DATASET_V1_ZH.md`，冻结主表见
`docs/LEVEL1_MAIN_TABLE_BENCHMARK_V1_ZH.md`。

## 数据与许可证

本仓库发布代码、manifest 和第三方光流权重。NAVSIM 与 Waymo 数据仍受其原始
访问条款约束，本仓库不重新分发原始数据。权重来源和上游许可证见
`weights/README.md`。
