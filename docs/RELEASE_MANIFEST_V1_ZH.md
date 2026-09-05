# IAC Benchmark 发布清单

本清单定义 GitHub `benchmark-release-v1` 中哪些文件属于可复现协议，哪些内容
必须留在私有评测环境。

## 发布内容

| 路径 | 作用 | 是否必需 |
|---|---|---:|
| `src/iac_new/` | 光流、标定地面几何、连续解码、可观测性和评分实现 | 是 |
| `scripts/validate_wam_submission.py` | 提交格式与泄漏审计 | 是 |
| `scripts/evaluate_continuous_decoder.py` | Step 1 图像侧解码入口 | 是 |
| `scripts/evaluate_continuous_motion_alignment.py` | 图像运动与 native action 对齐 | 是 |
| `scripts/score_iac_submission.py` | 能力分层记分板 | 是 |
| `configs/plane.json` | v3 冻结配置：448×256 解码、512×288 RAFT 推理 | 是 |
| `datasets/benchmark_v3_public.jsonl` | 1000 条脱敏 NAVSIM 主榜身份与协议元数据 | 是 |
| `datasets/benchmark_v3.audit.json` | 选集、分层和泄漏审计摘要 | 是 |
| `weights/` | RAFT-Large 权重、来源和 SHA-256 | 是 |
| `tests/` | 确定性协议与几何测试 | 推荐 |
| `docs/` | v3 协议、结果、审计与复现边界 | 是 |

## 明确排除

- 原始 NAVSIM/Waymo 图像、未来 GT、私有绝对路径；
- DriveWAM、LingBot-VA 或其他 WAM checkpoint；
- 生成视频、逐样本服务器日志和中间缓存；
- 768×432 分辨率消融及区域航向 shadow 配置；这些结果只在
  `docs/MEASUREMENT_ABLATION_20260905_ZH.md` 中作为实验记录；
- 旧版 580 条 NAVSIM+Waymo 数据的 pilot 结果不进入 v3 主榜；历史运行记录不随包发布。

## 复现边界

公开 manifest 只提供 `sample_id`、split、时间轴、标定和历史状态。评测端必须将
私有图像与 GT 按 `sample_id` join；提交方不得把 GT、realized state 或候选轨迹
注入图像侧解码器。所有不可观测区间必须 abstain，不能用 0 或插值掩盖。

## 发布前检查

```bash
python -m pip install -e .
PYTHONPATH=src:. python -m pytest -q
sha256sum -c weights/SHA256SUMS.txt
```

发布包版本由 `VERSION`、`pyproject.toml` 和 `src/iac_new/__init__.py` 三处共同
声明，当前均为 `1.0.0`。
