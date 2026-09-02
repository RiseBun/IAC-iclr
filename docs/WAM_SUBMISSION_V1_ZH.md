# WAM 提交与分层记分板 v1

本文件补全公开包里缺的「作者怎么交、我们怎么打分」。  
Step 1 探针和 `benchmark_v1` split 不变。新的统一准入是“future visual state +
native action”；生成形式不设限，详见
[`WAM_SCOPE_AND_UNIFIED_PROTOCOL_V1_ZH.md`](WAM_SCOPE_AND_UNIFIED_PROTOCOL_V1_ZH.md)。
CCFC/FCS 在缺少对应证据时仍为 `ineligible`，不得填 0。

## 0. 新提交的硬条件

每个新提交必须同时提供：

- 可复现的 native action；
- future visual state：8 帧 RGB，或通过固定 decoder 重建出的 8 帧 RGB；
- `history`、`condition/intervention`、精确时间戳、随机种子、model revision 和
  lineage。

若提交 future latent，还要提供 decoder revision、decoder checksum 和
latent-to-frame 重建协议。联合生成、共享 action/video head、semantic clear/risk
以及部署时是否默认生成视频都不是硬性条件。无法重建 RGB 的 latent 不能进入
Step 1 RAFT 图像指标；只有 action 的模型不能进入本协议主格。

当前 v1 脚本接口仍读取 `future_images` 路径。因此 latent 提交者应先用声明的
decoder 和重建协议生成 8 个固定时间点的 RGB 文件，再在 JSONL 中填写这些路径，
同时保留上述 decoder 元数据供审计；评测脚本不会偷偷替换 decoder。

下表中的 `externally_controlled_video`、`video_only`、`action_only` 是冻结试点和
历史审计用的兼容枚举，不代表新的主准入范围。

## 1. 作者提交什么

一行一个样本，JSONL。必须能对上公开 split 的 `sample_id`。

```json
{
  "sample_id": "<public sample_id>",
  "wam_model_id": "worlddrive_tadwm",
  "capability": "native_action_conditioned",
  "future_images_source": "wam_generated",
  "future_images": ["future_000.png", "future_001.png", "future_002.png", "future_003.png", "future_004.png", "future_005.png", "future_006.png", "future_007.png"],
  "future_times_s": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
  "action_trajectory": [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0], [6, 0, 0], [7, 0, 0]],
  "action_source": "native_action_head"
}
```

`capability` 只能是：

| 值 | 必须交 | 可打的格子 |
|---|---|---|
| `native_action_conditioned` | 8 帧生成未来 + native action `[8,3]` | L1、A→F、F→A、CCFC、FCS |
| `externally_controlled_video` | 8 帧生成未来 + 外控轨迹；`action_source=external_control` | 仅 A→F |
| `video_only` | 8 帧生成未来 | 无 IAC 主格 |
| `action_only` | 仅动作 | 无 IAC 主格 |

禁止字段：`realized_future_ego_state`、logged/oracle action 冒充 native。  
CCFC 另需同一 `counterfactual_group_id` 下两条 `branch_mode`（如 `clear`/`risk`）。

## 2. 评测服务器怎么跑

```bash
python scripts/validate_wam_submission.py \
  --public datasets/benchmark_v1_public.jsonl \
  --submission <submission.jsonl> \
  --output <audit.json>

python scripts/build_wam_level1_continuous_manifest.py \
  --base <private_benchmark_v1.jsonl> \
  --generated <submission.jsonl> \
  --output <joined.jsonl>

python scripts/evaluate_continuous_decoder.py \
  --manifest <joined.jsonl> \
  --config configs/plane.json \
  --output <decoder_scores.jsonl>

python scripts/evaluate_continuous_motion_alignment.py \
  --manifest <joined.jsonl> \
  --scores <decoder_scores.jsonl> \
  --reference-source action \
  --require-eight-frame-four-second \
  --disable-shape-fallback \
  --output <l1_report.json>

python scripts/score_iac_submission.py \
  --public datasets/benchmark_v1_public.jsonl \
  --submission <submission.jsonl> \
  --measurements <cells.json> \
  --output <scorecard.json>
```

公开 manifest 不能单独打分。动作只在最后比较阶段读取。

## 3. 记分板格子

| 格 | 含义 | 过线 |
|---|---|---|
| `l1` | 生成未来 vs native action 的形状对齐 | 主表 MAE/容差 + 覆盖 |
| `a2f` | 左/右或 clear/risk 图像是否随动作变 | bootstrap L1 下界 > 0.005 |
| `f2a` | 干预未来表征后 native action 是否变 | 内容敏感，不是 zero 开关 |
| `ccfc` | 同 history 语义双分支，Δimage↔Δaction | 正式 CCFC 合同 |
| `fcs` | 再加独立 rollout | 需 `realized_future_ego_state` |

未声称 = `ineligible`。声称但没测 = `missing`。小 n 已有结果 = `pilot`。禁止把 ineligible 写成 0 分。

当前官方试点记分板：`datasets/scorecard_v1.json`（`python scripts/score_iac_submission.py --frozen-pilots`）。CCFC/FCS 全员 ineligible。
