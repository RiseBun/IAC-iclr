# WAM 提交与分层记分板 v1

本文件补全公开包里缺的「作者怎么交、我们怎么打分」。  
当前主榜使用 `benchmark_v3` split；`benchmark_v1` 仅保留为历史兼容集。新的统一准入是“future visual state +
native action”；生成形式不设限，详见
[`WAM_SCOPE_AND_UNIFIED_PROTOCOL_V1_ZH.md`](WAM_SCOPE_AND_UNIFIED_PROTOCOL_V1_ZH.md)。
CCFC/FCS 是能力分层主榜列：模型支持就提交并计分，不支持就标记
`unavailable`，不得填 0。`ineligible` 只用于违反硬准入条件的提交。

## 0. 新提交的硬条件

每个新提交必须同时提供：

- 可复现的 native action；
- future visual state：至少 4 个未来点、覆盖约 4 秒的 RGB，或通过固定 decoder
  重建出的对应 RGB；模型必须保留原生时间戳。公开 Level-1 join/audit 接受
  **4 或 8 个未来帧**（history 另计，通常 4 帧；因此总图常为 8 或 12）；
  DriveWAM 原生 4 点/1 Hz 与标准 8 点/0.5 Hz 均合规；
- `history`、`condition/intervention`、精确时间戳、随机种子、model revision 和
  lineage。

若提交 future latent，还要提供 decoder revision、decoder checksum 和
latent-to-frame 重建协议。联合生成、共享 action/video head、semantic clear/risk
以及部署时是否默认生成视频都不是硬性条件。无法重建 RGB 的 latent 不能进入
Step 1 RAFT 图像指标；只有 action 的模型不能进入本协议主格。

当前 v1 脚本接口仍读取 `future_images` 路径。因此 latent 提交者应先用声明的
decoder 和重建协议生成与模型原生时间轴对应的 RGB 文件（至少 4 个点、覆盖 4 秒），再在 JSONL 中填写这些路径，
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
| `native_action_conditioned` | ≥4 个未来点覆盖 4 秒 + 同轴 native action | CFAC（及兼容旧 L1/A→F/F→A）；CCFC、FAU、FCS 需额外证据 |
| `externally_controlled_video` | ≥4 个未来点覆盖 4 秒 + 外控轨迹；`action_source=external_control` | 仅 A→F |
| `video_only` | ≥4 个未来点覆盖 4 秒 | 无 IAC 主格 |
| `action_only` | 仅动作 | 无 IAC 主格 |

禁止字段：`realized_future_ego_state`、logged/oracle action 冒充 native。  
CCFC 若要参评，另需同一 `counterfactual_group_id` 下两条 `branch_mode`，并固定
history、seed、nuisance；`clear`/`risk` 只是可选示例。干预必须改变模型的条件或
输入并重新生成 future 与 native action；评测端不得在生成后直接注入、覆盖或替换
action，再把结果声称为 CCFC。

停车（`stratum=stop`）只进入独立停车识别和 coverage 报告，不进入 CFAC 的运动
平均值；其余分层仍分别报告后再做 macro-average。

## 2. 评测服务器怎么跑

```bash
python scripts/validate_wam_submission.py \
  --public datasets/benchmark_v3_public.jsonl \
  --submission <submission.jsonl> \
  --output <audit.json>

python scripts/build_wam_level1_continuous_manifest.py \
  --base <private_benchmark_v3.jsonl> \
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
  --disable-shape-fallback \
  --output <l1_report.json>

python scripts/score_iac_submission.py \
  --public datasets/benchmark_v3_public.jsonl \
  --submission <submission.jsonl> \
  --measurements <cells.json> \
  --output <scorecard.json>
```

公开 manifest 不能单独打分。动作只在最后比较阶段读取。

## 3. 主榜指标与能力状态

主榜不把不同能力强行压成一个总分，而是并列展示以下列，并同时展示每列的
`n`、coverage 和状态。这样没有成对干预的模型仍可报告 CFAC/FAU，不会因缺少
CCFC 被判零分；但也不能把 `unavailable` 当作可比的低分。

主评分只使用已通过误差预算的形状/相对量：`lateral_speed_mps`、`yaw_rate_radps`、
`curvature_1pm`，以及归一化相对距离/弧长形状。`speed_mps`、`acceleration_mps2`
和未归一化的前向米制位移不进入主分，只作为诊断列。任何指标未达到可靠性门槛时，
必须从主分移除并标记为 `diagnostic_only`，不能用降权掩盖不可靠性。

| 主榜列 | 回答的问题 | 最小证据 | 不支持时 |
|---|---|---|---|
| `CFAC` | 单次推理中，想象运动与 native action 的形状是否一致 | 一条 future visual + 一条 native action | `unavailable` |
| `CCFC` | 两次固定条件推理中，干预引起的想象变化与动作变化是否一致 | 同 history/seed/nuisance 的成对分支；干预类型显式记录 | `unavailable` |
| `FAU_F` | 想象运动相对 history 是否接近私有 GT future | future visual + 私有 GT join | `unavailable` |
| `FAU_A` | native action 相对 history 是否接近私有 GT future | native action + 私有 GT join | `unavailable` |
| `FAU` | 想象与动作是否都接近真实未来 | `sqrt(FAU_F × FAU_A)` | `unavailable` |
| `FCS` | native action 在独立执行中是否成功 | 兼容 simulator、realized state、task label | `unavailable` |

`CCFC` 是正式主榜指标，但不是所有模型的硬性准入条件；排名时按能力列和
coverage 分层报告，不对缺失列做零填充或未经校准的总平均。

### 3.1 兼容的 CCFC 干预

不要求 semantic clear/risk。只要可重复、可审计，以下任一种都可以：
`left/right`、`slow/fast`、command 变化、future-latent swap。semantic 干预只是一种
`intervention_type`，不是 CCFC 的唯一形式。

## 4. 记分板格子（兼容旧字段）

| 格 | 含义 | 过线 |
|---|---|---|
| `l1` | 生成未来 vs native action 的形状对齐 | 主表 MAE/容差 + 覆盖 |
| `a2f` | 左/右或 clear/risk 图像是否随动作变 | bootstrap L1 下界 > 0.005 |
| `f2a` | 干预未来表征后 native action 是否变 | 内容敏感，不是 zero 开关 |
| `ccfc` | 同 history 的任意可重复双分支，Δimage↔Δaction | 主榜 CCFC；缺少双分支则 `unavailable` |
| `fau` | `sqrt(FAU_F × FAU_A)`，两侧都对私有 GT future | 主榜 FAU；缺 GT 则 `unavailable` |
| `fcs` | 再加独立 rollout | 主榜 FCS；无兼容环境则 `unavailable` |

不具备某项可选能力 = `unavailable`；声称具备但材料不完整 = `missing`；违反硬准入
= `ineligible`；小 n 已有结果 = `pilot`。禁止把这些状态写成 0 分。

v3 参考记分板示例：`datasets/scorecard_v3_example.json`。历史试点命令
`--frozen-pilots` 仅为兼容用途，不用于 v3 主榜。CCFC/FCS/FAU 是否可用按实际证据
填写；没有证据的列标记 `unavailable`，而不是伪造分数。
