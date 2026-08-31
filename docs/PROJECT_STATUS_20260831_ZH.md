# IAC 项目总状态与下一步计划

更新时间：2026-08-31  
仓库：`RiseBun/IAC-iclr`  
当前最新提交：`98f324b`

## 一句话结论

IAC 已从“分别测视频质量或任务成功率”收敛为候选盲、连续运动、反事实闭环的 WAM 评测框架：

```text
WAM action head / action condition
        ↓
   imagined future images
        ↓
candidate-blind IAC image probe
        ↓
continuous motion posterior
        ↓
independent closed-loop realization
        ↓
CCFC / FCS
```

当前最可靠的是 Level-1 图像侧横向/转向测量；Level-2/3 正式 WAM 因果指标尚未完成，因为还没有一个 WAM 在同一批样本上同时提供合格的未来图像、native action head 和独立 rollout。

## 1. 原始问题

现有 WAM 评测通常分别测未来视频质量和 action head 的轨迹/任务成功率，不能证明“模型想象的未来”真正驱动了“模型执行的动作”。IAC 要回答：

> WAM 想象的未来是否与其 action head 一致，并且在受控动作干预下，图像和动作是否产生一致的反事实变化？

我们不追求从单目前视图精确恢复全局米制轨迹，而是恢复足以比较的连续自车运动信号，并把因果性留给 action intervention 与独立闭环执行。

## 2. 冻结的方法结构

### Level-0：图像测量有效性

用真实 NAVSIM future image 与独立 logged ego state 对照，验证图像侧探针是否能读出增量运动信息。该层验证测量器，不是 WAM 因果性。

### Level-1：连续 future-action alignment

输入为 4 帧历史 + 8 帧 future image，future 时间为 `0.5, 1.0, ..., 4.0 s`。action waypoint 只能在最后比较阶段出现，不能输入图像解码器。

```text
history + future front-camera frames
  → RAFT-Large forward/backward flow
  → consistency 与动态抑制
  → ground-plane ego geometry / 相机标定
  → candidate-blind continuous decoder
  → observability / abstention
  → continuous motion posterior
  → 与 action waypoint 的运动学量比较
```

当前正式主信号：

- `lateral_speed`
- `yaw_rate`
- `curvature`

相对距离/相对前向进度是辅助诊断。绝对速度和绝对米制距离不作为正式主指标；单目尺度和逐 interval 噪声仍不稳定。事件仅用于解释和分桶，不能替代连续信号。

### Level-2：CCFC

对同一 history 构造 clear/risk 或 left/right paired intervention，比较 imagined motion profile 与 native action profile 的反事实变化：

```text
Δ imagined motion profile  ↔  Δ native action profile
```

报告方向、幅度、时间对齐、覆盖率和 CCFC。没有 native action head 或 action-to-image intervention 的模型不能进入正式 CCFC。

### Level-3：FCS

在 CCFC 之外，用独立 simulator rollout 得到 `realized_future_ego_state`、`task_score`、`task_success`，再检查 imagined future、action head 和实际执行是否一致。动作条件本身不能代替 realized state。

## 3. Level-1 证据

固定 NAVSIM 基础池为 78 条，另有 25 条 scene-aware non-overlap 窗口用于统计性结论，53 条 overlap 窗口只用于诊断。

78 样本的增量消融中，三个主分量均通过 history、matched-shuffle 和 time-reversal 三门：

| 分量 | history null | matched shuffle | time reversal |
|---|---:|---:|---:|
| lateral speed | 0.069 `[0.042, 0.099]` | 0.176 `[0.117, 0.240]` | 0.073 `[0.052, 0.096]` |
| yaw rate | 0.090 `[0.070, 0.109]` | **0.225** `[0.166, 0.289]` | 0.102 `[0.078, 0.128]` |
| curvature | 0.015 `[0.009, 0.020]` | 0.055 `[0.039, 0.071]` | 0.020 `[0.014, 0.026]` |

结论：图像中存在可泛化的横向/转向时间信息，`yaw_rate` 的时间顺序敏感性最强；这只证明图像测量增量成立，不证明 WAM 因果性。

纵向分支尝试过 RAFT、SEA-RAFT、NeuFlow、UniDepth/PnP。UniDepth+PnP 改善 metric forward progress 的尺度，但逐 interval speed 仍有伪影和重尾，因此速度降为辅助诊断。

## 4. WAM 与闭环证据

### DrivingWorld

- 已完成 20 scenes × 3 branches = 60 的 action-image intervention，动作注入可验证。
- 恢复复算：calibration 12 twins / 24 branches，CCFC `0.4833`、CC lift `0.4628`、diagonal Top-1 `0.8333`；holdout 8 twins / 16 branches，CCFC `0.4038`、CC lift `0.3612`、diagonal Top-1 `0.8125`。
- 旧 manifest 缺少完整 history image、相机标定和独立 realized state，只能作为 image-action response 证据。
- 当前接口没有干净的 external trajectory control，归入 `video_only` / legacy diagnostic。

### Epona

- 历史 generated compatibility `0.3279`，control compatibility `0.6352`。
- 有控制入口，但现有脚本仍是短 horizon（约 4 个 future frame），尚未对齐固定 8 帧 / 4 秒协议。
- 仍需独立 VAE/runtime 验证。

### DriveWAM

- 具有最清晰的 NAVSIM action-conditioning adapter，协议上属于 `native_action_conditioned`。
- 历史 intervention 显示 action-to-image 响应很弱，不能预先假设它是权威模型。
- 服务器已有 NAVSIM checkpoint，但 Wan2.2 基座不完整，仍缺 VAE、tokenizer、text encoder 等组件，并缺少 `easydict` / `flash_attn` 运行依赖。

### WorldDrive / SimWAM

- WorldDrive checkpoint 与代码已登记，尚未完成统一 runtime 验证。
- SimWAM 推理侧主要输出动作、不输出 future image，只能做 action-only 规划诊断。

## 5. 能力分层：覆盖多数 WAM

评测入口不再固定 DriveWAM 为 primary，而是按能力决定可报告范围：

| 能力层 | 必要条件 | 可报告内容 |
|---|---|---|
| `native_action_conditioned` | future image + external control + independent native action head | Level-1、CCFC、FCS |
| `externally_controlled_video` | future image + external control，无 native action head | 受限反事实图像响应 |
| `video_only` | 只能生成 future image | Level-1 图像测量 |
| `action_only` | 只能输出 action | 动作侧规划诊断 |

服务器审计当前给出：

```text
native_action_conditioned: drivewam_navsim, epona_nuplan, worlddrive_tadwm
video_only: drivingworld
action_only: simwam
```

同一个 scorer 通过 adapter 接收 `history + condition → future_images + native_action_head + lineage`。缺少字段就 fail-closed，不用候选轨迹、历史先验或文字事件补齐。

## 6. NAVSIM 闭环执行侧

从原生 NAVSIM pkl 重新导出了 4 history + 8 future 的 native records，并要求图像 anchor token 与 metric-cache token 严格相等：

- 固定 78 条样本中命中 5 个 twin；
- 每个 twin 有 `logged/left/right`，共 15 branches；
- 15/15 独立 PDM rollout 成功导出 `realized_future_ego_state`；
- 临时 PDM `task_success` 为 14/15 = 0.9333；
- action injection lineage 为 15/15。

服务器产物：

```text
/mnt/slurmfs-4090node1/homes/zchen897/work_dirs/closed_loop_recovered_20260829/native_78_cache_aligned_branches.jsonl
/mnt/slurmfs-4090node1/homes/zchen897/work_dirs/closed_loop_recovered_20260829/native_78_cache_aligned_rollout_realized.jsonl
/mnt/slurmfs-4090node1/homes/zchen897/work_dirs/closed_loop_recovered_20260829/native_78_closed_loop_branches.jsonl
```

这 15 条是 WAM 接入模板，但 future image 仍为 `wam_pending`，action 仍是 staging/native branch condition，不是某个 WAM 的真实 action head，因此不是正式 FCS 分数。

此前 9 twins / 27 branches 的 `sample_prev` scaffold 与图像 anchor 相差一个 native frame，只能作为执行链诊断；其结果为 27/27 rollout、task success `22/27 = 0.8148`，不进入正式指标。

## 7. 数据集状态

### NAVSIM

仓库已有：

```text
datasets/navsim_level1_v5/navsim_level1_v5_all_78.jsonl
datasets/navsim_level1_v5/navsim_level1_v5_eval_nonoverlap.jsonl
datasets/navsim_level1_v5/navsim_level1_v5_development_overlap.jsonl
```

当前 WAM 闭环先使用 5 个 exact-cache twin 验证协议；要做稳定统计，需要补齐 cache 或重新选择 25 个 non-overlap 窗口。

### Waymo

Waymo 已放在服务器独立存储盘：

```text
/mnt/slurmfs-4090node3/user_data/zchen897/iac_waymo
```

当前约 26G，已有 SEA-RAFT、RAFT、scale/no-FB 以及 e2e shard 报告。Waymo 尚未形成与 NAVSIM 同等级的最终固定 WAM twin / action-lineage 协议，目前作为 Level-1 外部验证和后续扩展集。

## 8. 当前不能宣称的内容

1. 不能说 IAC 已证明 WAM 因果性；
2. 不能把 DrivingWorld 旧 reciprocal CCFC 当成正式 FCS；
3. 不能把 9-twin scaffold 当作严格 5-twin 正式池；
4. 不能把 PDM static-cached-traffic 兼容模式当成真实交通参与者闭环的最终替代；
5. 不能把某个 WAM 的失败直接解释为 IAC 测量器失败，反之亦然；
6. 不能把速度或绝对米制距离重新升为正式主指标。

## 9. 下一步最短路径

1. **统一 adapter**：优先选择能保留 `source_key/branch_id`、输出 8 帧或可审计 4 秒时间轴、暴露 native action head 并支持固定 seed intervention 的 WAM。
2. **action-response gate**：在 5 个 strict twins 上跑 `logged/left/right` 或 `clear/risk`，先检验图像是否随动作条件变化。
3. **正式 join**：按 `branch_id` 合并 WAM future images、native action、独立 realized state、task score 和 lineage，之后才计算 CCFC/FCS。
4. **扩展统计**：补齐 NAVSIM cache 到 25 个 non-overlap 窗口，按相同 schema 整理 Waymo，并至少让两个 WAM 通过 response gate。

## 10. 关键代码入口

```text
src/iac_new/wam_adapters.py
scripts/inspect_wam_adapters.py
scripts/build_navsim_rollout_staging_manifest.py
scripts/run_navsim_counterfactual_rollouts.py
scripts/attach_counterfactual_rollouts.py
scripts/build_wam_level1_continuous_manifest.py
scripts/finalize_wam_realized_metrics.py
```

仓库当前 clean。最近关键提交：

```text
98f324b feat: gate WAM scoring by capability tier
acaef21 docs: distinguish strict cache-aligned WAM pool
139cac9 feat: stage cache-aligned WAM rollouts
```
