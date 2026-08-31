# 新 WAM 接入协议

目标是把“评测一个新 WAM”从一次研究工作变成一次性 adapter 接入。模型本身不需要修改 IAC；adapter 只负责把模型原生输入、未来图像、动作和分支 lineage 映射成统一 manifest。

## 1. 最小输出格式

每个分支一行 JSON（JSONL）：

```json
{
  "source_key": "dataset:scene:frame",
  "counterfactual_group_id": "dataset:scene:frame",
  "branch_id": "dataset:scene:frame::branch=logged",
  "branch_mode": "logged",
  "history_images": ["..."],
  "future_images": ["..."],
  "future_times_s": [0.2, 0.4, 0.6],
  "action_trajectory": [[x, y, yaw], [x, y, yaw]],
  "action_injection_verified": true,
  "realized_future_ego_state": null,
  "task_success": null
}
```

要求：

- `future_times_s` 必须是真实生成时间，不能为了凑协议重命名帧；
- 分支必须共享完全相同的 history 和 `counterfactual_group_id`；
- action 只能在最后比较阶段进入 scorer，不能泄漏到图像侧 probe；
- 没有独立 rollout 时，`realized_future_ego_state` 和 `task_success` 必须留空。

## 2. 能力声明

另提供一个 capability JSON：

```json
{
  "native_action_head": true,
  "external_trajectory_control": true,
  "time_alignment": "exact",
  "independent_rollout": true
}
```

`time_alignment` 只有三种值：

- `exact`：模型原生时间点就是统一评分轴；
- `continuous_resample`：保留原生帧和时间戳，由连续运动 posterior 映射到统一评分轴；
- `unsupported`：不能进入正式反事实评分。

例如 Epona 原生 5 Hz、覆盖 4 秒，应声明 `continuous_resample`，不能声明 `exact`。

## 3. 一条命令完成预检

```bash
python scripts/wam_onboarding_preflight.py \
  --manifest /path/to/wam_branches.jsonl \
  --capability-json /path/to/capability.json \
  --output /path/to/preflight.json
```

预检会自动给出：

- `image_probe_ready`：能否运行 Level-1；
- `counterfactual_image_ready`：是否具备受控 action→image 分支；
- `formal_level2_ready`：是否允许正式 CCFC；
- `fcs_ready`：是否有独立 rollout 和任务成功标签；
- `next_action`：下一步应运行 Level-1、CCFC、FCS，还是修复 adapter。

## 4. 接入后的实际工作量

同一模型架构只需实现一次 adapter。之后更换 checkpoint 或数据集只需重新生成 manifest 并运行预检；IAC scorer、统计 bootstrap、覆盖率和报告格式保持不变。

如果模型缺少 native action head、外部控制或独立 rollout，系统会自动降级为 Level-1 或 action-only 结果，不会伪造 CCFC/FCS 分数。
