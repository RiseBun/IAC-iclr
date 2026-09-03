# IAC Level-1 Benchmark v1：数据协议与冻结结果

更新时间：2026-09-02  
协议：`iac-level1-benchmark-v1`  
固定随机种子：`20260902`

## 1. 这次冻结解决什么问题

78 条样本适合 smoke test，但不能支撑可靠性结论：样本太少、运动模式不均衡，而且重叠窗口容易把同一段驾驶过程重复计数。本版本把 Level-1 图像侧测量基准固定为：

- 每个样本 **4 帧历史 + 8 帧未来**；未来时间为约 `0.5, 1.0, ..., 4.0 s`，精确时间戳保留在 manifest 中。
- 每个样本必须有前视图像、相机标定、历史自车状态和 8 个未来参考状态。
- 以物理场景/日志为隔离单位，`dev_v1` 与 `benchmark_v1` 不共享 scene group。
- 候选窗口按运动模式分层：`stop`、`braking`、`acceleration`、`lateral_turn`、`straight_cruise`。
- 未来图像是数据集的 native realized future，只能用于验证图像测量能力；真正的 WAM 因果评测时，必须替换成 WAM 生成的 future images。

因此，这个数据集能回答“图像侧能否稳定读出相对几何、横向/航向变化和轨迹形状”，不能单独回答“WAM 是否因果地使用了想象未来”。

Level-1 正式主表与有效性解释见 [LEVEL1_MAIN_TABLE_BENCHMARK_V1_ZH.md](LEVEL1_MAIN_TABLE_BENCHMARK_V1_ZH.md)。

## 2. 当前可复现规模

候选源和冻结集如下：

| 来源 | 候选有效窗口 | 日志/场景来源 | 冻结 dev | 冻结 benchmark |
|---|---:|---:|---:|---:|
| NAVSIM mini | 3,860 | 64 个原始 log，按 12 帧不重叠窗口导出 | 250 | 500 |
| Waymo Level-1 v1 | 200 | 50 个 segment；沿用已审计的 eval/development 角色 | 20 | 80 |
| **合计** | **4,060** | — | **270** | **580** |

Waymo 当前源只有 100 条 scene-disjoint evaluation 窗口，且为了保证 dev/benchmark 场景隔离，本次只使用其中 80 条作为 benchmark、20 条作为 dev。不能把同一 segment 的重叠窗口扩写成独立样本。

冻结后的总 manifest 为 850 条，其中 641 个 scene group；`benchmark_v1` 内部没有窗口重叠、没有重复 sample id。详细数字见：

- [benchmark_v1.audit.json](../datasets/benchmark_v1.audit.json)
- [dev_v1.audit.json](../datasets/dev_v1.audit.json)

## 3. 文件和使用方式

- [benchmark_v1_public.jsonl](../datasets/benchmark_v1_public.jsonl)：脱敏后的正式 Level-1 benchmark，580 条。
- [dev_v1_public.jsonl](../datasets/dev_v1_public.jsonl)：脱敏后的开发/阈值冻结集，270 条。
- [build_level1_benchmark_v1.py](../scripts/build_level1_benchmark_v1.py)：从源 manifest 重新生成上述文件的确定性脚本。

公开 manifest 不包含原始图像路径，不复制原始图像。评测服务器通过私有映射恢复图像；
发给 WAM 的输入协议只暴露：

1. 4 帧历史图像；
2. 时间戳、相机标定和允许的历史自车状态/waypoint；
3. 样本 id 与 split。

8 帧 future images、realized future state 和参考轨迹保留在评测服务器，用于计算
Step 1 与 FAU。CCFC 和 FCS 需要作者额外提交成对干预或独立 rollout；它们是主榜
的能力分层列，但没有相应接口的模型标记 `unavailable`，不会被填成 0。

## 4. 分层不是标签装饰

`stratum` 从 native future 的连续状态计算，不使用图像或 WAM 动作：

- `stop`：未来速度幅值低于 0.5 m/s；
- `braking` / `acceleration`：4 秒窗口首尾速度变化率分别低于 -1.0 或高于 1.0 m/s²；
- `lateral_turn`：横向位移绝对值至少 1.5 m，或航向变化至少 0.12 rad；
- `straight_cruise`：其余可观测窗口。

选择采用固定种子 round-robin，避免 580 条几乎全是直行/停车；稀有的制动、加速样本不通过复制制造。

## 5. 可信度边界

这个版本的“可信”来自数据协议，而不是宣称图像恢复了真实控制：

- scene/log 级隔离，避免相邻窗口把同一事件泄漏到两个 split；
- 时间戳单调、历史/未来帧数和 4 秒 horizon 强校验；
- 相机内参、畸变和 camera-to-ego 标定强校验；
- 记录 `rejected_by_reason`，当前有 23 个候选因 horizon 不符合 4 秒合同而被拒绝；
- 明确 `causal_claim_allowed: false`：native realized future 只验证 Level-1 measurement。

当前 benchmark 的运动分布为：

| 模式 | 数量（dev + benchmark） |
|---|---:|
| stop | 186 |
| braking | 109 |
| acceleration | 131 |
| lateral_turn | 190 |
| straight_cruise | 234 |

## 6. 下一步扩容，而不是放宽标准

当前 580 条足以替代 78 条作为第一版正式内部 benchmark，但还不是最终公开规模。扩容顺序固定为：

1. 从 Waymo raw segment 重新导出更多 **scene-disjoint** 4+8 窗口，并保留连续状态和标定；
2. 将 NAVSIM trainval 中可访问的更多 log 纳入同一脚本；
3. 目标是每个数据域至少 300 条 benchmark、每个稀有运动模式至少 50 条，并报告按域 macro-average，而不是把 NAVSIM 的数量优势混入总均值；
4. 另留一份不公开的 hidden holdout，等协议和 Level-1 探针冻结后只用于最终报告。

任何新增样本都必须重新生成 audit；若只能通过重复 scene 或重叠窗口增加数量，则不纳入 benchmark。
