# Actor 相对运动能力 V1

## 1. 要解决的量

方向事件只能回答“想象中向哪里走”。四类交互因果链还需要回答：

- 哪个 actor 与自车发生冲突；
- actor 的纵向/横向相对距离；
- closing speed 与横向侵入速度；
- TTC、进入自车走廊时间和 gap 状态；
- 这些量在当前图像中是否可观测。

目标不是从单目视频精确恢复所有三维轨迹，而是在证据充分时输出带区间的
交互状态，在证据不足时弃权。

## 2. 方法结构

```text
history state + 8 imagined frames + camera calibration
  |                         |
  |                         +-> instance mask / actor identity
  |                                  -> 8-frame association / occlusion tracking
  |
  +-> RAFT-Large forward/backward flow
        -> static-road support and ego geometry

actor ground contact / metric depth + calibrated ground plane
  -> per-frame actor position in instantaneous ego coordinates
  -> robust temporal fit
  -> d_long, d_lat, closing speed, lateral speed
  -> TTC / time-to-corridor / gap state
  -> observability, interval posterior, abstention
  -> interaction event posterior
  -> compare with action event and independently measured realization
```

这里有两条不同职责的视觉支路：

- `RAFT-Large` 支路提供静态地面、自车运动和局部连续运动证据；
- 长程 tracker 支路维持 actor 身份并处理遮挡。

RAFT 不能替代 actor association，tracker 也不能提供可靠米制尺度。二者必须由标定地面
几何连接。单目 metric depth 只能作为辅助或 challenger，不能当作真值。

## 3. 无泄漏数据接口

固定输入协议为 4 秒、8 帧，时刻 `0.5, 1.0, ..., 4.0 s`：

```text
future_images:       8 x RGB
future_timestamps:   精确时间戳
history_ego_state:   t <= 0 的 pose/speed/acceleration/yaw_rate
history_waypoints:   仅 t <= 0
camera:              intrinsics/extrinsics/distortion
```

禁止输入 action head 的未来 waypoint、候选轨迹或未来控制量。它们只能在事件提取完成后
作为被比较对象。否则评测器可以从动作反推出事件，`future -> action` 的因果检验失效。

actor 轨迹估计器的 JSONL 输入为：

```json
{"sample_id":"s1","actor_id":"vehicle-7","class_label":"vehicle","times_s":[0.5,1.0,1.5],"positions_ego_m":[[18.0,0.2],[16.5,0.1],[15.0,0.0]],"visibility":[true,true,true],"confidence":[0.9,0.9,0.8]}
```

`positions_ego_m[t]` 表示 actor 在该时刻瞬时自车坐标系中的 `[longitudinal, lateral]`。
它可以来自标定地面接触点，或带可靠性标签的米制深度回投影。

当位置来自 CoTracker3 时，先用 `observe(..., query_points=...)` 跟踪 detector/segmenter
提供的 actor anchor，再用 `actor_pixel_tracks_from_observation` 按 `query_index` 取出每个
actor 的轨迹。查询点坐标属于 tracker 的 resized 图像，因此投影前必须用同一 resize 比例
缩放相机内参；身份关联和遮挡标记由 detector/tracker 提供，不能由后续速度拟合猜测。

执行：

```bash
PYTHONPATH=src python scripts/estimate_actor_relative_motion.py \
  --tracks actor_tracks.jsonl --require-eight-frame-four-second \
  --output actor_relative_posteriors.jsonl
```

输出包含距离、closing/lateral speed 的 `q05/q50/q95`，纵向 `ttc_s`、结合横向走廊的
`corridor_conflict_ttc_s`、进入走廊时间、拟合残差、
可观测性和弃权原因，并明确记录 `candidate_bank_used=false`、
`future_action_used=false`。

## 4. 正式比较协议

真值必须独立来自数据集 actor state、LiDAR/radar 或人工校核，不能来自被评测 WAM 的动作。
评分 JSONL 每行至少包含：

```json
{"predicted_distance_m":15.1,"reference_distance_m":15.0,"predicted_closing_speed_mps":3.1,"reference_closing_speed_mps":3.0,"predicted_ttc_s":4.87,"reference_ttc_s":5.0,"observability":0.82,"abstain":false}
```

执行：

```bash
PYTHONPATH=src python scripts/evaluate_relative_motion_metrics.py \
  --records relative_motion_gold_pairs.jsonl \
  --output relative_motion_metrics.json
```

正式报告同时给出：

- distance MAE；
- closing-speed MAE；
- 接近/远离/静止方向准确率；
- 危险 TTC F1；
- coverage；
- coverage-risk 曲线。

`ttc_s` 仅表示纵向接近；只有 actor 已在走廊内，或预计在纵向 TTC 之前进入走廊时，
才产生 `corridor_conflict_ttc_s`。四类交互链的危险 TTC 应优先使用后者。

只报告高置信样本的误差会产生选择偏差，因此误差必须与 coverage 同时出现。缺字段、NaN、
时间跨度不足或支持不足都按弃权处理。
评分器会排除任何 `future_action_used=true` 或 `candidate_bank_used=true` 的记录，并将
`formal_ready=false` 写入报告；这种记录不能进入正式因果指标。

## 5. RAFT-Large 与 SEA-RAFT 的 78 样本 A/B

这次 A/B 使用现有冻结的 NAVSIM 78 样本、4 个未来时刻、约 2 秒协议。除 flow backend 外，
连续解码器和事件阈值一致。

| 指标 | RAFT-Large | SEA-RAFT | 结论 |
|---|---:|---:|---|
| 成功解码 | 78/78 | 77/78 | RAFT-Large 更稳健 |
| 联合误差 | 0.4314 | 0.6462 | RAFT-Large 更好 |
| 横向 MAE / m | 0.1350 | 0.2236 | RAFT-Large 更好 |
| yaw MAE / rad | 0.0090 | 0.0231 | RAFT-Large 更好 |
| speed relative error | 0.3072 | 0.2915 | SEA-RAFT 小幅更好 |
| 横向事件 accuracy | 0.9776 | 0.9675 | RAFT-Large 更好 |
| 横向事件 macro-F1 | 0.9744 | 0.9643 | RAFT-Large 更好 |
| 纵向诊断 accuracy | 0.6250 | 0.6721 | SEA-RAFT 更好，但仍非正式指标 |

因此：

- 正式图像事件基线继续使用 RAFT-Large；
- SEA-RAFT 保留为速度侧 challenger，不进入默认配置；
- 这组 A/B 不能证明 4 秒 actor 相对速度有效，因为它没有 8 帧 actor-level 真值。

## 6. 当前证据与下一道门

已完成：SEA-RAFT 后端修复、同协议 A/B、actor 相对运动拟合、区间/弃权协议和正式评分器。

尚未完成的是从原始图像自动产生稳定 actor track，以及在 78 个带独立 actor 真值的样本上
验证距离、closing speed 和 TTC。只有同时通过以下门槛，相对速度才能升级为正式因果指标：

1. actor association 在遮挡和重新出现时保持身份；
2. distance/closing-speed 在独立真值上达到预注册误差门槛；
3. 危险 TTC F1 不靠类别失衡获得；
4. coverage-risk 随高置信筛选单调改善；
5. 不读取未来 action waypoint。

下一实验应固定 78 个 4 秒/8 帧 actor-level 样本，运行：RAFT-Large 基线、加 history state、
加 8-frame tracker、加 metric depth/SpatialTrackerV2 challenger。不能直接沿用本次 2 秒
ego 事件结果冒充相对速度验证。
