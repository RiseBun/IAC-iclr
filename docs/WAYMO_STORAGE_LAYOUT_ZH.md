# Waymo 外置存储布局

## 服务器当前布局

Waymo 数据不放在项目所在的 `/mnt/slurmfs-4090node1`，因为该盘当前只剩约
`89 GB`。服务器已建立以下外置数据根目录：

```text
/mnt/slurmfs-4090node3/user_data/zchen897/iac_waymo
```

项目侧访问入口为软链接：

```text
iac_new/datasets/waymo_external
  -> /mnt/slurmfs-4090node3/user_data/zchen897/iac_waymo
```

当前外置盘可用空间约 `11 TB`。代码、JSONL manifest 和评估脚本可以继续使用
普通文件路径；manifest 中保存实际图像路径即可，不需要把数据复制回项目盘。

## 推荐目录

```text
iac_waymo/
  raw/          # 仅保存必要的官方原始分片；不要默认下载全集
  frames/       # 选定样本的 history 4 帧 + future 8 帧
  manifests/    # Waymo -> IAC JSONL 协议
  generated/    # WAM 生成的 8 帧图像
  cache/        # RAFT/结构检测中间结果
  reports/      # 审计、评估和校准报告
```

## Level-1 可复现数据协议

Waymo E2E v1.0.0 的 TFRecord 适合提供未来自车状态、驾驶意图和场景标签，
但每条 `E2EDFrame` 只有一个当前时刻的多相机图像，并不保证连续的未来 RGB 帧。
因此 Level-1 使用 Waymo Perception v2.0.1 的模块化 validation 数据补齐图像侧：

```text
gs://waymo_open_dataset_v_2_0_1/validation/camera_image/
gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/
gs://waymo_open_dataset_v_2_0_1/validation/vehicle_pose/
```

每个 segment 的 `camera_image` 是约 20 秒、10 Hz 的连续 JPEG；`vehicle_pose` 给出
逐帧 `world_from_vehicle` 位姿，`camera_calibration` 给出前视相机内外参。只下载这
三个组件，不下载 lidar、分割或检测标签即可满足 RAFT/AllTracker 和轨迹一致性评测。

转换命令（服务器上执行）：

```bash
python scripts/prepare_waymo_level1_samples.py \
  --input /mnt/slurmfs-4090node3/user_data/zchen897/iac_waymo/raw/perception_v2/validation \
  --output /mnt/slurmfs-4090node3/user_data/zchen897/iac_waymo/frames/level1
```

脚本默认使用 Waymo `FRONT=1` 相机，在 10 Hz 原始序列上每 5 帧取一个点，生成：

- `history_images`: `[-1.5, -1.0, -0.5, 0]` 秒，共 4 帧；
- `future_images`: `[0.5, 1.0, ..., 4.0]` 秒，共 8 帧；
- `history_ego_state` 与 `realized_future_ego_state`：规范化为
  `[x_m, y_m, yaw_rad, speed_mps, yaw_rate_rps]`，坐标原点是 anchor 车辆后轴；
- 每个样本的精确时间戳、JPEG 路径、相机标定和原始 segment ID。

脚本输出的 `manifest.jsonl` 是 IAC 的唯一训练/评测入口。JPEG 按 segment 去重保存，
滑动窗口样本只保存路径引用，不会重复复制图像。建议先下载 20--50 个 segment 做
能力验证，再扩展到 200--400 个 segment；完整 Perception validation 的 camera
image 组件约 225 GiB，没必要全量下载。

## 使用规则

1. 优先保存前视图像、ego pose、标定和时间戳；不保存无关相机和完整 TFRecord 副本。
2. Level-1 固定 `history=4`、`future=8`、`horizon=4 s`，图像使用 JPEG/WebP，避免
   生成大量 PNG 和 tensor 缓存。
3. 运行任务前确认外置盘挂载：

   ```bash
   readlink -f datasets/waymo_external
   df -h datasets/waymo_external
   ```

4. 软链接只存在于服务器工作区，不提交到 Git；换服务器时需要重新建立同名入口。

## 容量预算

第一轮建议使用 200--400 个 scene。按每个样本 12 张压缩图像、RAFT 缓存和 WAM
生成图像估算，预留 `20--60 GB` 足够；只有下载完整原始 TFRecord、所有相机或未压缩
中间结果时，才需要 TB 级空间。
