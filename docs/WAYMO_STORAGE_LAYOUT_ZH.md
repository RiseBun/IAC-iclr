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
