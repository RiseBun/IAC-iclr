# WAM 视觉—动作统一准入与评测协议 v1

本协议把 IAC 的适用范围固定为一条可验证的信息链，而不是某一种网络结构：

> **提交的 WAM 必须同时提供 native action 和 future visual state。**

future visual state 可以是直接生成的未来 RGB 帧，也可以是能够用固定 decoder
稳定重建为 RGB 的未来 latent。我们不要求视频与动作由同一个 head 联合生成，
也不要求模型提供 semantic clear/risk 或在部署时默认生成视频。

冻结 benchmark 的参考轴是 4 秒、8 个时间点（0.5 Hz 间隔）。但 WAM 提交不被
强制改写成 8 点：只要提供 **至少 4 个未来点并覆盖到约 4.0 s**，保留模型原生
时间戳即可。DriveWAM 的原生 4 点、1 Hz 轴因此合规；评测器按时间戳对齐，不把
插值点当作模型原生输出。

## 1. 最小准入条件

每个样本必须能复现以下字段：

```text
sample_id
history
condition / intervention
future visual output
native action
future timestamps
random seed
model revision
lineage
```

如果 future 是 latent，还必须提供：

```text
decoder revision
decoder checksum
latent-to-frame reconstruction protocol
```

评测服务器先把 latent 解码成固定格式的 RGB 帧，再进入图像侧探针。无法稳定
重建 RGB 的 latent 不能报告本协议的 RAFT 图像指标，但可以在另行注册的
action-only 试验中作为对照。

准入边界是信息边界，不是架构偏好：

| 提交形态 | 是否进入本协议 | 说明 |
|---|---:|---|
| RGB future + native action | 是 | 直接进入三个 Step |
| 可解码 future latent + native action | 是 | 先按固定 decoder 重建 RGB |
| joint / cascaded / 分离式生成 | 是 | 只要字段和 lineage 完整 |
| 只有 native action | 否 | 不能验证 foresight consistency |
| 只有 future video | 否 | 没有 action 侧比较对象 |
| 外部控制动作而非 native action | 否 | 只能做独立的 video/action 响应诊断 |

## 2. 三个 Step

### Step 1：Visual Motion Measurement

```text
future RGB（或 decoded latent）
  → RAFT-Large 前后向一致性
  → 动态抑制
  → 标定地面平面自车几何
  → candidate-blind continuous decoder
  → observability / abstention
  → lateral motion、yaw rate、curvature
```

动作和 waypoint 只在最后比较阶段读取，不能进入图像解码器。正式主表使用
lateral motion、yaw rate、curvature，以及 observability 和 coverage-risk；绝对
速度、加速度和米制前向距离保留为诊断项。

### Step 2：Action–Future Consistency（CCFC 主指标）

对固定 history、prompt、seed 和 nuisance，提交任意一种可重复的成对干预，例如：

```text
left / right
slow / fast
command 0 / command 2
future-latent swap
```

比较：

```text
ΔP_F(t) = P_F,branch1(t) − P_F,branch0(t)
ΔP_A(t) = P_A,branch1(t) − P_A,branch0(t)
CCFC    = consistency(ΔP_F, ΔP_A)
```

必须分别报告方向一致性、幅度一致性、时间对齐、coverage 和干预类型。

两条硬规则：

1. `stop` 样本只用于停车识别与 coverage，不进入 CFAC 的运动平均值；转弯、制动、
   加速和巡航分别报告后再做分层汇总。
2. 干预必须改变 WAM 的条件或输入，并重新生成 future visual 与 native action。
   评测端直接注入、覆盖或替换 action 的实验只能记为 action-response 诊断，不能
   记为 CCFC。

* 两侧 future 与 action 都随干预改变：报告 `CCFC`，并将其作为主榜指标列；
* 只有 future 表征改变 native action：报告 `F2A mechanism`；
* 只有 action 改变 future 图像：报告 `A→F response`；
* 没有可重复干预：该模型的 `CCFC` 为 `unavailable`，不能填 0，也不影响它报告
  `CFAC`/`FAU` 等其它可用主榜列。

semantic clear/risk 是一种高价值干预，但不是所有模型的硬性准入条件。

### Step 3：Independent Execution

```text
native action
  → independent simulator rollout
  → realized ego state
  → explicit task-success label
  → FCS
```

FCS 必须标注干预类型（例如 `FCS-command` 或 `FCS-semantic`）。缺少独立任务
标签或兼容模拟器时记为 `unavailable`，不能把缺失当作失败分数。

## 3. 失败关闭与报告规则

评测器对缺失字段、无法重建的 latent、未固定 seed 或 lineage 不完整的样本
fail-closed：该样本对应格子为 `unavailable`，并记录原因；只有违反硬准入条件时才为
`ineligible`。任何 Step 都不得用
logged/oracle action 或 waypoint 冒充 native action，也不得用 waypoint 冒充
realized state。

旧版 `capability` 枚举（如 `video_only`、`action_only`、
`externally_controlled_video`）仅为冻结试点和历史审计保留；新提交应按本协议
提供视觉未来与 native action，并在 manifest 中声明实际干预类型和 lineage。

## 4. 与冻结发布包的关系

Step 1 的 RAFT-Large 探针、`configs/plane.json`、公开 split 和现有脚本保持不变。
本文件只放宽“生成形式”和“干预语义”的准入，不放宽可验证的信息链：

```text
future visual state + native action + timestamp + seed + model revision + lineage
```

这是 IAC 能够覆盖更多 WAM、同时仍然回答“想象未来是否参与动作”的最小协议。
