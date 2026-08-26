# 近一年自动驾驶 WAM 结构与 IAC 事件体系调研

调研日期：2026-08-26

## 1. 调研问题与范围

本文回答两个问题：

1. 近一年公开的自动驾驶 World Action Model（AD-WAM）采用什么输入、输出、时间跨度、数据集与 action head？
2. IAC 要评测 WAM 是否真正利用预测未来，除转向与直行外还必须覆盖哪些事件？

“近一年”严格按首次公开日期限定为 2025-08-26 至 2026-08-26。主表只采用论文、官方项目页或官方仓库能够核对的信息。早于该时间窗但仍有直接参考价值的 VaViM/VaVAM、Epona 和 ACT-Bench 不计入主表，ACT-Bench 仅作为事件设计基线。

## 2. 第一性原理：什么才是我们要评测的 WAM

视频质量高和规划成功都不能证明动作由预测未来驱动。对 IAC 而言，关键不是模型是否同时有 video loss 与 action loss，而是推理计算图中是否存在可检验的未来到动作依赖。

将模型按信息流分为四类：

```text
F -> A       先预测未来，再由未来生成或修正动作
F <-> A      未来与动作联合生成，需要干预验证因果方向
A -> F       动作决定生成未来，未来不反向决定动作
train-only F 未来只提供训练监督，推理动作不读取未来
```

其中：

- `F -> A` 最直接符合 Foresight-Conditioned Success 和 Future-Use Intervention 的评测目标。
- `F <-> A` 只能说明耦合，必须交换、置换或干预未来后观察动作变化，才能证明未来被使用。
- `A -> F` 适合评测 action controllability，但不能单独证明动作由未来驱动。
- `train-only F` 可以证明视频监督改善表示学习，却不能声称部署动作由显式想象未来产生。

## 3. 近一年主要 AD-WAM

| 模型 | 输入 | 未来与动作输出 | Action head / 信息流 | 训练与评测数据 |
|---|---|---|---|---|
| [DriveVLA-W0](https://arxiv.org/abs/2510.12796) | 前视图像序列、语言指令、历史动作；时间消融的最佳配置为当前帧加 1 秒前帧 | AR 版本预测离散视觉 token；diffusion 版本预测下一帧；输出动作 token 序列 | 500M MoE action expert；比较 query+MLP、FAST 自回归和 flow matching。大规模数据下 AR 最好。图像生成在驾驶推理时绕过，属于 `train-only F` | NAVSIM v1/v2；内部 70M 帧数据 |
| [UniUGP](https://arxiv.org/abs/2512.09864) | 多帧图像、语言、历史轨迹、速度与加速度 | CoT、未来视频、20 点/5 秒轨迹，4 Hz；论文未明确视频输出帧数 | VLM understanding expert、flow planning expert、diffusion generation expert；边缘部署可关闭 generation expert | nuScenes、nuPlan、Waymo、DriveLM 及多个长尾数据集 |
| [VLA-World](https://arxiv.org/abs/2604.09059) | nuScenes 多视角图像与语言 | 0.5 秒未来单帧；高层动作与最终 3 秒轨迹 | 先预测 0.5 秒短轨迹和方向，再生成未来帧，对未来帧推理后修正 3 秒轨迹，形成 `A0 -> F -> A1` | nuScenes、nuScenes-GR-20K |
| [Latent-WAM](https://arxiv.org/abs/2603.24581) | 4 个时刻、左/前/右相机、ego command/velocity/acceleration | 未来 latent world status；4 秒多候选轨迹 | causal Transformer 训练 latent transition；trajectory query+MLP 解码候选轨迹。推理只保留 encoder 与 trajectory decoder，未来模块更接近训练监督 | NAVSIM v1/v2、HUGSIM |
| [DriveVA](https://arxiv.org/abs/2604.04198) | 4 个历史前视帧、ego velocity、文本/导航 | 8 个未来帧和 8 个 `(x,y,yaw)` 动作，2 Hz、4 秒 | 同一 DiT 对未来视频 latent 与动作 token 联合 flow matching；2 步采样，属于 `F <-> A` | NAVSIM v1；零样本 nuScenes、Bench2Drive |
| [DriveFuture](https://arxiv.org/abs/2605.09701) | 2 个时刻、前/后相机、ego state | 预测未来 latent；100 条候选轨迹，每条 8 点、4 秒、0.5 秒间隔 | 未来 latent 显式条件化 5 层 diffusion planner，再由 GTRS-Dense 选轨迹，属于 latent `F -> A` | NAVSIM v1/v2 navtest/navhard |
| [DriveWAM](https://arxiv.org/abs/2605.28544) | 当前前视帧、ego state、历史动作、VLM scene-evolving guidance | 4 个未来帧，1 Hz、4 秒；4 秒动作 chunk；PhysicalAI 动作标签保留 10 Hz | 先生成未来视频 latent，再由 MLP action decoder 进行 inverse-dynamics flow matching，明确 `F -> A` | NAVSIM v1、PhysicalAI-Autonomous-Vehicles |
| [Discrete-WAM](https://arxiv.org/abs/2606.05645) | 历史前视图像、ego state、导航命令 | 未来离散视觉 token；4 秒离散加速度 token | 高层 decision skeleton 后执行离散 acceleration token editing；通过两次积分恢复轨迹；支持 `A -> F`、联合 world-policy 和 policy-only 模式 | nuPlan 预训练、NAVSIM v1/v2 |
| [Metis](https://arxiv.org/abs/2606.15869) | 当前单帧前视图、ego state、文本 | 训练时 8 帧视频；8 个 `(x,y,yaw)` waypoint，2 Hz、4 秒 | 1B action DiT；非对称 mask 允许动作影响未来视频、禁止动作读取未来视频；推理只运行动作分支，属于 `A -> F / train-only F` | NAVSIM v1/v2、CityWalker |
| [UNIVERSE](https://arxiv.org/abs/2607.05133) | 4 个历史前视帧 | 8 个未来帧；4 秒轨迹 | 同一 DiT 使用两个 flow head，但 visibility mask 双向屏蔽未来视频与动作 token；默认 trajectory-only、2 步采样，属于 `train-only F` | NAVSIM；零样本 nuScenes、Bench2Drive |
| [SimWAM](https://arxiv.org/abs/2608.07468) | 当前前视帧、速度、加速度、yaw rate、导航 | 训练时 8 个未来帧；8 个 waypoint，2 Hz、4 秒 | 独立 lightweight action DiT 与 isolated mask；部署时删除视频分支，action expert 再用 FlowGRPO 强化，属于 `train-only F` | NAVSIM v1/v2；零样本 nuScenes |
| [WA-JEPA](https://arxiv.org/abs/2608.20974) | 4 个历史时刻的左/前/右/后相机、历史动作、ego state | 8 个未来 latent 帧与 8 个 `(x,y,yaw)`，2 Hz、4 秒 | MMDiT joint predictor 同时从噪声生成未来 latent 和动作；action stream 可注意未来 scene token，属于 latent `F <-> A` | nuPlan 预训练、NAVSIM v1/v2、零样本 HUGSIM |

## 4. Action head 的主流处理方法

### 4.1 连续轨迹回归

使用 learnable query 或 scene token，通过 MLP 直接输出 waypoint。优点是速度快、数值稳定；缺点是容易回归均值，难以表示多模态行为。DriveVLA-W0 的 query head 和 Latent-WAM 属于这一类。

### 4.2 自回归离散动作

将连续轨迹量化或用 FAST tokenizer 转换成 token，再由语言/视觉 backbone 自回归生成。它利用统一 token 接口，适合大规模数据，但会引入量化误差与序列延迟。DriveVLA-W0 的 AR head 属于这一类。

### 4.3 连续 flow/diffusion action expert

从高斯噪声开始，对完整 trajectory chunk 进行多步或少步去噪。它能够表达多模态轨迹，是 2026 年最常见的 action head。DriveVA、DriveWAM、DriveFuture、Metis、UNIVERSE、SimWAM 和 WA-JEPA 均采用这一方向，但未来到动作的信息通路并不相同。

### 4.4 离散 acceleration token editing

Discrete-WAM 不直接离散位置，而是对纵向和横向加速度建立离散词表，利用 soft-label interpolation 减少硬量化误差，再对加速度积分两次恢复轨迹。其 high-level decision skeleton 先约束机动模式，再生成低层动作。

### 4.5 多候选生成与 scorer

DriveFuture 等方法从 diffusion head 采样多条轨迹，再使用独立 scorer 选择。这提高了 NAVSIM 分数，但会引入新的因果问题：最终动作可能主要由 scorer 决定，而不是由 world latent 决定。因此 IAC 需要同时记录生成候选集合和最终选中动作。

## 5. 对 IAC 模型选择的直接结论

第一批正式评测对象：

1. `DriveWAM`：显式 `F -> A`，最适合验证 FUI。
2. `VLA-World`：存在可观察的“初始动作-未来-修正动作”链。
3. `DriveVA`：像素视频和动作联合生成，适合 Event-CC 与干预测试。
4. `WA-JEPA`：latent future 与动作联合生成，适合扩展 latent event probe。

结构性对照组：

1. `Metis`、`SimWAM`、`UNIVERSE`、`DriveVLA-W0`：视频监督有效，但默认动作推理不依赖显式未来。
2. `DriveFuture`：动作依赖 latent future，但不能直接使用 RAFT，需要 latent intervention 或可解码的 feature probe。
3. `Discrete-WAM`：适合评测 action-conditioned counterfactual world，但需区分 world-policy mode 与最终 policy-only action path。

因此，同一套 IAC 不能强迫所有模型走完全相同的像素接口。正式 benchmark 至少需要三个 adapter：

```text
pixel-future adapter  -> RAFT/event posterior
latent-future adapter -> frozen latent event probe or controlled intervention
action-only adapter   -> negative control; 不声称显式 foresight use
```

## 6. 为什么转向/直行不够

转向与直行描述的是粗粒度 route intent，而不是完整行为。NAVSIM 官方接口明确指出，left/right command 同时覆盖转弯、变道与急弯。因此，仅使用 `{straight, left, right}` 会把以下不同决策错误合并：

- 正常左转与向左变道；
- 绕过障碍与执行路线左转；
- 跟随道路曲率与主动选择目标车道；
- 为行人停车与为红灯停车；
- 正常减速与应对 cut-in 的紧急制动。

真正的因果评测必须同时表示“发生了什么”“为什么响应”“采取了什么动作”和“结果是否安全”。

## 7. 推荐的因子化事件表示

不要建立一个互斥的平铺类别表。单个片段通常同时包含多个事件，例如：

```text
pedestrian_crossing + yield + decelerate_to_stop + no_collision
```

推荐将事件后验定义为：

```text
E = (lateral, longitudinal, interaction, trigger, outcome)
```

### 7.1 横向机动 lateral

```text
keep_lane
turn_left / turn_right
lane_change_left / lane_change_right
merge_left / merge_right
diverge_left / diverge_right
avoid_left / avoid_right
u_turn
roundabout_enter / roundabout_exit
pull_over
```

### 7.2 纵向机动 longitudinal

```text
maintain_speed
accelerate
decelerate
stop
start
creep
reverse
emergency_brake
```

### 7.3 交互决策 interaction

```text
free_drive
follow
yield
gap_accept
overtake
give_way
maintain_headway
```

### 7.4 因果触发 trigger

```text
lead_vehicle_brake
vehicle_cut_in / vehicle_cut_out
pedestrian_crossing / pedestrian_popout
cyclist_crossing / cyclist_conflict
red_light / yellow_light / stop_sign
blocked_lane / stopped_vehicle
road_debris / construction_barrier
unprotected_turn / merge_gap
emergency_vehicle
door_opening
wrong_way_vehicle
```

### 7.5 结果 outcome

```text
no_collision / collision
safe_ttc / unsafe_ttc
drivable_area_compliant / off_road
direction_compliant / wrong_way
lane_keep_compliant / lane_violation
traffic_light_compliant / red_light_violation
comfortable / excessive_jerk
progress / blocked_or_no_progress
```

## 8. 场景条件不是动作事件

以下内容应作为分层统计的 context slice，而不是 action/event posterior 的互斥标签：

```text
天气：clear / rain / snow / fog
光照：day / dusk / night / tunnel / backlight
道路：urban / highway / ramp / intersection / narrow lane
密度：sparse / medium / dense
遮挡：large vehicle / static obstacle / temporary occlusion
对象属性：child / elderly / crowd
```

事故、施工、积水等只有在导致 `stop`、`yield`、`avoid` 或 `lane_change` 时，才形成可评分的动作因果链。

## 9. IAC V1 与 V2 的事件优先级

### 9.1 V1 必须新增

在当前 `{keep_lane, turn_left, turn_right}` 基础上，优先增加：

```text
lane_change_left / lane_change_right
accelerate / decelerate
stop / start / creep
merge_left / merge_right
yield / gap_accept
follow
emergency_brake
avoid_left / avoid_right
```

这些事件在 4 秒 WAM horizon 内通常可观察，并且能够产生明确的替代动作。

### 9.2 V2 交互与规则事件

```text
pedestrian/cyclist crossing response
vehicle cut-in response
lead vehicle sudden-brake response
red-light and stop-sign compliance
unprotected-turn gap acceptance
blocked-lane and construction detour
roundabout entry/exit
emergency-vehicle response
```

## 10. 首批应闭合的四条因果链

首批正式数据不应追求事件数量，而应优先闭合以下四条可干预链：

```text
cut-in / lead brake
  -> imagined conflict
  -> decelerate or emergency brake
  -> safe TTC / no collision

pedestrian crossing
  -> imagined occupancy conflict
  -> yield / stop
  -> restart after clearance

blocked lane
  -> imagined obstruction
  -> lane change / lateral avoidance / stop
  -> drivable-area and collision outcome

unprotected turn or merge
  -> imagined gap evolution
  -> yield or gap acceptance
  -> safe progress
```

它们比继续细化普通转弯更有价值，因为每条链都存在合理但不同的备选动作，能够区分“动作恰好正确”和“动作由模型预测未来驱动”。

## 11. 对指标实现的要求

对每个样本分别提取：

```text
p_img(E)   imagined future 的事件后验
p_act(E)   WAM 输出动作的事件后验
p_real(E)  实际执行/独立日志的事件后验
success    独立任务成功和安全结果
```

并至少报告：

1. imagined-event 与 action-event 的一致性，而不是只比较轨迹 ADE/FDE；
2. action intervention 后 imagined-event 是否按预期改变；
3. future swap/permutation 后 planner action 是否发生超出随机噪声的定向变化；
4. foresight 正确且 action 与其一致时的条件成功率；
5. observability、abstention 和各事件的有效样本数；
6. pixel、latent 和 action-only 三类模型分别统计，不混合排名。

## 12. 主要来源

- [DriveVLA-W0](https://arxiv.org/abs/2510.12796)
- [UniUGP](https://arxiv.org/abs/2512.09864)
- [Latent-WAM](https://arxiv.org/abs/2603.24581)
- [DriveVA](https://arxiv.org/abs/2604.04198)
- [VLA-World](https://arxiv.org/abs/2604.09059)
- [DriveFuture](https://arxiv.org/abs/2605.09701)
- [DriveWAM](https://arxiv.org/abs/2605.28544)
- [Discrete-WAM](https://arxiv.org/abs/2606.05645)
- [Metis](https://arxiv.org/abs/2606.15869)
- [UNIVERSE](https://arxiv.org/abs/2607.05133)
- [SimWAM](https://arxiv.org/abs/2608.07468)
- [WA-JEPA](https://arxiv.org/abs/2608.20974)
- [ACT-Bench](https://arxiv.org/abs/2412.05337)
- [nuPlan scenario taxonomy](https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/script/config/common/scenario_builder/scenario_mapping/nuplan_scenario_mapping.yaml)
- [NAVSIM agent input/output definition](https://github.com/autonomousvision/navsim/blob/main/docs/agents.md)
- [NAVSIM official repository](https://github.com/autonomousvision/navsim)

## 13. 最终结论

近一年 AD-WAM 的共同输出窗口正在收敛到约 4 秒，常见表示是 8 个未来帧/waypoint、2 Hz；但其因果结构没有收敛。联合训练、联合生成、未来条件动作和训练期视频监督是四个不同命题，不能使用同一个“WAM”标签替代。

IAC 的独特价值不应是再做一个视频质量指标或轨迹恢复器，而应是建立统一的事件接口，并通过 action intervention、future-use intervention 和 realized outcome 验证：模型想象了什么、因此做了什么、最终是否成功。
