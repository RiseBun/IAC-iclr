# IAC Benchmark · Level-1 Release v1

中文文档：[README_zh.md](README_zh.md)

This GitHub branch **is** the Level-1 package: clone it and run
`pip install -e .` from the repository root. Research notes and the old
78-sample workspace live on `main`; they are not part of this release.

IAC (Imagined-future and Action Consistency) is a reproducible benchmark for
testing whether a WAM's imagined future is aligned with the action it emits.
This release contains the **Level-1 continuous image-side measurement**: a
candidate-blind geometry probe extracts forward/lateral motion, heading and
curvature from an image sequence, then compares those signals with the WAM's
future action/trajectory. It is a measurement layer, not a claim of causality
by itself.

## Method architecture: Step 1 → Step 3

```mermaid
flowchart LR
    I["输入<br/>4 帧历史图像 + 自车状态<br/>WAM：8 帧未来图像 / 4 秒<br/>相机标定"] --> A
    subgraph S1["STEP 1 · 图像侧测量"]
        A["RAFT-Large + 前后向一致性<br/>动态抑制 + 地面平面自车几何"]
        N["光流 novelty<br/>不只是 dense flow：<br/>自车几何 + 候选盲 + abstention<br/>RAFT=位移 · UniDepth=尺度 · tracker=点关联"]
        O["输出 m_F(t)<br/>横向运动 · yaw · 曲率<br/>可观测性 / coverage-risk"]
        B["依据<br/>logged 自车状态 + history/shuffle/reversal 三门"]
        A --> O
        N -.-> A
        B -.-> O
    end
    O --> D
    subgraph S2["STEP 2 · 反事实一致性 CCFC"]
        D["成对 clear/risk 或 left/right 干预<br/>固定历史 + 随机种子"]
        C["比较 Δ 想象运动 ↔ Δ 原生动作"]
        Q["输出 CCFC<br/>方向 · 幅度 · 时间对齐 · 覆盖率"]
        E["依据<br/>候选盲解码 + 原生 lineage<br/>identity/time-order 对照"]
        D --> C --> Q
        E -.-> Q
    end
    Q --> R
    subgraph S3["STEP 3 · 前瞻条件成功 FCS"]
        R["独立模拟器闭环<br/>每条分支分别执行"]
        T["输出实际自车状态<br/>task score / task success → FCS"]
        U["依据<br/>状态必须来自模拟器<br/>waypoint 不能充当实际状态"]
        R --> T
        U -.-> T
    end
```

Editable diagram source: [`docs/IAC_FROZEN_PIPELINE_V1.mmd`](docs/IAC_FROZEN_PIPELINE_V1.mmd).

The levels are cumulative but answer different questions:

| Level | Design intent | Evidence boundary |
|---|---|---|
| **Step 1** | Establish a trustworthy, candidate-blind ruler for future motion | Image-derived motion agrees with an independent logged reference; this validates measurement, not WAM causality |
| **Step 2 / CCFC** | Test whether changing the imagined future changes the native action in the corresponding way | Compare `Δ imagined motion` with `Δ native action` under fixed history/seed; this is the foresight→action bridge, not task success |
| **Step 3 / FCS** | Test whether that consistency survives execution in the environment | Add an independent rollout, realized state and explicit task-success label; this is the causal-closure layer |

### What Step 1 contains

For each 4-history/8-future, 4-second window, the frozen probe uses:

```text
RAFT-Large forward/backward flow
  → consistency checks and dynamic suppression
  → calibrated ground-plane ego geometry
  → camera intrinsics/extrinsics and distortion
  → candidate-blind continuous decoder
  → observability gate and abstention
  → lateral motion, yaw rate and curvature posterior
  → final comparison with action-waypoint kinematics
```

The action or waypoint is withheld from the image decoder and is read only at
the final comparison stage. Absolute speed, acceleration and metric forward
distance remain diagnostic in v1; the formal ruler is lateral motion, yaw rate,
curvature, observability and coverage-risk.

### What CCFC means

**CCFC (Continuous Counterfactual Foresight Consistency)** measures whether a
WAM's imagined future is action-relevant. With the same history, prompt, seed
and nuisance settings, create paired interventions such as clear/risk or
left/right:

```text
ΔP_F(t) = P_F,risk(t) − P_F,clear(t)
ΔP_A(t) = P_A,risk(t) − P_A,clear(t)
CCFC    = consistency(ΔP_F, ΔP_A)
```

The report includes direction, magnitude, temporal alignment and coverage.
Wrong-identity and time-reversal controls test whether the response depends on
future content and its order rather than on a cache-presence artefact.

### What FCS means

**FCS (Foresight-Conditioned Success)** asks whether the foresight–action link
remains valid after the vehicle actually executes the action:

```text
imagined future → native action → independent simulator
                                  → realized state → task success
```

The realized state must come from the simulator, never from the WAM waypoint.
Missing task labels are `unavailable`, never zero. Thus FCS is not a prettier
video score and not a planner score alone: it is the final execution-level
check on the imagined-future/action chain.

## Release contents

```text
.
├── configs/         frozen RAFT-Large ground-plane configuration
├── datasets/        public, path-sanitised benchmark/dev manifests and audits
├── docs/            frozen benchmark protocol and Step 1 main table
├── scripts/         dataset builders, audits and scorers
├── src/iac_new/     reusable Step 1 geometry, flow and scoring library
├── tests/           deterministic unit and protocol tests
├── weights/         frozen RAFT-Large checkpoint + checksum
├── pyproject.toml
├── VERSION
├── README.md
└── README_zh.md
```

The role of each directory is fixed as follows:

| Path | Role | Required for reproduction |
|---|---|---|
| `configs/` | Frozen image-side geometry configuration (`plane.json`) | Yes |
| `datasets/` | Leakage-safe public manifests and split audits | Yes (metadata) |
| `docs/` | Frozen protocol, dataset contract and main-table definitions | Yes (protocol) |
| `scripts/` | Manifest builders, WAM-output audits and Step 1 scorers | Yes |
| `src/iac_new/` | Reusable flow, geometry, posterior and scoring implementation | Yes |
| `tests/` | Deterministic unit and protocol checks | Recommended |
| `weights/` | Frozen RAFT-Large checkpoint, provenance and SHA-256 checksums | Yes for the default probe |
| `pyproject.toml`, `VERSION` | Python package metadata and release identity | Yes |

The package intentionally contains no raw camera frames, private paths, WAM
checkpoints or generated experiment logs. Those inputs are mounted through the
private manifest interface at evaluation time.

Raw NAVSIM/Waymo frames, private absolute paths, WAM checkpoints and generated
logs are deliberately excluded. They stay in the data storage area and are
referenced through the manifest interface, which prevents accidental leakage
and keeps the GitHub package portable.

## Frozen protocol

- The private evaluation input contains four history frames (`t <= 0`) and
  eight future frames at `0.5, …, 4.0 s`. The public manifests in this
  repository do **not** contain future images; they contain only protocol
  metadata and sanitised sample identity.
- Exact timestamps, camera intrinsics/extrinsics and distortion are required.
- Scene/log groups are disjoint between benchmark and dev; split construction
  is deterministic and audited.
- Current internal freeze: 500 NAVSIM + 80 Waymo benchmark records and
  250 NAVSIM + 20 Waymo dev records (580/270 total). The Waymo expansion is
  continuing outside this release; it will become v2 after audit.
- Strata include stop, braking, acceleration, lateral-turn and straight-cruise
  so performance is not dominated by straight driving.

The public JSONL manifests retain sample identity, split, stratum, timestamps,
calibration and history state, while omitting raw image paths and future ground
truth. Use the private data root configured by your local environment to attach
frames at run time.

## Install and verify

```bash
python -m pip install -e .
PYTHONPATH=src:. python -m pytest tests -q
sha256sum -c weights/SHA256SUMS.txt
```

The test suite is deterministic and covers calibration interfaces, temporal
geometry, flow reliability, trajectory decoding, split isolation and the
Step 1 continuous scorer.

## Run Step 1 measurement

On the evaluation server, first run the frozen image decoder on WAM-generated
future-image records with `configs/plane.json`, producing a decoder-score JSONL.
The public manifests alone cannot be scored because their future images and
future reference states remain private. Action/trajectory fields are read only
at the final comparison stage and must not be passed to the image-side decoder.

```bash
python scripts/audit_wam_level1_outputs.py --generated <wam_generated_records.jsonl> \
  --output <wam_output_audit.json>
python scripts/evaluate_continuous_decoder.py \
  --manifest <private_wam_generated_records.jsonl> \
  --config configs/plane.json \
  --output <decoder_scores.jsonl>
python scripts/evaluate_continuous_motion_alignment.py \
  --manifest <private_evaluation_manifest.jsonl> \
  --scores <decoder_scores.jsonl> \
  --reference-source action \
  --require-eight-frame-four-second \
  --output <out_dir>
```

The report contains path-normalised lateral/heading/curvature errors,
candidate-blind observability and coverage-risk. Speed is diagnostic only in
v1; it is not a formal Step 1 score. CCFC and FCS are scorecard cells, not
separate packages: they stay `ineligible` until a submission provides
same-history semantic pairs and independent rollouts.

## Submit a WAM

Authors submit JSONL against `datasets/benchmark_v1_public.jsonl`. The schema,
capability tiers and scorecard rules are in
`docs/WAM_SUBMISSION_V1_ZH.md`. Validate locally with:

```bash
python scripts/validate_wam_submission.py \
  --public datasets/benchmark_v1_public.jsonl \
  --submission <submission.jsonl> \
  --output <audit.json>
python scripts/score_iac_submission.py --frozen-pilots --output scorecard.json
```

The official v1 board (`datasets/scorecard_v1.json`) records WorldDrive,
DriveWAM, Epona and DriveVA as pilots or ineligible cells. It does not invent
CCFC numbers.

## Step 1 metric summary

The following is the **new frozen `benchmark_v1` experiment**, evaluated on
580 records (500 NAVSIM + 80 Waymo), with strict shape gating and shape
fallback disabled. The reference is logged future ego state, so these numbers
validate the image measurement layer; they are not WAM causal scores.

| Metric | Result | Interpretation |
|---|---:|---|
| Non-stop shape coverage | **440/468 = 94.0%** | At least one shape interval is observable on most moving samples |
| Stop recognition | **92/112 = 82.1%** | Stop is reported by the dedicated stop layer, not by a fake velocity estimate |
| Lateral-speed MAE / within tolerance | **0.095 m/s / 98.4%** | Reliable lateral-motion amplitude (`0.50 m/s` tolerance) |
| Yaw-rate MAE / within tolerance | **0.029 rad/s / 97.0%** | Reliable heading-change measurement (`0.15 rad/s` tolerance) |
| Curvature MAE / within tolerance | **0.022 1/m / 86.1%** | Usable curvature measurement (`0.06 1/m` tolerance), with a heavier tail |
| Turn-layer yaw increment | **Gate passed** (106/114 evaluable) | Future content improves over history, matched-shuffle and reversal controls |
| Turn-layer curvature increment | **Gate passed** (106/114 evaluable) | Curvature uses the correct future and its temporal order |
| Full-pool curvature increment | **Gate passed** | The curvature signal remains specific on the mixed benchmark pool |

The formal Step 1 comparison therefore uses lateral motion, yaw rate and
curvature. Absolute speed, acceleration and metric forward distance remain
diagnostic only. The complete definitions, tolerances and bootstrap-gate
wording are frozen in
[`docs/LEVEL1_MAIN_TABLE_BENCHMARK_V1_ZH.md`](docs/LEVEL1_MAIN_TABLE_BENCHMARK_V1_ZH.md).

### Reliability and long-tail diagnostics

The 78-sample development audit is retained only to expose failure modes and
coverage. It reports mean interval observability **77.2%**, fully observable
sample rate **61.5%**, and overall core-pass rate **82.1%**. Strong turns are
observable (interval coverage **100%**) but have only **40.0%** core-pass rate;
the limitation is accumulated lateral error, not invisibility. On the
scene-aware non-overlap 25-sample audit, core-pass rate is **88.0%** and
interval coverage **58.5%**, but braking has only one sample. These diagnostics
must not be merged into the frozen 580-record headline or used to claim
causality.

## Dataset preparation

Use `scripts/build_level1_benchmark_v1.py` to construct deterministic splits
from private NAVSIM/Waymo records. `scripts/prepare_waymo_level1_samples.py`
converts Waymo Perception v2 shards to the 4+8-frame interface, and
`scripts/build_public_benchmark_manifest.py` creates a leakage-safe public
manifest. See `docs/BENCHMARK_DATASET_V1_ZH.md` for the data protocol and
`docs/LEVEL1_MAIN_TABLE_BENCHMARK_V1_ZH.md` for the frozen main table.

## Data and license

This repository contains code, manifests and third-party optical-flow weights.
NAVSIM and Waymo data remain subject to their original access terms and are
not redistributed here. Check `weights/README.md` for checkpoint provenance and
upstream licenses.
