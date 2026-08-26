"""Model capability declarations for the WAM causal evaluation protocol.

This module does not hide model-specific inference code.  It records the
capabilities that determine whether a repository can supply the required
action -> future-image intervention.  Keeping this explicit prevents an
action-predicting model from being accidentally treated as an
action-conditioned image generator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WAMCapability:
    model_id: str
    repository: str
    repository_path: str
    future_image_inference: bool
    external_trajectory_control: bool
    action_prediction: bool
    native_dataset: str
    native_history: str
    native_future: str
    checkpoint_source: str
    status: str
    reason: str
    control_path: str = "unknown"

    @property
    def suitable_for_counterfactual_image_cc(self) -> bool:
        return self.future_image_inference and self.external_trajectory_control

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["suitable_for_counterfactual_image_cc"] = self.suitable_for_counterfactual_image_cc
        return result


def inspect_known_wams(home: Path) -> list[WAMCapability]:
    """Return a conservative capability table for the candidate repositories."""

    entries = [
        WAMCapability(
            model_id="drivingworld",
            repository="YvanYin/DrivingWorld",
            repository_path=str(home / "wam_repro" / "DrivingWorld"),
            future_image_inference=True,
            external_trajectory_control=False,
            action_prediction=True,
            native_dataset="NuPlan/NavSim derived",
            native_history="repository-specific",
            native_future="repository-specific",
            checkpoint_source="local checkpoint",
            status="baseline",
            reason="Existing server baseline; retain for regression, but its current branch protocol is not a clean external trajectory-control interface.",
        ),
        WAMCapability(
            model_id="drivewam_navsim",
            repository="chenshi3/DriveWAM",
            repository_path=str(home / "DriveWAM"),
            future_image_inference=True,
            external_trajectory_control=True,
            action_prediction=True,
            native_dataset="NavSim / PhysicalAI-AV",
            native_history="4 frames in NavSim config",
            native_future="4 future image indices in NavSim config",
            checkpoint_source="Hugging Face chenchenshi/DriveWAM",
            status="checkpoint_available_base_pending",
            reason="The published NavSim checkpoint and the project adapter expose external trajectory injection. Runtime still requires the official LingBot-VA Base components and a loader smoke test.",
            control_path="inject supplied normalized trajectory into the clean action-conditioning chunk before video rollout",
        ),
        WAMCapability(
            model_id="epona_nuplan",
            repository="Kevin-thu/Epona",
            repository_path=str(home / "Epona"),
            future_image_inference=True,
            external_trajectory_control=True,
            action_prediction=True,
            native_dataset="NuPlan / NuScenes",
            native_history="configurable condition frames",
            native_future="variable autoregressive rollout",
            checkpoint_source="Hugging Face Kevin-thu/Epona",
            status="independent_candidate_vae_pending",
            reason="The official test_ctrl.py accepts user-provided poses and yaws and generates controlled future frames. Runtime also requires the published DCAE VAE checkpoint.",
        ),
        WAMCapability(
            model_id="worlddrive_tadwm",
            repository="TabGuigui/WorldDrive",
            repository_path=str(home / "WorldDrive"),
            future_image_inference=True,
            external_trajectory_control=True,
            action_prediction=True,
            native_dataset="NavSim / NuPlan / nuScenes",
            native_history="repository-specific",
            native_future="trajectory-aware video rollout",
            checkpoint_source="Hugging Face tabguigui/WorldDrive",
            status="comparison_candidate",
            reason="The official description exposes trajectory-conditioned video generation; retain it as a cross-implementation comparison after runtime validation.",
        ),
        WAMCapability(
            model_id="simwam",
            repository="H-EmbodVis/SimWAM",
            repository_path=str(home / "SimWAM"),
            future_image_inference=False,
            external_trajectory_control=False,
            action_prediction=True,
            native_dataset="NavSim / PhysicalAI-AV",
            native_history="repository-specific",
            native_future="future video used as training supervision",
            checkpoint_source="official repository",
            status="action_control",
            reason="Inference predicts actions directly and does not emit future frames, so it cannot be the main image-side counterfactual evaluator.",
        ),
    ]
    return [entry for entry in entries if Path(entry.repository_path).exists() or entry.model_id in {"drivingworld", "simwam"}]
