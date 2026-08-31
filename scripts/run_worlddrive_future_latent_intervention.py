#!/usr/bin/env python3
"""Intervene on WorldDrive's candidate-conditioned imagined futures.

The intervention holds history, ego status, top-k trajectories, trajectory
embeddings, and all model weights fixed.  It reverses only the association
between the five candidate trajectories and the five future-scene latents
inside the published stage-2 refiner, then records the resulting native action
head output.  This tests internal future-to-action mediation; it is not a
semantic hazard counterfactual.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from run_worlddrive_native_planner import (
    _history_digest,
    _history_tensor,
    _load_modules,
    _read_sample,
    _sha256,
    _status_tensor,
)


def _topk_state(
    *, visual: torch.Tensor, status: torch.Tensor, anchors: np.ndarray,
    modules: dict, device: torch.device, dtype: torch.dtype,
) -> dict:
    trajencoder = modules["trajencoder"]
    trajencoder_wm = modules["trajencoder_wm"]
    visual_latents = modules["adapters"](visual)
    batch_size = visual_latents.shape[0]
    trajectory_embeddings = trajencoder(traj_train=True)[None].repeat(batch_size, 1, 1)
    anchor_tensor = torch.from_numpy(anchors).to(device=device, dtype=dtype)
    stage1, final_rewards, offsets, (visual_tokens, _) = modules["trajplanner"](
        visual_latents, status, trajectory_embeddings, anchor_tensor,
        targets=None, eval_mode=True,
    )

    topk = 5
    topk_rewards, topk_indices = final_rewards.topk(k=topk, dim=1)
    selected = trajencoder_wm.traj_vocab[topk_indices]
    normalized = trajencoder_wm.norm_odo(selected)
    vocab_embeddings = trajencoder_wm.traj_vocab_encoder(
        normalized.reshape(batch_size, topk, -1)
    )
    batch_indices = torch.arange(batch_size, device=device)[:, None]
    selected_offsets = offsets[batch_indices, topk_indices]
    offset_embeddings = trajencoder_wm.traj_offset_encoder(
        selected_offsets.reshape(batch_size, topk, -1)
    )
    topk_trajectories = selected + selected_offsets.reshape(batch_size, topk, 8, 3)
    return {
        "stage1": stage1,
        "topk_rewards": topk_rewards,
        "topk_indices": topk_indices,
        "topk_trajectories": topk_trajectories,
        "topk_embeddings": vocab_embeddings + offset_embeddings,
        "visual_tokens": visual_tokens,
    }


def _score_with_future_assignment(
    *, topk_trajectories: torch.Tensor, topk_embeddings: torch.Tensor,
    visual_tokens: torch.Tensor, refiner, permutation: list[int],
) -> dict:
    batch_size, trajectory_count = topk_trajectories.shape[:2]
    if batch_size != 1 or trajectory_count != 5:
        raise ValueError(f"expected [1,5,...] candidates, got {topk_trajectories.shape}")

    visual_count = visual_tokens.shape[1]
    repeated_visual = visual_tokens[:, None].repeat(1, trajectory_count, 1, 1)
    repeated_visual = repeated_visual.reshape(batch_size * trajectory_count, visual_count, -1)
    trajectory_tokens = topk_embeddings[:, :, None].reshape(
        batch_size * trajectory_count, 1, -1
    )
    future_queries = refiner.future_scene_query.weight[None, None].repeat(
        batch_size, trajectory_count, 1, 1
    )
    future_queries = future_queries.reshape(
        batch_size * trajectory_count, refiner.h * refiner.w, refiner.visual_dim
    )
    position = refiner._get_positional_embeddings(
        refiner.h, refiner.w, device=repeated_visual.device
    )
    future_queries = future_queries + position[None].to(repeated_visual.dtype)
    native_futures = refiner.refine_decoder(
        future_queries, torch.cat([repeated_visual, trajectory_tokens], dim=1)
    )

    future_by_candidate = native_futures.reshape(
        batch_size, trajectory_count, refiner.h * refiner.w, refiner.visual_dim
    )
    permuted_futures = future_by_candidate[:, permutation].reshape(
        batch_size * trajectory_count, refiner.h * refiner.w, refiner.visual_dim
    )
    refined_tokens = refiner.traj_refine_decoder(trajectory_tokens, permuted_futures)
    refined_tokens = refined_tokens.reshape(batch_size, trajectory_count, -1)
    refined_tokens = refined_tokens + permuted_futures.mean(dim=1).reshape(
        batch_size, trajectory_count, -1
    )
    weights = refiner.refine_reward_heads[0](refined_tokens)
    selected_rank = int(weights[0, :, 0].argmax().item())
    selected = topk_trajectories[0, selected_rank]
    return {
        "trajectory": selected,
        "weights": weights,
        "selected_rank": selected_rank,
        "native_future_latents": native_futures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlddrive-root", type=Path, required=True)
    parser.add_argument("--planner-checkpoint", type=Path, required=True)
    parser.add_argument("--cogvideox-root", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    args = parser.parse_args()

    for path in (args.worlddrive_root, args.planner_checkpoint, args.cogvideox_root,
                 args.anchors, args.sample_root):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.samples < 1:
        raise ValueError("samples must be positive")
    sys.path.insert(0, str(args.worlddrive_root.resolve()))

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    anchors, modules, vae = _load_modules(
        checkpoint=args.planner_checkpoint,
        anchors_path=args.anchors,
        cogvideox_root=args.cogvideox_root,
        device=device,
        dtype=dtype,
    )
    checkpoint_sha256 = _sha256(args.planner_checkpoint)
    anchors_sha256 = _sha256(args.anchors)
    identity = list(range(5))
    reverse = list(reversed(identity))
    rows = []
    native_checks = []

    with torch.inference_mode():
        for sample_index in range(args.samples):
            sample_path = (
                args.sample_root / "drivewam_samples_logged" / f"sample_{sample_index:06d}.pkl"
            )
            sample = _read_sample(sample_path)
            item_seed = args.seed + sample_index
            torch.manual_seed(item_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(item_seed)
            history = _history_tensor(sample["images"], device, dtype)
            visual = vae.encode(history).latent_dist.sample() * vae.config.scaling_factor
            visual = visual.permute(0, 2, 1, 3, 4).contiguous()
            status = _status_tensor(sample, device, dtype)
            state = _topk_state(
                visual=visual, status=status, anchors=anchors, modules=modules,
                device=device, dtype=dtype,
            )
            native = _score_with_future_assignment(
                topk_trajectories=state["topk_trajectories"],
                topk_embeddings=state["topk_embeddings"],
                visual_tokens=state["visual_tokens"],
                refiner=modules["trajrefiner"], permutation=identity,
            )
            intervened = _score_with_future_assignment(
                topk_trajectories=state["topk_trajectories"],
                topk_embeddings=state["topk_embeddings"],
                visual_tokens=state["visual_tokens"],
                refiner=modules["trajrefiner"], permutation=reverse,
            )
            official_trajectory, _, _, official_weights = modules["trajrefiner"](
                state["topk_trajectories"], state["topk_embeddings"],
                state["visual_tokens"], None,
            )
            trajectory_error = float(
                (official_trajectory[0] - native["trajectory"]).abs().max().item()
            )
            weight_error = float(
                (official_weights - native["weights"]).abs().max().item()
            )
            if trajectory_error > 1e-5 or weight_error > 1e-5:
                raise RuntimeError(
                    f"manual native path mismatch at sample {sample_index}: "
                    f"trajectory={trajectory_error}, weights={weight_error}"
                )
            native_checks.append({"trajectory_max_abs": trajectory_error, "weight_max_abs": weight_error})

            logged = np.asarray(
                [item["pose"] for item in sample["future_trajectory"]], dtype=np.float32
            )
            common = {
                "sample_index": sample_index,
                "source_key": sample["metadata"].get("source_key", f"sample-{sample_index:06d}"),
                "history_sha256": _history_digest(sample["images"]),
                "ego_status": status[0].float().cpu().tolist(),
                "native_action_head_recorded": True,
                "planner_checkpoint": str(args.planner_checkpoint.resolve()),
                "planner_checkpoint_sha256": checkpoint_sha256,
                "anchors_sha256": anchors_sha256,
                "vae_seed": item_seed,
                "logged_reference_trajectory": logged.tolist(),
                "stage1_trajectory": state["stage1"][0].float().cpu().tolist(),
                "topk_trajectories": state["topk_trajectories"][0].float().cpu().tolist(),
                "topk_anchor_indices": state["topk_indices"][0].cpu().tolist(),
                "topk_stage1_rewards": state["topk_rewards"][0].float().cpu().tolist(),
                "counterfactual_group_id": f"worlddrive-future-latent-{sample_index:06d}",
                "intervention_type": "internal_future_latent_permutation",
                "intervention_target": "stage2_candidate_conditioned_future_scene_embed",
                "held_fixed": [
                    "history", "ego_status", "topk_trajectories", "topk_embeddings",
                    "visual_tokens", "model_weights",
                ],
                "semantic_hazard_intervention": False,
                "formal_foresight_mediation_input": True,
            }
            for branch_id, result, permutation, action_origin, unintervened in (
                ("future_native", native, identity,
                 "worlddrive_stage2_refined_planner", True),
                ("future_reverse", intervened, reverse,
                 "worlddrive_stage2_internal_future_latent_permutation", False),
            ):
                rows.append({
                    **common,
                    "sample_id": f"worlddrive-future-latent-{sample_index:06d}-{branch_id}",
                    "branch_id": branch_id,
                    "branch_role": branch_id,
                    "action_origin": action_origin,
                    "stage2_trajectory": result["trajectory"].float().cpu().tolist(),
                    "topk_refine_weights": result["weights"][0, :, 0].float().cpu().tolist(),
                    "stage2_selected_topk_rank": result["selected_rank"],
                    "future_latent_candidate_permutation": permutation,
                    "native_unintervened": unintervened,
                    "interventional_action_head_output": not unintervened,
                })
            del history, visual, status, state, native, intervened
            if device.type == "cuda":
                torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "groups": args.samples,
        "records": len(rows),
        "intervention": "internal_future_latent_permutation",
        "permutation": reverse,
        "manual_native_max_trajectory_error": max(x["trajectory_max_abs"] for x in native_checks),
        "manual_native_max_weight_error": max(x["weight_max_abs"] for x in native_checks),
        "semantic_hazard_intervention": False,
    }, indent=2))


if __name__ == "__main__":
    main()
