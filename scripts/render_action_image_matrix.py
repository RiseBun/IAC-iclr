#!/usr/bin/env python3
"""Render an audit sheet of action-image cross-score matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        ["C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        if bold else
        ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _heat(probability: float) -> tuple[int, int, int]:
    value = max(0.0, min(1.0, probability))
    low = (239, 242, 245)
    high = (43, 145, 105)
    return tuple(round(a + value * (b - a)) for a, b in zip(low, high))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.analysis.read_text(encoding="utf-8"))
    groups = [
        (str(row.get("twin_id") or row.get("group_id")), str(row.get("condition_family", "unknown")), row["action_image_matrix"])
        for row in report.get("results", [])
        if row.get("action_image_matrix")
    ]
    summary = report.get("action_image_matrix_summary", {})
    if not summary:
        matrices = [matrix for _, _, matrix in groups]
        summary = {
            "num_groups": len(groups),
            "num_branches": sum(int(matrix.get("num_branches", 0)) for matrix in matrices),
            "diagonal_top1_accuracy": float(sum(float(matrix.get("diagonal_top1_accuracy", 0.0)) for matrix in matrices) / len(matrices)) if matrices else 0.0,
            "mean_cc_margin": float(sum(float(matrix.get("mean_cc_margin", 0.0)) for matrix in matrices) / len(matrices)) if matrices else 0.0,
            "coverage": float(sum(float(matrix.get("coverage", 0.0)) for matrix in matrices) / len(matrices)) if matrices else 0.0,
        }
    columns = 2
    panel_width, panel_height = 760, 280
    header_height = 145
    rows = max(1, (len(groups) + columns - 1) // columns)
    canvas = Image.new("RGB", (columns * panel_width + 40, header_height + rows * panel_height + 25), "white")
    draw = ImageDraw.Draw(canvas)
    title = _font(30, bold=True)
    heading = _font(19, bold=True)
    body = _font(16)
    small = _font(13)
    draw.text((28, 20), "Counterfactual Action-Image Consistency", font=title, fill=(25, 31, 38))
    draw.text(
        (28, 67),
        f"groups={summary.get('num_groups')}  branches={summary.get('num_branches')}  "
        f"top-1={summary.get('diagonal_top1_accuracy', 0):.3f}  "
        f"CC margin={summary.get('mean_cc_margin', 0):.3f}  coverage={summary.get('coverage', 0):.3f}",
        font=heading,
        fill=(41, 55, 70),
    )
    draw.text(
        (28, 105),
        "Rows: generated futures. Columns: action hypotheses. Green diagonal border=matched; red=swap preferred.",
        font=body,
        fill=(83, 91, 101),
    )

    for index, (group_id, family, matrix) in enumerate(groups):
        column = index % columns
        row = index // columns
        left = 20 + column * panel_width
        top = header_height + row * panel_height
        right = left + panel_width - 18
        bottom = top + panel_height - 18
        draw.rectangle((left, top, right, bottom), fill=(250, 251, 252), outline=(205, 211, 218), width=1)
        draw.text((left + 18, top + 14), family, font=heading, fill=(25, 31, 38))
        draw.text((left + 18, top + 43), group_id[:16], font=small, fill=(102, 110, 120))
        draw.text(
            (left + 18, top + 68),
            f"top-1={matrix['diagonal_top1_accuracy']:.2f}   CC={matrix['mean_cc_margin']:+.3f}   "
            f"response TV={matrix['mean_pairwise_response_tv']:.3f}",
            font=body,
            fill=(45, 55, 65),
        )
        probabilities = matrix["probability_matrix"]
        energies = matrix["energy_matrix"]
        count = len(probabilities)
        cell = min(74, max(42, 190 // max(count, 1)))
        grid_left = left + 185
        grid_top = top + 108
        for action_index in range(count):
            label = chr(ord("A") + action_index) if action_index < 26 else str(action_index + 1)
            draw.text((grid_left + action_index * cell + 22, grid_top - 25), f"A{label}", font=small, fill=(55, 65, 75))
            draw.text((grid_left - 73, grid_top + action_index * cell + 20), f"F{label}", font=small, fill=(55, 65, 75))
        for future_index, row_values in enumerate(probabilities):
            valid_values = [value for value in row_values if value is not None]
            predicted = max(range(count), key=lambda item: row_values[item]) if valid_values else None
            for action_index, probability in enumerate(row_values):
                x0 = grid_left + action_index * cell
                y0 = grid_top + future_index * cell
                x1, y1 = x0 + cell - 4, y0 + cell - 4
                fill = (226, 229, 233) if probability is None else _heat(float(probability))
                draw.rectangle((x0, y0, x1, y1), fill=fill, outline=(190, 197, 205), width=1)
                if probability is not None:
                    draw.text((x0 + 8, y0 + 10), f"p {probability:.2f}", font=small, fill=(20, 31, 38))
                    draw.text((x0 + 8, y0 + 34), f"e {energies[future_index][action_index]:.2f}", font=small, fill=(55, 65, 75))
                if future_index == action_index:
                    color = (37, 140, 91) if predicted == action_index else (204, 67, 67)
                    draw.rectangle((x0 + 1, y0 + 1, x1 - 1, y1 - 1), outline=color, width=4)
        note_x = grid_left + count * cell + 22
        for branch_index, branch in enumerate(matrix["branches"]):
            label = chr(ord("A") + branch_index) if branch_index < 26 else str(branch_index + 1)
            rank = branch.get("matched_rank")
            margin = branch.get("cc_margin")
            text = f"F{label}: abstain" if rank is None else f"F{label}: rank {rank:g}, margin {margin:+.3f}"
            color = (95, 102, 110) if rank is None else ((37, 140, 91) if branch.get("matched_unique_top1") else (204, 67, 67))
            draw.text((note_x, grid_top + 8 + branch_index * 31), text, font=small, fill=color)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(json.dumps({"groups": len(groups), "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
