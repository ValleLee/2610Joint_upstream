"""Compare saved 14000-step depth against DA(rendered RGB) on the same 100 frames.

Both predictions are assessed against DA(GT RGB), not physical depth. The
saved JointTeacher predictions were already least-squares aligned to this
teacher by the validation code. Apply the same alignment to DA(rendered RGB).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from build_figures import PAIR_FILE, ROOT, RUN, VIS_NAME


def resize(arr: np.ndarray, hw: tuple[int, int], mode: str = "bilinear") -> np.ndarray:
    tensor = torch.as_tensor(arr.astype(np.float32))[None, None]
    if mode == "bilinear":
        return F.interpolate(tensor, size=hw, mode=mode, align_corners=False)[0, 0].numpy()
    return F.interpolate(tensor, size=hw, mode=mode)[0, 0].numpy()


def scores(teacher_path: Path, da_path: Path, pred_path: Path) -> tuple[float, float, float, float, int]:
    pred = np.load(pred_path).astype(np.float32)
    hw = pred.shape
    teacher_raw = np.load(teacher_path).astype(np.float32)
    da_raw = np.load(da_path).astype(np.float32)
    teacher = resize(np.where(np.isfinite(teacher_raw), teacher_raw, 0), hw)
    da = resize(np.where(np.isfinite(da_raw), da_raw, 0), hw)
    teacher_mask = resize(np.isfinite(teacher_raw) & (teacher_raw > 0), hw, "nearest") > 0.5
    da_mask = resize(np.isfinite(da_raw) & (da_raw > 0), hw, "nearest") > 0.5
    valid = teacher_mask & da_mask & np.isfinite(teacher) & (teacher > 0)
    valid &= np.isfinite(da) & (da > 0) & np.isfinite(pred) & (pred > 0)
    if valid.sum() < 100:
        raise ValueError(f"Insufficient common pixels: {pred_path}")
    design = np.column_stack((da[valid], np.ones(valid.sum(), dtype=np.float32)))
    scale, shift = np.linalg.lstsq(design, teacher[valid], rcond=None)[0]
    da_aligned = np.maximum(da * scale + shift, 1e-6)
    truth = teacher[valid]
    da_error = da_aligned[valid] - truth
    pred_error = pred[valid] - truth
    spread = float(np.percentile(truth, 99) - np.percentile(truth, 1))
    if spread <= 0:
        raise ValueError(f"Zero teacher range: {teacher_path}")
    da_rmse = float(np.sqrt(np.mean(np.square(da_error))))
    pred_rmse = float(np.sqrt(np.mean(np.square(pred_error))))
    return da_rmse, pred_rmse, da_rmse / spread, pred_rmse / spread, int(valid.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN,
                        help="Original run directory containing visualization/iter_014000")
    parser.add_argument("--pair-file", type=Path, default=PAIR_FILE,
                        help="Five-column validation list with accessible data paths")
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows = [line.split() for line in args.pair_file.read_text().splitlines() if line.strip()]
    vis_dir = args.run_dir / "visualization/iter_014000" / VIS_NAME
    records = []
    for i, row in enumerate(rows):
        pred_path = vis_dir / f"{row[0].replace('/', '_')}.npy"
        if not pred_path.exists():
            continue
        da_rmse, pred_rmse, da_nrmse, pred_nrmse, count = scores(
            Path(row[2]), Path(row[3]), pred_path
        )
        records.append({
            "val_pair_line_1based": i + 1,
            "scene_id": row[0].split("/DIFIX_DL3DV_Benchmark/")[1].split("/")[0],
            "frame": Path(row[0]).stem,
            "valid_pixels": count,
            "da_rendered_rmse": da_rmse,
            "joint_teacher_rmse": pred_rmse,
            "da_rendered_nrmse": da_nrmse,
            "joint_teacher_nrmse": pred_nrmse,
            "joint_teacher_wins": int(pred_rmse < da_rmse),
        })
    if len(records) != 100:
        raise RuntimeError(f"Expected 100 saved depth predictions, found {len(records)}")
    out = ROOT / "evidence/da_comparison_100.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "sample_count": len(records),
        "joint_teacher_wins": sum(r["joint_teacher_wins"] for r in records),
        "da_rendered_wins": sum(1 - r["joint_teacher_wins"] for r in records),
        "mean_da_rendered_nrmse": float(np.mean([r["da_rendered_nrmse"] for r in records])),
        "mean_joint_teacher_nrmse": float(np.mean([r["joint_teacher_nrmse"] for r in records])),
        "median_da_rendered_nrmse": float(np.median([r["da_rendered_nrmse"] for r in records])),
        "median_joint_teacher_nrmse": float(np.median([r["joint_teacher_nrmse"] for r in records])),
        "protocol": "100 saved validation visualizations at iter 14000; DA(GT RGB) proxy; 432x768; common valid mask; least-squares scale/shift alignment of DA(rendered RGB); saved JointTeacher NPY already teacher-aligned; RMSE divided by teacher q99-q1 per image",
    }
    (ROOT / "evidence/da_comparison_100_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
