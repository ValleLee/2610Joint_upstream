"""Rebuild the analysis figures from the six bundled source samples.

The repository already contains the image and evaluation originals. Set
JOINT_RUN_DIR and JOINT_PAIR_FILE and pass --refresh-sources only when
intentionally updating them from an external copy of the original run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
RUN = Path(os.environ.get("JOINT_RUN_DIR", PROJECT / "output/20261002_164657_train_rgbdepth_teacher_perimg"))
PAIR_FILE = Path(os.environ.get("JOINT_PAIR_FILE", PROJECT / "data/pairs/k24_ratio100_val2_ignore/val_pairs.txt"))
SOURCE = ROOT / "figures/source"
EVAL_COPY = ROOT / "evidence/evaluation"
FIGURES = ROOT / "figures"
VIS_NAME = "RGB+DepthCond_PerImg_Vis"
SAMPLES = {
    "garden": [(1, "frame_00002"), (20, "frame_00021"), (40, "frame_00041")],
    "sculpture": [(665, "frame_00320"), (680, "frame_00335"), (700, "frame_00355")],
}
STEPS = (14000,)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_once(src: Path, dst: Path, refresh: bool) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not dst.exists():
        shutil.copy2(src, dst)


def collect_sources(refresh: bool) -> None:
    # Once copied, all plotting inputs live in docs. Regeneration of figures
    # should not depend on the training output or source dataset still existing.
    expected = [SOURCE / f"{scene}_{frame}" / name
                for scene, samples in SAMPLES.items()
                for _, frame in samples
                for name in ("rendered_rgb.png", "gt_rgb.png", "rendered_depth.npy",
                             "da_teacher.npy", "da_rendered.npy", "da_teacher_original_vis.png",
                             "da_rendered_original_vis.png", "pred_rgb_014000.png",
                             "pred_depth_014000.npy", "pred_depth_014000_original.png")]
    if (not refresh and (ROOT / "evidence/provenance.json").is_file()
            and all(p.is_file() for p in expected)
            and list(EVAL_COPY.glob("eval-*-iter*.txt"))
            and (ROOT / "evidence/run_config.yaml").is_file()
            and (ROOT / "evidence/scheduler_config_iter014000.json").is_file()):
        return
    rows = [line.split() for line in PAIR_FILE.read_text().splitlines() if line.strip()]
    provenance = {
        "run": str(RUN),
        "pair_file": str(PAIR_FILE),
        "samples": {},
        "note": "Source files are byte-for-byte copies; colored depth and mosaics are derivatives.",
    }
    for scene, samples in SAMPLES.items():
        for index, frame in samples:
            row = rows[index]
            assert len(row) == 5 and Path(row[0]).stem == frame
            tag = f"{scene}_{frame}"
            out = SOURCE / tag
            paths = {
                "rendered_rgb.png": Path(row[0]),
                "rendered_depth.npy": Path(row[1]),
                "da_teacher.npy": Path(row[2]),
                "da_rendered.npy": Path(row[3]),
                "da_teacher_original_vis.png": Path(row[2]).parent.parent / "depth_vis" / f"{frame}_vis.png",
                "da_rendered_original_vis.png": Path(row[3]).parent.parent / "depth_vis" / f"{frame}_vis.png",
                "gt_rgb.png": Path(row[4]),
            }
            key = row[0].replace("/", "_")
            for step in STEPS:
                vis = RUN / "visualization" / f"iter_{step:06d}" / VIS_NAME
                paths[f"pred_rgb_{step:06d}.png"] = vis / f"{key}_rgb.png"
                paths[f"pred_depth_{step:06d}.npy"] = vis / f"{key}.npy"
                paths[f"pred_depth_{step:06d}_original.png"] = vis / f"{key}.png"
            for name, src in paths.items():
                copy_once(src, out / name, refresh)
            provenance["samples"][tag] = {
                "val_pair_line_1based": index + 1,
                "scene_id": row[0].split("/DIFIX_DL3DV_Benchmark/")[1].split("/")[0],
                "originals": {
                    name: {"path": str(src), "sha256": sha256(out / name)}
                    for name, src in paths.items()
                },
            }
    for src in sorted((RUN / "evaluation").glob("eval-*-iter*.txt")):
        copy_once(src, EVAL_COPY / src.name, refresh)
    copy_once(RUN / "config.yaml", ROOT / "evidence/run_config.yaml", refresh)
    copy_once(RUN / "checkpoint/iter_014000/scheduler/scheduler_config.json",
              ROOT / "evidence/scheduler_config_iter014000.json", refresh)
    (ROOT / "evidence").mkdir(parents=True, exist_ok=True)
    (ROOT / "evidence/provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
    )


def rgb_figure(scene: str) -> None:
    fig, axes = plt.subplots(len(SAMPLES[scene]), 3,
                             figsize=(12, 2.7 * len(SAMPLES[scene])), dpi=160)
    titles = ["GT RGB (target)", "Original rendered RGB", "Processed RGB @14000"]
    names = ["gt_rgb.png", "rendered_rgb.png", "pred_rgb_014000.png"]
    for row, (_, frame) in enumerate(SAMPLES[scene]):
        tag = f"{scene}_{frame}"
        for col, name in enumerate(names):
            ax = axes[row, col]
            ax.imshow(Image.open(SOURCE / tag / name).convert("RGB"))
            ax.set_axis_off()
            if row == 0:
                ax.set_title(titles[col], fontsize=11)
            if col == 0:
                ax.text(0.01, 0.03, frame, transform=ax.transAxes, color="white", fontsize=10,
                        bbox={"facecolor": "black", "alpha": 0.6, "pad": 2})
    fig.suptitle(f"JointTeacher RGB | {scene}", fontsize=14)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.02, wspace=0.015, hspace=0.06)
    fig.savefig(FIGURES / f"{scene}_rgb.png", dpi=160)
    plt.close(fig)


def normalized_depth(data: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Map each map's valid 1st/99th percentiles to a shared display range [0, 1]."""
    valid = np.isfinite(data) & (data > 0)
    if not valid.any():
        raise ValueError("Depth map has no positive finite values")
    lo, hi = (float(v) for v in np.percentile(data[valid], [1, 99]))
    display = np.full(data.shape, np.nan, dtype=np.float32)
    display[valid] = np.clip((data[valid] - lo) / max(hi - lo, 1e-6), 0, 1)
    return display, lo, hi


def inverse_for_display(depth: np.ndarray) -> np.ndarray:
    """Match DA's near-high direction for figures; training inputs remain raw."""
    valid = np.isfinite(depth) & (depth > 0)
    inverse = np.full(depth.shape, np.nan, dtype=np.float32)
    inverse[valid] = 1.0 / depth[valid]
    return inverse


def resize_bilinear(data: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    if data.shape == hw:
        return data.astype(np.float32, copy=False)
    tensor = torch.as_tensor(data.astype(np.float32))[None, None]
    return F.interpolate(tensor, size=hw, mode="bilinear", align_corners=False)[0, 0].numpy()


def affine_to_teacher(data: np.ndarray, teacher: np.ndarray) -> np.ndarray:
    valid = np.isfinite(data) & (data > 0) & np.isfinite(teacher) & (teacher > 0)
    design = np.column_stack((data[valid], np.ones(valid.sum(), dtype=np.float32)))
    scale, shift = np.linalg.lstsq(design, teacher[valid], rcond=None)[0]
    return data * scale + shift


def display_with_range(data: np.ndarray, lo: float, hi: float) -> np.ndarray:
    valid = np.isfinite(data) & (data > 0)
    display = np.full(data.shape, np.nan, dtype=np.float32)
    display[valid] = np.clip((data[valid] - lo) / max(hi - lo, 1e-6), 0, 1)
    return display


def depth_figure(scene: str) -> None:
    fig, axes = plt.subplots(len(SAMPLES[scene]), 3,
                             figsize=(12, 2.85 * len(SAMPLES[scene])), dpi=160)
    titles = ["DA teacher (GT proxy)", "Rendered depth: 1/d for display", "Processed depth @14000"]
    names = ["da_teacher.npy", "rendered_depth.npy", "pred_depth_014000.npy"]
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad((0.19, 0.19, 0.19))
    for row, (_, frame) in enumerate(SAMPLES[scene]):
        tag = f"{scene}_{frame}"
        arrays = [np.load(SOURCE / tag / name) for name in names]
        arrays[1] = inverse_for_display(arrays[1])
        for col, data in enumerate(arrays):
            ax = axes[row, col]
            display, lo, hi = normalized_depth(data)
            ax.imshow(display, cmap=cmap, vmin=0, vmax=1)
            ax.set_axis_off()
            if row == 0:
                ax.set_title(titles[col], fontsize=10.5)
            unit = "display 1/d" if col == 1 else "raw"
            ax.text(0.01, 0.03, f"{frame} | {unit} p1={lo:.2g}, p99={hi:.2g}", transform=ax.transAxes,
                    color="white", fontsize=9, bbox={"facecolor": "black", "alpha": 0.65, "pad": 2})
    fig.suptitle(f"JointTeacher depth | {scene} | same display scale, per-image percentile normalization", fontsize=13)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.12, wspace=0.015, hspace=0.06)
    cax = fig.add_axes((0.29, 0.052, 0.42, 0.017))
    colorbar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=cax, orientation="horizontal")
    colorbar.set_label("Within-image normalized value (0–1); gray = invalid", fontsize=9)
    fig.savefig(FIGURES / f"{scene}_depth.png", dpi=160)
    plt.close(fig)


def depth_vs_da_figure(scene: str) -> None:
    """Compare DA and model on a teacher scale; show render on its own scale."""
    fig, axes = plt.subplots(len(SAMPLES[scene]), 4,
                             figsize=(16, 2.85 * len(SAMPLES[scene])), dpi=160)
    titles = ["DA (GT RGB): teacher", "Render 1/depth (own scale)",
              "DA (rendered RGB)", "JointTeacher @14000"]
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad((0.19, 0.19, 0.19))
    panel_names = ("da_teacher_shared.png", "rendered_inverse_shared.png",
                   "da_rendered_shared.png", "processed_shared.png")
    for row, (_, frame) in enumerate(SAMPLES[scene]):
        tag = f"{scene}_{frame}"
        src = SOURCE / tag
        pred = np.load(src / "pred_depth_014000.npy")
        hw = pred.shape
        teacher = resize_bilinear(np.load(src / "da_teacher.npy"), hw)
        da_rendered = resize_bilinear(np.load(src / "da_rendered.npy"), hw)
        rendered = resize_bilinear(np.load(src / "rendered_depth.npy"), hw)
        rendered_inverse = inverse_for_display(rendered)
        aligned_da_rendered = affine_to_teacher(da_rendered, teacher)
        valid_teacher = np.isfinite(teacher) & (teacher > 0)
        lo, hi = (float(v) for v in np.percentile(teacher[valid_teacher], [1, 99]))
        arrays = [teacher, rendered_inverse, aligned_da_rendered, pred]
        panel_dir = FIGURES / "panels" / tag
        panel_dir.mkdir(parents=True, exist_ok=True)
        for col, (data, panel_name) in enumerate(zip(arrays, panel_names)):
            display = normalized_depth(data)[0] if col == 1 else display_with_range(data, lo, hi)
            plt.imsave(panel_dir / panel_name, display, cmap=cmap, vmin=0, vmax=1)
            ax = axes[row, col]
            ax.imshow(display, cmap=cmap, vmin=0, vmax=1)
            ax.set_axis_off()
            if row == 0:
                ax.set_title(titles[col], fontsize=10)
            if col == 0:
                ax.text(0.01, 0.03, f"{frame} | teacher q1={lo:.1f}, q99={hi:.1f}",
                        transform=ax.transAxes, color="white", fontsize=8.5,
                        bbox={"facecolor": "black", "alpha": 0.65, "pad": 2})
            if col == 1:
                ax.text(0.01, 0.03, "1/d, own q1–q99", transform=ax.transAxes,
                        color="white", fontsize=8.5,
                        bbox={"facecolor": "black", "alpha": 0.65, "pad": 2})
    fig.suptitle(f"JointTeacher vs DA | {scene} | shared teacher scale for columns 1, 3, 4", fontsize=13)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.12, wspace=0.015, hspace=0.06)
    cax = fig.add_axes((0.29, 0.052, 0.42, 0.017))
    colorbar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=cax, orientation="horizontal")
    colorbar.set_label("DA(GT RGB) q1–q99 for columns 1, 3, 4; render uses own q1–q99", fontsize=9)
    fig.savefig(FIGURES / f"{scene}_depth_vs_da.png", dpi=160)
    plt.close(fig)


def metrics_figure() -> None:
    rows = []
    for path in sorted(EVAL_COPY.glob("eval-*-iter*.txt")):
        step = int(re.search(r"iter(\d+)", path.name).group(1))
        body = path.read_text()
        values = re.search(r"\n(0\.\d+)\s+(0\.\d+)\s+(0\.\d+)", body)
        if values is None:
            raise ValueError(f"Cannot parse metrics from {path}")
        rows.append((step, *(float(v) for v in values.groups())))
    rows.sort()
    with (ROOT / "evidence/validation_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "joint_latent_mse", "rgb_latent_mse", "depth_latent_mse"])
        writer.writerows(rows)
    x = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=170)
    for col, name, color in [(1, "Joint", "#1d4ed8"), (2, "RGB", "#e45b2b"), (3, "Depth", "#16806b")]:
        ax.plot(x, [r[col] for r in rows], marker="o", markersize=3, linewidth=1.8,
                label=f"{name} latent MSE", color=color)
    best = min(rows, key=lambda r: r[1])
    ax.scatter([best[0]], [best[1]], s=85, facecolors="none", edgecolors="#1d4ed8", linewidth=1.8)
    ax.annotate(f"Lowest recorded joint: {best[1]:.6f} @ {best[0]}",
                (best[0], best[1]), xytext=(-180, 20), textcoords="offset points",
                arrowprops={"arrowstyle": "->"}, fontsize=9)
    ax.set_xlabel("Optimizer iteration")
    ax.set_ylabel("Fixed-timestep latent MSE (lower is better)")
    ax.set_title("JointTeacher validation on 715 pairs, one run")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "validation_mse.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-sources", action="store_true")
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    collect_sources(args.refresh_sources)
    for scene in SAMPLES:
        rgb_figure(scene)
        depth_figure(scene)
        depth_vs_da_figure(scene)
    metrics_figure()
    print(f"Wrote 7 figures and copied source artifacts under {ROOT}")


if __name__ == "__main__":
    main()
