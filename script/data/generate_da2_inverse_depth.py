#!/usr/bin/env python3
"""Generate native Depth-Anything-V2 maps for one or more RGB sources.

The saved ``depth_npy`` is the raw DA-V2 output.  DA-V2 is a relative
near-high estimator; this script deliberately does not apply a reciprocal or
``1 - x`` conversion.  The visualization uses the same native orientation.

For each scene the output layout is:

    <output>/<scene_id>/rgb_gt/depth_npy/frame_00002.npy
    <output>/<scene_id>/rgb_gt/depth_vis/frame_00002_vis.png
    <output>/<scene_id>/rgb_rendered/depth_npy/frame_00002.npy
    <output>/<scene_id>/rgb_rendered/depth_vis/frame_00002_vis.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = "0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3"
DEFAULT_FRAME = "frame_00002.png"
DEFAULT_GT_ROOT = Path("/data2/caoxiaodata/data/scp/DL3DV-10K-Benchmark")
DEFAULT_RENDER_ROOT = Path(
    "/data2/caoxiaodata/data/GeoQuery_training_data/DIFIX_DL3DV_Benchmark"
)
DEFAULT_OUTPUT_ROOT = Path("/data2/caoxiaodata/data/DAV2_depth_inverse")
DEFAULT_DA_ROOT = REPO_ROOT / "code/third_party/Depth-Anything-V2"
DEFAULT_CHECKPOINT = Path("/data1/lilinwei/checkpoints/depth_anything_v2_vitl.pth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", default=DEFAULT_SCENE)
    parser.add_argument("--frame", default=DEFAULT_FRAME)
    parser.add_argument(
        "--all-scenes",
        action="store_true",
        help="Process the scene-ID intersection of the GT and rendered roots.",
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Process every input frame file independently in each selected scene.",
    )
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--render-root", type=Path, default=DEFAULT_RENDER_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--da-root", type=Path, default=DEFAULT_DA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the discovered tasks and paths without loading the model or writing files.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def load_model(da_root: Path, checkpoint: Path, device: torch.device):
    da_root = da_root.resolve()
    checkpoint = checkpoint.resolve()
    if not da_root.is_dir():
        raise FileNotFoundError(f"Depth-Anything-V2 root not found: {da_root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Depth-Anything-V2 checkpoint not found: {checkpoint}")
    sys.path.insert(0, str(da_root))
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
    )
    state = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


def native_vis(raw: np.ndarray) -> np.ndarray:
    """Return an RGB Turbo visualization without changing DA orientation."""
    valid = np.isfinite(raw) & (raw > 0)
    if int(valid.sum()) < 10:
        raise RuntimeError("DA output has fewer than 10 valid positive pixels")
    lo, hi = np.percentile(raw[valid], [1.0, 99.0]).astype(np.float32)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise RuntimeError(f"Invalid DA visualization range: lo={lo}, hi={hi}")
    norm = np.clip((raw - lo) / (hi - lo), 0.0, 1.0)
    norm[~valid] = 0.0
    try:
        from matplotlib import colormaps

        rgb = (colormaps["turbo"](norm)[..., :3] * 255.0 + 0.5).astype(np.uint8)
    except Exception:
        bgr = cv2.applyColorMap((norm * 255.0 + 0.5).astype(np.uint8), cv2.COLORMAP_TURBO)
        rgb = bgr[..., ::-1]
    return rgb


def _input_frames(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {
        path.name
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    }


def discover_tasks(args: argparse.Namespace) -> list[dict[str, Path | str]]:
    """Discover all files independently in the GT and rendered branches."""
    if args.all_scenes:
        gt_scenes = {
            path.name for path in args.gt_root.iterdir() if path.is_dir()
        }
        render_scenes = {
            path.name for path in args.render_root.iterdir() if path.is_dir()
        }
        scene_ids = sorted(gt_scenes & render_scenes)
        missing_gt = sorted(render_scenes - gt_scenes)
        missing_render = sorted(gt_scenes - render_scenes)
        print(
            f"[discover] gt_scenes={len(gt_scenes)} rendered_scenes={len(render_scenes)} "
            f"matched_scenes={len(scene_ids)} missing_gt={len(missing_gt)} "
            f"missing_rendered={len(missing_render)}",
            flush=True,
        )
    else:
        scene_ids = [args.scene_id]

    tasks: list[dict[str, Path | str]] = []
    missing_scene_dirs: list[str] = []
    gt_file_count = 0
    rendered_file_count = 0
    for scene_id in scene_ids:
        gt_dir = args.gt_root / scene_id / "gaussian_splat/images_4"
        render_dir = args.render_root / scene_id / "lr/24/train_24"
        if not gt_dir.is_dir() or not render_dir.is_dir():
            missing_scene_dirs.append(scene_id)
            continue

        if args.all_frames:
            gt_frame_names = sorted(_input_frames(gt_dir))
            render_frame_names = sorted(_input_frames(render_dir))
        else:
            gt_frame_names = [args.frame] if (gt_dir / args.frame).is_file() else []
            render_frame_names = [args.frame] if (render_dir / args.frame).is_file() else []

        gt_file_count += len(gt_frame_names)
        rendered_file_count += len(render_frame_names)
        output_scene = args.output_root / scene_id

        for frame_name in gt_frame_names:
            gt_source = gt_dir / frame_name
            output_scene = args.output_root / scene_id
            stem = Path(frame_name).stem
            tasks.append(
                {
                    "branch": "rgb_gt",
                    "scene_id": scene_id,
                    "frame": frame_name,
                    "source": gt_source,
                    "npy": output_scene / "rgb_gt/depth_npy" / f"{stem}.npy",
                    "vis": output_scene / "rgb_gt/depth_vis" / f"{stem}_vis.png",
                }
            )
        for frame_name in render_frame_names:
            render_source = render_dir / frame_name
            stem = Path(frame_name).stem
            tasks.append(
                {
                    "branch": "rgb_rendered",
                    "scene_id": scene_id,
                    "frame": frame_name,
                    "source": render_source,
                    "npy": output_scene / "rgb_rendered/depth_npy" / f"{stem}.npy",
                    "vis": output_scene / "rgb_rendered/depth_vis" / f"{stem}_vis.png",
                }
            )

    if missing_scene_dirs:
        print(
            f"[discover] skipped scenes with missing input subdirectories: "
            f"{len(missing_scene_dirs)}",
            flush=True,
        )
    if not tasks:
        raise RuntimeError("No RGB tasks were discovered")
    print(
        f"[discover] rgb_gt_files={gt_file_count} "
        f"rgb_rendered_files={rendered_file_count} tasks={len(tasks)} "
        f"(branches are processed independently; original names are preserved)",
        flush=True,
    )
    return tasks


def process_one(model, source: Path, npy_path: Path, vis_path: Path, input_size: int, overwrite: bool) -> bool:
    if not source.is_file():
        raise FileNotFoundError(f"Input RGB not found: {source}")
    if not overwrite and npy_path.is_file() and vis_path.is_file():
        print(f"[skip] {npy_path} and {vis_path} already exist")
        return False

    bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"OpenCV could not read RGB input: {source}")
    with torch.inference_mode():
        raw = model.infer_image(bgr, input_size=input_size).astype(np.float32)

    npy_path.parent.mkdir(parents=True, exist_ok=True)
    vis_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, raw)
    Image.fromarray(native_vis(raw), mode="RGB").save(vis_path)
    valid = np.isfinite(raw) & (raw > 0)
    print(
        f"[write] source={source} shape={raw.shape} "
        f"raw_min={float(raw[valid].min()):.6g} "
        f"raw_max={float(raw[valid].max()):.6g} "
        f"npy={npy_path} vis={vis_path}"
    )
    return True


def main() -> None:
    args = parse_args()
    tasks = discover_tasks(args)
    print(f"[config] scene_selection={'all' if args.all_scenes else args.scene_id}")
    print(f"[config] frame_selection={'all-input-files' if args.all_frames else args.frame}")
    print(f"[config] gt_root={args.gt_root.resolve()}")
    print(f"[config] rendered_root={args.render_root.resolve()}")
    print(f"[config] output_root={args.output_root.resolve()}")
    print("[config] output orientation=native_DA_near_high; reciprocal_conversion=disabled")

    if args.dry_run:
        for index, task in enumerate(tasks[:10], start=1):
            print(
                f"[dry-run {index}/{len(tasks)}] branch={task['branch']} "
                f"scene={task['scene_id']} frame={task['frame']} "
                f"source={task['source']} npy={task['npy']} vis={task['vis']}"
            )
        if len(tasks) > 10:
            print(f"[dry-run] ... {len(tasks) - 10} more tasks")
        return

    device = resolve_device(args.device)
    print(f"[config] device={device} encoder=vitl input_size={args.input_size}")
    print(f"[config] checkpoint={args.checkpoint.resolve()}")
    model = load_model(args.da_root, args.checkpoint, device)

    written = 0
    skipped = 0
    errors: list[str] = []
    for index, task in enumerate(tasks, start=1):
        print(
            f"[progress {index}/{len(tasks)}] branch={task['branch']} "
            f"scene={task['scene_id']} frame={task['frame']}",
            flush=True,
        )
        try:
            did_write = process_one(
                model,
                task["source"],
                task["npy"],
                task["vis"],
                args.input_size,
                args.overwrite,
            )
            if did_write:
                written += 1
            else:
                skipped += 1
        except Exception as exc:
            message = f"{task['branch']} {task['scene_id']} {task['frame']}: {exc}"
            errors.append(message)
            print(f"[error] {message}", flush=True)

    print(
        f"[summary] tasks={len(tasks)} written={written} skipped={skipped} "
        f"errors={len(errors)} output_root={args.output_root.resolve()}",
        flush=True,
    )
    if errors:
        raise RuntimeError(f"{len(errors)} tasks failed; see [error] lines above")


if __name__ == "__main__":
    main()
