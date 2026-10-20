# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import logging
from glob import glob
import re

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm
import imageio.v2 as iio
from torchvision.transforms.functional import pil_to_tensor

try:
    import matplotlib

    _CMAP_TURBO = matplotlib.colormaps.get_cmap("turbo")
    _USE_TURBO = True
except Exception:
    _CMAP_TURBO = None
    _USE_TURBO = False

from marigold import MarigoldDepthPipeline, MarigoldDepthOutput
from src.util.depth_transform import get_depth_normalizer

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]


_SCENE_ID_RE = re.compile(r"[0-9a-f]{64}")


def _scene_id_from_path(path: str) -> str:
    m = _SCENE_ID_RE.search(path)
    if m is None:
        return "unknown_scene"
    return m.group(0)


def _read_png_u16_01(path: str) -> np.ndarray:
    arr = iio.imread(path)
    if arr.ndim != 2:
        raise RuntimeError(
            f"Expected 1-channel invdepth png, got shape={arr.shape} for {path}"
        )
    if arr.dtype != np.uint16:
        arr = arr.astype(np.uint16)
    return arr.astype(np.float32) / 65535.0


def _load_npy_f32(path: str) -> np.ndarray:
    x = np.load(path)
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    x = x.astype(np.float32)
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    return x


def _read_depth_any(path: str, *, invert: bool = False) -> tuple[np.ndarray, np.ndarray]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        x = _load_npy_f32(path)
    else:
        x = _read_png_u16_01(path).astype(np.float32)
    if invert:
        m0 = np.isfinite(x) & (x > 0)
        x = np.where(m0, 1.0 / (x + 1e-6), 0.0).astype(np.float32)
    m = np.isfinite(x) & (x > 0)
    x = np.where(np.isfinite(x), x, 0.0).astype(np.float32)
    return x, m


def _get_invalid_fill_value(depth_transform, *, is_inverse: bool = False):
    if getattr(depth_transform, "far_plane_at_max", False):
        return depth_transform.norm_min if is_inverse else depth_transform.norm_max
    return depth_transform.norm_max if is_inverse else depth_transform.norm_min


def _find_matching_file(base_name: str, folder: str, exts) -> str:
    for ext in exts:
        p = os.path.join(folder, base_name + ext)
        if os.path.exists(p):
            return p
    return ""


def _resize_keep_ratio_and_pad(x: torch.Tensor, out_h: int, out_w: int, *, mode: str):
    if x.ndim != 3:
        raise RuntimeError(f"Expected (C,H,W), got {tuple(x.shape)}")
    c, h, w = x.shape
    scale = min(out_h / h, out_w / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    x_in = x
    to_int = False
    to_bool = False
    if x.dtype in (torch.int32, torch.int64, torch.uint8):
        x_in = x.float()
        to_int = True
    elif x.dtype == torch.bool:
        x_in = x.float()
        to_bool = True
    elif not x.is_floating_point():
        x_in = x.float()

    interp_kwargs = {"mode": mode}
    if mode == "bilinear":
        interp_kwargs["align_corners"] = False
    x_resized = F.interpolate(
        x_in.unsqueeze(0), size=(new_h, new_w), **interp_kwargs
    ).squeeze(0)

    pad_h = max(0, out_h - new_h)
    pad_w = max(0, out_w - new_w)
    x_padded = F.pad(x_resized, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    if to_int:
        x_padded = x_padded.clamp(0, 255).round().to(torch.int32)
    elif to_bool:
        x_padded = x_padded > 0.5
    return x_padded


def _read_pairs_txt(pairs_txt_path: str):
    items = []
    with open(pairs_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split()
            if len(cols) < 2:
                raise RuntimeError(
                    f"Expected at least 2 columns in pairs txt (rgb invdepth [teacher]), got {len(cols)}: {line}"
                )
            rgb_rel = cols[0]
            inv_rel = cols[1]
            teacher_rel = cols[2] if len(cols) >= 3 else None
            items.append((rgb_rel, inv_rel, teacher_rel))
    return items


def _resolve_path(p: str, data_root: str | None) -> str:
    if os.path.isabs(p):
        return p
    if data_root is None:
        raise RuntimeError(
            f"Got relative path in pairs_txt but --data_root is not set: {p}"
        )
    return os.path.join(data_root, p)


def _name_base_from_path_tail(p: str, *, n_parts: int = 5) -> str:
    parts = os.path.normpath(p).split(os.sep)
    tail = parts[-int(n_parts) :] if len(parts) >= int(n_parts) else parts
    name = "_".join(tail)
    return os.path.splitext(name)[0]


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Marigold : RGB+InvDepth Conditioning (NO MASK) : Multi-image Inference"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Pipeline checkpoint (with model_index.json) or trainer checkpoint dir (containing unet/ and scheduler/).",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        required=True,
        help="Training config yaml path (used for depth_normalization).",
    )
    parser.add_argument(
        "--base_ckpt_dir",
        type=str,
        default=None,
        help="Base path for cfg.model.pretrained_path. Defaults to env BASE_CKPT_DIR if not set.",
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Dataset root. If set, relative paths in --pairs_txt are resolved relative to this root. Absolute paths in pairs txt do not need --data_root.",
    )
    parser.add_argument(
        "--pairs_txt",
        type=str,
        default=None,
        help="Optional pairs txt (columns: rgb invdepth [teacher]).",
    )
    parser.add_argument(
        "--input_rgb_dir",
        type=str,
        default=None,
        help="Path to the input RGB image folder.",
    )
    parser.add_argument(
        "--input_invdepth_dir",
        type=str,
        default=None,
        help="Path to the input invdepth folder (16-bit png mapped to [0,1]).",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=None,
        help="Diffusion denoising steps. If set to None, default will be read from checkpoint.",
    )
    parser.add_argument(
        "--processing_res",
        type=int,
        default=None,
        help="Resize max-edge before inference. 0 uses input resolution; None uses checkpoint default.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=1,
        help="Number of predictions to ensemble.",
    )
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with fp16.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="If set, output will be at processing resolution instead of input resolution.",
    )
    parser.add_argument(
        "--resample_method",
        choices=["bilinear", "bicubic", "nearest"],
        default="bilinear",
        help="Resampling method.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducibility. None means randomized inference.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--resize_to_hw",
        nargs=2,
        type=int,
        default=None,
        help="Optional: resize (keep ratio) + pad to fixed (H W) before inference.",
    )

    parser.add_argument(
        "--invdepth_from_depth",
        action="store_true",
        help="If set, interpret the 2nd column as depth and convert to invdepth by 1/(depth+eps) before normalization.",
    )

    parser.add_argument("--save_npy", dest="save_npy", action="store_true", default=True)
    parser.add_argument("--no_save_npy", dest="save_npy", action="store_false")
    parser.add_argument("--save_bw", dest="save_bw", action="store_true", default=True)
    parser.add_argument("--no_save_bw", dest="save_bw", action="store_false")
    parser.add_argument("--save_color", dest="save_color", action="store_true", default=True)
    parser.add_argument("--no_save_color", dest="save_color", action="store_false")
    parser.add_argument("--color_far_warm", action="store_true", default=False)
    parser.add_argument("--color_near_warm", action="store_true", default=False)
    parser.add_argument(
        "--color_norm",
        choices=["percentile", "minmax"],
        default="percentile",
    )
    parser.add_argument("--color_p_lo", type=float, default=1.0)
    parser.add_argument("--color_p_hi", type=float, default=99.0)

    parser.add_argument(
        "--name_mode",
        choices=["basename", "tail5"],
        default="tail5",
    )

    parser.add_argument(
        "--output_layout",
        choices=["global", "scene"],
        default="global",
    )

    args = parser.parse_args()

    if args.color_far_warm and args.color_near_warm:
        raise RuntimeError("--color_far_warm and --color_near_warm are mutually exclusive")

    cfg = OmegaConf.load(args.train_config)
    depth_transform = get_depth_normalizer(cfg.depth_normalization)

    if args.pairs_txt is not None:
        pairs_items = _read_pairs_txt(args.pairs_txt)
        use_pairs = True
    else:
        use_pairs = False
        if args.input_rgb_dir is None or args.input_invdepth_dir is None:
            raise RuntimeError(
                "Need either --pairs_txt (absolute or relative paths; use --data_root for relative) or --input_rgb_dir+--input_invdepth_dir"
            )

    rgb_filename_list = []
    if not use_pairs:
        rgb_filename_list = glob(os.path.join(args.input_rgb_dir, "*"))
        rgb_filename_list = [
            f
            for f in rgb_filename_list
            if os.path.splitext(f)[1].lower() in EXTENSION_LIST
        ]
        rgb_filename_list = sorted(rgb_filename_list)
        if len(rgb_filename_list) == 0:
            raise RuntimeError(f"No image found in '{args.input_rgb_dir}'")

    os.makedirs(args.output_dir, exist_ok=True)

    output_dir_color = None
    output_dir_bw = None
    output_dir_npy = None
    if args.output_layout == "global":
        output_dir_color = os.path.join(args.output_dir, "depth_colored")
        output_dir_bw = os.path.join(args.output_dir, "depth_bw")
        output_dir_npy = os.path.join(args.output_dir, "depth_npy")
        if args.save_color:
            os.makedirs(output_dir_color, exist_ok=True)
        if args.save_bw:
            os.makedirs(output_dir_bw, exist_ok=True)
        if args.save_npy:
            os.makedirs(output_dir_npy, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.half_precision:
        dtype = torch.float16
        variant = "fp16"
    else:
        dtype = torch.float32
        variant = None

    base_ckpt_dir = args.base_ckpt_dir or os.environ.get("BASE_CKPT_DIR", None)
    if base_ckpt_dir is None:
        base_ckpt_dir = ""

    model_index_path = os.path.join(args.checkpoint, "model_index.json")
    if os.path.exists(model_index_path):
        pipe: MarigoldDepthPipeline = MarigoldDepthPipeline.from_pretrained(
            args.checkpoint, variant=variant, torch_dtype=dtype
        )
    else:
        base_model_path = os.path.join(base_ckpt_dir, cfg.model.pretrained_path)
        pipe: MarigoldDepthPipeline = MarigoldDepthPipeline.from_pretrained(
            base_model_path, variant=variant, torch_dtype=dtype
        )

        unet_dir = os.path.join(args.checkpoint, "unet")
        if not os.path.isdir(unet_dir):
            raise FileNotFoundError(
                f"Checkpoint does not contain a pipeline (missing model_index.json) and no unet/ directory was found under: {args.checkpoint}"
            )
        pipe.unet = pipe.unet.from_pretrained(unet_dir, torch_dtype=dtype)

        scheduler_dir = os.path.join(args.checkpoint, "scheduler")
        if os.path.isdir(scheduler_dir):
            pipe.scheduler = pipe.scheduler.from_pretrained(scheduler_dir)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    try:
        pipe = pipe.to(device=device, dtype=dtype)
    except TypeError:
        pipe = pipe.to(device)
        pipe.unet.to(device=device, dtype=dtype)

    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    resample_method = args.resample_method

    resize_to_hw = None
    if args.resize_to_hw is not None:
        resize_to_hw = (int(args.resize_to_hw[0]), int(args.resize_to_hw[1]))

    with torch.no_grad():
        iterable = pairs_items if use_pairs else rgb_filename_list

        created_scene_dirs: set[str] = set()

        for item in tqdm(iterable, desc="Depth Inference", leave=True):
            if use_pairs:
                rgb_rel, inv_rel, _teacher_rel = item
                rgb_path = _resolve_path(rgb_rel, args.data_root)
                inv_path = _resolve_path(inv_rel, args.data_root)
                if args.name_mode == "basename":
                    rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
                else:
                    rgb_name_base = _name_base_from_path_tail(rgb_path, n_parts=5)
            else:
                rgb_path = item
                rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
                inv_path = _find_matching_file(
                    rgb_name_base, args.input_invdepth_dir, [".png"]
                )
                if not inv_path:
                    raise RuntimeError(
                        f"Missing invdepth for {rgb_name_base}: expected {os.path.join(args.input_invdepth_dir, rgb_name_base + '.png')}"
                    )

            if not os.path.exists(rgb_path):
                raise RuntimeError(f"RGB path not found: {rgb_path}")
            if not os.path.exists(inv_path):
                raise RuntimeError(f"InvDepth path not found: {inv_path}")

            if args.output_layout == "scene":
                scene_id = _scene_id_from_path(rgb_path)
                scene_root = os.path.join(args.output_dir, scene_id)
                if scene_root not in created_scene_dirs:
                    os.makedirs(scene_root, exist_ok=True)
                    created_scene_dirs.add(scene_root)
                output_dir_color = os.path.join(scene_root, "colored")
                output_dir_bw = os.path.join(scene_root, "bw")
                output_dir_npy = os.path.join(scene_root, "npy")
                if args.save_color:
                    os.makedirs(output_dir_color, exist_ok=True)
                if args.save_bw:
                    os.makedirs(output_dir_bw, exist_ok=True)
                if args.save_npy:
                    os.makedirs(output_dir_npy, exist_ok=True)

            input_image = Image.open(rgb_path).convert("RGB")

            inv01, inv_valid_np = _read_depth_any(
                inv_path, invert=bool(args.invdepth_from_depth)
            )
            inv_raw = torch.from_numpy(inv01).float().unsqueeze(0)  # [1,H,W]
            inv_valid = torch.from_numpy(inv_valid_np.astype(bool)).unsqueeze(0)  # [1,H,W]
            inv_fill = _get_invalid_fill_value(depth_transform, is_inverse=True)

            if resize_to_hw is not None:
                out_h, out_w = resize_to_hw
                rgb_int = pil_to_tensor(input_image).to(torch.int32)  # [3,H,W]
                rgb_int = _resize_keep_ratio_and_pad(
                    rgb_int, out_h, out_w, mode="bilinear"
                )
                rgb_in = rgb_int.unsqueeze(0)  # [1,3,H,W]

                inv_raw_rs = _resize_keep_ratio_and_pad(
                    inv_raw, out_h, out_w, mode="bilinear"
                )
                inv_valid_rs = _resize_keep_ratio_and_pad(
                    inv_valid, out_h, out_w, mode="nearest"
                )

                invdepth_norm = depth_transform(
                    inv_raw_rs.unsqueeze(0), inv_valid_rs.unsqueeze(0)
                ).clone()
                invdepth_norm[~inv_valid_rs.unsqueeze(0)] = inv_fill
            else:
                rgb_in = input_image

                invdepth_norm = depth_transform(
                    inv_raw.unsqueeze(0), inv_valid.unsqueeze(0)
                ).clone()
                invdepth_norm[~inv_valid.unsqueeze(0)] = inv_fill

            if args.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(args.seed)

            pipe_out: MarigoldDepthOutput = pipe(
                rgb_in,
                invdepth_norm=invdepth_norm,
                denoising_steps=denoise_steps,
                ensemble_size=ensemble_size,
                processing_res=processing_res,
                match_input_res=match_input_res,
                batch_size=args.batch_size,
                color_map=None,
                show_progress_bar=True,
                resample_method=resample_method,
                generator=generator,
            )

            depth_pred: np.ndarray = pipe_out.depth_np
            pred_name_base = rgb_name_base + "_infer"

            if args.save_npy:
                npy_save_path = os.path.join(output_dir_npy, f"{pred_name_base}.npy")
                np.save(npy_save_path, depth_pred)

            if args.save_bw:
                depth_u16 = (depth_pred * 65535.0).astype(np.uint16)
                png_u16_path = os.path.join(output_dir_bw, f"{pred_name_base}.png")
                iio.imwrite(png_u16_path, depth_u16)

                depth_u8 = (depth_pred * 255.0).astype(np.uint8)
                preview_save_path = os.path.join(
                    output_dir_bw, f"{pred_name_base}_preview8.png"
                )
                Image.fromarray(depth_u8).save(preview_save_path)

            if args.save_color:
                colored_save_path = os.path.join(output_dir_color, f"{pred_name_base}.png")
                if _USE_TURBO:
                    depth_vis = depth_pred.astype(np.float32)
                    finite = np.isfinite(depth_vis)
                    if finite.any():
                        if args.color_norm == "minmax":
                            min_d = float(np.min(depth_vis[finite]))
                            max_d = float(np.max(depth_vis[finite]))
                        else:
                            min_d = float(np.nanpercentile(depth_vis[finite], args.color_p_lo))
                            max_d = float(np.nanpercentile(depth_vis[finite], args.color_p_hi))
                    else:
                        min_d, max_d = 0.0, 1.0

                    if (not np.isfinite(min_d)) or (not np.isfinite(max_d)) or (max_d - min_d < 1e-6):
                        depth_normalized = np.zeros_like(depth_vis, dtype=np.float32)
                    else:
                        depth_normalized = (depth_vis - min_d) / (max_d - min_d)
                        depth_normalized = np.clip(depth_normalized, 0.0, 1.0)
                        depth_normalized = np.nan_to_num(depth_normalized, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
                    if args.color_near_warm:
                        depth_normalized = 1.0 - depth_normalized
                    rgb_colored_np = (
                        _CMAP_TURBO(depth_normalized)[..., :3] * 255.0 + 0.5
                    ).astype(np.uint8)
                    Image.fromarray(rgb_colored_np).save(colored_save_path)
                else:
                    if pipe_out.depth_colored is not None:
                        pipe_out.depth_colored.save(colored_save_path)

    logging.info(f"Done. Outputs saved to {args.output_dir}")
