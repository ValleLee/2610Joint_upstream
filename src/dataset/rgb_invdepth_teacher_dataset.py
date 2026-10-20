import os
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from .base_depth_dataset import BaseDepthDataset, DatasetMode


def _read_png_u16_01(path: str) -> np.ndarray:
    im = np.asarray(Image.open(path))
    if im.dtype == np.uint16:
        x = im.astype(np.float32) / 65535.0
    else:
        x = im.astype(np.float32) / 255.0
    return x


def _load_npy_f32(path: str) -> np.ndarray:
    x = np.load(path)
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    x = x.astype(np.float32)
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    return x


def _normalize_representation(name: str) -> str:
    rep = str(name).strip().lower()
    if rep not in {"depth", "invdepth"}:
        raise RuntimeError(
            f"Unknown representation={name}; expected one of ['depth', 'invdepth']"
        )
    return rep


def _should_invert(source_is_depth: bool, target_representation: str) -> bool:
    target_is_inverse = _normalize_representation(target_representation) == "invdepth"
    return bool(source_is_depth) == bool(target_is_inverse)


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


def _resize_like(x: torch.Tensor, ref_hw: tuple[int, int], *, mode: str) -> torch.Tensor:
    if x.shape[-2:] == ref_hw:
        return x
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if mode == "nearest":
        y = F.interpolate(x.float(), size=ref_hw, mode=mode)
        y = (y > 0.5)
    else:
        y = F.interpolate(x, size=ref_hw, mode=mode, align_corners=False)
    return y.squeeze(0)


class RGBInvDepthTeacherDataset(BaseDepthDataset):
    def __init__(self, mode: DatasetMode, filename_ls_path: str, dataset_dir: str, **kwargs):
        condition_from_depth = kwargs.pop("condition_from_depth", None)
        teacher_from_depth = kwargs.pop("teacher_from_depth", None)
        condition_representation = kwargs.pop("condition_representation", None)
        teacher_representation = kwargs.pop("teacher_representation", None)
        legacy_invdepth_from_depth = bool(kwargs.pop("invdepth_from_depth", False))
        if condition_from_depth is None:
            condition_from_depth = legacy_invdepth_from_depth
        if teacher_from_depth is None:
            teacher_from_depth = False
        self.condition_from_depth = bool(condition_from_depth)
        self.teacher_from_depth = bool(teacher_from_depth)
        if condition_representation is None:
            condition_representation = "invdepth"
        if teacher_representation is None:
            teacher_representation = "invdepth" if self.teacher_from_depth else "depth"
        self.condition_representation = _normalize_representation(condition_representation)
        self.teacher_representation = _normalize_representation(teacher_representation)
        self.condition_is_inverse = self.condition_representation == "invdepth"
        self.teacher_is_inverse = self.teacher_representation == "invdepth"
        self.depth_representation_is_inverse = self.teacher_is_inverse
        self.invdepth_from_depth = self.condition_from_depth
        super().__init__(mode=mode, filename_ls_path=filename_ls_path, dataset_dir=dataset_dir, **kwargs)

    def _get_data_item(self, index):
        filename_line = self.filenames[index]
        if len(filename_line) != 5:
            raise RuntimeError(
                f"Expected 5 columns (rendered RGB, rendered depth, DA depth, rendered DA depth, GT RGB), got {len(filename_line)}: {filename_line}"
            )

        rgb_rel_path = filename_line[0]
        inv_rel_path = filename_line[1]
        teacher_rel_path = filename_line[2]
        gt_rgb_rel_path = filename_line[4]

        rasters = {}
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))
        rasters["gt_rgb_norm"] = self._load_rgb_data(rgb_rel_path=gt_rgb_rel_path)["rgb_norm"]

        ref_hw = tuple(rasters["rgb_norm"].shape[-2:])
        rasters["gt_rgb_norm"] = _resize_like(rasters["gt_rgb_norm"], ref_hw, mode="bilinear")

        teacher_path = os.path.join(self.dataset_dir, teacher_rel_path)
        teacher01, teacher_valid = _read_depth_any(
            teacher_path,
            invert=_should_invert(
                source_is_depth=self.teacher_from_depth,
                target_representation=self.teacher_representation,
            ),
        )
        teacher_ts = torch.from_numpy(teacher01.copy()).float().unsqueeze(0)
        rasters["depth_raw_linear"] = teacher_ts.clone()
        rasters["depth_filled_linear"] = teacher_ts.clone()
        rasters["teacher_target_linear"] = teacher_ts.clone()

        inv_path = os.path.join(self.dataset_dir, inv_rel_path)
        inv01, inv_valid = _read_depth_any(
            inv_path,
            invert=_should_invert(
                source_is_depth=self.condition_from_depth,
                target_representation=self.condition_representation,
            ),
        )
        inv_ts = torch.from_numpy(inv01.copy()).float().unsqueeze(0)
        rasters["invdepth_raw_linear"] = inv_ts
        rasters["condition_raw_linear"] = inv_ts.clone()

        teacher_valid_ts = torch.from_numpy(teacher_valid.astype(bool))
        if teacher_valid_ts.ndim == 2:
            teacher_valid_ts = teacher_valid_ts.unsqueeze(0)
        inv_valid_ts = torch.from_numpy(inv_valid.astype(bool))
        if inv_valid_ts.ndim == 2:
            inv_valid_ts = inv_valid_ts.unsqueeze(0)

        rasters["valid_mask_raw"] = teacher_valid_ts.clone()
        rasters["valid_mask_filled"] = teacher_valid_ts.clone()
        rasters["valid_mask_invdepth"] = inv_valid_ts.clone()
        rasters["valid_mask_teacher_target"] = teacher_valid_ts.clone()
        rasters["valid_mask_condition"] = inv_valid_ts.clone()

        rasters["invdepth_raw_linear"] = _resize_like(
            rasters["invdepth_raw_linear"], ref_hw, mode="bilinear"
        )
        rasters["condition_raw_linear"] = _resize_like(
            rasters["condition_raw_linear"], ref_hw, mode="bilinear"
        )
        rasters["depth_raw_linear"] = _resize_like(
            rasters["depth_raw_linear"], ref_hw, mode="bilinear"
        )
        rasters["depth_filled_linear"] = _resize_like(
            rasters["depth_filled_linear"], ref_hw, mode="bilinear"
        )
        rasters["teacher_target_linear"] = _resize_like(
            rasters["teacher_target_linear"], ref_hw, mode="bilinear"
        )
        rasters["valid_mask_raw"] = _resize_like(
            rasters["valid_mask_raw"], ref_hw, mode="nearest"
        )
        rasters["valid_mask_filled"] = _resize_like(
            rasters["valid_mask_filled"], ref_hw, mode="nearest"
        )
        rasters["valid_mask_invdepth"] = _resize_like(
            rasters["valid_mask_invdepth"], ref_hw, mode="nearest"
        )
        rasters["valid_mask_teacher_target"] = _resize_like(
            rasters["valid_mask_teacher_target"], ref_hw, mode="nearest"
        )
        rasters["valid_mask_condition"] = _resize_like(
            rasters["valid_mask_condition"], ref_hw, mode="nearest"
        )

        other = {
            "index": index,
            "rgb_relative_path": rgb_rel_path,
            "condition_relative_path": inv_rel_path,
            "invdepth_relative_path": inv_rel_path,
            "teacher_relative_path": teacher_rel_path,
            "gt_rgb_relative_path": gt_rgb_rel_path,
        }

        return rasters, other

    def _training_preprocess(self, rasters):
        rasters = super()._training_preprocess(rasters)

        condition = rasters["condition_raw_linear"]
        condition_valid = rasters.get("valid_mask_condition", None)
        if condition_valid is None:
            condition_valid = torch.isfinite(condition) & (condition > 0)

        rasters["condition_norm"] = self.depth_transform(condition, condition_valid).clone()
        # Backward-compatible alias for older trainer/pipeline code paths.
        rasters["invdepth_norm"] = rasters["condition_norm"].clone()

        teacher_target = rasters["teacher_target_linear"]
        teacher_valid = rasters.get("valid_mask_teacher_target", None)
        if teacher_valid is None:
            teacher_valid = torch.isfinite(teacher_target) & (teacher_target > 0)

        rasters["teacher_target_norm"] = self.depth_transform(
            teacher_target, teacher_valid
        ).clone()

        if self.move_invalid_to_far_plane:
            cond_fill = self._get_invalid_fill_value(is_inverse=self.condition_is_inverse)
            teacher_fill = self._get_invalid_fill_value(is_inverse=self.teacher_is_inverse)
            rasters["condition_norm"][~condition_valid] = cond_fill
            rasters["invdepth_norm"][~condition_valid] = cond_fill
            rasters["teacher_target_norm"][~teacher_valid] = teacher_fill
            if "depth_filled_norm" in rasters:
                rasters["depth_filled_norm"][~teacher_valid] = teacher_fill

        return rasters
