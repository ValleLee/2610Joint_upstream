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
# More information about Marigold:
#   https://marigoldmonodepth.github.io
#   https://marigoldcomputervision.github.io
# Efficient inference pipelines are now part of diffusers:
#   https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage
#   https://huggingface.co/docs/diffusers/api/pipelines/marigold
# Examples of trained models and live demos:
#   https://huggingface.co/prs-eth
# Related projects:
#   https://rollingdepth.github.io/
#   https://marigolddepthcompletion.github.io/
# Citation (BibTeX):
#   https://github.com/prs-eth/Marigold#-citation
# If you find Marigold useful, we kindly ask you to cite our papers.
# --------------------------------------------------------------------------

import io
import logging
import numpy as np
import os
import random
import tarfile
import torch
import torch.nn.functional as F

from PIL import Image
from enum import Enum
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize  # kept for compatibility
from typing import Union

from src.util.depth_transform import DepthNormalizerBase

# CHANGE: 兼容 OmegaConf 的 ListConfig/DictConfig（YAML 读取常见）
try:
    from omegaconf import ListConfig, DictConfig
except Exception:  # OmegaConf 不一定安装
    ListConfig = None
    DictConfig = None


class DatasetMode(Enum):
    RGB_ONLY = "rgb_only"
    EVAL = "evaluate"
    TRAIN = "train"


class DepthFileNameMode(Enum):
    """Prediction file naming modes"""

    id = 1  # id.png
    rgb_id = 2  # rgb_id.png
    i_d_rgb = 3  # i_d_1_rgb.png
    rgb_i_d = 4


class BaseDepthDataset(Dataset):
    def __init__(
        self,
        mode: DatasetMode,
        filename_ls_path: str,
        dataset_dir: str,
        disp_name: str,
        min_depth: float,
        max_depth: float,
        has_filled_depth: bool,
        name_mode: DepthFileNameMode,
        depth_transform: Union[DepthNormalizerBase, None] = None,
        augmentation_args: dict = None,
        resize_to_hw=None,
        move_invalid_to_far_plane: bool = True,
        rgb_transform=lambda x: x / 255.0 * 2 - 1,  #  [0, 255] -> [-1, 1],
        **kwargs,
    ) -> None:
        super().__init__()
        self.mode = mode
        # dataset info
        self.filename_ls_path = os.path.join(dataset_dir, filename_ls_path)
        self.dataset_dir = dataset_dir
        assert os.path.exists(
            self.dataset_dir
        ), f"Dataset does not exist at: {self.dataset_dir}"
        self.disp_name = disp_name
        self.has_filled_depth = has_filled_depth
        self.name_mode: DepthFileNameMode = name_mode
        self.min_depth = min_depth
        self.max_depth = max_depth

        # training arguments
        self.depth_transform: DepthNormalizerBase = depth_transform
        self.augm_args = augmentation_args
        self.resize_to_hw = resize_to_hw
        self.rgb_transform = rgb_transform
        self.move_invalid_to_far_plane = move_invalid_to_far_plane

        # Load filenames
        with open(self.filename_ls_path, "r") as f:
            self.filenames = [
                s.split() for s in f.readlines()
            ]  # [['rgb.png', 'depth.tif'], [], ...]

        # Tar dataset
        self.tar_obj = None
        self.is_tar = (
            True
            if os.path.isfile(dataset_dir) and tarfile.is_tarfile(dataset_dir)
            else False
        )

        # CHANGE: 解析 & 规范化 resize_to_hw，兼容 OmegaConf / 字符串 / 各类容器
        if self.resize_to_hw is not None:
            self.resize_to_hw = self._coerce_hw(self.resize_to_hw)
            assert (
                isinstance(self.resize_to_hw, (list, tuple)) and len(self.resize_to_hw) == 2
            ), f"bad resize_to_hw: {self.resize_to_hw}"

    def __len__(self):
        return len(self.filenames)

    # ---------- helpers ----------
    # CHANGE: 新增健壮的解析函数，兼容 ListConfig/DictConfig/字符串 等
    def _coerce_hw(self, hw):
        """
        将各种可能的 hw 表达（list/tuple/torch.Size/np.array/OmegaConf/字符串/可迭代）转为 [int(h), int(w)]
        支持字符串格式："512,768" 或 "[512, 768]" 或 "(512,768)"
        支持 OmegaConf: ListConfig([512,768]) 或 DictConfig({'h':512,'w':768}) / {'height':..,'width':..}
        """
        import re
        import numpy as np

        # ---- OmegaConf: ListConfig ----
        if ListConfig is not None and isinstance(hw, ListConfig):
            hw = list(hw)  # 转普通 list

        # ---- OmegaConf: DictConfig ----
        if DictConfig is not None and isinstance(hw, DictConfig):
            keys = {str(k).lower(): k for k in hw.keys()}
            if ('h' in keys and 'w' in keys) or ('height' in keys and 'width' in keys):
                hk = keys['h'] if 'h' in keys else keys['height']
                wk = keys['w'] if 'w' in keys else keys['width']
                return [int(hw[hk]), int(hw[wk])]
            # 也可能是 {'0':512,'1':768}
            if '0' in hw and '1' in hw:
                return [int(hw['0']), int(hw['1'])]
            # 兜底：按插入顺序取前两个值
            vals = list(hw.values())
            assert len(vals) >= 2, f"bad DictConfig for resize_to_hw: {hw}"
            return [int(vals[0]), int(vals[1])]

        # ---- 常见 python/numpy/torch 类型 ----
        if isinstance(hw, (list, tuple)):
            assert len(hw) == 2, f"bad resize_to_hw length: {hw}"
            return [int(hw[0]), int(hw[1])]

        try:
            from torch import Size as TorchSize
            if isinstance(hw, TorchSize):
                assert len(hw) == 2, f"bad resize_to_hw length: {hw}"
                return [int(hw[0]), int(hw[1])]
        except Exception:
            pass

        if isinstance(hw, np.ndarray):
            assert hw.size == 2, f"bad resize_to_hw length: {hw}"
            return [int(hw[0]), int(hw[1])]

        # ---- 字符串 ----
        if isinstance(hw, str):
            s = hw.strip()
            # 去掉可能的 [] 或 ()
            s = re.sub(r"^[\[\(\s]+", "", s)
            s = re.sub(r"[\]\)\s]+$", "", s)
            # 用逗号或空白分割
            parts = re.split(r"[,\s]+", s)
            parts = [p for p in parts if p != ""]
            assert len(parts) == 2, f"bad resize_to_hw string: {hw}"
            return [int(float(parts[0])), int(float(parts[1]))]

        # ---- 可迭代对象兜底（某些自定义容器）----
        try:
            it = list(hw)
            if len(it) == 2:
                return [int(it[0]), int(it[1])]
        except Exception:
            pass

        # ---- 不支持的类型 ----
        raise AssertionError(f"Unsupported resize_to_hw type: {type(hw)} -> {hw}")

    # CHANGE: 新增“等比缩放 + 中心 pad 到 (H,W)”的通用函数
    def _resize_keep_ratio_and_pad(self, x: torch.Tensor, out_h: int, out_w: int, *, mode: str):
        """
        x: (C,H,W) float/int/bool tensor
        out_h,out_w: 目标输出尺寸（H,W）
        mode: 'bilinear' 或 'nearest'（mask/整型用 nearest）
        返回：与 x 同 dtype 的 (C,out_h,out_w)
        """
        C, H, W = x.shape
        # 计算等比缩放尺寸
        scale = min(out_h / H, out_w / W)
        new_h = int(round(H * scale))
        new_w = int(round(W * scale))

        # 1) resize（插值在 NCHW 上做）
        x_in = x
        to_int = False
        to_bool = False
        if x.dtype in (torch.int32, torch.int64):
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

        # 2) center pad 到目标尺寸
        pad_h = max(0, out_h - new_h)
        pad_w = max(0, out_w - new_w)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        x_padded = F.pad(
            x_resized,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0.0,
        )

        # 3) 还原 dtype
        if to_int:
            x_padded = x_padded.clamp(0, 255).round().to(torch.int32)
        elif to_bool:
            x_padded = (x_padded > 0.5)

        return x_padded

    def _resize_rasters_to_hw(self, rasters: dict):
        if self.resize_to_hw is None:
            return rasters

        out_h, out_w = int(self.resize_to_hw[0]), int(self.resize_to_hw[1])
        out = {}
        for k, v in rasters.items():
            if not torch.is_tensor(v):
                out[k] = v
                continue
            if v.ndim == 2:
                v = v.unsqueeze(0)
            mode = "nearest" if ("mask" in k) else "bilinear"
            out[k] = self._resize_keep_ratio_and_pad(v, out_h, out_w, mode=mode)

        for k, v in out.items():
            if torch.is_tensor(v):
                assert v.shape[-2:] == (out_h, out_w), f"{k} -> {v.shape[-2:]} != {(out_h, out_w)}"

        return out

    # ---------- dataset io ----------
    def __getitem__(self, index):
        rasters, other = self._get_data_item(index)
        if DatasetMode.TRAIN != self.mode:
            rasters = self._resize_rasters_to_hw(rasters)
        if DatasetMode.TRAIN == self.mode:
            rasters = self._training_preprocess(rasters)
        else:
            ref_hw = None
            for _k in ("rgb_int", "rgb_norm", "depth_raw_linear"):
                if _k in rasters and torch.is_tensor(rasters[_k]) and rasters[_k].ndim >= 3:
                    ref_hw = tuple(rasters[_k].shape[-2:])
                    break
            if ref_hw is not None:
                bad = []
                for k, v in rasters.items():
                    if not torch.is_tensor(v) or v.ndim < 3:
                        continue
                    if tuple(v.shape[-2:]) != ref_hw:
                        bad.append((k, tuple(v.shape), ref_hw))
                if len(bad) > 0:
                    logging.warning(
                        f"[dataset] raster shape mismatch after resize_to_hw, index={index}, other={other}, bad={bad}"
                    )
        outputs = rasters
        outputs.update(other)
        return outputs

    def _get_data_item(self, index):
        rgb_rel_path, depth_rel_path, filled_rel_path = self._get_data_path(index=index)

        rasters = {}
        # RGB
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))

        # Depth
        if DatasetMode.RGB_ONLY != self.mode:
            depth_data = self._load_depth_data(
                depth_rel_path=depth_rel_path, filled_rel_path=filled_rel_path
            )
            rasters.update(depth_data)
            # masks
            rasters["valid_mask_raw"] = self._get_valid_mask(
                rasters["depth_raw_linear"]
            ).clone()
            rasters["valid_mask_filled"] = self._get_valid_mask(
                rasters["depth_filled_linear"]
            ).clone()

        other = {"index": index, "rgb_relative_path": rgb_rel_path}
        return rasters, other

    def _load_rgb_data(self, rgb_rel_path):
        rgb = self._read_rgb_file(rgb_rel_path)  # numpy CHW int
        rgb_norm = rgb / 255.0 * 2.0 - 1.0
        return {
            "rgb_int": torch.from_numpy(rgb).int(),
            "rgb_norm": torch.from_numpy(rgb_norm).float(),
        }

    def _load_depth_data(self, depth_rel_path, filled_rel_path):
        outputs = {}
        depth_raw = self._read_depth_file(depth_rel_path).squeeze()
        depth_raw_linear = torch.from_numpy(depth_raw.copy()).float().unsqueeze(0)  # (1,H,W)
        outputs["depth_raw_linear"] = depth_raw_linear.clone()

        if self.has_filled_depth:
            depth_filled = self._read_depth_file(filled_rel_path).squeeze()
            depth_filled_linear = torch.from_numpy(depth_filled.copy()).float().unsqueeze(0)
            outputs["depth_filled_linear"] = depth_filled_linear
        else:
            outputs["depth_filled_linear"] = depth_raw_linear.clone()
        return outputs

    def _get_data_path(self, index):
        filename_line = self.filenames[index]
        rgb_rel_path = filename_line[0]
        depth_rel_path, filled_rel_path = None, None
        if DatasetMode.RGB_ONLY != self.mode:
            depth_rel_path = filename_line[1]
            if self.has_filled_depth:
                filled_rel_path = filename_line[2]
        return rgb_rel_path, depth_rel_path, filled_rel_path

    def _read_image(self, img_rel_path) -> np.ndarray:
        if self.is_tar:
            if self.tar_obj is None:
                self.tar_obj = tarfile.open(self.dataset_dir)
            image_to_read = self.tar_obj.extractfile("./" + img_rel_path)
            image_to_read = image_to_read.read()
            image_to_read = io.BytesIO(image_to_read)
        else:
            image_to_read = os.path.join(self.dataset_dir, img_rel_path)
        image = Image.open(image_to_read)  # [H, W, C]
        image = np.asarray(image)
        return image

    def _read_rgb_file(self, rel_path) -> np.ndarray:
        rgb = self._read_image(rel_path)
        rgb = np.transpose(rgb, (2, 0, 1)).astype(int)  # [C,H,W]
        return rgb

    def _read_depth_file(self, rel_path):
        # depth_in: uint16(mm) -> float(m)
        depth_in = self._read_image(rel_path)
        depth_decoded = depth_in.astype(np.float32) / 1000.0
        return depth_decoded

    def _get_valid_mask(self, depth: torch.Tensor):
        """
        depth 为米；有效掩码要避开“远平面/无效值”
        """
        eps = 1e-6
        far_sentinel = 65.535  # 65535 mm
        valid_mask = (
            torch.isfinite(depth)
            & (depth > max(self.min_depth, eps))
            & (depth < min(self.max_depth, far_sentinel - 1e-3))
        )
        return valid_mask.bool()

    def _get_invalid_fill_value(self, *, is_inverse: bool = False):
        if getattr(self.depth_transform, "far_plane_at_max", False):
            return self.depth_transform.norm_min if is_inverse else self.depth_transform.norm_max
        return self.depth_transform.norm_max if is_inverse else self.depth_transform.norm_min

    def _training_preprocess(self, rasters):
        # Augmentation
        if self.augm_args is not None:
            rasters = self._augment_data(rasters)

        # Normalization
        rasters["depth_raw_norm"] = self.depth_transform(
            rasters["depth_raw_linear"], rasters["valid_mask_raw"]
        ).clone()
        rasters["depth_filled_norm"] = self.depth_transform(
            rasters["depth_filled_linear"], rasters["valid_mask_filled"]
        ).clone()

        # Set invalid pixel to far plane
        if self.move_invalid_to_far_plane:
            fill_value = self._get_invalid_fill_value(
                is_inverse=getattr(self, "depth_representation_is_inverse", False)
            )
            rasters["depth_filled_norm"][~rasters["valid_mask_filled"]] = fill_value

        rasters = self._resize_rasters_to_hw(rasters)

        return rasters

    def _augment_data(self, rasters_dict):
        # lr flipping
        lr_flip_p = self.augm_args.lr_flip_p
        if random.random() < lr_flip_p:
            rasters_dict = {k: v.flip(-1) for k, v in rasters_dict.items()}
        return rasters_dict

    def __del__(self):
        if hasattr(self, "tar_obj") and self.tar_obj is not None:
            self.tar_obj.close()
            self.tar_obj = None


def get_pred_name(rgb_basename, name_mode, suffix=".png"):
    if DepthFileNameMode.rgb_id == name_mode:
        pred_basename = "pred_" + rgb_basename.split("_")[1]
    elif DepthFileNameMode.i_d_rgb == name_mode:
        pred_basename = rgb_basename.replace("_rgb.", "_pred.")
    elif DepthFileNameMode.id == name_mode:
        pred_basename = "pred_" + rgb_basename
    elif DepthFileNameMode.rgb_i_d == name_mode:
        pred_basename = "pred_" + "_".join(rgb_basename.split("_")[1:])
    else:
        raise NotImplementedError
    # change suffix
    pred_basename = os.path.splitext(pred_basename)[0] + suffix
    return pred_basename
