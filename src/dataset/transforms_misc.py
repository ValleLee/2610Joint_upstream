# data/transforms_misc.py
import cv2, torch
import torch.nn.functional as F
import numpy as np

def resize_keep_ratio_max(img: np.ndarray, dep: np.ndarray, max_h: int, max_w: int):
    h, w = img.shape[:2]
    scale = min(max_h / h, max_w / w)
    if scale != 1.0:
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        # RGB 用线性/AREA均可；放大用LINEAR，缩小用AREA
        interp_rgb = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
        img = cv2.resize(img, (new_w, new_h), interpolation=interp_rgb)
        # 深度建议最近邻，避免值混合
        dep = cv2.resize(dep, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return img, dep

def pad_to_hw(img_t: torch.Tensor, dep_t: torch.Tensor, out_h: int, out_w: int):
    _, h, w = img_t.shape
    pad_h = max(0, out_h - h)
    pad_w = max(0, out_w - w)
    # 右下角补；如需对称，可改 (left,right,top,bottom)
    img_t = F.pad(img_t, (0, pad_w, 0, pad_h), value=0.0)
    dep_t = F.pad(dep_t, (0, pad_w, 0, pad_h), value=0.0)
    return img_t, dep_t
