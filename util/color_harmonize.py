"""
Color‑Harmonization utilities for StyleID+
-----------------------------------------
Implements a lightweight Reinhard LAB mean / std transfer.

Usage
-----
from util.color_harmonize import harmonize_lab

out_img = harmonize_lab(stylized_img, style_img)
# • stylized_img : uint8 ndarray H×W×3   (BGR or RGB ok, auto‑detect)
# • style_img    : uint8 ndarray H×W×3
# returns uint8 ndarray RGB
"""

import cv2
import numpy as np

def _bgr2lab(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

def _lab2bgr(img_lab):
    img_lab = np.clip(img_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

def harmonize_lab(stylized_rgb: np.ndarray, style_rgb: np.ndarray) -> np.ndarray:
    """Align the mean and std of stylized image to that of the style reference in LAB space."""
    # ensure BGR for OpenCV
    styl_bgr  = stylized_rgb[:, :, ::-1] if stylized_rgb.shape[2] == 3 else stylized_rgb
    style_bgr = style_rgb[:, :, ::-1]    if style_rgb.shape[2] == 3 else style_rgb

    styl_lab  = _bgr2lab(styl_bgr)
    style_lab = _bgr2lab(style_bgr)

    # channel‑wise mean / std
    mu_s,  sigma_s  = cv2.meanStdDev(style_lab)
    mu_t,  sigma_t  = cv2.meanStdDev(styl_lab)

    sigma_t[sigma_t == 0] = 1   # avoid /0
    alpha = 1
    sigma_s = alpha * sigma_s + (1-alpha) * sigma_t
    # apply (lab - μ_t)/σ_t * σ_s + μ_s
    result_lab = (styl_lab - mu_t.reshape(1,1,3)) / sigma_t.reshape(1,1,3) \
                 * sigma_s.reshape(1,1,3) + mu_s.reshape(1,1,3)

    result_bgr = _lab2bgr(result_lab)
    return result_bgr[:, :, ::-1]   # return RGB
