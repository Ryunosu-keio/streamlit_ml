# features_pupil_gpu.py
# -*- coding: utf-8 -*-
"""
GPU対応版 features_pupil.py
- OpenCV CUDA と CuPy を用いて Laplacian, Sobel, Canny, FFT をGPU化
- GPUが利用できない環境では自動的にCPU実装へフォールバック
"""

import math
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis

# ===== CuPy (FFT用) =====
try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = np
    _HAS_CUPY = False

# ===== OpenCV CUDA 対応確認 =====
_HAS_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0

# ------------------------------------------------------------
# 色空間変換
# ------------------------------------------------------------
def srgb_to_linear(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0).astype(np.float32)
    a = 0.04045
    return np.where(u <= a, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)

def to_linear_luminance(img_bgr_uint8: np.ndarray) -> np.ndarray:
    b = img_bgr_uint8.astype(np.float32) / 255.0
    b_lin = srgb_to_linear(b)
    R = b_lin[..., 2]; G = b_lin[..., 1]; B = b_lin[..., 0]
    Y_lin = 0.2126 * R + 0.7152 * G + 0.0722 * B
    return np.clip(Y_lin, 0.0, 1.0)

def to_nonlinear_gray(img_bgr_uint8: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return gray

def to_saturation(img_bgr_uint8: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    Cmax = img_rgb.max(axis=2)
    Cmin = img_rgb.min(axis=2)
    delta = Cmax - Cmin
    S = np.zeros_like(Cmax, dtype=np.float32)
    mask = Cmax > 0
    S[mask] = delta[mask] / Cmax[mask]
    return np.clip(S, 0.0, 1.0)

# ------------------------------------------------------------
# ユーティリティ
# ------------------------------------------------------------
def _vals(arr: np.ndarray, mask) -> np.ndarray:
    return arr[mask].astype(np.float32) if mask is not None else arr.ravel().astype(np.float32)

# ------------------------------------------------------------
# 統計系
# ------------------------------------------------------------
def stat_mean(arr, mask):    v = _vals(arr, mask); return float(v.mean()) if v.size else np.nan
def stat_std(arr, mask):     v = _vals(arr, mask); return float(v.std()) if v.size else np.nan
def stat_min(arr, mask):     v = _vals(arr, mask); return float(v.min()) if v.size else np.nan
def stat_max(arr, mask):     v = _vals(arr, mask); return float(v.max()) if v.size else np.nan
def stat_median(arr, mask):  v = _vals(arr, mask); return float(np.median(v)) if v.size else np.nan
def stat_skew(arr, mask):    v = _vals(arr, mask); return float(skew(v)) if v.size else np.nan
def stat_kurt(arr, mask):    v = _vals(arr, mask); return float(kurtosis(v)) if v.size else np.nan

def stat_entropy_256(arr01, mask):
    v = (_vals(arr01, mask) * 255.0).clip(0, 255).astype(np.uint8)
    if v.size == 0: return np.nan
    hist, _ = np.histogram(v, bins=256, range=(0, 255))
    p = hist.astype(np.float64) / max(1, hist.sum())
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

# ------------------------------------------------------------
# コントラスト（RMS, GLCM, FFT）
# ------------------------------------------------------------
def rms_contrast(arr01, mask):
    v = _vals(arr01, mask)
    if v.size == 0: return np.nan
    mu = v.mean()
    return float(v.std() / mu) if mu != 0 else 0.0

def glcm_prop(arr01, mask, prop):
    v8 = (arr01 * 255.0).clip(0, 255).astype(np.uint8)
    if mask is not None:
        v8 = np.where(mask, v8, np.median(v8)).astype(np.uint8)
    glcm = graycomatrix(v8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    return float(graycoprops(glcm, prop)[0, 0])

def bandpowers_cpd(arr01, mask, px_per_deg: float, bands=((0.25,0.5),(0.5,1),(1,2),(2,4),(4,8))):
    # CuPyによるGPU FFT
    xp = cp if _HAS_CUPY else np
    a = xp.asarray(arr01)
    if mask is not None:
        a = xp.where(xp.asarray(mask), a, xp.median(a))
    H, W = a.shape
    F = xp.fft.fft2(a)
    A = xp.abs(F) ** 2
    fy = xp.fft.fftfreq(H) * H / px_per_deg
    fx = xp.fft.fftfreq(W) * W / px_per_deg
    FY, FX = xp.meshgrid(fy, fx, indexing='ij')
    FR = xp.sqrt(FX**2 + FY**2)
    out = {}
    for (f1, f2) in bands:
        band_mask = (FR >= f1) & (FR < f2)
        power = float(xp.mean(A[band_mask])) if xp.any(band_mask) else np.nan
        out[f"{f1:.2f}-{f2:.2f}cpd"] = power
    if _HAS_CUPY:
        cp.get_default_memory_pool().free_all_blocks()
    return out

# ------------------------------------------------------------
# シャープネス系 (OpenCV CUDA)
# ------------------------------------------------------------
def laplacian_var(arr01, mask):
    g = (arr01 * 255.0).astype(np.uint8)
    if _HAS_CUDA:
        gpu = cv2.cuda_GpuMat(); gpu.upload(g)
        L = cv2.cuda.createLaplacianFilter(cv2.CV_8UC1, cv2.CV_32F, ksize=3).apply(gpu)
        L_cpu = L.download()
    else:
        L_cpu = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    return float(L_cpu[mask].var()) if mask is not None else float(L_cpu.var())

def tenengrad(arr01, mask):
    g = (arr01 * 255.0).astype(np.uint8)
    if _HAS_CUDA:
        gpu = cv2.cuda_GpuMat(); gpu.upload(g)
        gx = cv2.cuda.Sobel(gpu, cv2.CV_32F, 1, 0, ksize=3).download()
        gy = cv2.cuda.Sobel(gpu, cv2.CV_32F, 0, 1, ksize=3).download()
    else:
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    M2 = gx**2 + gy**2
    return float(M2[mask].mean()) if mask is not None else float(M2.mean())

def edge_density(arr01, mask):
    g = (arr01 * 255.0).astype(np.uint8)
    if _HAS_CUDA:
        gpu = cv2.cuda_GpuMat(); gpu.upload(g)
        edges = cv2.cuda.Canny(gpu, 100, 200).download()
    else:
        edges = cv2.Canny(g, 100, 200)
    roi = edges[mask] if mask is not None else edges
    return float(np.mean(roi > 0)) if roi.size else np.nan

# ------------------------------------------------------------
# 彩度
# ------------------------------------------------------------
def sat_mean_std(S01, mask):
    v = _vals(S01, mask)
    return (float(v.mean()), float(v.std())) if v.size else (np.nan, np.nan)

# ------------------------------------------------------------
# マスクとpx/deg
# ------------------------------------------------------------
def make_masks(h, w, screen_w_mm, dist_mm, res_x, center_deg, parafovea_deg):
    fov_x_deg = math.degrees(2.0 * math.atan((screen_w_mm / 2.0) / dist_mm))
    deg_per_px = fov_x_deg / float(res_x)
    center_r_px = int(round(center_deg / deg_per_px))
    para_r_px   = int(round(parafovea_deg / deg_per_px))
    cx, cy = w // 2, h // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask_center     = dist <= center_r_px
    mask_parafovea  = (dist > center_r_px) & (dist <= para_r_px)
    mask_periphery  = dist > para_r_px
    return {"center": mask_center, "parafovea": mask_parafovea, "periphery": mask_periphery}

def _px_per_deg_from_geometry(screen_w_mm, dist_mm, res_x):
    fov_x_deg = math.degrees(2.0 * math.atan((screen_w_mm / 2.0) / dist_mm))
    deg_per_px = fov_x_deg / float(res_x)
    return 1.0 / deg_per_px

# ------------------------------------------------------------
# 複数領域の一括特徴量
# ------------------------------------------------------------
def compute_features_for_image(img_bgr, masks, screen_w_mm, dist_mm, res_x):
    px_per_deg = _px_per_deg_from_geometry(screen_w_mm, dist_mm, res_x)
    out = {}
    for region_name, mask in masks.items():
        Y_lin = to_linear_luminance(img_bgr)
        S = to_saturation(img_bgr)
        feats = {
            f"{region_name}_bL_mean": stat_mean(Y_lin, mask),
            f"{region_name}_bL_std": stat_std(Y_lin, mask),
            f"{region_name}_bL_entropy": stat_entropy_256(Y_lin, mask),
            f"{region_name}_c_rms": rms_contrast(Y_lin, mask),
            f"{region_name}_c_glcm_contrast": glcm_prop(Y_lin, mask, "contrast"),
            f"{region_name}_sh_laplacian_var": laplacian_var(Y_lin, mask),
            f"{region_name}_sh_tenengrad": tenengrad(Y_lin, mask),
            f"{region_name}_sh_edge_density": edge_density(Y_lin, mask),
        }
        # FFT系帯域パワー
        for band, pwr in bandpowers_cpd(Y_lin, mask, px_per_deg).items():
            feats[f"{region_name}_c_bandpow_{band}"] = pwr
        # 彩度
        muS, sdS = sat_mean_std(S, mask)
        feats[f"{region_name}_sa_mean"] = muS
        feats[f"{region_name}_sa_std"] = sdS
        out.update(feats)
    return out
