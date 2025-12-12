# features_pupil.py
# -*- coding: utf-8 -*-
import math
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis

# ------------------------------------------------------------
#  色空間・輝度
# ------------------------------------------------------------
def srgb_to_linear(u: np.ndarray) -> np.ndarray:
    """
    sRGB(0–1) -> 線形(0–1)
    """
    u = np.clip(u, 0.0, 1.0).astype(np.float32)
    a = 0.04045
    return np.where(u <= a, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)

def to_linear_luminance(img_bgr_uint8: np.ndarray) -> np.ndarray:
    """
    OpenCV BGR(uint8) -> 線形輝度 Y_lin (0–1)
    Rec.709/sRGB: Y = 0.2126R + 0.7152G + 0.0722B （R,G,Bは線形）
    """
    b = img_bgr_uint8.astype(np.float32) / 255.0
    b_lin = srgb_to_linear(b)
    R = b_lin[..., 2]
    G = b_lin[..., 1]
    B = b_lin[..., 0]
    Y_lin = 0.2126 * R + 0.7152 * G + 0.0722 * B
    return np.clip(Y_lin, 0.0, 1.0)

def to_nonlinear_gray(img_bgr_uint8: np.ndarray) -> np.ndarray:
    """
    OpenCVのグレイ変換（luma近似・非線形のまま） -> 0–1
    """
    gray = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return gray

def to_saturation(img_bgr_uint8: np.ndarray) -> np.ndarray:
    """
    BGR(uint8) -> HSVに準拠したS(0–1)。RGB最大値が0の画素は0。
    """
    img_rgb = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    Cmax = img_rgb.max(axis=2)
    Cmin = img_rgb.min(axis=2)
    delta = Cmax - Cmin
    S = np.zeros_like(Cmax, dtype=np.float32)
    mask = Cmax > 0
    S[mask] = delta[mask] / Cmax[mask]
    return np.clip(S, 0.0, 1.0)

# ------------------------------------------------------------
#  ユーティリティ
# ------------------------------------------------------------
def _vals(arr: np.ndarray, mask) -> np.ndarray:
    """
    マスクがあれば該当画素、なければ全画素を1次元で返す
    """
    return arr[mask].astype(np.float32) if mask is not None else arr.ravel().astype(np.float32)

# ------------------------------------------------------------
#  輝度・統計（bL_ / bnL_）
# ------------------------------------------------------------
def stat_mean(arr, mask):    v = _vals(arr, mask); return float(v.mean()) if v.size else np.nan
def stat_std(arr, mask):     v = _vals(arr, mask); return float(v.std()) if v.size else np.nan
def stat_min(arr, mask):     v = _vals(arr, mask); return float(v.min()) if v.size else np.nan
def stat_max(arr, mask):     v = _vals(arr, mask); return float(v.max()) if v.size else np.nan
def stat_median(arr, mask):  v = _vals(arr, mask); return float(np.median(v)) if v.size else np.nan
def stat_skew(arr, mask):    v = _vals(arr, mask); return float(skew(v)) if v.size else np.nan
def stat_kurt(arr, mask):    v = _vals(arr, mask); return float(kurtosis(v)) if v.size else np.nan

def stat_entropy_256(arr01, mask):
    """
    0–1 を 0–255 量子化してエントロピーを算出
    """
    v = (_vals(arr01, mask) * 255.0).clip(0, 255).astype(np.uint8)
    if v.size == 0: return np.nan
    hist, _ = np.histogram(v, bins=256, range=(0, 255))
    p = hist.astype(np.float64) / max(1, hist.sum())
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

# ------------------------------------------------------------
#  コントラスト（c_）
# ------------------------------------------------------------
def rms_contrast(arr01, mask):
    """
    μで正規化したRMS: std/mean
    """
    v = _vals(arr01, mask)
    if v.size == 0: return np.nan
    mu = v.mean()
    if mu == 0: return 0.0
    return float(v.std() / mu)

def local_rms_mean(arr01, mask, block=32):
    """
    ブロック毎のRMSコントラストを計算し、その平均を返す
    """
    a = arr01
    h, w = a.shape
    vals = []
    for y in range(0, h, block):
        for x in range(0, w, block):
            patch = a[y:y+block, x:x+block]
            if patch.size == 0: continue
            if mask is not None:
                pm = mask[y:y+block, x:x+block]
                pv = patch[pm]
                if pv.size == 0: continue
            else:
                pv = patch.ravel()
            mu = pv.mean()
            if mu > 0:
                vals.append(pv.std() / mu)
    return float(np.mean(vals)) if vals else np.nan

def glcm_prop(arr01, mask, prop):
    """
    GLCMプロパティ（contrast, dissimilarity, homogeneity, energy, correlation）
    """
    v8 = (arr01 * 255.0).clip(0, 255).astype(np.uint8)
    if mask is not None:
        v8 = np.where(mask, v8, np.median(v8)).astype(np.uint8)
    glcm = graycomatrix(v8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    return float(graycoprops(glcm, prop)[0, 0])

def bandpowers_cpd(arr01, mask, px_per_deg: float, bands=((0.25,0.5),(0.5,1),(1,2),(2,4),(4,8))):
    """
    2D-FFT の等方平均パワーを cycles/deg の帯域ごとに平均
    """
    a = arr01
    if mask is not None:
        a = np.where(mask, a, np.median(a))
    H, W = a.shape
    F = np.fft.fft2(a)
    A = np.abs(F) ** 2
    fy = np.fft.fftfreq(H) * H / px_per_deg  # cpd
    fx = np.fft.fftfreq(W) * W / px_per_deg  # cpd
    FY, FX = np.meshgrid(fy, fx, indexing='ij')
    FR = np.sqrt(FX**2 + FY**2)

    out = {}
    for (f1, f2) in bands:
        band_mask = (FR >= f1) & (FR < f2)
        power = float(A[band_mask].mean()) if np.any(band_mask) else np.nan
        out[f"{f1:.2f}-{f2:.2f}cpd"] = power
    return out

# ------------------------------------------------------------
#  シャープネス（sh_）
# ------------------------------------------------------------
def laplacian_var(arr01, mask):
    g = (arr01 * 255.0).astype(np.uint8)
    L = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    return float(L[mask].var()) if mask is not None else float(L.var())

def tenengrad(arr01, mask):
    g = (arr01 * 255.0).astype(np.uint8)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    M2 = gx**2 + gy**2
    return float(M2[mask].mean()) if mask is not None else float(M2.mean())

def grad_entropy(arr01, mask):
    g = (arr01 * 255.0).astype(np.uint8)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    m = mag[mask] if mask is not None else mag.ravel()
    if m.size == 0: return np.nan
    m = np.clip(m, 0, None)
    m = (m / (m.max() + 1e-6)) * 255.0
    hist, _ = np.histogram(m.astype(np.uint8), bins=256, range=(0, 255))
    p = hist.astype(np.float64) / max(1, hist.sum())
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def edge_density(arr01, mask):
    g = (arr01 * 255.0).astype(np.uint8)
    edges = cv2.Canny(g, 100, 200)
    if mask is not None:
        roi = edges[mask]
        return float(np.mean(roi > 0)) if roi.size else np.nan
    return float(np.mean(edges > 0))

def sharpness_histdiff(arr01, mask, sigma=1.0, bins=128):
    """
    Δ = 8近傍との平均絶対差。元画像とガウシアンぼかし後のΔの
    ヒストグラム差（L1距離/画素数）を返す。大きいほどシャープ。
    """
    def mean_abs_diff8_valid(a01):
        a = a01.astype(np.float32)
        H, W = a.shape
        out = np.zeros_like(a, dtype=np.float32)

        c = a[1:-1, 1:-1]
        acc = np.zeros_like(c, dtype=np.float32)

        acc += np.abs(c - a[0:-2, 1:-1])   # 上
        acc += np.abs(c - a[2:  , 1:-1])   # 下
        acc += np.abs(c - a[1:-1, 0:-2])   # 左
        acc += np.abs(c - a[1:-1, 2:  ])   # 右
        acc += np.abs(c - a[0:-2, 0:-2])   # 左上
        acc += np.abs(c - a[0:-2, 2:  ])   # 右上
        acc += np.abs(c - a[2:  , 0:-2])   # 左下
        acc += np.abs(c - a[2:  , 2:  ])   # 右下

        out[1:-1, 1:-1] = acc / 8.0
        return out

    delta   = mean_abs_diff8_valid(arr01)
    blurred = cv2.GaussianBlur(arr01.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    delta_b = mean_abs_diff8_valid(blurred)

    v1 = delta[mask] if mask is not None else delta.ravel()
    v2 = delta_b[mask] if mask is not None else delta_b.ravel()
    if v1.size == 0 or v2.size == 0:
        return np.nan

    h1, _ = np.histogram(v1, bins=bins, range=(0.0, 1.0), density=False)
    h2, _ = np.histogram(v2, bins=bins, range=(0.0, 1.0), density=False)

    sf = np.abs(h1 - h2).sum() / float(v1.size)
    return float(sf)

# ------------------------------------------------------------
#  彩度（sa_）
# ------------------------------------------------------------
def sat_mean_std(S01, mask):
    v = _vals(S01, mask)
    if v.size == 0: return np.nan, np.nan
    return float(v.mean()), float(v.std())

# ------------------------------------------------------------
#  1領域の特徴量（接頭辞は呼び出し側で付与）
# ------------------------------------------------------------
def compute_region_features(img_bgr_uint8: np.ndarray, mask, px_per_deg: float, include_color: bool = True) -> dict:
    feats = {}

    # 輝度：線形と非線形
    Y_lin = to_linear_luminance(img_bgr_uint8)   # 0–1
    Y_nl  = to_nonlinear_gray(img_bgr_uint8)     # 0–1

    # bL_（線形輝度）
    feats["bL_mean"]     = stat_mean(Y_lin, mask)
    feats["bL_std"]      = stat_std(Y_lin, mask)
    feats["bL_entropy"]  = stat_entropy_256(Y_lin, mask)
    feats["bL_min"]      = stat_min(Y_lin, mask)
    feats["bL_max"]      = stat_max(Y_lin, mask)
    feats["bL_median"]   = stat_median(Y_lin, mask)
    feats["bL_skewness"] = stat_skew(Y_lin, mask)
    feats["bL_kurtosis"] = stat_kurt(Y_lin, mask)

    # bnL_（非線形グレイスケール）
    feats["bnL_mean"]     = stat_mean(Y_nl, mask)
    feats["bnL_std"]      = stat_std(Y_nl, mask)
    feats["bnL_entropy"]  = stat_entropy_256(Y_nl, mask)
    feats["bnL_min"]      = stat_min(Y_nl, mask)
    feats["bnL_max"]      = stat_max(Y_nl, mask)
    feats["bnL_median"]   = stat_median(Y_nl, mask)
    feats["bnL_skewness"] = stat_skew(Y_nl, mask)
    feats["bnL_kurtosis"] = stat_kurt(Y_nl, mask)

    # コントラスト（c_）— 線形輝度ベース
    feats["c_rms_contrast"]        = rms_contrast(Y_lin, mask)
    feats["c_local_rms_mean"]      = local_rms_mean(Y_lin, mask)
    feats["c_glcm_contrast"]       = glcm_prop(Y_lin, mask, "contrast")
    feats["c_glcm_dissimilarity"]  = glcm_prop(Y_lin, mask, "dissimilarity")
    feats["c_glcm_homogeneity"]    = glcm_prop(Y_lin, mask, "homogeneity")
    feats["c_glcm_energy"]         = glcm_prop(Y_lin, mask, "energy")
    feats["c_glcm_correlation"]    = glcm_prop(Y_lin, mask, "correlation")

    # 帯域パワー（cpd）
    for band, power in bandpowers_cpd(Y_lin, mask, px_per_deg).items():
        feats[f"c_bandpow_{band}"] = power

    # シャープネス（sh_）
    feats["sh_laplacian_var"]  = laplacian_var(Y_lin, mask)
    feats["sh_tenengrad"]      = tenengrad(Y_lin, mask)
    feats["sh_grad_entropy"]   = grad_entropy(Y_lin, mask)
    feats["sh_edge_density"]   = edge_density(Y_lin, mask)
    feats["sh_sharpness_factor"] = sharpness_histdiff(Y_lin, mask, sigma=1.0, bins=128)

    # # 彩度（sa_）
    if include_color:
        S = to_saturation(img_bgr_uint8)  # 0–1
        muS, sdS = sat_mean_std(S, mask)
        feats["sa_saturation_mean"] = muS
        feats["sa_saturation_std"]  = sdS

    return feats

# ------------------------------------------------------------
#  マスク生成（中心/傍中心/周辺）・px/deg計算
# ------------------------------------------------------------
def make_masks(h: int, w: int,
               screen_w_mm: float,
               dist_mm: float,
               res_x: int,
               center_deg: float,
               parafovea_deg: float) -> dict:
    """
    幾何パラメータから度/pxを算出し、画像中心起点の円形マスクを生成
    """
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

def make_all_masks() -> dict:
    """
    画像全体を1領域として扱いたいときのヘルパー
    """
    return {"all": None}

def _px_per_deg_from_geometry(screen_w_mm: float, dist_mm: float, res_x: int) -> float:
    """
    横方向 px/deg を返す（FFTのcpd計算で使用）
    """
    fov_x_deg = math.degrees(2.0 * math.atan((screen_w_mm / 2.0) / dist_mm))
    deg_per_px = fov_x_deg / float(res_x)
    return 1.0 / deg_per_px

# ------------------------------------------------------------
#  画像全体/複数領域の一括特徴量
# ------------------------------------------------------------
def compute_features_for_image(
    img_bgr: np.ndarray,
    masks: dict,
    screen_w_mm: float,
    dist_mm: float,
    res_x: int,
    include_color_feats: bool = True,
    band_edges_cpd=((0.25,0.5),(0.5,1),(1,2),(2,4),(4,8)),
) -> dict:
    """
    画像(BGR)と任意のマスク群を受け取り、<region>_<feature> 形式で dict を返す。
    - masks には {"center": mask, "parafovea": mask, "periphery": mask} などを渡す。
    - 全体のみなら make_all_masks() を使うか {"all": None} を渡す。
    """
    px_per_deg = _px_per_deg_from_geometry(screen_w_mm, dist_mm, res_x)
    out = {}

    for region_name, mask in masks.items():
        feats = compute_region_features(
            img_bgr_uint8=img_bgr,
            mask=mask,
            px_per_deg=px_per_deg,
            include_color=include_color_feats,
        )

        # カスタム帯域が指定されているなら上書き
        if band_edges_cpd is not None:
            Y_lin = to_linear_luminance(img_bgr)  # 0–1
            bandpow = bandpowers_cpd(Y_lin, mask, px_per_deg, bands=band_edges_cpd)
            # 既存 c_bandpow_ を置換
            for k in list(feats.keys()):
                if k.startswith("c_bandpow_"):
                    feats.pop(k)
            for band, power in bandpow.items():
                feats[f"c_bandpow_{band}"] = power

        # 地域接頭辞を付与
        for k, v in feats.items():
            out[f"{region_name}_{k}"] = v

    return out
