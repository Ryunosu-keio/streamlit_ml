# app_keepmodel2.py
# ============================================================
# 2段モデルを残す版（全体統合・安定化）
# 1段目: 画像特徴 -> pupil (回帰)  or  shrink (分類)   ★ここを切替可能に
# 2段目: (param interaction + *_orig) -> 重要特徴量 (MultiOutput, GridSearch optional)
# 高速探索: 2段目で特徴推定→1段目で {瞳孔 or 縮瞳確率} 推定
# 最終選抜: 実画像で 画質(Q=SSIM/PSNR) + (optional) HF_ratio を計算して選抜
#
# 追加要件:
#  - 画質指標を SSIM(Y) / PSNR(Y) から選択
#  - A/B/C 表示:
#      A: 元画像
#      C: モデルが吐いた加工（best）
#      B: 「brightnessのみ」で A→B を作り、mean(Y) を C と一致させる
#      (BとCは平均輝度が同じ)
#    さらに A/B/C それぞれの予想縮瞳（Stage1予測: 回帰なら pupil、分類なら P(shrink)）を表示
#
# Fix/Improve:
#  - 学習結果を session_state に保持（切り替えで再学習しない）
#  - 画質はY(輝度)で計算（進捗バー日本語）
#  - HF_ratioはON/OFF切替 + 低解像度（downscale）で高速化
#  - パレート最適（目的 vs 画質Q）を散布図で可視化（front & knee）
#  - matplotlibの表示文字は英語のみ（文字化け回避）  ← ここは維持
#  - feasible_mask の長さズレを確実に防止（try/except + reset_index + 保険）
#  - 新画像の *_orig は fallback から借りない：
#      新画像（加工前）から特徴量を計算し、それを *_orig にマッピングして使う
# ============================================================

import warnings
warnings.simplefilter("ignore")

import hashlib
import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import cv2
from PIL import Image

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score, roc_auc_score

from xgboost import XGBRegressor, XGBClassifier

# ==== features_pupil / GPU 対応 =====================================
import warnings as _warnings

def cuda_available():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False

_warnings.filterwarnings(
    "ignore",
    message="CUDA path could not be detected.*",
    module="cupy.*",
)

if cuda_available():
    # import features_pupil_gpu as fp
    import features_pupil as fp
    USING_GPU = False  # 一旦無効化
    print("[INFO] Using GPU version (features_pupil_gpu)")
else:
    import features_pupil as fp
    USING_GPU = False
    print("[INFO] Using CPU version (features_pupil)")

# ==== 画面・観察距離など（features_pupil 用） =======================
SCREEN_W_MM = 260
DIST_MM     = 450
RES_X       = 6000
CENTER_DEG  = 2
PARAFOVEA_DEG = 5

ROI_REGIONS = ("center", "parafovea", "periphery")

ROI_AREA_WEIGHTS = {"center": 0.04, "parafovea": 0.20, "periphery": 0.76}
ROI_PUPIL_WEIGHTS = {"center": 0.5, "parafovea": 0.3, "periphery": 0.2}

NON_FEATURE_COLS = [
    "folder_name",
    "平均_変化率",
    "平均_変化量_z",
    "両眼.注視Z座標[mm]",
    "pattern_id",
    "param1", "param2", "param3",
    "param1_val", "param2_val", "param3_val",
]

# ============================================================
# Image processing ops
# ============================================================
def slide_brightness(image: Image.Image, shift: float) -> Image.Image:
    img = np.array(image).astype("float32") / 255.0
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + shift / 255.0, 0.0, 1.0)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(np.round(out * 255).astype("uint8"))

def adjust_contrast_adachi(image: Image.Image, scale: float) -> Image.Image:
    img = np.array(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = cv2.convertScaleAbs(hsv[:, :, 2], alpha=float(scale))
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(out.astype("uint8"))

def adjust_sharpness(image: Image.Image, sharpness: float) -> Image.Image:
    img = np.array(image)
    s = float(sharpness)
    kernel = np.array(
        [[-s, -s, -s],
         [-s, 1 + 8*s, -s],
         [-s, -s, -s]], dtype=np.float32
    )
    out = cv2.filter2D(img, -1, kernel)
    return Image.fromarray(np.clip(out, 0, 255).astype("uint8"))

def adjust_gamma(image: Image.Image, gamma: float) -> Image.Image:
    g = float(gamma)
    if g <= 0:
        return image
    lut = np.array([((i / 255.0) ** g) * 255 for i in range(256)]).astype("uint8")
    img = np.array(image.convert("RGB"))
    out = cv2.LUT(img, lut)
    return Image.fromarray(out)

def stretch_rgb_clahe(image: Image.Image, clipLimit: float = 2.0, tile: int = 8) -> Image.Image:
    img = np.array(image).astype("uint8")
    tile = int(max(4, min(64, tile)))
    clahe = cv2.createCLAHE(clipLimit=float(clipLimit), tileGridSize=(tile, tile))
    out = img.copy()
    for ch in range(3):
        out[:, :, ch] = clahe.apply(out[:, :, ch])
    return Image.fromarray(out)

def apply_one_op(image: Image.Image, op: str, val: float) -> Image.Image:
    if op == "brightness":
        return slide_brightness(image, shift=val)
    if op == "contrast":
        return adjust_contrast_adachi(image, scale=val)
    if op == "gamma":
        return adjust_gamma(image, gamma=val)
    if op == "sharpness":
        return adjust_sharpness(image, sharpness=val)
    if op == "equalization":
        tile = int(round(val))
        return stretch_rgb_clahe(image, clipLimit=2.0, tile=tile)
    return image

def apply_processing_sequence(image: Image.Image, ops, vals) -> Image.Image:
    out = image.copy()
    for op, v in zip(ops, vals):
        if op is None or op == "None":
            continue
        out = apply_one_op(out, str(op), float(v))
    return out

# ============================================================
# SSIM(Y) / PSNR(Y) + (optional) HF_ratio + mean(Y)
# ============================================================
def _to_y01(pil: Image.Image) -> np.ndarray:
    rgb = np.array(pil.convert("RGB")).astype("float32") / 255.0
    y = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    return y

def compute_ssim_y(img_ref: Image.Image, img_proc: Image.Image) -> float:
    ref = _to_y01(img_ref)
    proc = _to_y01(img_proc)

    K1, K2, L = 0.01, 0.03, 1.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    kernel = (11, 11)
    sigma = 1.5

    mu1 = cv2.GaussianBlur(ref, kernel, sigma)
    mu2 = cv2.GaussianBlur(proc, kernel, sigma)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(ref * ref, kernel, sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(proc * proc, kernel, sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(ref * proc, kernel, sigma) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(np.mean(ssim_map))

def compute_psnr_y(img_ref: Image.Image, img_proc: Image.Image) -> float:
    ref = _to_y01(img_ref)
    proc = _to_y01(img_proc)
    mse = float(np.mean((ref - proc) ** 2))
    if mse <= 1e-12:
        return float("inf")
    L = 1.0
    psnr = 10.0 * np.log10((L * L) / mse)
    return float(psnr)

def mean_y_255(pil: Image.Image) -> float:
    return float(np.mean(_to_y01(pil) * 255.0))

def compute_quality(img_ref: Image.Image, img_proc: Image.Image, metric_key: str) -> float:
    metric_key = (metric_key or "SSIM").upper()
    if metric_key == "PSNR":
        return compute_psnr_y(img_ref, img_proc)
    return compute_ssim_y(img_ref, img_proc)

def normalize_quality_to_01(q: float, metric_key: str, psnr_min: float, psnr_max: float) -> float:
    metric_key = (metric_key or "SSIM").upper()
    if metric_key == "SSIM":
        return float(np.clip(q, 0.0, 1.0))
    # PSNR -> 0..1
    lo = float(min(psnr_min, psnr_max - 1e-6))
    hi = float(max(psnr_max, lo + 1e-6))
    q01 = (float(q) - lo) / (hi - lo)
    return float(np.clip(q01, 0.0, 1.0))

def match_mean_y_with_brightness(
    img: Image.Image,
    target_mean_y_255: float,
    tol: float = 0.2,
    max_iter: int = 18,
    lo: float = -120.0,
    hi: float = 120.0,
):
    """
    Aにbrightness(=slide_brightness)だけをかけて mean(Y) を target に合わせる。
    二分探索で近づけ、(best_img, best_shift, best_mean, best_err) を返す。
    """
    base_mean = mean_y_255(img)
    if abs(base_mean - target_mean_y_255) <= tol:
        return img.copy(), 0.0, base_mean, abs(base_mean - target_mean_y_255)

    best_img = img.copy()
    best_shift = 0.0
    best_mean = base_mean
    best_err = abs(base_mean - target_mean_y_255)

    l, r = float(lo), float(hi)
    for _ in range(max_iter):
        mid = (l + r) / 2.0
        cand = slide_brightness(img, shift=mid)
        m = mean_y_255(cand)
        err = abs(m - target_mean_y_255)

        if err < best_err:
            best_err = err
            best_img = cand
            best_shift = mid
            best_mean = m

        if err <= tol:
            break

        if m < target_mean_y_255:
            l = mid
        else:
            r = mid

    return best_img, float(best_shift), float(best_mean), float(best_err)

def hf_ratio_laplacian(img_ref: Image.Image, img_proc: Image.Image, downscale: int = 4) -> float:
    downscale = int(max(1, downscale))

    ref = (_to_y01(img_ref) * 255.0).astype("uint8")
    proc = (_to_y01(img_proc) * 255.0).astype("uint8")

    if downscale > 1:
        h, w = ref.shape[:2]
        nh = max(32, h // downscale)
        nw = max(32, w // downscale)
        ref = cv2.resize(ref, (nw, nh), interpolation=cv2.INTER_AREA)
        proc = cv2.resize(proc, (nw, nh), interpolation=cv2.INTER_AREA)

    lap_ref = cv2.Laplacian(ref, cv2.CV_32F)
    lap_proc = cv2.Laplacian(proc, cv2.CV_32F)
    v_ref = float(np.var(lap_ref))
    v_proc = float(np.var(lap_proc))
    if v_ref < 1e-9:
        return float("inf") if v_proc > 0 else 1.0
    return v_proc / v_ref

# ============================================================
# Data loading / parsing
# ============================================================
@st.cache_data
def load_and_parse_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    def parse_params_ordered(name):
        if pd.isna(name):
            return {"param1": "None", "param1_val": 0.0,
                    "param2": "None", "param2_val": 0.0,
                    "param3": "None", "param3_val": 0.0}

        s = str(name)
        s = s.replace(".jpg", "").replace(".JPG", "").replace(".jpeg", "").replace(".JPEG", "")
        s = s.replace(".png", "").replace(".PNG", "")

        parts = s.split("_")
        valid_ops = ["brightness", "contrast", "gamma", "sharpness", "equalization"]
        params = []
        for part in parts:
            for op in valid_ops:
                if part.startswith(op):
                    try:
                        val = float(part.replace(op, ""))
                        params.append((op, val))
                    except ValueError:
                        pass
                    break
        while len(params) < 3:
            params.append(("None", 0.0))

        return {"param1": params[0][0], "param1_val": params[0][1],
                "param2": params[1][0], "param2_val": params[1][1],
                "param3": params[2][0], "param3_val": params[2][1]}

    df_full = df.copy()
    if "image_name" in df_full.columns:
        params_df = pd.DataFrame([parse_params_ordered(n) for n in df_full["image_name"]])
        params_df["pattern_id"] = params_df["param1"] + " → " + params_df["param2"] + " → " + params_df["param3"]

        for c in params_df.columns:
            if c in df_full.columns:
                df_full = df_full.drop(columns=[c])
        df_full = pd.concat([df_full, params_df], axis=1)
    else:
        if "pattern_id" not in df_full.columns:
            df_full["pattern_id"] = "no_pattern"

    return df_full

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    valid_ops = ["brightness", "contrast", "gamma", "sharpness", "equalization"]
    X = {}
    for i in range(1, 4):
        op_col = f"param{i}"
        val_col = f"param{i}_val"
        if op_col not in df.columns or val_col not in df.columns:
            continue
        for op in valid_ops:
            mask = (df[op_col] == op).astype(float)
            X[f"step{i}_{op}"] = mask * df[val_col].astype(float)
    return pd.DataFrame(X, index=df.index) if X else pd.DataFrame(index=df.index)

def compute_sample_weights(df: pd.DataFrame) -> pd.Series:
    key = df["pattern_id"].astype(str)
    freq = key.value_counts()
    w = 1.0 / key.map(freq).astype(float)
    w *= len(w) / w.sum()
    return w

# ============================================================
# Feature aggregation (ROI -> all_area / all_pupil)
# ============================================================
def make_weighted_globals_for_single(roi_feats: dict) -> dict:
    metric_map = {}
    for k, v in roi_feats.items():
        for r in ROI_REGIONS:
            prefix = r + "_"
            if k.startswith(prefix):
                feat_name = k[len(prefix):]
                metric_map.setdefault(feat_name, {})[r] = v
                break

    out = {}
    for feat_name, region_vals in metric_map.items():
        # area
        num, den = 0.0, 0.0
        for r, val in region_vals.items():
            w = ROI_AREA_WEIGHTS.get(r, 0.0)
            num += float(val) * w
            den += w
        out[f"all_area_{feat_name}"] = num / den if den > 0 else np.nan

        # pupil
        num, den = 0.0, 0.0
        for r, val in region_vals.items():
            w = ROI_PUPIL_WEIGHTS.get(r, 0.0)
            num += float(val) * w
            den += w
        out[f"all_pupil_{feat_name}"] = num / den if den > 0 else np.nan

    return out

# ============================================================
# New image feature computation (for *_orig, and before features)
# ============================================================
def compute_features_for_pil(pil_img: Image.Image) -> dict:
    img_rgb = pil_img.convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    roi_masks = fp.make_masks(h, w, SCREEN_W_MM, DIST_MM, RES_X, CENTER_DEG, PARAFOVEA_DEG)
    feats_roi = fp.compute_features_for_image(
        img_bgr, roi_masks, screen_w_mm=SCREEN_W_MM, dist_mm=DIST_MM, res_x=RES_X
    )

    all_masks = fp.make_all_masks()
    feats_all = fp.compute_features_for_image(
        img_bgr, all_masks, screen_w_mm=SCREEN_W_MM, dist_mm=DIST_MM, res_x=RES_X
    )

    feats_area_pupil = make_weighted_globals_for_single(feats_roi)
    feats = {**feats_roi, **feats_all, **feats_area_pupil}
    return feats

def build_orig_vector_from_new_image(
    new_feats: dict,
    orig_cols: list,
    X2_means: pd.Series,
    fallback_df: pd.DataFrame,
    fallback_idx,
) -> pd.Series:
    out = pd.Series(index=orig_cols, dtype=float)

    for c in orig_cols:
        v = np.nan
        if c.endswith("_orig"):
            base = c[:-5]
            if base in new_feats:
                v = new_feats[base]
        elif c.endswith("_orig_area"):
            base = c[:-9]
            if base in new_feats:
                v = new_feats[base]
        elif c.endswith("_orig_pupil"):
            base = c[:-10]
            if base in new_feats:
                v = new_feats[base]
        out[c] = v

    # 1) training mean
    out = out.fillna(X2_means.reindex(orig_cols))

    # 2) last resort: fallback row
    if out.isna().any():
        for c in orig_cols:
            if pd.isna(out[c]) and c in fallback_df.columns:
                out[c] = fallback_df.loc[fallback_idx, c]

    # 3) still NaN -> 0
    out = out.fillna(0.0)
    return out

def image_basic_stats(pil_img: Image.Image) -> pd.DataFrame:
    rgb = np.array(pil_img.convert("RGB")).astype(np.float32)
    y = (_to_y01(pil_img) * 255.0).astype(np.float32)

    rows = []
    for name, arr in [
        ("R", rgb[:, :, 0]),
        ("G", rgb[:, :, 1]),
        ("B", rgb[:, :, 2]),
        ("Y", y),
    ]:
        rows.append({
            "channel": name,
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        })
    return pd.DataFrame(rows)

def build_x_from_feats(feats: dict, selected: list, img_feature_means: pd.Series) -> pd.Series:
    x = pd.Series(index=selected, dtype=float)
    miss = []
    for f in selected:
        if f in feats:
            x[f] = float(feats[f])
        else:
            x[f] = np.nan
            miss.append(f)
    if miss:
        st.warning(f"特徴量欠損: {miss} → 学習データ平均で補完します。")
    return x.fillna(img_feature_means.reindex(selected)).fillna(0.0)

def predict_stage1_from_x(m1, x: pd.Series, stage1_task: str) -> dict:
    X = x.values.reshape(1, -1)
    if stage1_task == "reg":
        return {"pupil": float(m1.predict(X)[0]), "p_shrink": np.nan}
    else:
        return {"pupil": np.nan, "p_shrink": float(m1.predict_proba(X)[:, 1][0])}

# ============================================================
# 18 patterns / param ranges
# ============================================================
def generate_allowed_patterns():
    ops = ["brightness", "contrast", "gamma", "sharpness", "equalization"]
    patterns = []
    for p1 in ops:
        for p2 in ops:
            for p3 in ops:
                pat = [p1, p2, p3]
                if len(set(pat)) < 3:
                    continue
                if "brightness" in pat and p1 != "brightness":
                    continue
                if "equalization" in pat and p3 != "equalization":
                    continue
                if "brightness" in pat and "equalization" in pat:
                    continue
                patterns.append(f"{p1}_{p2}_{p3}")
    return patterns

def get_param_range(df, step, op, q_low=0.1, q_high=0.9):
    col_op = f"param{step}"
    col_val = f"param{step}_val"

    defaults = {
        "gamma": (0.7, 1.3),
        "contrast": (0.7, 1.3),
        "sharpness": (0.0, 1.5),
        "brightness": (-80.0, 80.0),
        "equalization": (5.0, 40.0),
    }
    vmin, vmax = defaults.get(op, (0.0, 1.0))

    if col_op in df.columns and col_val in df.columns:
        mask = df[col_op] == op
        if mask.any():
            v = df.loc[mask, col_val].astype(float)
            vmin = float(v.quantile(q_low))
            vmax = float(v.quantile(q_high))
            if vmin == vmax:
                vmin -= abs(vmin) * 0.1 + 1e-3
                vmax += abs(vmax) * 0.1 + 1e-3

    dvmin, dvmax = defaults.get(op, (vmin, vmax))
    vmin = max(vmin, dvmin)
    vmax = min(vmax, dvmax)
    return float(vmin), float(vmax)

# ============================================================
# Stage1 ラベル作成（分類用）
# ============================================================
def make_y_class(df: pd.DataFrame, pupil_col: str, group_col: str | None, mode: str = "within_subject_median") -> pd.Series:
    """
    二値ラベル（1=縮瞳側, 0=非縮瞳側）を作る。
    mode:
      - within_subject_median: 各被験者(folder_nameなど)内の中央値より小さい=1
      - global_median: 全体中央値より小さい=1
    """
    y = df[pupil_col].astype(float)
    if mode == "within_subject_median" and group_col is not None and group_col in df.columns:
        med = df.groupby(group_col)[pupil_col].transform("median").astype(float)
        return (y <= med).astype(int)
    else:
        return (y <= float(np.nanmedian(y.values))).astype(int)

def safe_auc(y_true, y_prob) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return np.nan

# ============================================================
# Models & grid search
# ============================================================
RF_PARAM_GRID_STAGE1 = {
    "n_estimators": [50, 75, 100, 125, 150],
    "max_depth": [5, 10, 15],
    "min_samples_leaf": [1, 3, 5],
}
XGB_PARAM_GRID_STAGE1 = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.03, 0.06],
}

RF_PARAM_GRID_STAGE2 = {
    "n_estimators": [150, 300],
    "max_depth": [None, 8],
    "min_samples_leaf": [1, 3],
}
XGB_PARAM_GRID_STAGE2 = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.03, 0.06],
}

def iter_param_grid(param_grid: dict):
    keys = list(param_grid.keys())
    if not keys:
        yield {}
        return
    from itertools import product
    for values in product(*[param_grid[k] for k in keys]):
        yield dict(zip(keys, values))

def create_stage1_model(model_type: str, params: dict, task: str):
    """
    task: 'reg' or 'clf'
    """
    params = params or {}
    if task == "reg":
        if model_type == "RandomForest":
            base = {"n_estimators": 300, "random_state": 42, "n_jobs": -1}
            base.update(params)
            return RandomForestRegressor(**base)
        else:
            base = {"objective": "reg:squarederror", "random_state": 42, "n_jobs": -1}
            base.update(params)
            return XGBRegressor(**base)
    else:
        if model_type == "RandomForest":
            base = {"n_estimators": 400, "random_state": 42, "n_jobs": -1, "class_weight": "balanced"}
            base.update(params)
            return RandomForestClassifier(**base)
        else:
            base = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "random_state": 42,
                "n_jobs": -1,
            }
            base.update(params)
            return XGBClassifier(**base)

def create_stage2_base(model_type: str, params: dict):
    if model_type == "RandomForest":
        base = {"n_estimators": 300, "random_state": 42, "n_jobs": -1}
        base.update(params or {})
        return RandomForestRegressor(**base)
    else:
        base = {"objective": "reg:squarederror", "random_state": 42, "n_jobs": -1}
        base.update(params or {})
        return XGBRegressor(**base)

def _get_splitter(groups):
    if groups is not None:
        splitter = GroupKFold(n_splits=5)
        return splitter, True
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    return splitter, False

def score_stage1(task: str, y_true, y_pred_or_prob) -> float:
    if task == "reg":
        return float(r2_score(y_true, y_pred_or_prob))
    else:
        return safe_auc(y_true, y_pred_or_prob)

def grid_search_stage1(X, y, w, groups, model_type, task: str):
    param_grid = RF_PARAM_GRID_STAGE1 if model_type == "RandomForest" else XGB_PARAM_GRID_STAGE1
    total = int(np.prod([len(v) for v in param_grid.values()])) if param_grid else 1
    prog = st.progress(0.0, text=f"Stage1 グリッドサーチ... (0/{total})")

    splitter, is_group = _get_splitter(groups)
    best_score, best_params = -1e18, None
    best_train, best_test = None, None

    done = 0
    for params in iter_param_grid(param_grid):
        tr_scores, te_scores = [], []
        split_iter = splitter.split(X, y, groups) if is_group else splitter.split(X, y)
        for tr_idx, te_idx in split_iter:
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
            w_tr = w.iloc[tr_idx]

            m = create_stage1_model(model_type, params, task=task)
            if task == "reg":
                m.fit(X_tr, y_tr, sample_weight=w_tr)
                tr_pred = m.predict(X_tr)
                te_pred = m.predict(X_te)
                tr_scores.append(r2_score(y_tr, tr_pred))
                te_scores.append(r2_score(y_te, te_pred))
            else:
                m.fit(X_tr, y_tr, sample_weight=w_tr)
                tr_prob = m.predict_proba(X_tr)[:, 1]
                te_prob = m.predict_proba(X_te)[:, 1]
                tr_scores.append(score_stage1(task, y_tr, tr_prob))
                te_scores.append(score_stage1(task, y_te, te_prob))

        mean_te = float(np.nanmean(te_scores))
        if mean_te > best_score:
            best_score, best_params = mean_te, params
            best_train, best_test = tr_scores, te_scores

        done += 1
        prog.progress(done / total, text=f"Stage1 グリッドサーチ... ({done}/{total})")

    final = create_stage1_model(model_type, best_params, task=task)
    final.fit(X, y, sample_weight=w)
    prog.progress(1.0, text="Stage1 グリッドサーチ完了")

    cv = {"mean_train": float(np.nanmean(best_train)), "std_train": float(np.nanstd(best_train)),
          "mean_test": float(np.nanmean(best_test)), "std_test": float(np.nanstd(best_test))}
    return final, best_params, cv

def train_stage1_fixed_params(X, y, w, groups, model_type, params, task: str):
    splitter, is_group = _get_splitter(groups)
    splits = list(splitter.split(X, y, groups)) if is_group else list(splitter.split(X, y))
    prog = st.progress(0.0, text="Stage1 学習中...")

    tr_scores, te_scores = [], []
    for i, (tr_idx, te_idx) in enumerate(splits):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        w_tr = w.iloc[tr_idx]

        m = create_stage1_model(model_type, params or {}, task=task)
        m.fit(X_tr, y_tr, sample_weight=w_tr)

        if task == "reg":
            tr_scores.append(r2_score(y_tr, m.predict(X_tr)))
            te_scores.append(r2_score(y_te, m.predict(X_te)))
        else:
            tr_scores.append(score_stage1(task, y_tr, m.predict_proba(X_tr)[:, 1]))
            te_scores.append(score_stage1(task, y_te, m.predict_proba(X_te)[:, 1]))

        prog.progress((i + 1) / len(splits), text=f"Stage1 学習中... ({i+1}/{len(splits)})")

    final = create_stage1_model(model_type, params or {}, task=task)
    final.fit(X, y, sample_weight=w)
    prog.progress(1.0, text="Stage1 学習完了")

    cv = {"mean_train": float(np.nanmean(tr_scores)), "std_train": float(np.nanstd(tr_scores)),
          "mean_test": float(np.nanmean(te_scores)), "std_test": float(np.nanstd(te_scores))}
    return final, cv

def grid_search_stage2(X2, Y2, w, groups, model_type):
    param_grid = RF_PARAM_GRID_STAGE2 if model_type == "RandomForest" else XGB_PARAM_GRID_STAGE2
    total = int(np.prod([len(v) for v in param_grid.values()])) if param_grid else 1
    prog = st.progress(0.0, text=f"Stage2 グリッドサーチ... (0/{total})")

    splitter, is_group = _get_splitter(groups)
    best_score, best_params = -1e18, None
    best_Yte_all, best_pred_all = None, None

    done = 0
    for params in iter_param_grid(param_grid):
        cv_scores, Yte_list, Ypred_list = [], [], []
        split_iter = splitter.split(X2, Y2, groups) if is_group else splitter.split(X2, Y2)
        for tr_idx, te_idx in split_iter:
            X_tr, X_te = X2.iloc[tr_idx], X2.iloc[te_idx]
            Y_tr, Y_te = Y2.iloc[tr_idx], Y2.iloc[te_idx]
            w_tr = w.iloc[tr_idx]

            base = create_stage2_base(model_type, params)
            mo = MultiOutputRegressor(base)
            mo.fit(X_tr, Y_tr, sample_weight=w_tr)

            Y_pred = mo.predict(X_te)
            cv_scores.append(r2_score(Y_te, Y_pred, multioutput="uniform_average"))
            Yte_list.append(Y_te)
            Ypred_list.append(Y_pred)

        mean_cv = float(np.mean(cv_scores))
        if mean_cv > best_score:
            best_score, best_params = mean_cv, params
            best_Yte_all = pd.concat(Yte_list, axis=0)
            best_pred_all = np.vstack(Ypred_list)

        done += 1
        prog.progress(done / total, text=f"Stage2 グリッドサーチ... ({done}/{total})")

    base_final = create_stage2_base(model_type, best_params)
    mo2 = MultiOutputRegressor(base_final)
    mo2.fit(X2, Y2, sample_weight=w)

    r2_each = r2_score(best_Yte_all, best_pred_all, multioutput="raw_values")
    prog.progress(1.0, text="Stage2 グリッドサーチ完了")
    return mo2, best_params, r2_each, best_score

def train_stage2_simple(X2, Y2, w, groups, model_type):
    splitter, is_group = _get_splitter(groups)
    splits = list(splitter.split(X2, Y2, groups)) if is_group else list(splitter.split(X2, Y2))
    prog = st.progress(0.0, text="Stage2 学習中...")

    cv_scores, Yte_list, Ypred_list = [], [], []
    for i, (tr_idx, te_idx) in enumerate(splits):
        X_tr, X_te = X2.iloc[tr_idx], X2.iloc[te_idx]
        Y_tr, Y_te = Y2.iloc[tr_idx], Y2.iloc[te_idx]
        w_tr = w.iloc[tr_idx]

        base = create_stage2_base(model_type, {})
        mo = MultiOutputRegressor(base)
        mo.fit(X_tr, Y_tr, sample_weight=w_tr)

        Y_pred = mo.predict(X_te)
        cv_scores.append(r2_score(Y_te, Y_pred, multioutput="uniform_average"))
        Yte_list.append(Y_te)
        Ypred_list.append(Y_pred)

        prog.progress((i + 1) / len(splits), text=f"Stage2 学習中... ({i+1}/{len(splits)})")

    Yte_all = pd.concat(Yte_list, axis=0)
    Ypred_all = np.vstack(Ypred_list)
    r2_each = r2_score(Yte_all, Ypred_all, multioutput="raw_values")

    base_final = create_stage2_base(model_type, {})
    mo2 = MultiOutputRegressor(base_final)
    mo2.fit(X2, Y2, sample_weight=w)

    prog.progress(1.0, text="Stage2 学習完了")
    return mo2, {}, r2_each, float(np.mean(cv_scores))

# ============================================================
# Pareto / knee
#  - maximize_x: 画質Qは常に↑
#  - yは task により (Pupil:↓) or (P(shrink):↑)
# ============================================================
def pareto_front_mask(df: pd.DataFrame, x_col: str, y_col: str, maximize_x=True, maximize_y=True) -> np.ndarray:
    X = df[x_col].values
    Y = df[y_col].values
    keep = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if not keep[i]:
            continue
        for j in range(len(df)):
            if i == j or not keep[j]:
                continue
            better_x = (X[j] >= X[i]) if maximize_x else (X[j] <= X[i])
            better_y = (Y[j] >= Y[i]) if maximize_y else (Y[j] <= Y[i])
            strict = ((X[j] > X[i]) if maximize_x else (X[j] < X[i])) or ((Y[j] > Y[i]) if maximize_y else (Y[j] < Y[i]))
            if better_x and better_y and strict:
                keep[i] = False
                break
    return keep

def knee_point_on_front(front: pd.DataFrame, x_col: str, y_col: str, maximize_y: bool):
    f = front.sort_values(x_col, ascending=True).reset_index(drop=True)
    x = f[x_col].values.astype(float)
    y = f[y_col].values.astype(float)

    xn = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-9)
    yn = (y - np.nanmin(y)) / (np.nanmax(y) - np.nanmin(y) + 1e-9)

    if maximize_y:
        d = np.sqrt((1 - xn) ** 2 + (1 - yn) ** 2)
    else:
        d = np.sqrt((1 - xn) ** 2 + (yn - 0) ** 2)

    idx = int(np.nanargmin(d))
    return f.iloc[idx].copy()

# ============================================================
# session_state utilities
# ============================================================
def df_fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(str(df.shape).encode("utf-8"))
    h.update(("|".join(df.columns)).encode("utf-8"))
    sample = df.select_dtypes(include=[np.number]).head(50).fillna(0).values.tobytes()
    h.update(sample)
    return h.hexdigest()[:16]

def get_state():
    if "trained" not in st.session_state:
        st.session_state.trained = {}
    return st.session_state.trained

# ============================================================
# main
# ============================================================
def main():
    st.set_page_config(page_title="画像加工レコメンダ（Model2）", layout="wide")

    st.markdown("""
    <style>
      html, body, [class*="css"] { font-size: 18px !important; }
      h1, h2, h3 { font-size: 1.25em !important; }
    </style>
    """, unsafe_allow_html=True)

    # matplotlib は英語固定（文字化け回避）
    plt.rcParams["font.family"] = "DejaVu Sans"

    st.title("🧪 画像特徴 →（回帰/分類）→ 画像加工レコメンダ（Model2）")
    st.caption(f"features_pupil バックエンド: {'GPU' if USING_GPU else 'CPU'}")

    # ---------- Data ----------
    st.sidebar.header("📁 入力データ")
    uploaded_file = st.sidebar.file_uploader("実験データ（CSV / Excel）", type=["csv", "xlsx", "xls"])
    if uploaded_file is None:
        st.info("左のサイドバーからデータセットをアップロードしてください。")
        return

    df_full = load_and_parse_data(uploaded_file)
    fp_df = df_fingerprint(df_full)

    # Exclude subjects
    if "folder_name" in df_full.columns:
        all_subjects = sorted(df_full["folder_name"].dropna().unique().tolist())
        excluded = st.sidebar.multiselect("除外する folder_name", options=all_subjects)
        if excluded:
            df_full = df_full[~df_full["folder_name"].isin(excluded)].copy()

    # CV setting
    st.sidebar.subheader("🧪 CV 設定")
    use_group_default = ("folder_name" in df_full.columns)
    use_group = st.sidebar.checkbox("GroupKFold を使う", value=use_group_default)

    groups = None
    group_col = None
    if use_group:
        cand = []
        for c in df_full.columns:
            nunique = df_full[c].nunique(dropna=True)
            if 1 < nunique < len(df_full):
                cand.append(c)
        if cand:
            default = cand.index("folder_name") if "folder_name" in cand else 0
            group_col = st.sidebar.selectbox("グループ列", options=cand, index=default)
            groups = df_full[group_col]
        else:
            st.sidebar.warning("有効なグループ列が見つかりません。KFold に切り替えます。")
            groups = None
            group_col = None

    sample_weights = compute_sample_weights(df_full)

    tab1, tab2 = st.tabs(["📊 データ概要", "🧬 推奨（Model2）"])

    with tab1:
        st.subheader("データセット概要")
        st.write(f"行数: **{len(df_full)}**")
        st.dataframe(df_full.head(), use_container_width=True)

    with tab2:
        st.header("🧬 推奨（Model2）")

        num_cols = df_full.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            st.error("数値列が見つかりません。")
            return

        default_pupil = "corrected_pupil" if "corrected_pupil" in num_cols else num_cols[0]
        pupil_col = st.selectbox("目的変数（瞳孔列）", options=num_cols, index=num_cols.index(default_pupil))

        # ★ Stage1 を回帰/分類で切替
        st.markdown("### 🧠 Stage1 のタスク")
        stage1_task_label = st.radio(
            "縮瞳モデルをどう扱う？",
            ["回帰（瞳孔を直接予測）", "分類（縮瞳側かどうかを予測）"],
            index=0,
            horizontal=True
        )
        stage1_task = "reg" if "回帰" in stage1_task_label else "clf"

        # 分類時のラベル設定
        if stage1_task == "clf":
            st.caption("分類は「縮瞳側=1 / 非縮瞳側=0」の二値を作って学習します（被験者内中央値が推奨）。")
            y_mode = st.radio(
                "二値ラベルの作り方",
                ["被験者内中央値（推奨）", "全体中央値"],
                index=0,
                horizontal=True
            )
            y_class_mode = "within_subject_median" if "被験者内" in y_mode else "global_median"
        else:
            y_class_mode = None

        dir_choice = st.radio(
            "望ましい方向（z の符号付けに使用）",
            ["小さいほど良い（縮瞳）", "大きいほど良い（散瞳）"],
            index=0,
            horizontal=True
        )
        sign_dir = -1.0 if "小さい" in dir_choice else 1.0

        feat_choice = st.radio(
            "特徴量グループ",
            ["全体（all）", "領域重み（all_area）", "瞳孔重み（all_pupil）", "ROI別（center/parafovea/periphery）"],
            index=0,
            horizontal=True
        )
        feat_group = {
            "全体（all）": "all",
            "領域重み（all_area）": "all_area",
            "瞳孔重み（all_pupil）": "all_pupil",
            "ROI別（center/parafovea/periphery）": "ROI",
        }[feat_choice]

        # Candidate columns
        if feat_group == "all":
            candidate_cols = [c for c in num_cols if c.startswith("all_")
                              and not c.startswith("all_area_") and not c.startswith("all_pupil_")
                              and not c.endswith("_orig") and c not in NON_FEATURE_COLS and c != pupil_col]
        elif feat_group == "all_area":
            candidate_cols = [c for c in num_cols if c.startswith("all_area_")
                              and not c.endswith("_orig") and c not in NON_FEATURE_COLS and c != pupil_col]
        elif feat_group == "all_pupil":
            candidate_cols = [c for c in num_cols if c.startswith("all_pupil_")
                              and not c.endswith("_orig") and c not in NON_FEATURE_COLS and c != pupil_col]
        else:
            candidate_cols = [c for c in num_cols if (c.startswith("center_") or c.startswith("parafovea_") or c.startswith("periphery_"))
                              and "_orig" not in c and c not in NON_FEATURE_COLS and c != pupil_col]

        if not candidate_cols:
            st.error("候補特徴量が見つかりません。")
            return

        st.caption(f"候補特徴量数: {len(candidate_cols)}")

        # ---- top_k slider robust ----
        max_k = min(30, len(candidate_cols))
        min_k = 3 if max_k >= 3 else 1
        if max_k <= min_k:
            top_k = int(max_k)
            st.info(f"top_k を {top_k} に固定しました（候補が少ないためスライダーを出せません）。")
        else:
            default_k = min(10, max_k)
            top_k = st.slider("Top-k（z の計算に使用）", min_k, max_k, default_k)

        n_trials_per_pattern = st.slider("パターンごとの試行回数（高速探索）", 200, 5000, 1000, 200)

        m1_label = st.radio("モデル1（特徴量 → 出力）", ["ランダムフォレスト", "XGBoost"], index=0, horizontal=True)
        m2_label = st.radio("モデル2（加工パラメータ+orig → 特徴量）", ["ランダムフォレスト", "XGBoost"], index=0, horizontal=True)
        model1_type = "RandomForest" if m1_label == "ランダムフォレスト" else "XGBoost"
        model2_type = "RandomForest" if m2_label == "ランダムフォレスト" else "XGBoost"

        use_grid1 = st.checkbox("Stage1 GridSearch（ハイパラ探索）", value=True)
        use_grid2 = st.checkbox("Stage2 GridSearch（ハイパラ探索）", value=True)

        # 目的関数
        if stage1_task == "reg":
            obj_choice = st.radio(
                "高速探索の目的関数",
                ["瞳孔を最小化（推奨）", "z を最大化（従来）"],
                index=0,
                horizontal=True
            )
            objective_mode = "pupil" if "瞳孔" in obj_choice else "z"
        else:
            obj_choice = st.radio(
                "高速探索の目的関数（分類）",
                ["縮瞳確率を最大化（推奨）", "z を最大化（従来）"],
                index=0,
                horizontal=True
            )
            objective_mode = "pshrink" if "縮瞳確率" in obj_choice else "z"

        st.markdown("### 🎛 品質評価（上位候補のみ実画像で評価）")

        quality_metric = st.radio(
            "画質指標（A vs 加工後で計算）",
            ["SSIM(Y)", "PSNR(Y)"],
            index=0,
            horizontal=True
        )
        quality_metric_key = "SSIM" if "SSIM" in quality_metric else "PSNR"

        # PSNR 正規化レンジ（Jに使う）
        if quality_metric_key == "PSNR":
            psnr_norm_min = st.slider("PSNR 正規化下限（J用）", 0.0, 40.0, 20.0, 0.5)
            psnr_norm_max = st.slider("PSNR 正規化上限（J用）", 20.0, 80.0, 60.0, 0.5)
            if psnr_norm_max <= psnr_norm_min:
                psnr_norm_max = psnr_norm_min + 1.0
        else:
            psnr_norm_min, psnr_norm_max = 20.0, 60.0

        hf_enabled = st.checkbox("HF_ratio を有効化（任意）", value=False)
        hf_downscale = st.slider("HF 計算の縮小率（downscale）", 1, 8, 4, 1) if hf_enabled else 4

        qm_choice = st.radio(
            "最終選抜モード",
            ["パレート（目的 vs 画質Q）", "制約（Q>=閾値 & HF<=閾値）", "合成スコア J"],
            index=0,
        )
        quality_mode = {"制約": "constraint", "合成": "composite"}.get(qm_choice[:2], "pareto")

        # 閾値（SSIM/PSNRで切替）
        if quality_metric_key == "SSIM":
            q_th = st.slider("SSIM(Y) の閾値", 0.5, 1.0, 0.7, 0.01)
        else:
            q_th = st.slider("PSNR(Y) の閾値 [dB]", 10.0, 60.0, 25.0, 0.5)

        hf_th = st.slider("HF_ratio の上限（1.0=同等、増えるほど高周波が増加）", 1.0, 10.0, 2.0, 0.1)
        max_candidates_for_quality = st.slider("画質/HF を評価する候補数", 100, 5000, 1000, 100)

        # mean(Y) 制約（Aの方針）
        st.markdown("#### 平均輝度（mean Y）制約（A）")
        luma_mode = st.radio(
            "mean(Y) をどれくらい固定する？",
            ["固定（推奨：ほぼ変えない）", "許容（±eps）", "無視"],
            index=0,
            horizontal=True
        )
        if "固定" in luma_mode:
            # 「固定」は現実的にeps=1.0くらいを内蔵扱い
            eps_y = 1.0
            st.caption("固定は実質 |ΔmeanY| <= 1.0（Yは0-255スケール）として扱います。")
        elif "許容" in luma_mode:
            eps_y = st.slider("許容幅 eps（|ΔmeanY| <= eps）", 0.0, 10.0, 1.0, 0.1)
        else:
            eps_y = 1e9

        alpha = st.number_input("alpha", value=1.0, step=0.1)
        beta  = st.number_input("beta（画質ペナルティ）", value=1.0, step=0.1)
        gamma = st.number_input("gamma（HF ペナルティ）", value=0.5, step=0.1)
        lambda_luma = st.number_input("lambda（meanY ペナルティ）", value=1.0, step=0.1)

        # Image input
        st.subheader("新規画像入力（品質評価 / A-B-C 表示用）")
        if st.button("🧹 画像アップロードをクリア"):
            st.session_state["new_img_key"] = str(np.random.randint(0, 10**9))

        if "new_img_key" not in st.session_state:
            st.session_state["new_img_key"] = "new_img"

        new_image_file = st.file_uploader(
            "新規画像（jpg / jpeg / png）",
            type=["jpg", "jpeg", "png"],
            key=st.session_state["new_img_key"]
        )

        st.caption("画像がない場合、画質評価（Q/HF/meanY）はスキップし、高速探索の目的関数だけで選びます。")
        fallback_idx = st.selectbox("フォールバック行（画像なし/特徴量欠損時のみ使用）", options=df_full.index)

        # -------- Training button --------
        state = get_state()

        def train_key():
            return (fp_df, pupil_col, feat_group, top_k, model1_type, model2_type,
                    use_grid1, use_grid2, bool(groups is not None), stage1_task, y_class_mode)

        if st.button("🚀 学習（Model1 & Model2）"):
            key = train_key()
            with st.spinner("学習中..."):
                # ---- Stage1 ----
                X_all = df_full[candidate_cols].copy()

                if stage1_task == "reg":
                    y1 = df_full[pupil_col].copy()
                else:
                    y1 = make_y_class(df_full, pupil_col=pupil_col, group_col=group_col, mode=y_class_mode)

                if use_grid1:
                    m1_full, best_p1, cv1_full = grid_search_stage1(
                        X_all, y1, sample_weights, groups, model1_type, task=stage1_task
                    )
                else:
                    best_p1 = {}
                    m1_full, cv1_full = train_stage1_fixed_params(
                        X_all, y1, sample_weights, groups, model1_type, best_p1, task=stage1_task
                    )

                imp = getattr(m1_full, "feature_importances_", None)
                if imp is None:
                    imp = np.ones(len(candidate_cols), dtype=float)

                imp_df = pd.DataFrame({"feature": candidate_cols, "importance": imp}).sort_values(
                    "importance", ascending=False
                ).reset_index(drop=True)
                selected = imp_df["feature"].head(top_k).tolist()

                # retrain on selected
                X_sel = df_full[selected].copy()
                m1_sel, cv1_sel = train_stage1_fixed_params(
                    X_sel, y1, sample_weights, groups, model1_type, best_p1, task=stage1_task
                )

                # z weights: importance * sign(corr)
                imp_sel = getattr(m1_sel, "feature_importances_", np.ones(len(selected), dtype=float))
                signs = []
                y_for_corr = df_full[pupil_col].astype(float)  # zは元の連続瞳孔で符号付け（分類でも同じ）
                for f in selected:
                    r = df_full[f].corr(y_for_corr)
                    s = 0.0 if (pd.isna(r) or r == 0) else sign_dir * float(np.sign(r))
                    signs.append(s)
                signs = np.array(signs)
                w_raw = imp_sel * signs
                if np.sum(np.abs(w_raw)) > 0:
                    thr = 0.01 * np.max(np.abs(w_raw))
                    w_raw[np.abs(w_raw) < thr] = 0.0
                    if np.sum(np.abs(w_raw)) > 0:
                        w_raw = w_raw / np.sum(np.abs(w_raw))
                z_w = pd.Series(w_raw, index=selected)

                feat_mean = df_full[selected].mean()
                feat_std = df_full[selected].std().replace(0, 1.0)
                img_feature_means = df_full[selected].mean()

                # ---- Stage2 ----
                X_param = create_interaction_features(df_full)
                orig_cols = [c for c in df_full.columns if c.endswith("_orig") or c.endswith("_orig_area") or c.endswith("_orig_pupil")]
                X_orig = df_full[orig_cols].copy() if orig_cols else pd.DataFrame(index=df_full.index)
                X2 = pd.concat([X_param, X_orig], axis=1) if not X_orig.empty else X_param.copy()

                if X2.empty:
                    st.error("Stage2 の入力が空です（param / _orig が見つかりません）。")
                    st.stop()

                Y2 = df_full[selected].copy()

                if use_grid2:
                    m2, best_p2, r2_each2, r2_mean2 = grid_search_stage2(
                        X2, Y2, sample_weights, groups, model2_type
                    )
                else:
                    m2, best_p2, r2_each2, r2_mean2 = train_stage2_simple(
                        X2, Y2, sample_weights, groups, model2_type
                    )

                state[key] = {
                    "candidate_cols": candidate_cols,
                    "selected": selected,
                    "m1_full": m1_full,
                    "m1_sel": m1_sel,
                    "cv1_full": cv1_full,
                    "cv1_sel": cv1_sel,
                    "best_p1": best_p1,
                    "imp_df": imp_df,
                    "z_w": z_w,
                    "feat_mean": feat_mean,
                    "feat_std": feat_std,
                    "img_feature_means": img_feature_means,
                    "m2": m2,
                    "best_p2": best_p2,
                    "r2_each2": r2_each2,
                    "r2_mean2": r2_mean2,
                    "X2_cols": X2.columns.tolist(),
                    "X2_means": X2.mean(),
                    "orig_cols": orig_cols,
                    "stage1_task": stage1_task,
                    "y_class_mode": y_class_mode,
                    "group_col": group_col,
                }

            st.success("学習が完了しました（結果は session_state に保持されます）。")

        # -------- Show trained results --------
        key = train_key()
        trained = state.get(key)

        if trained is None:
            st.info("先に「学習（Model1 & Model2）」を押してください。")
            return

        st.subheader("Stage1 重要度（全特徴）")
        st.dataframe(trained["imp_df"].head(30), use_container_width=True)

        st.subheader("Stage1 CV")
        cv1f = trained["cv1_full"]
        cv1s = trained["cv1_sel"]
        if trained["stage1_task"] == "reg":
            st.write(f"全特徴の Test R2: **{cv1f['mean_test']:.3f} ± {cv1f['std_test']:.3f}**")
            st.write(f"Top-k の Test R2: **{cv1s['mean_test']:.3f} ± {cv1s['std_test']:.3f}**")
        else:
            st.write(f"全特徴の Test AUC: **{cv1f['mean_test']:.3f} ± {cv1f['std_test']:.3f}**")
            st.write(f"Top-k の Test AUC: **{cv1s['mean_test']:.3f} ± {cv1s['std_test']:.3f}**")
            st.caption("※ fold内で正例/負例が片方しかない場合、AUCは NaN になり、平均は NaN を除いて計算します。")

        st.subheader("z の重み（Top-k）")
        z_w_df = pd.DataFrame({"feature": trained["selected"], "weight": [trained["z_w"][f] for f in trained["selected"]]})
        st.dataframe(z_w_df, use_container_width=True)

        st.subheader("Stage2 CV（特徴量の予測）")
        r2_df2 = pd.DataFrame({"feature": trained["selected"], "Test_R2": trained["r2_each2"]})
        st.dataframe(r2_df2, use_container_width=True)
        st.caption(f"平均 Test R2: {trained['r2_mean2']:.3f}")

        # ============================================================
        # Recommend
        # ============================================================
        if st.button("🔍 推奨実行（高速探索 → 画質評価 → A/B/C表示）"):
            selected = trained["selected"]
            m1 = trained["m1_sel"]
            m2 = trained["m2"]
            z_w = trained["z_w"]
            feat_mean = trained["feat_mean"]
            feat_std = trained["feat_std"]
            img_feature_means = trained["img_feature_means"]

            orig_cols = trained["orig_cols"]
            X2_cols = trained["X2_cols"]
            X2_means = trained["X2_means"]
            stage1_task = trained["stage1_task"]

            # --- compute features of new image (BEFORE) if available ---
            new_img_pil = None
            feats_before = {}

            if new_image_file is not None:
                new_img_pil = Image.open(new_image_file).convert("RGB")
                try:
                    feats_before = compute_features_for_pil(new_img_pil)
                except Exception as e:
                    st.error(f"新規画像の特徴量計算に失敗しました: {e}")
                    st.stop()
            else:
                feats_before = {f: df_full.loc[fallback_idx, f] for f in selected if f in df_full.columns}

            # x_before for stage1 (selected features)
            x_before = pd.Series(index=selected, dtype=float)
            miss = []
            for f in selected:
                if f in feats_before:
                    x_before[f] = float(feats_before[f])
                else:
                    miss.append(f)
                    x_before[f] = np.nan
            if miss:
                st.warning(f"新規画像で取得できない特徴量があります: {miss} → 学習データ平均で補完します。")
            x_before = x_before.fillna(img_feature_means.reindex(selected)).fillna(0.0)

            # z は常に表示
            z_before = float(np.sum([z_w[f] * ((x_before[f] - feat_mean[f]) / feat_std[f]) for f in selected]))

            st.subheader("加工前の予測（A）")
            if stage1_task == "reg":
                pupil_before = float(m1.predict(x_before.values.reshape(1, -1))[0])
                st.write(f"予測瞳孔: **{pupil_before:.3f}**")
                st.write(f"z スコア: **{z_before:.3f}**")
            else:
                p_shrink_before = float(m1.predict_proba(x_before.values.reshape(1, -1))[:, 1][0])
                st.write(f"縮瞳確率 P(shrink): **{p_shrink_before:.3f}**")
                st.write(f"z スコア: **{z_before:.3f}**")

            # --- new image *_orig vector for stage2 (IMPORTANT FIX) ---
            if new_img_pil is not None and orig_cols:
                orig_vec = build_orig_vector_from_new_image(
                    new_feats=feats_before,
                    orig_cols=orig_cols,
                    X2_means=X2_means,
                    fallback_df=df_full,
                    fallback_idx=fallback_idx,
                )
                st.caption("Stage2 の *_orig は新規画像から計算しています（フォールバックから借りません）。")
            else:
                orig_vec = pd.Series(index=orig_cols, dtype=float)
                for c in orig_cols:
                    if c in df_full.columns:
                        orig_vec[c] = df_full.loc[fallback_idx, c]
                orig_vec = orig_vec.fillna(X2_means.reindex(orig_cols)).fillna(0.0)

            # --- Fast search ---
            allowed = generate_allowed_patterns()
            sim_records = []

            with st.spinner("高速探索（Model2で特徴推定 → Stage1で目的推定）..."):
                prog = st.progress(0.0, text="0%")
                total_steps = len(allowed)

                for pi, pat in enumerate(allowed):
                    op1, op2, op3 = pat.split("_")
                    v1min, v1max = get_param_range(df_full, 1, op1)
                    v2min, v2max = get_param_range(df_full, 2, op2)
                    v3min, v3max = get_param_range(df_full, 3, op3)

                    vals1 = np.random.uniform(v1min, v1max, n_trials_per_pattern)
                    vals2 = np.random.uniform(v2min, v2max, n_trials_per_pattern)
                    vals3 = np.random.uniform(v3min, v3max, n_trials_per_pattern)

                    sim_X2 = pd.DataFrame(0.0, index=range(n_trials_per_pattern), columns=X2_cols)

                    # put *_orig
                    for c in orig_cols:
                        if c in sim_X2.columns:
                            sim_X2[c] = float(orig_vec.get(c, np.nan))

                    sim_X2 = sim_X2.fillna(X2_means)

                    # put params
                    c1, c2, c3 = f"step1_{op1}", f"step2_{op2}", f"step3_{op3}"
                    if c1 in sim_X2.columns: sim_X2[c1] = vals1
                    if c2 in sim_X2.columns: sim_X2[c2] = vals2
                    if c3 in sim_X2.columns: sim_X2[c3] = vals3

                    Y_pred_feats = m2.predict(sim_X2)  # (N, k)

                    # Stage1 output
                    if stage1_task == "reg":
                        pupil_preds = m1.predict(Y_pred_feats)
                        p_shrink = None
                    else:
                        p_shrink = m1.predict_proba(Y_pred_feats)[:, 1]
                        pupil_preds = None

                    # z
                    scores_z = np.zeros(n_trials_per_pattern, dtype=float)
                    for i in range(n_trials_per_pattern):
                        feat_vec = pd.Series(Y_pred_feats[i, :], index=selected)
                        zv = 0.0
                        for f in selected:
                            zv += z_w[f] * ((feat_vec[f] - feat_mean[f]) / feat_std[f])
                        scores_z[i] = zv

                    df_pat = pd.DataFrame({
                        "pattern": pat,
                        "Score_z": scores_z,
                        "step1_op": op1, "step1_val": vals1,
                        "step2_op": op2, "step2_val": vals2,
                        "step3_op": op3, "step3_val": vals3,
                    })
                    if stage1_task == "reg":
                        df_pat["Pupil"] = pupil_preds
                        df_pat["P_shrink"] = np.nan
                    else:
                        df_pat["Pupil"] = np.nan
                        df_pat["P_shrink"] = p_shrink

                    # objective
                    if objective_mode == "z":
                        df_pat["Objective"] = df_pat["Score_z"]
                    elif objective_mode == "pupil":
                        df_pat["Objective"] = -df_pat["Pupil"]  # smaller pupil is better
                    else:
                        df_pat["Objective"] = df_pat["P_shrink"]  # maximize shrink probability

                    sim_records.append(df_pat)

                    prog.progress((pi + 1) / total_steps, text=f"{pi+1}/{total_steps} パターン")

                sim_all = pd.concat(sim_records, ignore_index=True)

            st.subheader("高速探索サマリー（18パターン）")

            def top5_mean(x):
                k = max(1, int(len(x) * 0.05))
                return x.nlargest(k).mean()

            summary = (sim_all.groupby("pattern")["Objective"]
                       .agg(max_obj="max", top5_mean=top5_mean)
                       .reset_index()
                       .sort_values(["top5_mean", "max_obj"], ascending=False))
            st.dataframe(summary, use_container_width=True)

            # --- Quality evaluation (only if image exists) ---
            if new_img_pil is None:
                st.warning("画像がないため画質評価をスキップします。目的関数の最大で選びます。")
                best = sim_all.loc[sim_all["Objective"].idxmax()].copy()
                best["Q"] = np.nan
                best["HF_ratio"] = np.nan
                best["J"] = np.nan
                best["feasible"] = False
                best["delta_meanY"] = np.nan
                best["abs_delta_meanY"] = np.nan
            else:
                cand = sim_all.sort_values("Objective", ascending=False).head(max_candidates_for_quality).copy()
                cand = cand.reset_index(drop=True)

                with st.spinner("画質評価（Q / HF / meanY）..."):
                    total_q = len(cand)
                    q_prog = st.progress(0.0, text=f"画質評価: 0/{total_q}")

                    q_list, q01_list, hf_list, J_list, feasible_mask = [], [], [], [], []
                    dmy_list, admy_list = [], []

                    meanA = mean_y_255(new_img_pil)

                    for ii in range(total_q):
                        row = cand.iloc[ii]

                        try:
                            ops = [row["step1_op"], row["step2_op"], row["step3_op"]]
                            vals = [row["step1_val"], row["step2_val"], row["step3_val"]]
                            img_proc = apply_processing_sequence(new_img_pil, ops, vals)

                            # quality
                            q = compute_quality(new_img_pil, img_proc, metric_key=quality_metric_key)
                            q01 = normalize_quality_to_01(q, quality_metric_key, psnr_norm_min, psnr_norm_max)

                            # meanY constraint
                            meanC = mean_y_255(img_proc)
                            delta_mean = float(meanC - meanA)
                            abs_delta = float(abs(delta_mean))

                            if "無視" in luma_mode:
                                ok_luma = True
                                penalty_luma = 0.0
                            else:
                                ok_luma = (abs_delta <= float(eps_y))
                                penalty_luma = max(0.0, abs_delta - float(eps_y))

                            # HF
                            if hf_enabled:
                                hf = hf_ratio_laplacian(new_img_pil, img_proc, downscale=hf_downscale)
                                penalty_hf = max(0.0, float(hf) - hf_th)
                            else:
                                hf = np.nan
                                penalty_hf = 0.0

                            obj = float(row["Objective"])

                            # J: objective↑、画質↑、HF↑はペナルティ、meanY逸脱はペナルティ
                            J = alpha * obj - beta * (1.0 - float(q01)) - lambda_luma * float(penalty_luma) - (gamma * penalty_hf if hf_enabled else 0.0)

                            # feasible: Q>=th & HF<=th & meanY OK
                            if hf_enabled:
                                feasible = (float(q) >= float(q_th)) and (float(hf) <= float(hf_th)) and ok_luma
                            else:
                                feasible = (float(q) >= float(q_th)) and ok_luma

                        except Exception:
                            q = np.nan
                            q01 = 0.0
                            hf = np.nan
                            J = -1e18
                            feasible = False
                            delta_mean = np.nan
                            abs_delta = np.nan

                        q_list.append(q)
                        q01_list.append(q01)
                        hf_list.append(hf)
                        J_list.append(J)
                        feasible_mask.append(feasible)
                        dmy_list.append(delta_mean)
                        admy_list.append(abs_delta)

                        if (ii + 1) % 10 == 0 or (ii + 1) == total_q:
                            q_prog.progress((ii + 1) / max(1, total_q),
                                            text=f"画質評価: {ii+1}/{total_q}")

                    # safety for length mismatch
                    if len(feasible_mask) != len(cand):
                        m = min(len(feasible_mask), len(cand))
                        q_list = q_list[:m]
                        q01_list = q01_list[:m]
                        hf_list = hf_list[:m]
                        J_list = J_list[:m]
                        feasible_mask = feasible_mask[:m]
                        dmy_list = dmy_list[:m]
                        admy_list = admy_list[:m]
                        cand = cand.iloc[:m].reset_index(drop=True)

                    cand["Q"] = q_list
                    cand["Q01"] = q01_list
                    cand["HF_ratio"] = hf_list
                    cand["delta_meanY"] = dmy_list
                    cand["abs_delta_meanY"] = admy_list
                    cand["J"] = J_list
                    cand["feasible"] = feasible_mask

                    q_prog.progress(1.0, text="画質評価: 完了")

                st.subheader("評価済み候補（上位）")
                show_cols = ["pattern", "Objective", "Score_z", "Pupil", "P_shrink",
                             "Q", "HF_ratio", "delta_meanY", "abs_delta_meanY", "J", "feasible",
                             "step1_op", "step1_val", "step2_op", "step2_val", "step3_op", "step3_val"]
                st.dataframe(cand[show_cols].head(200), use_container_width=True)

                # ---- Pareto visualization (objective vs Q) ----
                st.subheader("散布図: 画質Q vs 目的（パレート最適を強調）")

                plot_df = cand.dropna(subset=["Q"]).copy()
                if stage1_task == "reg":
                    plot_df = plot_df.dropna(subset=["Pupil"]).copy()
                    y_col = "Pupil"
                    maximize_y = False
                    y_label = "Predicted pupil  ↓"
                    title = "Pareto optimal points (pupil vs Q)"
                else:
                    plot_df = plot_df.dropna(subset=["P_shrink"]).copy()
                    y_col = "P_shrink"
                    maximize_y = True
                    y_label = "P(shrink)  ↑"
                    title = "Pareto optimal points (P(shrink) vs Q)"

                if len(plot_df) >= 2:
                    front_mask = pareto_front_mask(plot_df, x_col="Q", y_col=y_col, maximize_x=True, maximize_y=maximize_y)
                    front = plot_df.loc[front_mask].copy()

                    knee = None
                    if len(front) >= 2:
                        knee = knee_point_on_front(front, x_col="Q", y_col=y_col, maximize_y=maximize_y)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(plot_df["Q"], plot_df[y_col], alpha=0.25, label="All candidates")
                    ax.scatter(front["Q"], front[y_col], alpha=0.9, label="Pareto front")

                    if knee is not None:
                        ax.scatter([knee["Q"]], [knee[y_col]], marker="*", s=200, label="Knee point")

                    ax.set_xlabel(f"{quality_metric_key}(Y)  ↑")
                    ax.set_ylabel(y_label)
                    ax.set_title(title)
                    ax.grid(True, linestyle="--", alpha=0.4)
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.info("パレートフロントを描くための有効点が足りません。")

                # ---- Select best ----
                if quality_mode == "constraint":
                    feasible = cand[cand["feasible"]].copy()
                    if feasible.empty:
                        st.warning("制約を満たす候補がありません。代わりに J 最大を採用します。")
                        best = cand.loc[cand["J"].idxmax()].copy()
                    else:
                        best = feasible.loc[feasible["Objective"].idxmax()].copy()
                elif quality_mode == "composite":
                    best = cand.loc[cand["J"].idxmax()].copy()
                else:
                    plot_df = cand.dropna(subset=["Q"]).copy()
                    if stage1_task == "reg":
                        plot_df = plot_df.dropna(subset=["Pupil"]).copy()
                        y_col = "Pupil"
                        maximize_y = False
                    else:
                        plot_df = plot_df.dropna(subset=["P_shrink"]).copy()
                        y_col = "P_shrink"
                        maximize_y = True

                    if len(plot_df) < 2:
                        st.warning("パレート選択に必要な有効点が足りません。代わりに J 最大を採用します。")
                        best = cand.loc[cand["J"].idxmax()].copy()
                    else:
                        front_mask = pareto_front_mask(plot_df, x_col="Q", y_col=y_col, maximize_x=True, maximize_y=maximize_y)
                        front = plot_df.loc[front_mask].copy()
                        best = knee_point_on_front(front, x_col="Q", y_col=y_col, maximize_y=maximize_y)

            # ============================================================
            # Best display + A/B/C + predictions
            # ============================================================
            st.divider()
            st.subheader("👑 最良の加工条件（C）")

            ops_best = [best["step1_op"], best["step2_op"], best["step3_op"]]
            vals_best = [best["step1_val"], best["step2_val"], best["step3_val"]]

            lines = [
                f"- パターン: **{best['pattern'].replace('_',' → ')}**",
                f"- step1: **{ops_best[0]}** = `{float(vals_best[0]):.3f}`",
                f"- step2: **{ops_best[1]}** = `{float(vals_best[1]):.3f}`",
                f"- step3: **{ops_best[2]}** = `{float(vals_best[2]):.3f}`",
                f"- {quality_metric_key}(Y): **{float(best.get('Q', np.nan)):.4f}**",
                f"- HF_ratio: **{float(best.get('HF_ratio', np.nan)):.3f}**",
                f"- ΔmeanY (C-A): **{float(best.get('delta_meanY', np.nan)):+.2f}** (abs={float(best.get('abs_delta_meanY', np.nan)):.2f})",
            ]
            if stage1_task == "reg":
                lines.insert(4, f"- 予測瞳孔（C候補）: **{float(best.get('Pupil', np.nan)):.3f}**")
            else:
                lines.insert(4, f"- 縮瞳確率 P(shrink)（C候補）: **{float(best.get('P_shrink', np.nan)):.3f}**")

            st.markdown("  \n".join(lines))

            if new_img_pil is not None:
                # A: original
                img_a = new_img_pil

                # C: model processing
                img_c = apply_processing_sequence(img_a, ops_best, vals_best)

                # B: brightness-only to match mean(Y) of C
                target_mean = mean_y_255(img_c)
                img_b, b_shift, mean_b, mean_err = match_mean_y_with_brightness(
                    img_a,
                    target_mean_y_255=target_mean,
                    tol=0.2
                )

                mean_a = mean_y_255(img_a)
                mean_c = target_mean  # mean_y_255(img_c)

                st.subheader("A / B / C（BとCは平均輝度が同じ）")

                colA, colB, colC = st.columns(3)
                with colA:
                    st.image(img_a, caption="A: Original", use_container_width=True)
                    st.write(f"mean(Y): **{mean_a:.2f}**")
                with colB:
                    st.image(img_b, caption="B: Brightness-only (mean matched to C)", use_container_width=True)
                    st.write(f"mean(Y): **{mean_b:.2f}**")
                    st.write(f"brightness shift: `{b_shift:+.2f}`  (err={mean_err:.2f})")
                with colC:
                    st.image(img_c, caption="C: Model processing (best)", use_container_width=True)
                    st.write(f"mean(Y): **{mean_c:.2f}**")

                # Q(A,B) and Q(A,C)
                q_ab = compute_quality(img_a, img_b, metric_key=quality_metric_key)
                q_ac = compute_quality(img_a, img_c, metric_key=quality_metric_key)
                st.markdown(f"- {quality_metric_key}(A,B): **{q_ab:.4f}**")
                st.markdown(f"- {quality_metric_key}(A,C): **{q_ac:.4f}**")

                # ---- Stage1 predictions for A/B/C ----
                st.subheader("A/B/C の予想縮瞳（Stage1予測）")

                try:
                    feats_a = feats_before if isinstance(feats_before, dict) and feats_before else compute_features_for_pil(img_a)
                except Exception:
                    feats_a = compute_features_for_pil(img_a)

                feats_b = compute_features_for_pil(img_b)
                feats_c = compute_features_for_pil(img_c)

                x_a = build_x_from_feats(feats_a, selected, img_feature_means)
                x_b = build_x_from_feats(feats_b, selected, img_feature_means)
                x_c = build_x_from_feats(feats_c, selected, img_feature_means)

                pred_a = predict_stage1_from_x(m1, x_a, stage1_task)
                pred_b = predict_stage1_from_x(m1, x_b, stage1_task)
                pred_c = predict_stage1_from_x(m1, x_c, stage1_task)

                pred_rows = []
                if stage1_task == "reg":
                    pred_rows = [
                        {"image": "A (original)", "pred_pupil": pred_a["pupil"], "pred_P(shrink)": np.nan},
                        {"image": "B (brightness-only, mean matched)", "pred_pupil": pred_b["pupil"], "pred_P(shrink)": np.nan},
                        {"image": "C (model best)", "pred_pupil": pred_c["pupil"], "pred_P(shrink)": np.nan},
                    ]
                    st.caption("回帰モード：値が小さいほど縮瞳側（予測瞳孔）")
                else:
                    pred_rows = [
                        {"image": "A (original)", "pred_pupil": np.nan, "pred_P(shrink)": pred_a["p_shrink"]},
                        {"image": "B (brightness-only, mean matched)", "pred_pupil": np.nan, "pred_P(shrink)": pred_b["p_shrink"]},
                        {"image": "C (model best)", "pred_pupil": np.nan, "pred_P(shrink)": pred_c["p_shrink"]},
                    ]
                    st.caption("分類モード：P(shrink) が大きいほど縮瞳側")

                st.dataframe(pd.DataFrame(pred_rows), use_container_width=True)

                st.subheader("基本統計（A/B/C）")
                dfA = image_basic_stats(img_a); dfA["image"] = "A"
                dfB = image_basic_stats(img_b); dfB["image"] = "B"
                dfC = image_basic_stats(img_c); dfC["image"] = "C"
                stats_df = pd.concat([dfA, dfB, dfC], axis=0, ignore_index=True)
                stats_df = stats_df[["image", "channel", "mean", "std", "min", "max"]]
                st.dataframe(stats_df, use_container_width=True)

if __name__ == "__main__":
    main()

# python -m streamlit run app_keepmodel2.py
