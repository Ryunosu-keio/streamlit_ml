# app_keepmodel2.py
# ============================================================
# 2段モデルを残す版（全体統合・安定化）
# 1段目: 画像特徴 -> pupil (回帰)  or  shrink (分類)   ★ここを切替可能に
# 2段目: (param interaction + *_orig) -> 重要特徴量 (MultiOutput, GridSearch optional)
# 高速探索: 2段目で特徴推定→1段目で {瞳孔 or 縮瞳確率} 推定
# 最終選抜: 実画像で 画質(Q=SSIM/PSNR) + (optional) HF_ratio を計算して選抜
#
# 追加要件:
#  - 画質指標を SSIM / PSNR から選択（※輝度は “Y” ではなく、特徴量で用いた輝度(bL相当)で計算）
#  - A/B/C 表示:
#      A: 元画像
#      C: モデルが吐いた加工（best）
#      B: 「brightnessのみ」で A→B を作り、mean(luma) を C と一致させる
#      (BとCは平均輝度が同じ)
#    さらに A/B/C それぞれの予想縮瞳（Stage1予測: 回帰なら pupil、分類なら P(shrink)）を表示
#
# 追加（構造解明パート）:
#  - 「縮瞳と画像特徴量の関係」をまず重回帰で確認するタブを追加
#  - 輝度のみ vs (コントラスト + シャープネス + 交互作用あり) を比較
#  - 特徴量グループ（all / all_area / all_pupil / ROI別）ごとに R^2, p値, train/test散布図を出す
#
# 追加（今回の要望）:
#  - Model1 に「重回帰（手動式）」を追加（Stage1）
#  - 重回帰に使う項を手動で決める（ベース特徴量＋ 交互作用項/二乗項/平方根項 を用意）
#    ※ Stage2 は「ベース特徴量」を予測。重回帰の多項は推定ベース特徴量から組み立てて予測に使う。
#
# Fix/Improve:
#  - 学習結果を session_state に保持（切り替えで再学習しない）
#  - matplotlibの表示文字は英語のみ（文字化け回避） ← 維持
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
from sklearn.model_selection import GroupKFold, KFold, GroupShuffleSplit, train_test_split
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor, XGBClassifier

# ==== optional deps ===================================================
SM_AVAILABLE = False
LAZY_AVAILABLE = False
OPTUNA_AVAILABLE = False

try:
    import statsmodels.api as sm
    SM_AVAILABLE = True
except Exception:
    SM_AVAILABLE = False

try:
    from lazypredict.Supervised import LazyRegressor, LazyClassifier
    LAZY_AVAILABLE = True
except Exception:
    LAZY_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

# ==== features_pupil / GPU 対応 =====================================
import warnings as _warnings

def cuda_available():
    try:
        return Exception
    except Exception:
        return False

_warnings.filterwarnings(
    "ignore",
    message="CUDA path could not be detected.*",
    module="cupy.*",
)

if cuda_available():
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
# RES_X       = 6000   実験画像横幅
RES_X = 1500
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
# Luminance (bL相当)  ※ここが今回のキモ：Yではなく “特徴量と同じ輝度” を使う
#  - sRGB -> Linear -> Rec.709係数で線形輝度
#  - 返り値は 0..1
# ============================================================
def srgb_to_linear(u01: np.ndarray) -> np.ndarray:
    u01 = np.clip(u01.astype(np.float32), 0.0, 1.0)
    a = 0.04045
    return np.where(u01 <= a, u01 / 12.92, ((u01 + 0.055) / 1.055) ** 2.4)

def to_feature_luma01(pil: Image.Image) -> np.ndarray:
    """
    特徴量計算で使った輝度(bL相当)に寄せる：
      - sRGB(0..1) -> linear
      - linear RGB -> linear luminance (Rec.709)
    """
    rgb = np.array(pil.convert("RGB")).astype(np.float32) / 255.0
    lin = srgb_to_linear(rgb)
    # Rec.709
    luma = 0.2126 * lin[:, :, 0] + 0.7152 * lin[:, :, 1] + 0.0722 * lin[:, :, 2]
    return np.clip(luma, 0.0, 1.0)

def mean_luma_255(pil: Image.Image) -> float:
    return float(np.mean(to_feature_luma01(pil) * 255.0))

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
# SSIM/PSNR (輝度は feature-luma で計算) + HF_ratio + mean(luma)
# ============================================================
def compute_ssim_luma(img_ref: Image.Image, img_proc: Image.Image) -> float:
    ref = to_feature_luma01(img_ref).astype(np.float32)
    proc = to_feature_luma01(img_proc).astype(np.float32)

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

def compute_psnr_luma(img_ref: Image.Image, img_proc: Image.Image) -> float:
    ref = to_feature_luma01(img_ref).astype(np.float32)
    proc = to_feature_luma01(img_proc).astype(np.float32)
    mse = float(np.mean((ref - proc) ** 2))
    if mse <= 1e-12:
        return float("inf")
    L = 1.0
    psnr = 10.0 * np.log10((L * L) / mse)
    return float(psnr)

def compute_quality(img_ref: Image.Image, img_proc: Image.Image, metric_key: str) -> float:
    metric_key = (metric_key or "SSIM").upper()
    if metric_key == "PSNR":
        return compute_psnr_luma(img_ref, img_proc)
    return compute_ssim_luma(img_ref, img_proc)

def normalize_quality_to_01(q: float, metric_key: str, psnr_min: float, psnr_max: float) -> float:
    metric_key = (metric_key or "SSIM").upper()
    if metric_key == "SSIM":
        return float(np.clip(q, 0.0, 1.0))
    lo = float(min(psnr_min, psnr_max - 1e-6))
    hi = float(max(psnr_max, lo + 1e-6))
    q01 = (float(q) - lo) / (hi - lo)
    return float(np.clip(q01, 0.0, 1.0))

def match_mean_luma_with_brightness(
    img: Image.Image,
    target_mean_luma_255: float,
    tol: float = 0.2,
    max_iter: int = 18,
    lo: float = -120.0,
    hi: float = 120.0,
):
    """
    Aにbrightness(=slide_brightness)だけをかけて mean(luma) を target に合わせる。
    二分探索で近づけ、(best_img, best_shift, best_mean, best_err) を返す。
    """
    base_mean = mean_luma_255(img)
    if abs(base_mean - target_mean_luma_255) <= tol:
        return img.copy(), 0.0, base_mean, abs(base_mean - target_mean_luma_255)

    best_img = img.copy()
    best_shift = 0.0
    best_mean = base_mean
    best_err = abs(base_mean - target_mean_luma_255)

    l, r = float(lo), float(hi)
    for _ in range(max_iter):
        mid = (l + r) / 2.0
        cand = slide_brightness(img, shift=mid)
        m = mean_luma_255(cand)
        err = abs(m - target_mean_luma_255)

        if err < best_err:
            best_err = err
            best_img = cand
            best_shift = mid
            best_mean = m

        if err <= tol:
            break

        if m < target_mean_luma_255:
            l = mid
        else:
            r = mid

    return best_img, float(best_shift), float(best_mean), float(best_err)

def hf_ratio_laplacian(img_ref: Image.Image, img_proc: Image.Image, downscale: int = 4) -> float:
    downscale = int(max(1, downscale))

    ref = (to_feature_luma01(img_ref) * 255.0).astype("uint8")
    proc = (to_feature_luma01(img_proc) * 255.0).astype("uint8")

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
    luma = (to_feature_luma01(pil_img) * 255.0).astype(np.float32)

    rows = []
    for name, arr in [
        ("R", rgb[:, :, 0]),
        ("G", rgb[:, :, 1]),
        ("B", rgb[:, :, 2]),
        ("Luma(bL)", luma),
    ]:
        rows.append({
            "channel": name,
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        })
    return pd.DataFrame(rows)

def build_x_from_feats(feats: dict, selected: list, img_feature_means: pd.Series, tag: str = "") -> pd.Series:
    feats = feats or {}
    x = pd.Series(index=selected, dtype=float)

    miss = []
    for f in selected:
        if f in feats:
            x[f] = float(feats[f])
        else:
            x[f] = np.nan
            miss.append(f)

    if len(miss) > 0:
        st.error(f"[{tag}] selectedに対して特徴量が欠損 {len(miss)}/{len(selected)}。"
                 f" 例: {miss[:20]}")
        st.write(f"[{tag}] feats keys 例:", list(feats.keys())[:30])
        st.stop()  # ★平均埋めで誤魔化さず止める

    return x



# ============================================================
# Stage1 prediction helpers (single/batch)  ※列順事故を防ぐ
# ============================================================
def predict_stage1_from_x(m1, x: pd.Series, stage1_task: str) -> dict:
    X = x.to_frame().T  # 1行DataFrame（列名あり）
    if hasattr(m1, "feature_names_in_"):
        X = X.reindex(columns=list(m1.feature_names_in_), fill_value=0.0)

    if stage1_task == "reg":
        return {"pupil": float(m1.predict(X)[0]), "p_shrink": np.nan}
    else:
        proba = m1.predict_proba(X)
        return {"pupil": np.nan, "p_shrink": float(proba[:, 1][0])}

def safe_auc(y_true, y_prob) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return np.nan

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

# ============================================================
# Models & search (RF/XGB)
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

def create_stage1_model_manual(model_type: str, params: dict, task: str):
    """
    manual: RF / XGB
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

def _safe_fit_with_weights(model, X, y, w):
    """
    sample_weight対応してないモデルが混ざることを想定して安全にfitする
    """
    try:
        model.fit(X, y, sample_weight=w)
        return True
    except TypeError:
        model.fit(X, y)
        return False
    except Exception:
        model.fit(X, y)
        return False

def grid_search_stage1_manual(X, y, w, groups, model_type, task: str):
    param_grid = RF_PARAM_GRID_STAGE1 if model_type == "RandomForest" else XGB_PARAM_GRID_STAGE1
    total = int(np.prod([len(v) for v in param_grid.values()])) if param_grid else 1
    prog = st.progress(0.0, text=f"Stage1 GridSearch... (0/{total})")

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

            m = create_stage1_model_manual(model_type, params, task=task)
            if task == "reg":
                _safe_fit_with_weights(m, X_tr, y_tr, w_tr)
                tr_pred = m.predict(X_tr)
                te_pred = m.predict(X_te)
                tr_scores.append(r2_score(y_tr, tr_pred))
                te_scores.append(r2_score(y_te, te_pred))
            else:
                _safe_fit_with_weights(m, X_tr, y_tr, w_tr)
                tr_prob = m.predict_proba(X_tr)[:, 1]
                te_prob = m.predict_proba(X_te)[:, 1]
                tr_scores.append(safe_auc(y_tr, tr_prob))
                te_scores.append(safe_auc(y_te, te_prob))

        mean_te = float(np.nanmean(te_scores))
        if mean_te > best_score:
            best_score, best_params = mean_te, params
            best_train, best_test = tr_scores, te_scores

        done += 1
        prog.progress(done / total, text=f"Stage1 GridSearch... ({done}/{total})")

    final = create_stage1_model_manual(model_type, best_params, task=task)
    _safe_fit_with_weights(final, X, y, w)
    prog.progress(1.0, text="Stage1 GridSearch done")

    cv = {"mean_train": float(np.nanmean(best_train)), "std_train": float(np.nanstd(best_train)),
          "mean_test": float(np.nanmean(best_test)), "std_test": float(np.nanstd(best_test))}
    return final, best_params, cv

def train_stage1_fixed_params_generic(X, y, w, groups, model, task: str, model_name: str = "model"):
    splitter, is_group = _get_splitter(groups)
    splits = list(splitter.split(X, y, groups)) if is_group else list(splitter.split(X, y))
    prog = st.progress(0.0, text=f"Stage1 training ({model_name})...")

    tr_scores, te_scores = [], []
    weight_used_any = False

    for i, (tr_idx, te_idx) in enumerate(splits):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        w_tr = w.iloc[tr_idx]

        # clone-ish
        try:
            import sklearn.base
            m = sklearn.base.clone(model)
        except Exception:
            m = model.__class__(**getattr(model, "get_params", lambda: {})())

        used_w = _safe_fit_with_weights(m, X_tr, y_tr, w_tr)
        weight_used_any = weight_used_any or used_w

        if task == "reg":
            tr_scores.append(r2_score(y_tr, m.predict(X_tr)))
            te_scores.append(r2_score(y_te, m.predict(X_te)))
        else:
            tr_scores.append(safe_auc(y_tr, m.predict_proba(X_tr)[:, 1]))
            te_scores.append(safe_auc(y_te, m.predict_proba(X_te)[:, 1]))

        prog.progress((i + 1) / len(splits), text=f"Stage1 training ({model_name})... ({i+1}/{len(splits)})")

    # final fit
    used_w_final = _safe_fit_with_weights(model, X, y, w)
    weight_used_any = weight_used_any or used_w_final
    prog.progress(1.0, text="Stage1 training done")

    cv = {"mean_train": float(np.nanmean(tr_scores)), "std_train": float(np.nanstd(tr_scores)),
          "mean_test": float(np.nanmean(te_scores)), "std_test": float(np.nanstd(te_scores)),
          "sample_weight_used": bool(weight_used_any)}
    return model, cv

def train_stage2_simple(X2, Y2, w, groups, model_type):
    splitter, is_group = _get_splitter(groups)
    splits = list(splitter.split(X2, Y2, groups)) if is_group else list(splitter.split(X2, Y2))
    prog = st.progress(0.0, text="Stage2 training...")

    cv_scores, Yte_list, Ypred_list = [], [], []
    for i, (tr_idx, te_idx) in enumerate(splits):
        X_tr, X_te = X2.iloc[tr_idx], X2.iloc[te_idx]
        Y_tr, Y_te = Y2.iloc[tr_idx], Y2.iloc[te_idx]
        w_tr = w.iloc[tr_idx]

        base = create_stage2_base(model_type, {})
        mo = MultiOutputRegressor(base)
        try:
            mo.fit(X_tr, Y_tr, sample_weight=w_tr)
        except TypeError:
            mo.fit(X_tr, Y_tr)

        Y_pred = mo.predict(X_te)
        cv_scores.append(r2_score(Y_te, Y_pred, multioutput="uniform_average"))
        Yte_list.append(Y_te)
        Ypred_list.append(Y_pred)

        prog.progress((i + 1) / len(splits), text=f"Stage2 training... ({i+1}/{len(splits)})")

    Yte_all = pd.concat(Yte_list, axis=0)
    Ypred_all = np.vstack(Ypred_list)
    r2_each = r2_score(Yte_all, Ypred_all, multioutput="raw_values")

    base_final = create_stage2_base(model_type, {})
    mo2 = MultiOutputRegressor(base_final)
    try:
        mo2.fit(X2, Y2, sample_weight=w)
    except TypeError:
        mo2.fit(X2, Y2)

    prog.progress(1.0, text="Stage2 training done")
    return mo2, {}, r2_each, float(np.mean(cv_scores))

def grid_search_stage2(X2, Y2, w, groups, model_type):
    param_grid = RF_PARAM_GRID_STAGE2 if model_type == "RandomForest" else XGB_PARAM_GRID_STAGE2
    total = int(np.prod([len(v) for v in param_grid.values()])) if param_grid else 1
    prog = st.progress(0.0, text=f"Stage2 GridSearch... (0/{total})")

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
            try:
                mo.fit(X_tr, Y_tr, sample_weight=w_tr)
            except TypeError:
                mo.fit(X_tr, Y_tr)

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
        prog.progress(done / total, text=f"Stage2 GridSearch... ({done}/{total})")

    base_final = create_stage2_base(model_type, best_params)
    mo2 = MultiOutputRegressor(base_final)
    try:
        mo2.fit(X2, Y2, sample_weight=w)
    except TypeError:
        mo2.fit(X2, Y2)

    r2_each = r2_score(best_Yte_all, best_pred_all, multioutput="raw_values")
    prog.progress(1.0, text="Stage2 GridSearch done")
    return mo2, best_params, r2_each, best_score

# ============================================================
# Pareto / knee
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

def knee_point_on_front(
    front: pd.DataFrame,
    x_col: str,
    y_col: str,
    maximize_y: bool,
    mode: str = "knee",
    x_min: float | None = None,   # ★追加：Q(SSIM/PSNR)の下限
):
    """
    パレートフロントから最適点を選択
      - x_col: 通常は画質Q
      - x_min: 指定すると x_col >= x_min の点だけで選ぶ（満たす点が無ければ None）
    """
    f = front.dropna(subset=[x_col, y_col]).copy()
    if x_min is not None:
        f = f[f[x_col].astype(float) >= float(x_min)].copy()
        if len(f) == 0:
            return None

    f = f.sort_values(x_col, ascending=True).reset_index(drop=True)
    x = f[x_col].values.astype(float)
    y = f[y_col].values.astype(float)

    if len(f) <= 2:
        idx = int(np.nanargmax(y)) if maximize_y else int(np.nanargmin(y))
        return f.iloc[idx].copy()

    if mode == "extreme":
        idx = int(np.nanargmax(y)) if maximize_y else int(np.nanargmin(y))
        return f.iloc[idx].copy()

    x_minv, x_maxv = np.nanmin(x), np.nanmax(x)
    y_minv, y_maxv = np.nanmin(y), np.nanmax(y)
    x_range = x_maxv - x_minv if (x_maxv - x_minv) > 1e-9 else 1.0
    y_range = y_maxv - y_minv if (y_maxv - y_minv) > 1e-9 else 1.0

    xn = (x - x_minv) / x_range
    yn = (y - y_minv) / y_range

    if not maximize_y:
        yn = 1.0 - yn

    p1 = np.array([xn[0], yn[0]])
    p2 = np.array([xn[-1], yn[-1]])
    vec_line = p2 - p1
    vec_points = np.vstack([xn, yn]).T - p1
    cross_prod = vec_points[:, 0] * vec_line[1] - vec_points[:, 1] * vec_line[0]
    dist = np.abs(cross_prod)

    idx = int(np.nanargmax(dist))
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
# 重回帰（構造解明タブ用）
# ============================================================
def _group_split(X, y, groups, test_size=0.2, random_state=42):
    if groups is None:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)
        tr_idx = X_tr.index
        te_idx = X_te.index
        return tr_idx, te_idx
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))
    tr_idx = X.index[tr_idx]
    te_idx = X.index[te_idx]
    return tr_idx, te_idx

def _suggest_by_keywords(cols, include_any):
    out = []
    for c in cols:
        lc = c.lower()
        if any(k in lc for k in include_any):
            out.append(c)
    return out

def run_ols_with_pvalues(X: pd.DataFrame, y: pd.Series, w: pd.Series | None = None):
    """
    statsmodelsのOLS/WLSで係数とp値を返す
    """
    if not SM_AVAILABLE:
        raise RuntimeError("statsmodels is not available")

    X_ = X.copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_ = y.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X_ = sm.add_constant(X_, has_constant="add")

    if w is not None:
        ww = w.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        model = sm.WLS(y_.values, X_.values, weights=ww)
    else:
        model = sm.OLS(y_.values, X_.values)

    res = model.fit()

    params = pd.Series(res.params, index=X_.columns, name="coef")
    pvals = pd.Series(res.pvalues, index=X_.columns, name="pval")
    return res, params, pvals

def build_interactions(dfX: pd.DataFrame, terms_a: list, terms_b: list, prefix="int"):
    out = {}
    for a in terms_a:
        for b in terms_b:
            if a == b:
                continue
            out[f"{prefix}:{a}*{b}"] = dfX[a].astype(float) * dfX[b].astype(float)
    return pd.DataFrame(out, index=dfX.index) if out else pd.DataFrame(index=dfX.index)

def regression_block(
    df_full: pd.DataFrame,
    pupil_col: str,
    candidate_cols: list,
    groups: pd.Series | None,
    sample_weights: pd.Series | None,
    title: str,
    default_brightness_kw=("bl",),
    default_contrast_kw=("contrast", "rms_contrast", "weber", "michelson", "glcm_contrast"),
    default_sharp_kw=("sharp", "lap", "tenengrad", "acut", "hf", "highfreq","grad_entropy")
):
    st.subheader(title)
    if not SM_AVAILABLE:
        st.warning("statsmodels が見つかりません。重回帰を使う場合は `pip install statsmodels` を入れてください。")
        return

    y = df_full[pupil_col].astype(float)
    Xcand = df_full[candidate_cols].copy()
    Xcand = Xcand.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    st.caption("重回帰で使う説明変数セットを選びます（デフォルトはキーワードで自動候補）。")

    bright_sug = [c for c in candidate_cols if "bnl" not in c.lower() and any(k in c.lower() for k in default_brightness_kw)]
    if not bright_sug:
        bright_sug = _suggest_by_keywords(candidate_cols, ["luma", "lumin", "mean", "y", "bl"])
    bright_cols = st.multiselect("輝度（Brightness）候補", options=candidate_cols, default=bright_sug[:1])

    cont_sug = [c for c in candidate_cols if any(k in c.lower() for k in default_contrast_kw)]
    cont_cols = st.multiselect("コントラスト（Contrast）候補", options=candidate_cols, default=cont_sug[:3])

    sharp_sug = [c for c in candidate_cols if any(k in c.lower() for k in default_sharp_kw)]
    sharp_cols = st.multiselect("シャープネス（Sharpness）候補", options=candidate_cols, default=sharp_sug[:3])

    use_interactions = st.checkbox("交互作用（Contrast×Sharpness など）を入れる", value=True)
    standardize = st.checkbox("説明変数を標準化（推奨）", value=True)
    test_size = st.slider("train/test 分割（test比）", 0.1, 0.5, 0.2, 0.05)

    if len(bright_cols) == 0:
        st.warning("輝度の説明変数が0です。最低1つ選んでください。")
        return

    tr_idx, te_idx = _group_split(Xcand, y, groups, test_size=float(test_size), random_state=42)
    X_tr_all = Xcand.loc[tr_idx].copy()
    X_te_all = Xcand.loc[te_idx].copy()
    y_tr = y.loc[tr_idx].copy()
    y_te = y.loc[te_idx].copy()
    w_tr = sample_weights.loc[tr_idx] if (sample_weights is not None) else None

    def _prep(Xtr, Xte):
        if not standardize:
            return Xtr, Xte
        scaler = StandardScaler()
        Xtr_s = pd.DataFrame(scaler.fit_transform(Xtr), index=Xtr.index, columns=Xtr.columns)
        Xte_s = pd.DataFrame(scaler.transform(Xte), index=Xte.index, columns=Xte.columns)
        return Xtr_s, Xte_s

    Xa_tr = X_tr_all[bright_cols].copy()
    Xa_te = X_te_all[bright_cols].copy()
    Xa_tr, Xa_te = _prep(Xa_tr, Xa_te)

    resA, coefA, pA = run_ols_with_pvalues(Xa_tr, y_tr, w_tr)

    def _predict_sm(res, Xdf):
        X_ = sm.add_constant(Xdf, has_constant="add")
        return pd.Series(res.predict(X_.values), index=Xdf.index)

    yhatA_tr = _predict_sm(resA, Xa_tr)
    yhatA_te = _predict_sm(resA, Xa_te)
    r2A_tr = float(r2_score(y_tr, yhatA_tr))
    r2A_te = float(r2_score(y_te, yhatA_te))

    colsB = list(dict.fromkeys(bright_cols + cont_cols + sharp_cols))
    Xb_tr = X_tr_all[colsB].copy()
    Xb_te = X_te_all[colsB].copy()

    if use_interactions and (len(cont_cols) + len(sharp_cols)) > 0:
        inter1 = build_interactions(X_tr_all, cont_cols, sharp_cols, prefix="intCS")
        inter2 = build_interactions(X_tr_all, bright_cols, cont_cols + sharp_cols, prefix="intBL")
        inter1_te = build_interactions(X_te_all, cont_cols, sharp_cols, prefix="intCS")
        inter2_te = build_interactions(X_te_all, bright_cols, cont_cols + sharp_cols, prefix="intBL")

        Xb_tr = pd.concat([Xb_tr, inter1, inter2], axis=1)
        Xb_te = pd.concat([Xb_te, inter1_te, inter2_te], axis=1)

    Xb_tr, Xb_te = _prep(Xb_tr, Xb_te)

    resB, coefB, pB = run_ols_with_pvalues(Xb_tr, y_tr, w_tr)

    yhatB_tr = _predict_sm(resB, Xb_tr)
    yhatB_te = _predict_sm(resB, Xb_te)
    r2B_tr = float(r2_score(y_tr, yhatB_tr))
    r2B_te = float(r2_score(y_te, yhatB_te))

    st.markdown("#### 結果（R²）")
    st.write(f"Model A (Brightness only)  Train R²: **{r2A_tr:.3f}** / Test R²: **{r2A_te:.3f}**")
    st.write(f"Model B (+Contrast/+Sharpness/+Interactions)  Train R²: **{r2B_tr:.3f}** / Test R²: **{r2B_te:.3f}**")
    st.caption("p値はWLS/OLSの仮定に依存します（まず構造を見る用途として使用）。")

    def _coef_table(coef, pval):
        dfc = pd.concat([coef, pval], axis=1).reset_index().rename(columns={"index": "term"})
        dfc = dfc.sort_values("pval", ascending=True).reset_index(drop=True)
        return dfc

    st.markdown("#### 係数とp値（Model A）")
    st.dataframe(_coef_table(coefA, pA).head(80), use_container_width=True)

    st.markdown("#### 係数とp値（Model B）")
    st.dataframe(_coef_table(coefB, pB).head(120), use_container_width=True)

    st.markdown("#### Scatter (train/test)")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_tr, yhatA_tr, alpha=0.35, label="Train (Model A)")
    ax.scatter(y_te, yhatA_te, alpha=0.70, label="Test (Model A)")
    ax.set_xlabel("True pupil")
    ax.set_ylabel("Predicted pupil")
    ax.set_title("OLS/WLS prediction (Model A)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_tr, yhatB_tr, alpha=0.35, label="Train (Model B)")
    ax.scatter(y_te, yhatB_te, alpha=0.70, label="Test (Model B)")
    ax.set_xlabel("True pupil")
    ax.set_ylabel("Predicted pupil")
    ax.set_title("OLS/WLS prediction (Model B)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    st.pyplot(fig)

# ============================================================
# Stage1: 重回帰（手動式） for Model1
#   - Stage2は「ベース特徴量」を予測し、
#     それを元に（交互作用/二乗/平方根）を組み立てて Stage1 に入力する
# ============================================================
def _sqrt_safe(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(x, 0.0))

def build_manual_terms_from_base(
    X_base: pd.DataFrame,
    include_linear: bool,
    include_square: bool,
    include_sqrt: bool,
    interaction_pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    """
    X_base: columns = base features
    出力: design matrix columns = 手動作成した項
    """
    out = {}

    if include_linear:
        for c in X_base.columns:
            out[f"lin:{c}"] = X_base[c].astype(float)

    if include_square:
        for c in X_base.columns:
            out[f"sq:{c}"] = X_base[c].astype(float) ** 2

    if include_sqrt:
        for c in X_base.columns:
            out[f"sqrt:{c}"] = _sqrt_safe(X_base[c].astype(float).values)

    # interactions
    for a, b in interaction_pairs:
        if (a in X_base.columns) and (b in X_base.columns) and (a != b):
            out[f"int:{a}*{b}"] = X_base[a].astype(float) * X_base[b].astype(float)

    return pd.DataFrame(out, index=X_base.index) if out else pd.DataFrame(index=X_base.index)

def parse_bases_from_term(term: str) -> list[str]:
    # "lin:feat" / "sq:feat" / "sqrt:feat" / "int:a*b"
    if term.startswith(("lin:", "sq:", "sqrt:")):
        return [term.split(":", 1)[1]]
    if term.startswith("int:") and "*" in term:
        rhs = term.split(":", 1)[1]
        a, b = rhs.split("*", 1)
        return [a, b]
    return []

class Stage1ManualOLS:
    """
    statsmodels OLS/WLS wrapper (regression only)
    """
    def __init__(
        self,
        base_features: list[str],
        use_terms: list[str],
        standardize_X: bool = True,
        use_WLS: bool = True,
    ):
        self.base_features = list(base_features)
        self.use_terms = list(use_terms)
        self.standardize_X = bool(standardize_X)
        self.use_WLS = bool(use_WLS)

        self.scaler = None
        self.res_ = None
        self.params_ = None
        self.term_names_ = None
        self.feature_names_in_ = None  # sklearn互換的に
        self.is_manual_ols_ = True

    def _make_design(self, X_base_df: pd.DataFrame) -> pd.DataFrame:
        # まず全部作る → use_termsで絞る
        # interactionは use_terms を見て必要なペアだけ作る（無駄を減らす）
        bases = self.base_features
        Xb = X_base_df.reindex(columns=bases).copy()
        Xb = Xb.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 必要なinteraction pairsを use_terms から抽出
        pairs = []
        for t in self.use_terms:
            if t.startswith("int:") and "*" in t:
                rhs = t.split(":", 1)[1]
                a, b = rhs.split("*", 1)
                pairs.append((a, b))

        # どの種類を入れるかは use_terms を見て判定
        include_linear = any(t.startswith("lin:") for t in self.use_terms)
        include_square = any(t.startswith("sq:") for t in self.use_terms)
        include_sqrt = any(t.startswith("sqrt:") for t in self.use_terms)

        Xd_all = build_manual_terms_from_base(
            X_base=Xb,
            include_linear=include_linear,
            include_square=include_square,
            include_sqrt=include_sqrt,
            interaction_pairs=pairs,
        )

        # 絞り込み（順序を固定）
        Xd = Xd_all.reindex(columns=self.use_terms).copy()
        Xd = Xd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return Xd

    def fit(self, X_base_df: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None):
        if not SM_AVAILABLE:
            raise RuntimeError("statsmodels が必要です（pip install statsmodels）")

        Xd = self._make_design(X_base_df)
        y_ = y.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if self.standardize_X:
            self.scaler = StandardScaler()
            Xs = pd.DataFrame(self.scaler.fit_transform(Xd), index=Xd.index, columns=Xd.columns)
        else:
            self.scaler = None
            Xs = Xd

        self.term_names_in_ = np.array(list(Xd.columns), dtype=object)
        self.feature_names_in_ = np.array(list(Xs.columns), dtype=object)

        Xs_ = sm.add_constant(Xs, has_constant="add")

        if self.use_WLS and (sample_weight is not None):
            ww = sample_weight.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
            model = sm.WLS(y_.values, Xs_.values, weights=ww)
        else:
            model = sm.OLS(y_.values, Xs_.values)

        self.res_ = model.fit()
        self.params_ = pd.Series(self.res_.params, index=Xs_.columns, name="coef")
        return self

    def predict(self, X_base_df: pd.DataFrame) -> np.ndarray:
        Xd = self._make_design(X_base_df)
        if self.standardize_X and (self.scaler is not None):
            Xs = pd.DataFrame(self.scaler.transform(Xd), index=Xd.index, columns=Xd.columns)
        else:
            Xs = Xd

        # 学習時列順に合わせる
        if self.term_names_in_ is not None:
            Xs = Xs.reindex(columns=list(self.term_names_in_), fill_value=0.0)
        
        Xs_ = sm.add_constant(Xs, has_constant="add")
        return np.asarray(self.res_.predict(Xs_.values), dtype=float)

def cv_manual_ols(
    model_cfg: dict,
    X_base: pd.DataFrame,
    y: pd.Series,
    w: pd.Series | None,
    groups: pd.Series | None,
):
    splitter, is_group = _get_splitter(groups)
    splits = list(splitter.split(X_base, y, groups)) if is_group else list(splitter.split(X_base, y))

    tr_scores, te_scores = [], []
    prog = st.progress(0.0, text="Stage1 (Manual OLS) CV...")

    for i, (tr_idx, te_idx) in enumerate(splits):
        X_tr, X_te = X_base.iloc[tr_idx], X_base.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        w_tr = w.iloc[tr_idx] if w is not None else None

        m = Stage1ManualOLS(**model_cfg)
        m.fit(X_tr, y_tr, sample_weight=w_tr)
        tr_scores.append(r2_score(y_tr, m.predict(X_tr)))
        te_scores.append(r2_score(y_te, m.predict(X_te)))

        prog.progress((i + 1) / len(splits), text=f"Stage1 (Manual OLS) CV... ({i+1}/{len(splits)})")

    # final fit
    m_final = Stage1ManualOLS(**model_cfg)
    m_final.fit(X_base, y, sample_weight=w if model_cfg.get("use_WLS", True) else None)

    prog.progress(1.0, text="Stage1 (Manual OLS) done")
    cv = {
        "mean_train": float(np.nanmean(tr_scores)),
        "std_train": float(np.nanstd(tr_scores)),
        "mean_test": float(np.nanmean(te_scores)),
        "std_test": float(np.nanstd(te_scores)),
        "sample_weight_used": bool(model_cfg.get("use_WLS", True) and (w is not None)),
    }
    return m_final, cv

def base_importance_from_manual_ols(m: Stage1ManualOLS) -> pd.Series:
    """
    手動OLSの係数から、ベース特徴量の寄与をざっくり集計：
      importance(base) = sum(|coef(term)|) over terms that include base
    """
    if m is None or m.params_ is None:
        return pd.Series(dtype=float)

    coef = m.params_.copy()
    # 定数は除外
    coef = coef.drop(index=[c for c in coef.index if c == "const"], errors="ignore")

    imp = {}
    for term, val in coef.items():
        bases = parse_bases_from_term(term)
        for b in bases:
            imp[b] = imp.get(b, 0.0) + float(abs(val))
    return pd.Series(imp).sort_values(ascending=False)

# ============================================================
# LazyPredict (Stage1選択) + Optuna (任意)
# ============================================================
def lazypredict_select_model(X: pd.DataFrame, y: pd.Series, task: str, groups: pd.Series | None):
    if not LAZY_AVAILABLE:
        raise RuntimeError("lazypredict not available")

    tr_idx, te_idx = _group_split(X, y, groups, test_size=0.2, random_state=42)
    X_tr, X_te = X.loc[tr_idx], X.loc[te_idx]
    y_tr, y_te = y.loc[tr_idx], y.loc[te_idx]

    if task == "reg":
        lazy = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None, predictions=True)
        models_df, preds = lazy.fit(X_tr, X_te, y_tr, y_te)
        if "R-Squared" in models_df.columns:
            best_name = models_df["R-Squared"].idxmax()
        elif "R2" in models_df.columns:
            best_name = models_df["R2"].idxmax()
        else:
            best_name = models_df.iloc[0].name
        best_model = lazy.models[best_name]
        return best_name, best_model, models_df
    else:
        lazy = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None, predictions=True)
        models_df, preds = lazy.fit(X_tr, X_te, y_tr, y_te)
        if "ROC AUC" in models_df.columns:
            best_name = models_df["ROC AUC"].idxmax()
        elif "Accuracy" in models_df.columns:
            best_name = models_df["Accuracy"].idxmax()
        else:
            best_name = models_df.iloc[0].name
        best_model = lazy.models[best_name]
        return best_name, best_model, models_df

def optuna_tune_stage1(
    base_model,
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    groups: pd.Series | None,
    task: str,
    n_trials: int = 30,
):
    if not OPTUNA_AVAILABLE:
        raise RuntimeError("optuna not available")

    model_name = base_model.__class__.__name__.lower()
    splitter, is_group = _get_splitter(groups)

    def objective(trial):
        if "randomforest" in model_name:
            if task == "reg":
                params = dict(
                    n_estimators=trial.suggest_int("n_estimators", 100, 800),
                    max_depth=trial.suggest_int("max_depth", 3, 30),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                    random_state=42,
                    n_jobs=-1,
                )
                m = RandomForestRegressor(**params)
            else:
                params = dict(
                    n_estimators=trial.suggest_int("n_estimators", 200, 1000),
                    max_depth=trial.suggest_int("max_depth", 3, 30),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                )
                m = RandomForestClassifier(**params)

        elif "xgb" in model_name:
            max_depth = trial.suggest_int("max_depth", 2, 8)
            lr = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
            n_estimators = trial.suggest_int("n_estimators", 100, 800)
            subsample = trial.suggest_float("subsample", 0.6, 1.0)
            colsample = trial.suggest_float("colsample_bytree", 0.6, 1.0)
            reg_lambda = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)

            if task == "reg":
                m = XGBRegressor(
                    objective="reg:squarederror",
                    random_state=42, n_jobs=-1,
                    max_depth=max_depth, learning_rate=lr, n_estimators=n_estimators,
                    subsample=subsample, colsample_bytree=colsample, reg_lambda=reg_lambda,
                )
            else:
                m = XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42, n_jobs=-1,
                    max_depth=max_depth, learning_rate=lr, n_estimators=n_estimators,
                    subsample=subsample, colsample_bytree=colsample, reg_lambda=reg_lambda,
                )
        else:
            m = base_model

        scores = []
        split_iter = splitter.split(X, y, groups) if is_group else splitter.split(X, y)
        for tr_idx, te_idx in split_iter:
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
            w_tr = w.iloc[tr_idx] if w is not None else None

            try:
                m_ = m.__class__(**m.get_params())
            except Exception:
                m_ = m

            if w_tr is not None:
                _safe_fit_with_weights(m_, X_tr, y_tr, w_tr)
            else:
                m_.fit(X_tr, y_tr)

            if task == "reg":
                pred = m_.predict(X_te)
                scores.append(r2_score(y_te, pred))
            else:
                prob = m_.predict_proba(X_te)[:, 1]
                scores.append(safe_auc(y_te, prob))

        return float(np.nanmean(scores))

    study = optuna.create_study(direction="maximize")
    prog = st.progress(0.0, text="Optuna tuning...")

    def cb(study, trial):
        prog.progress(min(1.0, (trial.number + 1) / max(1, n_trials)), text=f"Optuna tuning... ({trial.number+1}/{n_trials})")

    study.optimize(objective, n_trials=int(n_trials), callbacks=[cb])

    best_params = study.best_params
    st.caption(f"Optuna best score: {study.best_value:.4f}")

    if "randomforest" in model_name:
        if task == "reg":
            final = RandomForestRegressor(**{**best_params, "random_state": 42, "n_jobs": -1})
        else:
            final = RandomForestClassifier(**{**best_params, "random_state": 42, "n_jobs": -1, "class_weight": "balanced"})
    elif "xgb" in model_name:
        if task == "reg":
            final = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1, **best_params)
        else:
            final = XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42, n_jobs=-1, **best_params)
    else:
        final = base_model

    return final, best_params, study.best_value

# ============================================================
# main
# ============================================================
def main():
    st.set_page_config(page_title="画像加工レコメンダ（Model2）編集用", layout="wide")

    st.markdown("""
    <style>
      html, body, [class*="css"] { font-size: 18px !important; }
      h1, h2, h3 { font-size: 1.25em !important; }
    </style>
    """, unsafe_allow_html=True)

    # matplotlib は英語固定（文字化け回避）
    plt.rcParams["font.family"] = "DejaVu Sans"

    st.title("🧪 画像特徴 →（回帰/分類）→ 画像加工レコメンダ（Model2）編集用")
    st.caption(f"features_pupil backend: {'GPU' if USING_GPU else 'CPU'}")

    # ---------- sidebar: deps ----------
    st.sidebar.header("🧩 追加機能の依存関係")
    st.sidebar.write(f"- statsmodels: {'OK' if SM_AVAILABLE else 'NG'}")
    st.sidebar.write(f"- lazypredict: {'OK' if LAZY_AVAILABLE else 'NG'}")
    st.sidebar.write(f"- optuna: {'OK' if OPTUNA_AVAILABLE else 'NG'}")
    if not SM_AVAILABLE:
        st.sidebar.caption("重回帰: pip install statsmodels")
    if not LAZY_AVAILABLE:
        st.sidebar.caption("LazyPredict: pip install lazypredict")
    if not OPTUNA_AVAILABLE:
        st.sidebar.caption("Optuna: pip install optuna")

    # ---------- Data ----------
    st.sidebar.header("📁 入力データ")
    uploaded_file = st.sidebar.file_uploader("実験データ（CSV / Excel）", type=["csv", "xlsx", "xls"])
    if uploaded_file is None:
        st.info("左のサイドバーからデータセットをアップロードしてください。")
        return

    df_full = load_and_parse_data(uploaded_file)

    # ===== bnL を完全に除去（学習に入れない）=====
    drop_cols = [c for c in df_full.columns if "bnL" in c]
    df_full = df_full.drop(columns=drop_cols, errors="ignore")

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

    tabs = st.tabs(["📊 データ概要", "🧩 構造解析（重回帰）", "🧬 推奨（Model2）"])

    # =========================
    # tab1: overview
    # =========================
    with tabs[0]:
        st.subheader("データセット概要")
        st.write(f"行数: **{len(df_full)}**")
        st.dataframe(df_full.head(), use_container_width=True)

    # =========================
    # tab2: regression analysis
    # =========================
    with tabs[1]:
        st.header("🧩 構造解析（重回帰）")
        st.caption("目的：縮瞳（瞳孔）と画像特徴量の関係を、まず線形モデルで見て議論の芯を作る。")

        num_cols = df_full.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            st.error("数値列が見つかりません。")
        else:
            default_pupil = "corrected_pupil" if "corrected_pupil" in num_cols else num_cols[0]
            pupil_col_reg = st.selectbox("目的変数（瞳孔列）", options=num_cols, index=num_cols.index(default_pupil), key="pupil_col_reg")

            feat_choice_reg = st.radio(
                "特徴量グループ（重回帰）",
                ["全体（all）", "領域重み（all_area）", "瞳孔重み（all_pupil）", "ROI別（center/parafovea/periphery）"],
                index=0,
                horizontal=True,
                key="feat_choice_reg"
            )
            feat_group_reg = {
                "全体（all）": "all",
                "領域重み（all_area）": "all_area",
                "瞳孔重み（all_pupil）": "all_pupil",
                "ROI別（center/parafovea/periphery）": "ROI",
            }[feat_choice_reg]

            if feat_group_reg == "all":
                candidate_cols_reg = [c for c in num_cols if c.startswith("all_")
                                      and not c.startswith("all_area_") and not c.startswith("all_pupil_")
                                      and not c.endswith("_orig") and c not in NON_FEATURE_COLS and c != pupil_col_reg]
            elif feat_group_reg == "all_area":
                candidate_cols_reg = [c for c in num_cols if c.startswith("all_area_")
                                      and not c.endswith("_orig") and c not in NON_FEATURE_COLS and c != pupil_col_reg]
            elif feat_group_reg == "all_pupil":
                candidate_cols_reg = [c for c in num_cols if c.startswith("all_pupil_")
                                      and not c.endswith("_orig") and c not in NON_FEATURE_COLS and c != pupil_col_reg]
            else:
                candidate_cols_reg = [c for c in num_cols if (c.startswith("center_") or c.startswith("parafovea_") or c.startswith("periphery_"))
                                      and "_orig" not in c and c not in NON_FEATURE_COLS and c != pupil_col_reg]

            st.write(f"候補特徴量数: **{len(candidate_cols_reg)}**")

            if st.button("📌 重回帰を実行（このグループ）"):
                regression_block(
                    df_full=df_full,
                    pupil_col=pupil_col_reg,
                    candidate_cols=candidate_cols_reg,
                    groups=groups,
                    sample_weights=sample_weights,
                    title=f"Regression: {feat_choice_reg}",
                )

            st.divider()
            st.subheader("4グループまとめて比較（R²だけ一覧）")
            st.caption("まずは『どのグループが線形に説明しやすいか』をざっくり確認します（説明変数は各グループの自動候補）。")

            if st.button("📊 4グループのR²比較（自動候補で簡易）"):
                if not SM_AVAILABLE:
                    st.warning("statsmodels が必要です。pip install statsmodels")
                else:
                    rows = []
                    for fg_label, fg in [
                        ("all", "all"),
                        ("all_area", "all_area"),
                        ("all_pupil", "all_pupil"),
                        ("ROI", "ROI"),
                    ]:
                        if fg == "all":
                            cand_cols = [c for c in num_cols if c.startswith("all_")
                                         and not c.startswith("all_area_") and not c.startswith("all_pupil_")
                                         and not c.endswith("_orig") and c not in NON_FEATURE_COLS and c != pupil_col_reg]
                        elif fg == "all_area":
                            cand_cols = [c for c in num_cols if c.startswith("all_area_")
                                         and not c.endswith("_orig") and c not in NON_FEATURE_COLS and c != pupil_col_reg]
                        elif fg == "all_pupil":
                            cand_cols = [c for c in num_cols if c.startswith("all_pupil_")
                                         and not c.endswith("_orig") and c not in NON_FEATURE_COLS and c != pupil_col_reg]
                        else:
                            cand_cols = [c for c in num_cols if (c.startswith("center_") or c.startswith("parafovea_") or c.startswith("periphery_"))
                                         and "_orig" not in c and c not in NON_FEATURE_COLS and c != pupil_col_reg]

                        if len(cand_cols) < 3:
                            continue

                        bright = [c for c in cand_cols if "bnl" not in c.lower() and "bl" in c.lower()]
                        if not bright:
                            bright = _suggest_by_keywords(cand_cols, ["luma", "lumin", "mean", "y", "bl"])
                        bright = bright[:1] if bright else cand_cols[:1]

                        cont = _suggest_by_keywords(cand_cols, ["contrast", "rms_contrast", "weber", "michelson", "glcm_contrast"])[:3]
                        sharp = _suggest_by_keywords(cand_cols, ["sharp", "lap", "tenengrad", "acut", "hf", "highfreq"])[:3]

                        Xcand = df_full[cand_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                        y = df_full[pupil_col_reg].astype(float)

                        tr_idx, te_idx = _group_split(Xcand, y, groups, test_size=0.2, random_state=42)
                        y_tr = y.loc[tr_idx]
                        y_te = y.loc[te_idx]
                        w_tr = sample_weights.loc[tr_idx] if sample_weights is not None else None

                        Xa_tr = Xcand.loc[tr_idx, bright].copy()
                        Xa_te = Xcand.loc[te_idx, bright].copy()
                        sc = StandardScaler()
                        Xa_tr = pd.DataFrame(sc.fit_transform(Xa_tr), index=Xa_tr.index, columns=Xa_tr.columns)
                        Xa_te = pd.DataFrame(sc.transform(Xa_te), index=Xa_te.index, columns=Xa_te.columns)

                        resA, _, _ = run_ols_with_pvalues(Xa_tr, y_tr, w_tr)
                        yhatA_te = pd.Series(resA.predict(sm.add_constant(Xa_te, has_constant="add").values), index=Xa_te.index)
                        r2A_te = float(r2_score(y_te, yhatA_te))

                        colsB = list(dict.fromkeys(bright + cont + sharp))
                        Xb_tr = Xcand.loc[tr_idx, colsB].copy()
                        Xb_te = Xcand.loc[te_idx, colsB].copy()

                        if (len(cont) + len(sharp)) > 0:
                            inter_tr = build_interactions(Xcand.loc[tr_idx], cont, sharp, prefix="intCS")
                            inter_te = build_interactions(Xcand.loc[te_idx], cont, sharp, prefix="intCS")
                            Xb_tr = pd.concat([Xb_tr, inter_tr], axis=1)
                            Xb_te = pd.concat([Xb_te, inter_te], axis=1)

                        sc2 = StandardScaler()
                        Xb_tr = pd.DataFrame(sc2.fit_transform(Xb_tr), index=Xb_tr.index, columns=Xb_tr.columns)
                        Xb_te = pd.DataFrame(sc2.transform(Xb_te), index=Xb_te.index, columns=Xb_te.columns)

                        resB, _, _ = run_ols_with_pvalues(Xb_tr, y_tr, w_tr)
                        yhatB_te = pd.Series(resB.predict(sm.add_constant(Xb_te, has_constant="add").values), index=Xb_te.index)
                        r2B_te = float(r2_score(y_te, yhatB_te))

                        rows.append({"group": fg_label, "R2_test_brightness_only": r2A_te, "R2_test_plus_CS_int": r2B_te})

                    st.dataframe(pd.DataFrame(rows).sort_values("R2_test_plus_CS_int", ascending=False), use_container_width=True)

    # =========================
    # tab3: recommender
    # =========================
    with tabs[2]:
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
            top_k = st.slider("Top-k（z の計算/Stage2ターゲットに使用）", min_k, max_k, default_k)

        n_trials_per_pattern = st.slider("パターンごとの試行回数（高速探索）", 200, 5000, 1000, 200)

        # --- Stage1 model selection: Manual vs LazyPredict vs Manual OLS ---
        st.markdown("### 🤖 Stage1 モデル選択")
        model1_mode = st.radio(
            "モデル1の決め方",
            ["手動（RF/XGB）", "自動（LazyPredict）", "重回帰（手動式）"],
            index=0,
            horizontal=True,
            help="重回帰（手動式）は、ベース特徴量＋(交互作用/二乗/平方根)の項を手動で選びます（回帰のみ）。"
        )

        # ---- Manual OLS UI (only for regression) ----
        ols_base_feats = []
        ols_use_terms = []
        ols_cfg = None

        if model1_mode == "重回帰（手動式）":
            if stage1_task != "reg":
                st.error("重回帰（手動式）は Stage1=回帰 のときのみ使えます。分類にしたいなら RF/XGB/Lazy を選んでください。")
                st.stop()
            if not SM_AVAILABLE:
                st.error("statsmodels が必要です（pip install statsmodels）。")
                st.stop()

            st.markdown("#### 🧩 重回帰（手動式）: ベース特徴量と項を選ぶ")
            # ベース特徴量の推奨候補
            sug_b = [c for c in candidate_cols if ("bnl" not in c.lower()) and ("bl" in c.lower())]
            if not sug_b:
                sug_b = _suggest_by_keywords(candidate_cols, ["luma", "lumin", "mean", "y", "bl"])
            sug_c = _suggest_by_keywords(candidate_cols, ["contrast", "rms_contrast", "weber", "michelson", "glcm_contrast"])
            sug_s = _suggest_by_keywords(candidate_cols, ["sharp", "lap", "tenengrad", "acut", "hf", "highfreq", "grad_entropy"])
            default_base = list(dict.fromkeys((sug_b[:1] + sug_c[:2] + sug_s[:2])))[:10]

            ols_base_feats = st.multiselect(
                "ベース特徴量（Stage2が予測し、重回帰の項を作る元になります）",
                options=candidate_cols,
                default=default_base
            )
            if len(ols_base_feats) < 1:
                st.warning("ベース特徴量が0です。最低1つ選んでください。")
                st.stop()

            colA, colB, colC = st.columns(3)
            with colA:
                inc_lin = st.checkbox("線形項（x）を用意", value=True)
            with colB:
                inc_sq = st.checkbox("二乗項（x^2）を用意", value=True)
            with colC:
                inc_sqrt = st.checkbox("平方根項（sqrt(x)）を用意", value=False)

            st.caption("交互作用は 'int:a*b' 形式で作ります（順序違いは別項扱い）。")
            use_all_pairs = st.checkbox("交互作用は『全ペア（ベース特徴量の組合せ）』を用意", value=True)
            manual_pairs = []
            if use_all_pairs:
                pairs = []
                for i in range(len(ols_base_feats)):
                    for j in range(i + 1, len(ols_base_feats)):
                        pairs.append((ols_base_feats[i], ols_base_feats[j]))
                manual_pairs = pairs
            else:
                # 手動でペアを選ぶ
                pair_strs = []
                for i in range(len(ols_base_feats)):
                    for j in range(i + 1, len(ols_base_feats)):
                        pair_strs.append(f"{ols_base_feats[i]} * {ols_base_feats[j]}")
                chosen = st.multiselect("交互作用ペア（手動）", options=pair_strs, default=pair_strs[: min(5, len(pair_strs))])
                pairs = []
                for s in chosen:
                    a, b = s.split(" * ", 1)
                    pairs.append((a, b))
                manual_pairs = pairs

            # 「用意される項一覧」を一旦生成
            Xtmp = df_full[ols_base_feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            all_terms_df = build_manual_terms_from_base(
                X_base=Xtmp,
                include_linear=inc_lin,
                include_square=inc_sq,
                include_sqrt=inc_sqrt,
                interaction_pairs=manual_pairs,
            )
            all_terms = list(all_terms_df.columns)

            st.caption(f"用意される項数: {len(all_terms)}（この中から実際に使う項を選びます）")
            # デフォルト：線形は全部、二乗はbLだけ、交互作用はコントラスト×シャープネスっぽいものを少し
            default_terms = []
            if inc_lin:
                default_terms += [t for t in all_terms if t.startswith("lin:")]
            if inc_sq:
                default_terms += [t for t in all_terms if t.startswith("sq:")][:min(3, len(all_terms))]
            if len(manual_pairs) > 0:
                default_terms += [t for t in all_terms if t.startswith("int:")][:min(3, len(all_terms))]

            ols_use_terms = st.multiselect(
                "実際に重回帰に入れる項（手動）",
                options=all_terms,
                default=list(dict.fromkeys(default_terms))[: min(20, len(all_terms))]
            )
            if len(ols_use_terms) < 1:
                st.warning("重回帰に入れる項が0です。最低1つ選んでください。")
                st.stop()

            col1, col2, col3 = st.columns(3)
            with col1:
                ols_standardize = st.checkbox("重回帰: Xを標準化（推奨）", value=True)
            with col2:
                ols_use_wls = st.checkbox("重回帰: WLS（sample_weight）を使う", value=True)
            with col3:
                st.write("")

            ols_cfg = dict(
                base_features=ols_base_feats,
                use_terms=ols_use_terms,
                standardize_X=ols_standardize,
                use_WLS=ols_use_wls,
            )

            with st.expander("（確認）選んだ項一覧"):
                st.write(ols_use_terms)

        # ---- RF/XGB / Lazy UI ----
        if model1_mode == "手動（RF/XGB）":
            m1_label = st.radio("モデル1（特徴量 → 出力）", ["ランダムフォレスト", "XGBoost"], index=0, horizontal=True)
            model1_type = "RandomForest" if m1_label == "ランダムフォレスト" else "XGBoost"
            use_grid1 = st.checkbox("Stage1 GridSearch（ハイパラ探索）", value=True)
            use_optuna1 = False
            optuna_trials = 0

        elif model1_mode == "自動（LazyPredict）":
            model1_type = "AUTO_LAZY"
            use_grid1 = False
            use_optuna1 = st.checkbox("Optunaでチューニング（任意）", value=False)
            optuna_trials = st.slider("Optuna 試行回数", 10, 100, 30, 5) if use_optuna1 else 0
            if not LAZY_AVAILABLE:
                st.warning("LazyPredict がありません。`pip install lazypredict` を入れてください。")
            if use_optuna1 and not OPTUNA_AVAILABLE:
                st.warning("Optuna がありません。`pip install optuna` を入れてください。")
                use_optuna1 = False
                optuna_trials = 0
            model1_type = "AUTO_LAZY"

        else:
            # manual OLS mode
            model1_type = "MANUAL_OLS"
            use_grid1 = False
            use_optuna1 = False
            optuna_trials = 0

        m2_label = st.radio("モデル2（加工パラメータ+orig → 特徴量）", ["ランダムフォレスト", "XGBoost"], index=0, horizontal=True)
        model2_type = "RandomForest" if m2_label == "ランダムフォレスト" else "XGBoost"
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
            ["SSIM(Luma)", "PSNR(Luma)"],
            index=0,
            horizontal=True
        )
        quality_metric_key = "SSIM" if "SSIM" in quality_metric else "PSNR"

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

        pareto_selection_mode = "knee"
        if quality_mode == "pareto":
            st.markdown("#### 🎯 パレート選抜の方針")
            pareto_mode_choice = st.radio(
                "どの基準で選びますか？",
                ["バランス重視 (屈曲点)", "縮瞳/散瞳優先 (効果最大)"],
                index=0,
                horizontal=True,
            )
            pareto_selection_mode = "knee" if "バランス" in pareto_mode_choice else "extreme"

        if quality_metric_key == "SSIM":
            q_th = st.slider("SSIM(Luma) の閾値", 0.2, 1.0, 0.3, 0.01)
        else:
            q_th = st.slider("PSNR(Luma) の閾値 [dB]", 10.0, 60.0, 25.0, 0.5)

        hf_th = st.slider("HF_ratio の上限（1.0=同等、増えるほど高周波が増加）", 1.0, 10.0, 2.0, 0.1)
        max_candidates_for_quality = st.slider("画質/HF を評価する候補数", 100, 5000, 1000, 100)

        st.markdown("#### 平均輝度（mean Luma(bL)）制約")
        luma_mode = st.radio(
            "mean(luma) をどれくらい固定する？",
            ["固定（推奨：ほぼ変えない）", "許容（±eps）", "無視"],
            index=0,
            horizontal=True
        )
        if "固定" in luma_mode:
            eps_luma = 1.0
            st.caption("固定は実質 |ΔmeanLuma| <= 1.0（0-255スケール）として扱います。")
        elif "許容" in luma_mode:
            eps_luma = st.slider("許容幅 eps（|ΔmeanLuma| <= eps）", 0.0, 10.0, 1.0, 0.1)
        else:
            eps_luma = 1e9

        alpha = st.number_input("alpha（効果）", value=1.0, step=0.1)
        beta  = st.number_input("beta（画質）", value=1.0, step=0.1)
        gamma = st.number_input("gamma（HFペナルティ）", value=0.5, step=0.1)
        lambda_luma = st.number_input("lambda（meanLumaペナルティ）", value=1.0, step=0.1)

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

        st.caption("画像がない場合、画質評価（Q/HF/meanLuma）はスキップし、高速探索の目的関数だけで選びます。")
        fallback_idx = st.selectbox("フォールバック行（画像なし/特徴量欠損時のみ使用）", options=df_full.index)

        # -------- Training button --------
        state = get_state()

        def train_key():
            # OLS設定は key に入れる（同じ設定なら再学習しない）
            ols_key = None
            if model1_mode == "重回帰（手動式）":
                ols_key = (
                    tuple(ols_base_feats),
                    tuple(ols_use_terms),
                    bool(ols_cfg["standardize_X"]),
                    bool(ols_cfg["use_WLS"]),
                )
            return (
                fp_df, pupil_col, feat_group, top_k,
                model1_mode, model1_type, model2_type,
                use_grid2, bool(groups is not None),
                stage1_task, y_class_mode,
                use_optuna1, int(optuna_trials),
                ols_key
            )

        if st.button("🚀 学習（Model1 & Model2）"):
            key = train_key()
            with st.spinner("学習中..."):
                # ---- Stage1 data ----
                X_all = df_full[candidate_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)

                if stage1_task == "reg":
                    y1 = df_full[pupil_col].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                else:
                    y1 = make_y_class(df_full, pupil_col=pupil_col, group_col=group_col, mode=y_class_mode)

                best_p1 = {}
                lazy_report = None
                lazy_best_name = None
                cv1_full = {"mean_train": np.nan, "std_train": np.nan, "mean_test": np.nan, "std_test": np.nan, "sample_weight_used": np.nan}

                # ========== Stage1 training ==========
                if model1_mode == "重回帰（手動式）":
                    # Stage2ターゲットは「ベース特徴量」
                    selected = list(ols_base_feats)

                    # Fit manual OLS
                    model_cfg = dict(
                        base_features=ols_cfg["base_features"],
                        use_terms=ols_cfg["use_terms"],
                        standardize_X=ols_cfg["standardize_X"],
                        use_WLS=ols_cfg["use_WLS"],
                    )
                    X_base = df_full[selected].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    m1_full, cv1_sel = cv_manual_ols(model_cfg, X_base, y1, sample_weights, groups)
                    m1_sel = m1_full  # 同一

                    # importance: 係数集計
                    imp_base = base_importance_from_manual_ols(m1_sel)
                    # 欠けがあれば0埋め
                    imp_series = pd.Series({f: float(imp_base.get(f, 0.0)) for f in selected})
                    imp_df = pd.DataFrame({"feature": selected, "importance": imp_series.values}).sort_values(
                        "importance", ascending=False
                    ).reset_index(drop=True)

                    best_p1 = {"manual_ols": True, "n_terms": len(ols_use_terms)}
                    cv1_full = cv1_sel

                elif model1_mode == "自動（LazyPredict）":
                    if not LAZY_AVAILABLE:
                        st.error("LazyPredict が見つかりません。pip install lazypredict")
                        st.stop()

                    st.info("LazyPredict で Stage1 モデルを自動選択します。")
                    with st.spinner("LazyPredict running..."):
                        lazy_best_name, lazy_best_model, lazy_df = lazypredict_select_model(X_all, y1, stage1_task, groups)
                        lazy_report = lazy_df
                        st.success(f"LazyPredict best: {lazy_best_name}")
                        st.dataframe(lazy_df.head(25), use_container_width=True)

                    m1_full = lazy_best_model
                    if use_optuna1 and OPTUNA_AVAILABLE:
                        st.info("Optuna tuning...")
                        try:
                            tuned, best_params_opt, best_score_opt = optuna_tune_stage1(
                                m1_full, X_all, y1, sample_weights, groups, stage1_task, n_trials=int(optuna_trials)
                            )
                            m1_full = tuned
                            best_p1 = best_params_opt
                        except Exception as e:
                            st.warning(f"Optuna tuning failed: {e} （チューニングなしで続行）")

                    imp = getattr(m1_full, "feature_importances_", None)
                    if imp is None:
                        # 重要度がないモデル：相関絶対値で代用
                        if stage1_task == "reg":
                            yy = df_full[pupil_col].astype(float)
                        else:
                            yy = df_full[pupil_col].astype(float)  # z用は連続
                        imp = np.array([abs(df_full[c].corr(yy)) for c in candidate_cols], dtype=float)
                        imp = np.nan_to_num(imp, nan=0.0)

                    imp_df = pd.DataFrame({"feature": candidate_cols, "importance": imp}).sort_values(
                        "importance", ascending=False
                    ).reset_index(drop=True)
                    selected = imp_df["feature"].head(top_k).tolist()

                    # retrain on selected
                    X_sel = df_full[selected].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    try:
                        import sklearn.base
                        m1_sel = sklearn.base.clone(m1_full)
                    except Exception:
                        m1_sel = m1_full.__class__(**getattr(m1_full, "get_params", lambda: {})())

                    m1_sel, cv1_sel = train_stage1_fixed_params_generic(
                        X_sel, y1, sample_weights, groups, m1_sel, task=stage1_task, model_name=str(lazy_best_name)
                    )

                else:
                    # manual RF/XGB
                    if use_grid1:
                        m1_full, best_p1, cv1_full = grid_search_stage1_manual(
                            X_all, y1, sample_weights, groups, model1_type, task=stage1_task
                        )
                    else:
                        m1_full = create_stage1_model_manual(model1_type, {}, task=stage1_task)
                        m1_full, cv1_full = train_stage1_fixed_params_generic(
                            X_all, y1, sample_weights, groups, m1_full, task=stage1_task, model_name=model1_type
                        )
                        best_p1 = {}

                    imp = getattr(m1_full, "feature_importances_", None)
                    if imp is None:
                        imp = np.ones(len(candidate_cols), dtype=float)

                    imp_df = pd.DataFrame({"feature": candidate_cols, "importance": imp}).sort_values(
                        "importance", ascending=False
                    ).reset_index(drop=True)
                    selected = imp_df["feature"].head(top_k).tolist()

                    X_sel = df_full[selected].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    m1_sel = create_stage1_model_manual(model1_type, best_p1, task=stage1_task)
                    m1_sel, cv1_sel = train_stage1_fixed_params_generic(
                        X_sel, y1, sample_weights, groups, m1_sel, task=stage1_task, model_name=model1_type
                    )

                # ========== z weights（ベース特徴量に対して計算）==========
                # selected は Stage2ターゲット（ベース特徴量）
                if model1_mode == "重回帰（手動式）":
                    imp_sel = np.array([float(imp_df.set_index("feature").loc[f, "importance"]) if f in imp_df["feature"].values else 0.0 for f in selected])
                else:
                    imp_sel = getattr(m1_sel, "feature_importances_", np.ones(len(selected), dtype=float))

                # zの符号は連続瞳孔との相関で決める（分類でも同じ）
                y_for_corr = df_full[pupil_col].astype(float)
                signs = []
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

                Y2 = df_full[selected].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)

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
                    "selected": selected,                 # Stage2ターゲット（ベース特徴量）
                    "m1_full": m1_full,
                    "m1_sel": m1_sel,                     # 予測用 Stage1
                    "cv1_full": cv1_full,
                    "cv1_sel": cv1_full if model1_mode == "重回帰（手動式）" else cv1_sel,
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
                    "lazy_report": lazy_report,
                    "lazy_best_name": lazy_best_name,
                    "model1_mode": model1_mode,
                    "model1_type": model1_type,
                    "ols_cfg": ols_cfg,
                }

            st.success("学習が完了しました（結果は session_state に保持されます）。")

        # -------- Show trained results --------
        key = train_key()
        trained = state.get(key)

        if trained is None:
            st.info("先に「学習（Model1 & Model2）」を押してください。")
            return

        if trained.get("model1_mode") == "自動（LazyPredict）" and trained.get("lazy_report") is not None:
            st.subheader("LazyPredict レポート（参考）")
            st.dataframe(trained["lazy_report"].head(25), use_container_width=True)
            st.caption(f"Selected model: {trained.get('lazy_best_name')}")

        st.subheader("Stage1 重要度")
        st.dataframe(trained["imp_df"].head(30), use_container_width=True)

        st.subheader("Stage1 CV")
        cv1f = trained["cv1_full"]
        cv1s = trained["cv1_sel"]
        if trained["stage1_task"] == "reg":
            st.write(f"Test R2: **{cv1s['mean_test']:.3f} ± {cv1s['std_test']:.3f}**")
        else:
            st.write(f"Test AUC: **{cv1s['mean_test']:.3f} ± {cv1s['std_test']:.3f}**")
            st.caption("※ fold内で正例/負例が片方しかない場合、AUCは NaN になり、平均は NaN を除いて計算します。")

        if "sample_weight_used" in cv1s and (trained.get("model1_mode") == "自動（LazyPredict）"):
            if not cv1s["sample_weight_used"]:
                st.warning("選ばれたモデルが sample_weight に未対応の可能性があり、学習で重みが使われていない場合があります。")

        st.subheader("z の重み（Stage2ターゲット=ベース特徴量）")
        z_w_df = pd.DataFrame({"feature": trained["selected"], "weight": [trained["z_w"][f] for f in trained["selected"]]})
        st.dataframe(z_w_df, use_container_width=True)

        st.subheader("Stage2 CV（特徴量の予測）")
        r2_df2 = pd.DataFrame({"feature": trained["selected"], "Test_R2": trained["r2_each2"]})
        st.dataframe(r2_df2, use_container_width=True)
        st.caption(f"平均 Test R2: {trained['r2_mean2']:.3f}")

        # ============================================================
        # Recommend
        # ============================================================
        def stage1_predict_from_base_features(m1, X_base_df: pd.DataFrame, stage1_task: str, model1_mode: str):
            """
            X_base_df: columns = trained['selected'] (ベース特徴量)
            return: np.ndarray (pred)  or prob for clf
            """
            if model1_mode == "重回帰（手動式）":
                # Stage1ManualOLS only supports reg
                return m1.predict(X_base_df)
            else:
                # sklearn-like
                if hasattr(m1, "feature_names_in_"):
                    X_in = X_base_df.reindex(columns=list(m1.feature_names_in_), fill_value=0.0)
                else:
                    X_in = X_base_df.copy()
                if stage1_task == "reg":
                    return m1.predict(X_in)
                else:
                    return m1.predict_proba(X_in)[:, 1]

        if st.button("🔍 推奨実行（高速探索 → 画質評価 → A/B/C表示）"):
            selected = trained["selected"]          # ベース特徴量
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
            model1_mode_tr = trained["model1_mode"]

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

            # base features vector for Stage1 (selected base features)
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

            # z は常に表示（ベース特徴量で計算）
            z_before = float(np.sum([z_w[f] * ((x_before[f] - feat_mean[f]) / feat_std[f]) for f in selected]))

            st.subheader("加工前の予測（A）")
            baseA_df = x_before.to_frame().T
            if stage1_task == "reg":
                predA = float(stage1_predict_from_base_features(m1, baseA_df, stage1_task, model1_mode_tr)[0])
                st.write(f"予測瞳孔: **{predA:.3f}**")
            else:
                predA = float(stage1_predict_from_base_features(m1, baseA_df, stage1_task, model1_mode_tr)[0])
                st.write(f"縮瞳確率 P(shrink): **{predA:.3f}**")
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

            # ============================================================
            # Fast search
            # ============================================================
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

                    Y_pred_df = pd.DataFrame(Y_pred_feats, columns=selected)  # ★列名つきDFへ

                    # Stage1 output
                    if stage1_task == "reg":
                        pupil_preds = m1.predict(Y_pred_df)
                        p_shrink = None
                    else:
                        p_shrink = m1.predict_proba(Y_pred_df)[:, 1]
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

                    prog.progress((pi + 1) / total_steps, text=f"{pi+1}/{total_steps} patterns")

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
                best["delta_meanLuma"] = np.nan
                best["abs_delta_meanLuma"] = np.nan
            else:
                cand = sim_all.sort_values("Objective", ascending=False).head(max_candidates_for_quality).copy()
                cand = cand.reset_index(drop=True)

                with st.spinner("画質評価（Q / HF / meanLuma）..."):
                    total_q = len(cand)
                    q_prog = st.progress(0.0, text=f"Quality eval: 0/{total_q}")

                    q_list, q01_list, hf_list, J_list, feasible_mask = [], [], [], [], []
                    dml_list, adml_list = [], []

                    meanA = mean_luma_255(new_img_pil)

                    for ii in range(total_q):
                        row = cand.iloc[ii]

                        try:
                            ops = [row["step1_op"], row["step2_op"], row["step3_op"]]
                            vals = [row["step1_val"], row["step2_val"], row["step3_val"]]
                            img_proc = apply_processing_sequence(new_img_pil, ops, vals)

                            # quality
                            q = compute_quality(new_img_pil, img_proc, metric_key=quality_metric_key)
                            q01 = normalize_quality_to_01(q, quality_metric_key, psnr_norm_min, psnr_norm_max)

                            # meanLuma constraint
                            meanC = mean_luma_255(img_proc)
                            delta_mean = float(meanC - meanA)
                            abs_delta = float(abs(delta_mean))

                            if "無視" in luma_mode:
                                ok_luma = True
                                penalty_luma = 0.0
                            else:
                                ok_luma = (abs_delta <= float(eps_luma))
                                penalty_luma = max(0.0, abs_delta - float(eps_luma))

                            # HF
                            if hf_enabled:
                                hf = hf_ratio_laplacian(new_img_pil, img_proc, downscale=hf_downscale)
                                penalty_hf = max(0.0, float(hf) - hf_th)
                            else:
                                hf = np.nan
                                penalty_hf = 0.0

                            obj = float(row["Objective"])

                            # J: objective↑、画質↑、HF↑はペナルティ、meanLuma逸脱はペナルティ
                            J = alpha * obj - beta * (1.0 - float(q01)) - lambda_luma * float(penalty_luma) - (gamma * penalty_hf if hf_enabled else 0.0)

                            # feasible
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
                        dml_list.append(delta_mean)
                        adml_list.append(abs_delta)

                        if (ii + 1) % 10 == 0 or (ii + 1) == total_q:
                            q_prog.progress((ii + 1) / max(1, total_q),
                                            text=f"Quality eval: {ii+1}/{total_q}")

                    # safety for length mismatch
                    if len(feasible_mask) != len(cand):
                        m = min(len(feasible_mask), len(cand))
                        q_list = q_list[:m]
                        q01_list = q01_list[:m]
                        hf_list = hf_list[:m]
                        J_list = J_list[:m]
                        feasible_mask = feasible_mask[:m]
                        dml_list = dml_list[:m]
                        adml_list = adml_list[:m]
                        cand = cand.iloc[:m].reset_index(drop=True)

                    cand["Q"] = q_list
                    cand["Q01"] = q01_list
                    cand["HF_ratio"] = hf_list
                    cand["delta_meanLuma"] = dml_list
                    cand["abs_delta_meanLuma"] = adml_list
                    cand["J"] = J_list
                    cand["feasible"] = feasible_mask

                    q_prog.progress(1.0, text="Quality eval done")

                st.subheader("評価済み候補（上位）")
                show_cols = ["pattern", "Objective", "Score_z", "Pupil", "P_shrink",
                             "Q", "HF_ratio", "delta_meanLuma", "abs_delta_meanLuma", "J", "feasible",
                             "step1_op", "step1_val", "step2_op", "step2_val", "step3_op", "step3_val"]
                st.dataframe(cand[show_cols].head(200), use_container_width=True)

                # ---- Pareto visualization (objective vs Q) ----
                st.subheader("Scatter: Q vs objective (highlight Pareto front)")

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
                        # knee = knee_point_on_front(front, x_col="Q", y_col=y_col, maximize_y=maximize_y)
                        # knee = knee_point_on_front(front, x_col="Q", y_col=y_col, maximize_y=maximize_y,
                        #    mode=pareto_selection_mode, x_min=float(q_th))
                        knee = knee_point_on_front(
                                front,
                                x_col="Q",
                                y_col=y_col,
                                maximize_y=maximize_y,
                                mode=pareto_selection_mode,
                                x_min=None # ★追加：SSIM/PSNR 閾値無視
                            )


                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(plot_df["Q"], plot_df[y_col], alpha=0.25, label="All candidates")
                    ax.scatter(front["Q"], front[y_col], alpha=0.9, label="Pareto front")

                    # if knee is not None:
                    #     ax.scatter([knee["Q"]], [knee[y_col]], marker="*", s=200, label="Knee point")
                    if pareto_selection_mode == "knee" and knee is not None:
                        ax.scatter([knee["Q"]], [knee[y_col]], marker="*", s=200, label="Knee Point")
                    elif pareto_selection_mode == "extreme" :
                        ax.scatter([knee["Q"]], [knee[y_col]], marker="*", s=200, label="Max Point")


                    ax.set_xlabel(f"{quality_metric_key}(Luma)  ↑")
                    ax.set_ylabel(y_label)
                    ax.set_title(title)
                    ax.grid(True, linestyle="--", alpha=0.4)
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.info("Not enough valid points to plot Pareto front.")

                # ---- Select best ----
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
                # パレートモード（事前に選択済みの mode を使用）
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
                    
                    if len(front) == 0:
                        st.warning("パレートフロントが見つかりません。J最大を採用します。")
                        best = cand.loc[cand["J"].idxmax()].copy()
                    else:
                        # ★★★ 事前に選択された mode を使用 ★★★
                        # best = knee_point_on_front(front, x_col="Q", y_col=y_col, maximize_y=maximize_y, mode=pareto_selection_mode, x_min=float(q_th))
                        best = knee_point_on_front(
                                front,
                                x_col="Q",
                                y_col=y_col,
                                maximize_y=maximize_y,
                                mode=pareto_selection_mode,
                                x_min=float(q_th),  # ★追加：SSIM/PSNR 閾値無視
                            )

                        if best is None:
                            # 閾値以上のPareto点が無い（＝そもそもQ>=閾値の候補がほぼ無い等）
                            # ここは好みでフォールバックを決める
                            st.warning("パレートフロント上に Q>=閾値 を満たす点がありません。Q>=閾値 の中で Objective 最大を探します。")
                            feasible_q = cand[cand["Q"].astype(float) >= float(q_th)].copy()
                            if feasible_q.empty:
                                st.warning("候補全体でも Q>=閾値 がありません。J最大にフォールバックします。")
                                best = cand.loc[cand["J"].idxmax()].copy()
                            else:
                                best = feasible_q.loc[feasible_q["Objective"].idxmax()].copy()

                        # 選択された方針を表示
                        if pareto_selection_mode == "knee":
                            st.info("💡 画質と予測値のバランスが良い点（カーブの屈曲点）を選びました。")
                        else:
                            if stage1_task == "reg":
                                st.info("💡 画質を多少犠牲にしても、予測瞳孔径が最小の点（縮瞳効果最大）を選びました。")
                            else:
                                st.info("💡 画質を多少犠牲にしても、縮瞳確率P(shrink)が最大の点（縮瞳効果最大）を選びました。")
            # ============================================================
            # Best display + A/B/C + predictions
            # ============================================================
            st.divider()
            st.subheader("👑 最良の加工条件（C）")

            ops_best = [best["step1_op"], best["step2_op"], best["step3_op"]]
            vals_best = [best["step1_val"], best["step2_val"], best["step3_val"]]

            lines = [
                f"- pattern: **{best['pattern'].replace('_',' → ')}**",
                f"- step1: **{ops_best[0]}** = `{float(vals_best[0]):.3f}`",
                f"- step2: **{ops_best[1]}** = `{float(vals_best[1]):.3f}`",
                f"- step3: **{ops_best[2]}** = `{float(vals_best[2]):.3f}`",
                f"- {quality_metric_key}(Luma): **{float(best.get('Q', np.nan)):.4f}**",
                f"- HF_ratio: **{float(best.get('HF_ratio', np.nan)):.3f}**",
                f"- ΔmeanLuma (C-A): **{float(best.get('delta_meanLuma', np.nan)):+.2f}** (abs={float(best.get('abs_delta_meanLuma', np.nan)):.2f})",
            ]
            if stage1_task == "reg":
                lines.insert(4, f"- Pred pupil (C): **{float(best.get('Pupil', np.nan)):.3f}**")
            else:
                lines.insert(4, f"- P(shrink) (C): **{float(best.get('P_shrink', np.nan)):.3f}**")

            st.markdown("  \n".join(lines))

            if new_img_pil is not None:
                # A: original
                img_a = new_img_pil

                # C: model processing
                img_c = apply_processing_sequence(img_a, ops_best, vals_best)

                # B: brightness-only to match mean(luma) of C
                target_mean = mean_luma_255(img_c)
                img_b, b_shift, mean_b, mean_err = match_mean_luma_with_brightness(
                    img_a,
                    target_mean_luma_255=target_mean,
                    tol=0.2
                )

                mean_a = mean_luma_255(img_a)
                mean_c = target_mean

                st.subheader("A / B / C（BとCは平均輝度が同じ：Luma(bL)）")

                colA, colB, colC = st.columns(3)
                with colA:
                    st.image(img_a, caption="A: Original", use_container_width=True)
                    st.write(f"mean(Luma): **{mean_a:.2f}**")
                with colB:
                    st.image(img_b, caption="B: Brightness-only (mean matched to C)", use_container_width=True)
                    st.write(f"mean(Luma): **{mean_b:.2f}**")
                    st.write(f"brightness shift: `{b_shift:+.2f}`  (err={mean_err:.2f})")
                with colC:
                    st.image(img_c, caption="C: Model processing (best)", use_container_width=True)
                    st.write(f"mean(Luma): **{mean_c:.2f}**")

                # Q(A,B) and Q(A,C)
                q_ab = compute_quality(img_a, img_b, metric_key=quality_metric_key)
                q_ac = compute_quality(img_a, img_c, metric_key=quality_metric_key)
                st.markdown(f"- {quality_metric_key}(A,B): **{q_ab:.4f}**")
                st.markdown(f"- {quality_metric_key}(A,C): **{q_ac:.4f}**")

                # ---- Stage1 predictions for A/B/C ----
                st.subheader("A/B/C の予想縮瞳（Stage1 prediction）")

                feats_a = feats_before if isinstance(feats_before, dict) and feats_before else compute_features_for_pil(img_a)
                feats_b = compute_features_for_pil(img_b)
                feats_c = compute_features_for_pil(img_c)

                x_a = build_x_from_feats(feats_a, selected, img_feature_means,tag="A")
                x_b = build_x_from_feats(feats_b, selected, img_feature_means, tag="B")
                x_c = build_x_from_feats(feats_c, selected, img_feature_means, tag="C")

                # ★ 入力特徴量の中身を必ず表示（要求）
                feat_tbl = pd.DataFrame({
                    "A": x_a,
                    "B": x_b,
                    "C": x_c,
                })
                feat_tbl["B-A"] = feat_tbl["B"] - feat_tbl["A"]
                feat_tbl["C-A"] = feat_tbl["C"] - feat_tbl["A"]
                feat_tbl = feat_tbl.reset_index().rename(columns={"index": "feature"})
                st.markdown("#### Stage1に入れた特徴量（A/B/C）")
                st.dataframe(feat_tbl, use_container_width=True)

                # ★ A/B/Cで本当に違いがあるか簡易チェック
                same_ab = np.allclose(x_a.values, x_b.values, rtol=0, atol=1e-12)
                same_ac = np.allclose(x_a.values, x_c.values, rtol=0, atol=1e-12)
                if same_ab and same_ac:
                    st.error("A/B/C の Stage1入力特徴量が同一です（= 予測が同じのは当然）。特徴量名の不一致 or 欠損→平均補完を疑ってください。")

                # ★ 予測（列名維持版の predict_stage1_from_x を使う）
                pred_a = predict_stage1_from_x(m1, x_a, stage1_task)
                pred_b = predict_stage1_from_x(m1, x_b, stage1_task)
                pred_c = predict_stage1_from_x(m1, x_c, stage1_task)

                # ★ 縮瞳値（Δpred）を計算して表示（要求）
                if stage1_task == "reg":
                    # shrink量を「Aからどれだけ小さくなったか」で定義（+が縮瞳）
                    shrink_b = float(pred_a["pupil"] - pred_b["pupil"])
                    shrink_c = float(pred_a["pupil"] - pred_c["pupil"])

                    pred_rows = [
                        {"image": "A (original)", "pred_pupil": pred_a["pupil"], "shrink_vs_A": 0.0},
                        {"image": "B (brightness-only, mean matched)", "pred_pupil": pred_b["pupil"], "shrink_vs_A": shrink_b},
                        {"image": "C (model best)", "pred_pupil": pred_c["pupil"], "shrink_vs_A": shrink_c},
                    ]
                    st.caption("Regression mode: pred_pupil が小さいほど縮瞳。shrink_vs_A = pred_pupil(A) - pred_pupil(X)（+が縮瞳）")
                else:
                    # 縮瞳確率の増分（+が縮瞳側）
                    d_b = float(pred_b["p_shrink"] - pred_a["p_shrink"])
                    d_c = float(pred_c["p_shrink"] - pred_a["p_shrink"])

                    pred_rows = [
                        {"image": "A (original)", "pred_P(shrink)": pred_a["p_shrink"], "delta_vs_A": 0.0},
                        {"image": "B (brightness-only, mean matched)", "pred_P(shrink)": pred_b["p_shrink"], "delta_vs_A": d_b},
                        {"image": "C (model best)", "pred_P(shrink)": pred_c["p_shrink"], "delta_vs_A": d_c},
                    ]
                    st.caption("Classification mode: pred_P(shrink) が大きいほど縮瞳側。delta_vs_A = P(X) - P(A)（+が縮瞳側）")

                st.dataframe(pd.DataFrame(pred_rows), use_container_width=True)

                # ---- Basic stats for A/B/C ----
                st.subheader("Basic stats (A/B/C)")
                dfA = image_basic_stats(img_a); dfA["image"] = "A"
                dfB = image_basic_stats(img_b); dfB["image"] = "B"
                dfC = image_basic_stats(img_c); dfC["image"] = "C"
                stats_df = pd.concat([dfA, dfB, dfC], axis=0, ignore_index=True)
                stats_df = stats_df[["image", "channel", "mean", "std", "min", "max"]]
                st.dataframe(stats_df, use_container_width=True)

if __name__ == "__main__":
    main()

# python -m streamlit run app_keepmodel2.py
