# app_keep_model2.py
# ============================================================
# 2段モデルを残す版
# 1段目: 画像特徴 -> pupil (RF / XGB, GroupKFold, GridSearch optional)
# 2段目: (param interaction + *_orig) -> 重要特徴量 (MultiOutput, GridSearch optional)
# ランダム探索(18 patterns)は 2段目で特徴を推定して高速に回す
# その後、上位候補に対して画像を実際に加工し SSIM(Y)+HF_ratio を計算して選抜
#
# 改善:
#  - 学習結果を session_state に保持（切り替えで再学習しない）
#  - SSIMはY(輝度)で計算
#  - HF_ratio(高周波増幅)でガビガビ抑制
#  - 制約/合成J/パレート選択
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

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score

from xgboost import XGBRegressor

# ==== features_pupil / GPU 対応 =====================================
# ※あなたの元コードは try/except で両方 features_pupil を import していたので修正
try:
    import features_pupil as fp  # GPU版がある想定
    if cv2.cuda.getCudaEnabledDeviceCount() == 0:
        raise ImportError("CUDA device not found")
    USING_GPU = True
except Exception:
    import features_pupil as fp
    USING_GPU = False

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

# =========================
# 画像加工
# =========================
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

# =========================
# Q: SSIM(Y) + ガビガビ抑制(HF_ratio)
# =========================
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

def hf_ratio_laplacian(img_ref: Image.Image, img_proc: Image.Image, downscale: float = 0.25) -> float:
    """
    HF ratio via Laplacian variance on Y channel.
    Speed-up: compute on downscaled image (default 0.25).
    """
    y_ref = _to_y01(img_ref).astype("float32")
    y_prc = _to_y01(img_proc).astype("float32")

    ds = float(downscale)
    if ds < 1.0:
        # INTER_AREA is good for downsampling
        y_ref = cv2.resize(y_ref, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA)
        y_prc = cv2.resize(y_prc, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA)

    lap_ref = cv2.Laplacian(y_ref, cv2.CV_32F, ksize=3)
    lap_prc = cv2.Laplacian(y_prc, cv2.CV_32F, ksize=3)

    v_ref = float(np.var(lap_ref))
    v_prc = float(np.var(lap_prc))

    if v_ref < 1e-12:
        return float("inf") if v_prc > 0 else 1.0
    return v_prc / v_ref


# =========================
# データ読み込み
# =========================
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

# =========================
# 特徴量集約（ROI→all_area/all_pupil）
# =========================
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

# =========================
# 18 patterns / パラメータ範囲
# =========================
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

    # クリップ
    dvmin, dvmax = defaults.get(op, (vmin, vmax))
    vmin = max(vmin, dvmin)
    vmax = min(vmax, dvmax)
    return float(vmin), float(vmax)

# =========================
# モデル & GridSearch（あなたの実装を維持）
# =========================
RF_PARAM_GRID_STAGE1 = {"n_estimators": [50, 75, 100, 125, 150],
                        "max_depth": [5, 10, 15],
                        "min_samples_leaf": [1, 3, 5]}
XGB_PARAM_GRID_STAGE1 = {"n_estimators": [100, 200],
                         "max_depth": [3, 5],
                         "learning_rate": [0.03, 0.06]}

RF_PARAM_GRID_STAGE2 = {"n_estimators": [150, 300],
                        "max_depth": [None, 8],
                        "min_samples_leaf": [1, 3]}
XGB_PARAM_GRID_STAGE2 = {"n_estimators": [100, 200],
                         "max_depth": [3, 5],
                         "learning_rate": [0.03, 0.06]}

def iter_param_grid(param_grid: dict):
    keys = list(param_grid.keys())
    if not keys:
        yield {}
        return
    from itertools import product
    for values in product(*[param_grid[k] for k in keys]):
        yield dict(zip(keys, values))

def create_base_regressor(model_type: str, params: dict):
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

def grid_search_stage1(X, y, w, groups, model_type):
    param_grid = RF_PARAM_GRID_STAGE1 if model_type == "RandomForest" else XGB_PARAM_GRID_STAGE1
    total = int(np.prod([len(v) for v in param_grid.values()])) if param_grid else 1
    prog = st.progress(0.0, text=f"1段目 GridSearch... (0/{total})")

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

            m = create_base_regressor(model_type, params)
            m.fit(X_tr, y_tr, sample_weight=w_tr)
            tr_scores.append(r2_score(y_tr, m.predict(X_tr)))
            te_scores.append(r2_score(y_te, m.predict(X_te)))

        mean_te = float(np.mean(te_scores))
        if mean_te > best_score:
            best_score, best_params = mean_te, params
            best_train, best_test = tr_scores, te_scores

        done += 1
        prog.progress(done / total, text=f"1段目 GridSearch... ({done}/{total})")

    final = create_base_regressor(model_type, best_params)
    final.fit(X, y, sample_weight=w)
    prog.progress(1.0, text="1段目 GridSearch 完了 ✅")

    cv = {"mean_train": float(np.mean(best_train)), "std_train": float(np.std(best_train)),
          "mean_test": float(np.mean(best_test)), "std_test": float(np.std(best_test))}
    return final, best_params, cv

def train_stage1_fixed_params(X, y, w, groups, model_type, params):
    splitter, is_group = _get_splitter(groups)
    splits = list(splitter.split(X, y, groups)) if is_group else list(splitter.split(X, y))
    prog = st.progress(0.0, text="1段目 学習中...")

    tr_scores, te_scores = [], []
    for i, (tr_idx, te_idx) in enumerate(splits):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        w_tr = w.iloc[tr_idx]
        m = create_base_regressor(model_type, params or {})
        m.fit(X_tr, y_tr, sample_weight=w_tr)
        tr_scores.append(r2_score(y_tr, m.predict(X_tr)))
        te_scores.append(r2_score(y_te, m.predict(X_te)))
        prog.progress((i + 1) / len(splits), text=f"1段目 学習中...({i+1}/{len(splits)})")

    final = create_base_regressor(model_type, params or {})
    final.fit(X, y, sample_weight=w)
    prog.progress(1.0, text="1段目 学習完了 ✅")

    cv = {"mean_train": float(np.mean(tr_scores)), "std_train": float(np.std(tr_scores)),
          "mean_test": float(np.mean(te_scores)), "std_test": float(np.std(te_scores))}
    return final, cv

def grid_search_stage2(X2, Y2, w, groups, model_type):
    param_grid = RF_PARAM_GRID_STAGE2 if model_type == "RandomForest" else XGB_PARAM_GRID_STAGE2
    total = int(np.prod([len(v) for v in param_grid.values()])) if param_grid else 1
    prog = st.progress(0.0, text=f"2段目 GridSearch... (0/{total})")

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

            base = create_base_regressor(model_type, params)
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
        prog.progress(done / total, text=f"2段目 GridSearch... ({done}/{total})")

    base_final = create_base_regressor(model_type, best_params)
    mo2 = MultiOutputRegressor(base_final)
    mo2.fit(X2, Y2, sample_weight=w)

    r2_each = r2_score(best_Yte_all, best_pred_all, multioutput="raw_values")
    prog.progress(1.0, text="2段目 GridSearch 完了 ✅")
    return mo2, best_params, r2_each, best_score

def train_stage2_simple(X2, Y2, w, groups, model_type):
    splitter, is_group = _get_splitter(groups)
    splits = list(splitter.split(X2, Y2, groups)) if is_group else list(splitter.split(X2, Y2))
    prog = st.progress(0.0, text="2段目 学習中...")

    cv_scores, Yte_list, Ypred_list = [], [], []
    for i, (tr_idx, te_idx) in enumerate(splits):
        X_tr, X_te = X2.iloc[tr_idx], X2.iloc[te_idx]
        Y_tr, Y_te = Y2.iloc[tr_idx], Y2.iloc[te_idx]
        w_tr = w.iloc[tr_idx]

        base = create_base_regressor(model_type, {})
        mo = MultiOutputRegressor(base)
        mo.fit(X_tr, Y_tr, sample_weight=w_tr)

        Y_pred = mo.predict(X_te)
        cv_scores.append(r2_score(Y_te, Y_pred, multioutput="uniform_average"))
        Yte_list.append(Y_te)
        Ypred_list.append(Y_pred)

        prog.progress((i + 1) / len(splits), text=f"2段目 学習中...({i+1}/{len(splits)})")

    Yte_all = pd.concat(Yte_list, axis=0)
    Ypred_all = np.vstack(Ypred_list)
    r2_each = r2_score(Yte_all, Ypred_all, multioutput="raw_values")

    base_final = create_base_regressor(model_type, {})
    mo2 = MultiOutputRegressor(base_final)
    mo2.fit(X2, Y2, sample_weight=w)

    prog.progress(1.0, text="2段目 学習完了 ✅")
    return mo2, {}, r2_each, float(np.mean(cv_scores))

# =========================
# パレート/knee（簡易）
# =========================
def pareto_front(df: pd.DataFrame, x_col: str, y_col: str, maximize_x=True, maximize_y=True):
    # O(N^2)だが候補1000程度なら十分
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
    return df.loc[keep].copy()

def knee_point(front: pd.DataFrame, q_col: str, p_col: str):
    # q: maximize, p: minimize を想定
    f = front.sort_values(q_col, ascending=True).reset_index(drop=True)
    q = f[q_col].values.astype(float)
    p = f[p_col].values.astype(float)

    # 正規化
    qn = (q - q.min()) / (q.max() - q.min() + 1e-9)
    pn = (p - p.min()) / (p.max() - p.min() + 1e-9)

    # (qn, pn) を結ぶ折れ線で、(0,1)と(1,0)の線からの距離最大をkneeとする簡易
    # ここでは「q高 & p低」ほど良いので、理想点(1,0)への距離最小でも良い
    d = np.sqrt((1 - qn) ** 2 + (pn - 0) ** 2)
    idx = int(np.argmin(d))
    return f.iloc[idx].copy()

# =========================
# session_state ユーティリティ
# =========================
def df_fingerprint(df: pd.DataFrame) -> str:
    # 軽量ハッシュ（厳密でなくてOK）：形+列名+先頭/末尾の数値
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

# =========================
# main
# =========================
def main():
    st.set_page_config(page_title="加工推薦（Model2あり）", layout="wide")

    st.markdown("""
    <style>
      html, body, [class*="css"] { font-size: 18px !important; }
      h1, h2, h3 { font-size: 1.25em !important; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🧪 画像特徴 → 縮瞳 → 加工推薦（モデル2あり：高速探索 + SSIM(Y)+HFで品質制御）")
    st.caption(f"features_pupil: {'GPU' if USING_GPU else 'CPU'}")

    # ---------- Data ----------
    st.sidebar.header("📁 データ入力")
    uploaded_file = st.sidebar.file_uploader("実験データ(CSV/Excel)", type=["csv", "xlsx", "xls"])
    if uploaded_file is None:
        st.info("左からデータをアップロードしてください。")
        return

    df_full = load_and_parse_data(uploaded_file)
    fp_df = df_fingerprint(df_full)

    # 被験者除外
    if "folder_name" in df_full.columns:
        all_subjects = sorted(df_full["folder_name"].dropna().unique().tolist())
        excluded = st.sidebar.multiselect("学習に使わない folder_name", options=all_subjects)
        if excluded:
            df_full = df_full[~df_full["folder_name"].isin(excluded)].copy()

    # GroupKFold設定
    st.sidebar.subheader("🧪 CV設定")
    use_group = st.sidebar.checkbox("GroupKFold を使う", value=("folder_name" in df_full.columns))
    groups = None
    if use_group:
        cand = []
        for c in df_full.columns:
            nunique = df_full[c].nunique(dropna=True)
            if 1 < nunique < len(df_full):
                cand.append(c)
        if cand:
            default = cand.index("folder_name") if "folder_name" in cand else 0
            group_col = st.sidebar.selectbox("Group列", options=cand, index=default)
            groups = df_full[group_col]
        else:
            st.sidebar.warning("Group列が見つからず KFold にします")
            groups = None

    sample_weights = compute_sample_weights(df_full)

    # ---------- Tabs ----------
    tab1, tab2 = st.tabs(["📊 データ概要", "🧬 推薦（モデル2あり）"])

    with tab1:
        st.subheader("データセット概要")
        st.write(f"行数: **{len(df_full)}**")
        st.dataframe(df_full.head(), use_container_width=True)

    with tab2:
        st.header("🧬 推薦（モデル2あり）")

        num_cols = df_full.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            st.error("数値列がありません。")
            return

        default_pupil = "corrected_pupil" if "corrected_pupil" in num_cols else num_cols[0]
        pupil_col = st.selectbox("ターゲット（pupil列）", options=num_cols, index=num_cols.index(default_pupil))

        direction = st.radio("良い方向", ["値が小さいほど良い（縮瞳）", "値が大きいほど良い（散瞳）"], index=0, horizontal=True)
        sign_dir = -1.0 if "小さい" in direction else 1.0

        feat_group = st.radio("特徴量グループ", ["all", "all_area", "all_pupil", "ROI"], index=0, horizontal=True)

        # 候補列
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

        st.caption(f"候補特徴量: {len(candidate_cols)} 列")

        top_k = st.slider("上位k（z計算に使う）", 3, min(30, len(candidate_cols)), min(10, len(candidate_cols)))
        n_trials_per_pattern = st.slider("1パターンあたり試行数（高速探索なので大きめ可）", 200, 5000, 1000, 200)

        model1_type = st.radio("モデル1（特徴→pupil）", ["RandomForest", "XGBoost"], index=0, horizontal=True)
        model2_type = st.radio("モデル2（param+orig→特徴）", ["RandomForest", "XGBoost"], index=0, horizontal=True)
        use_grid1 = st.checkbox("1段目 GridSearch", value=True)
        use_grid2 = st.checkbox("2段目 GridSearch", value=True)

        # 探索目的
        objective_mode = st.radio(
            "探索の目的（高速探索で使うスコア）",
            ["z最大化（従来）", "pupil最小化（推奨）"],
            index=1,
            horizontal=True
        )

        # 品質制御
        st.markdown("### 🎛 画質制御（上位候補だけ実画像で評価）")
        quality_mode = st.radio(
            "最終選抜方法",
            ["制約（SSIM>=th & HF_ratio<=th の中で最良）",
             "合成J（α*目的 − β*(1-SSIM) − γ*penalty(HF)）",
             "パレート（Qとpupilのトレードオフ）"],
            index=0
        )
        ssim_th = st.slider("SSIM(Y) 下限", 0.5, 1.0, 0.7, 0.01)
        use_hf = st.checkbox("Compute HF_ratio (Laplacian)", value=False)
        # HF を使うときだけ意味があるので、使わない時は downscale/hf_th を固定でもOK
        hf_downscale = st.slider("HF downscale (lower=faster)", 0.10, 1.00, 0.25, 0.05, disabled=not use_hf)
        hf_th = st.slider("HF_ratio upper bound (1.0=same)", 1.0, 10.0, 2.0, 0.1, disabled=not use_hf)
        max_candidates_for_quality = st.slider("実画像でQ計算する候補数（上位から）", 100, 5000, 1000, 100)

        alpha = st.number_input("α", value=1.0, step=0.1)
        beta  = st.number_input("β", value=1.0, step=0.1)
        gamma = st.number_input("γ（HFペナルティ重み）", value=0.5, step=0.1)

        # 画像入力（品質計算に必要）
        st.subheader("新しい画像の入力（Q計算/BeforeAfter表示）")
        if st.button("🧹 画像アップロードをクリア"):
            # keyを変える（Streamlitのextensionエラー対策にも効く）
            st.session_state["new_img_key"] = str(np.random.randint(0, 10**9))

        if "new_img_key" not in st.session_state:
            st.session_state["new_img_key"] = "new_img"

        new_image_file = st.file_uploader(
            "新しい画像（jpg/jpeg/png）",
            type=["jpg", "jpeg", "png"],
            key=st.session_state["new_img_key"]
        )

        st.markdown("画像なしでも動作します（その場合Q計算はスキップし、最終候補はスコアで決定）。")
        fallback_idx = st.selectbox("fallback行（画像なし時/ *_orig 借用にも使用）", options=df_full.index)

        # -------- 学習ボタン（ここでだけ学習する） --------
        state = get_state()

        def train_key():
            # 学習条件キー（切り替え操作でも state に保持される）
            return (fp_df, pupil_col, feat_group, top_k, model1_type, model2_type, use_grid1, use_grid2, bool(groups is not None))

        if st.button("🚀 学習（モデル1 & モデル2）"):
            key = train_key()
            with st.spinner("学習中..."):
                # ---- Stage1 ----
                X_all = df_full[candidate_cols].copy()
                y = df_full[pupil_col].copy()

                if use_grid1:
                    m1_full, best_p1, cv1_full = grid_search_stage1(X_all, y, sample_weights, groups, model1_type)
                else:
                    best_p1 = {}
                    m1_full, cv1_full = train_stage1_fixed_params(X_all, y, sample_weights, groups, model1_type, best_p1)

                imp = m1_full.feature_importances_
                imp_df = pd.DataFrame({"feature": candidate_cols, "importance": imp}).sort_values("importance", ascending=False).reset_index(drop=True)
                selected = imp_df["feature"].head(top_k).tolist()

                # 上位k再学習（固定パラメータでOK）
                X_sel = df_full[selected].copy()
                m1_sel, cv1_sel = train_stage1_fixed_params(X_sel, y, sample_weights, groups, model1_type, best_p1)

                # z重み（importance * sign(corr)）
                imp_sel = m1_sel.feature_importances_
                signs = []
                for f in selected:
                    r = df_full[f].corr(y)
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
                    st.error("2段目の説明変数（param/_orig）が空です。")
                    st.stop()
                Y2 = df_full[selected].copy()

                if use_grid2:
                    m2, best_p2, r2_each2, r2_mean2 = grid_search_stage2(X2, Y2, sample_weights, groups, model2_type)
                else:
                    m2, best_p2, r2_each2, r2_mean2 = train_stage2_simple(X2, Y2, sample_weights, groups, model2_type)

                # 保存（切り替えで消えない）
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
                }

            st.success("学習完了。以後、切り替え操作をしても学習結果は保持されます。")

        # --------- 学習結果表示（あれば） ---------
        key = train_key()
        trained = state.get(key)

        if trained is None:
            st.info("まず『学習（モデル1 & モデル2）』を押してください。")
            return

        st.subheader("1段目（全部入り）重要度")
        st.dataframe(trained["imp_df"].head(30), use_container_width=True)

        st.subheader("1段目 CV")
        cv1f = trained["cv1_full"]
        cv1s = trained["cv1_sel"]
        st.write(f"全部入り Test R²: **{cv1f['mean_test']:.3f} ± {cv1f['std_test']:.3f}**")
        st.write(f"上位k     Test R²: **{cv1s['mean_test']:.3f} ± {cv1s['std_test']:.3f}**")

        st.subheader("z重み（上位k）")
        z_w_df = pd.DataFrame({"feature": trained["selected"], "weight": [trained["z_w"][f] for f in trained["selected"]]})
        st.dataframe(z_w_df, use_container_width=True)

        st.subheader("2段目 CV（特徴量予測）")
        r2_df2 = pd.DataFrame({"feature": trained["selected"], "Test_R2": trained["r2_each2"]})
        st.dataframe(r2_df2, use_container_width=True)
        st.caption(f"平均 Test R²: {trained['r2_mean2']:.3f}")

        # -------- 推薦探索 --------
        if st.button("🔍 推薦探索（高速探索→上位だけQ計算）"):
            selected = trained["selected"]
            m1 = trained["m1_sel"]
            m2 = trained["m2"]
            z_w = trained["z_w"]
            feat_mean = trained["feat_mean"]
            feat_std = trained["feat_std"]
            img_feature_means = trained["img_feature_means"]

            # 新画像特徴（加工前の表示用）
            new_img_pil = None
            feats_before = {}
            if new_image_file is not None:
                new_img_pil = Image.open(new_image_file).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(new_img_pil), cv2.COLOR_RGB2BGR)
                h, w = img_bgr.shape[:2]
                roi_masks = fp.make_masks(h, w, SCREEN_W_MM, DIST_MM, RES_X, CENTER_DEG, PARAFOVEA_DEG)
                feats_roi = fp.compute_features_for_image(img_bgr, roi_masks,
                                                         screen_w_mm=SCREEN_W_MM, dist_mm=DIST_MM, res_x=RES_X)
                all_masks = fp.make_all_masks()
                feats_all = fp.compute_features_for_image(img_bgr, all_masks,
                                                         screen_w_mm=SCREEN_W_MM, dist_mm=DIST_MM, res_x=RES_X)
                feats_area_pupil = make_weighted_globals_for_single(feats_roi)
                feats_before = {**feats_roi, **feats_all, **feats_area_pupil}
            else:
                # fallback
                feats_before = {f: df_full.loc[fallback_idx, f] for f in selected if f in df_full.columns}

            x_before = pd.Series(index=selected, dtype=float)
            miss = []
            for f in selected:
                if f in feats_before:
                    x_before[f] = float(feats_before[f])
                else:
                    miss.append(f)
                    x_before[f] = np.nan
            if miss:
                st.warning(f"新画像から取れない特徴量: {miss} → 平均で補完")
            x_before = x_before.fillna(img_feature_means)

            pupil_before = float(m1.predict(x_before.values.reshape(1, -1))[0])
            z_before = float(np.sum([z_w[f] * ((x_before[f] - feat_mean[f]) / feat_std[f]) for f in selected]))

            st.subheader("加工前")
            st.write(f"予測 pupil: **{pupil_before:.3f}**")
            st.write(f"z: **{z_before:.3f}**")

            # --- 高速探索（モデル2で特徴を推定） ---
            allowed = generate_allowed_patterns()
            sim_records = []

            # 2段目入力用 *_orig（暫定：fallbackから借用）
            orig_cols = trained["orig_cols"]
            orig_vec = pd.Series(index=orig_cols, dtype=float)
            if orig_cols:
                for c in orig_cols:
                    orig_vec[c] = df_full.loc[fallback_idx, c] if c in df_full.columns else np.nan

            X2_cols = trained["X2_cols"]
            X2_means = trained["X2_means"]

            with st.spinner("高速探索中（モデル2で特徴予測）..."):
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
                    # orig
                    for c in orig_cols:
                        sim_X2[c] = orig_vec.get(c, np.nan)
                    sim_X2 = sim_X2.fillna(X2_means)

                    # params
                    c1, c2, c3 = f"step1_{op1}", f"step2_{op2}", f"step3_{op3}"
                    if c1 in sim_X2.columns: sim_X2[c1] = vals1
                    if c2 in sim_X2.columns: sim_X2[c2] = vals2
                    if c3 in sim_X2.columns: sim_X2[c3] = vals3

                    Y_pred_feats = m2.predict(sim_X2)  # (N, k)
                    pupil_preds = m1.predict(Y_pred_feats)

                    # z
                    scores_z = []
                    for i in range(n_trials_per_pattern):
                        feat_vec = pd.Series(Y_pred_feats[i, :], index=selected)
                        zv = 0.0
                        for f in selected:
                            zv += z_w[f] * ((feat_vec[f] - feat_mean[f]) / feat_std[f])
                        scores_z.append(zv)
                    scores_z = np.array(scores_z)

                    df_pat = pd.DataFrame({
                        "pattern": pat,
                        "Score_z": scores_z,
                        "Pupil": pupil_preds,
                        "step1_op": op1, "step1_val": vals1,
                        "step2_op": op2, "step2_val": vals2,
                        "step3_op": op3, "step3_val": vals3,
                    })
                    for j, f in enumerate(selected):
                        df_pat[f] = Y_pred_feats[:, j]

                    # 目的（高速探索でのランキング用）
                    if objective_mode.startswith("z"):
                        df_pat["Objective"] = df_pat["Score_z"]
                    else:
                        df_pat["Objective"] = -df_pat["Pupil"]  # pupil小さいほど良い = -pupil大きいほど良い

                    sim_records.append(df_pat)

                    prog.progress((pi + 1) / total_steps, text=f"{pi+1}/{total_steps} patterns")

                sim_all = pd.concat(sim_records, ignore_index=True)

            st.subheader("高速探索：18パターン要約")
            def top5_mean(x):
                k = max(1, int(len(x) * 0.05))
                return x.nlargest(k).mean()

            summary = (sim_all.groupby("pattern")["Objective"]
                       .agg(max_obj="max", top5_mean=top5_mean)
                       .reset_index()
                       .sort_values(["top5_mean", "max_obj"], ascending=False))
            st.dataframe(summary, use_container_width=True)

            # --- 上位候補に対してだけ Q計算（実画像がある場合） ---
            if new_img_pil is None:
                st.warning("画像が無いのでQ計算はスキップ。高速探索のObjective最大を採用します。")
                best = sim_all.loc[sim_all["Objective"].idxmax()].copy()
                best["SSIM"] = np.nan
                best["HF_ratio"] = np.nan
            else:
                cand = sim_all.sort_values("Objective", ascending=False).head(max_candidates_for_quality).copy()

                with st.spinner("上位候補に対して実画像で SSIM(Y) + HF_ratio を計算中..."):
                    ssim_list, hf_list, J_list = [], [], []
                    feasible_mask = []

                    for _, row in cand.iterrows():
                        ops = [row["step1_op"], row["step2_op"], row["step3_op"]]
                        vals = [row["step1_val"], row["step2_val"], row["step3_val"]]
                        img_proc = apply_processing_sequence(new_img_pil, ops, vals)

                        q = compute_ssim_y(new_img_pil, img_proc)

                        if use_hf:
                            hf = hf_ratio_laplacian(new_img_pil, img_proc, downscale=hf_downscale)
                            penalty_hf = max(0.0, hf - hf_th)
                        else:
                            hf = np.nan          # もしくは 1.0 でもOK
                            penalty_hf = 0.0

                        ssim_list.append(q)
                        hf_list.append(hf)

                        obj = float(row["Objective"])
                        J = alpha * obj - beta * (1.0 - q) - gamma * penalty_hf

                        # feasible 判定も HF を使う時だけ
                        if use_hf:
                            feasible_mask.append((q >= ssim_th) and (hf <= hf_th))
                        else:
                            feasible_mask.append(q >= ssim_th)

                        J_list.append(J)

                        feasible_mask.append((q >= ssim_th) and (hf <= hf_th))

                    cand["SSIM"] = ssim_list
                    cand["HF_ratio"] = hf_list
                    cand["J"] = J_list
                    cand["feasible"] = feasible_mask

                st.subheader("Q計算した候補（上位）")
                st.dataframe(
                    cand[["pattern","Objective","Score_z","Pupil","SSIM","HF_ratio","J","feasible",
                          "step1_op","step1_val","step2_op","step2_val","step3_op","step3_val"]].head(200),
                    use_container_width=True
                )

                # 散布図
                # --- Pareto (maximize SSIM, minimize Pupil) ---
                front = pareto_front(cand, x_col="SSIM", y_col="Pupil", maximize_x=True, maximize_y=False)
                cand["is_pareto"] = cand.index.isin(front.index)

                # knee on the front (uses SSIM maximize, Pupil minimize assumption)
                knee = knee_point(front, q_col="SSIM", p_col="Pupil")

                st.subheader("Pareto front (SSIM vs Pupil)")
                st.dataframe(
                    front[["pattern","Pupil","SSIM","Objective","HF_ratio","J",
                        "step1_op","step1_val","step2_op","step2_val","step3_op","step3_val"]]
                    .sort_values(["SSIM","Pupil"], ascending=[False, True])
                    .reset_index(drop=True),
                    use_container_width=True
                )

                # --- Plot (English only) ---
                fig, ax = plt.subplots(figsize=(8, 6))

                # All candidates
                ax.scatter(
                    cand["SSIM"], cand["Pupil"],
                    alpha=0.25, s=18,
                    label="Candidates"
                )

                # Pareto-optimal points
                ax.scatter(
                    front["SSIM"], front["Pupil"],
                    alpha=0.95, s=55,
                    edgecolors="black", linewidths=0.6,
                    label="Pareto-optimal"
                )

                # Knee / selected point
                ax.scatter(
                    [float(knee["SSIM"])], [float(knee["Pupil"])],
                    marker="*", s=220,
                    edgecolors="black", linewidths=0.8,
                    label="Selected (knee)"
                )

                ax.set_xlabel("SSIM(Y) (higher is better)")
                ax.set_ylabel("Predicted pupil (lower is better)")
                ax.grid(True, linestyle="--", alpha=0.4)
                ax.legend()
                st.pyplot(fig)


                # 選抜
                if quality_mode.startswith("制約"):
                    feasible = cand[cand["feasible"]].copy()
                    if feasible.empty:
                        st.warning("制約を満たす候補がありません。J最大で代替します。")
                        best = cand.loc[cand["J"].idxmax()].copy()
                    else:
                        best = feasible.loc[feasible["Objective"].idxmax()].copy()
                elif quality_mode.startswith("合成"):
                    best = cand.loc[cand["J"].idxmax()].copy()
                else:
                    # パレート：SSIM(最大) & pupil(最小)
                    front = pareto_front(cand, x_col="SSIM", y_col="Pupil", maximize_x=True, maximize_y=False)
                    best = knee_point(front, q_col="SSIM", p_col="Pupil")
                    st.subheader("パレートフロント（抽出）")
                    st.dataframe(front[["pattern","Pupil","SSIM","HF_ratio","J"]].sort_values("SSIM", ascending=False),
                                 use_container_width=True)

            # --- ベスト表示 ---
            st.divider()
            st.subheader("👑 ベスト加工案（モデル2あり）")

            ops_best = [best["step1_op"], best["step2_op"], best["step3_op"]]
            vals_best = [best["step1_val"], best["step2_val"], best["step3_val"]]

            st.markdown(
                f"- pattern: **{best['pattern'].replace('_',' → ')}**  \n"
                f"- step1: **{ops_best[0]}** = `{vals_best[0]:.3f}`  \n"
                f"- step2: **{ops_best[1]}** = `{vals_best[1]:.3f}`  \n"
                f"- step3: **{ops_best[2]}** = `{vals_best[2]:.3f}`  \n"
                f"- predicted pupil: **{float(best['Pupil']):.3f}**  \n"
                f"- SSIM(Y): **{float(best.get('SSIM', np.nan)):.3f}**  \n"
                f"- HF_ratio: **{float(best.get('HF_ratio', np.nan)):.3f}**"
            )

            if new_img_pil is not None:
                img_after = apply_processing_sequence(new_img_pil, ops_best, vals_best)
                c1, c2 = st.columns(2)
                with c1:
                    st.image(new_img_pil, caption="Before", use_container_width=True)
                with c2:
                    st.image(img_after, caption="After", use_container_width=True)

if __name__ == "__main__":
    main()
