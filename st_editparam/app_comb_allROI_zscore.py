# app_comb_allROI_zscore_gs_v2.py
# Updated: 可視化プロセス追加 & 再学習を避けるためのセッション管理

import warnings
warnings.simplefilter("ignore")

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import cv2
from PIL import Image

from xgboost import XGBRegressor

# ==== features_pupil / GPU 対応 =====================================
try:
    import features_pupil as fp
    if cv2.cuda.getCudaEnabledDeviceCount() == 0:
        raise ImportError("CUDA device not found")
    print("[INFO] Using GPU version (features_pupil_gpu)")
except Exception:
    import features_pupil as fp
    print("[INFO] Using CPU version (features_pupil)")

# ==== 画面・観察距離など（features_pupil 用） =======================
SCREEN_W_MM = 260
DIST_MM     = 450
RES_X       = 6000
CENTER_DEG  = 2
PARAFOVEA_DEG = 5

# ROI 名と重み（面積 / 瞳孔）
ROI_REGIONS = ("center", "parafovea", "periphery")

# 面積比（目安：中心 ≒ 4%, 傍心 ≒ 20%, 周辺 ≒ 76%）
ROI_AREA_WEIGHTS = {
    "center": 0.04,
    "parafovea": 0.20,
    "periphery": 0.76,
}

# 瞳孔反映用の重み（仮）
ROI_PUPIL_WEIGHTS = {
    "center": 0.5,
    "parafovea": 0.3,
    "periphery": 0.2,
}

# 画像特徴量として使わない列（メタ列）
NON_FEATURE_COLS = [
    "folder_name",
    "平均_変化率",
    "平均_変化量_z",
    "両眼.注視Z座標[mm]",
    "pattern_id",
    "param1", "param2", "param3",
    "param1_val", "param2_val", "param3_val",
]

# ==========================================
# 画像加工ライブラリ（輝度・コントラスト等）
# ==========================================
def slide_brightness(image: Image.Image, shift: float) -> Image.Image:
    img_np = np.array(image).astype("float32") / 255.0
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + shift / 255.0, 0.0, 1.0)
    img_np = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(np.round(img_np * 255).astype("uint8"))


def adjust_contrast_adachi(image: Image.Image, scale: float) -> Image.Image:
    img_np = np.array(image)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = cv2.convertScaleAbs(hsv[:, :, 2], alpha=scale)
    img_np = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(img_np.astype("uint8"))


def adjust_sharpness(image: Image.Image, sharpness: float) -> Image.Image:
    img_array = np.array(image)
    kernel = np.array(
        [
            [-sharpness, -sharpness, -sharpness],
            [-sharpness, 1 + 8 * sharpness, -sharpness],
            [-sharpness, -sharpness, -sharpness],
        ],
        dtype=np.float32,
    )
    img_sharpness = cv2.filter2D(img_array, -1, kernel)
    return Image.fromarray(img_sharpness)


def adjust_gamma(image: Image.Image, gamma: float) -> Image.Image:
    image = image.convert("RGB")
    gamma_correction = lambda v: int(((v / 255.0) ** gamma) * 255)
    return image.point(gamma_correction)


def stretch_rgb_clahe(image: Image.Image, clipLimit: float = 2.0, tile: int = 8) -> Image.Image:
    img_np = np.array(image).astype("float32") / 255.0
    tile = int(tile)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tile, tile))
    for i in range(3):
        img_np[:, :, i] = clahe.apply((img_np[:, :, i] * 255).astype("uint8")) / 255.0
    return Image.fromarray(np.round(img_np * 255).astype("uint8"))


def apply_one_op(image: Image.Image, op: str, val: float) -> Image.Image:
    if op == "brightness":
        return slide_brightness(image, shift=val)
    elif op == "contrast":
        return adjust_contrast_adachi(image, scale=val)
    elif op == "gamma":
        return adjust_gamma(image, gamma=val)
    elif op == "sharpness":
        return adjust_sharpness(image, sharpness=val)
    elif op == "equalization":
        tile = max(4, min(64, int(round(val))))
        return stretch_rgb_clahe(image, clipLimit=2.0, tile=tile)
    else:
        return image


def apply_processing_sequence(image: Image.Image, ops, vals) -> Image.Image:
    out = image.copy()
    for op, v in zip(ops, vals):
        if op is None or op == "None":
            continue
        out = apply_one_op(out, op, float(v))
    return out


# ==========================================
# データ読み込み & パース
# ==========================================
@st.cache_data
def load_and_parse_data(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    def parse_params_ordered(name):
        if pd.isna(name):
            return {
                "param1": "None", "param1_val": 0.0,
                "param2": "None", "param2_val": 0.0,
                "param3": "None", "param3_val": 0.0,
            }

        clean_name = str(name).replace(".jpg", "").replace(".JPG", "")
        parts = clean_name.split("_")
        valid_ops = ["brightness", "contrast", "gamma", "sharpness", "equalization"]

        params = []
        for part in parts:
            for op in valid_ops:
                if part.startswith(op):
                    try:
                        val_str = part.replace(op, "")
                        val = float(val_str)
                        params.append((op, val))
                    except ValueError:
                        pass
                    break

        while len(params) < 3:
            params.append(("None", 0.0))

        return {
            "param1": params[0][0], "param1_val": params[0][1],
            "param2": params[1][0], "param2_val": params[1][1],
            "param3": params[2][0], "param3_val": params[2][1],
        }

    if "image_name" in df.columns:
        parsed_list = [parse_params_ordered(n) for n in df["image_name"]]
        params_df = pd.DataFrame(parsed_list)

        params_df["pattern_id"] = (
            params_df["param1"] + " → " + params_df["param2"] + " → " + params_df["param3"]
        )

        cols_to_use = params_df.columns.tolist()
        df = df.drop(columns=[c for c in cols_to_use if c in df.columns], errors="ignore")
        df_full = pd.concat([df, params_df], axis=1)
    else:
        df_full = df.copy()
        if "pattern_id" not in df_full.columns:
            df_full["pattern_id"] = "no_pattern"

    return df_full


def create_interaction_features(df):
    valid_ops = ["brightness", "contrast", "gamma", "sharpness", "equalization"]
    X_dict = {}
    for i in range(1, 4):
        col_op = f"param{i}"
        col_val = f"param{i}_val"
        if col_op not in df.columns or col_val not in df.columns:
            continue
        for op in valid_ops:
            mask = (df[col_op] == op).astype(float)
            X_dict[f"step{i}_{op}"] = mask * df[col_val]
    if not X_dict:
        return pd.DataFrame(index=df.index)
    return pd.DataFrame(X_dict, index=df.index)


def compute_sample_weights(df):
    key = df["pattern_id"]
    freq = key.value_counts()
    w = key.map(freq).astype(float)
    w = 1.0 / w
    w *= len(w) / w.sum()
    return w


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
    if col_op not in df.columns or col_val not in df.columns:
        if op == "gamma":
            vmin, vmax = 0.7, 1.3
        elif op == "equalization":
            vmin, vmax = 5.0, 40.0
        elif op == "brightness":
            vmin, vmax = -80, 80
        elif op == "contrast":
            vmin, vmax = 0.7, 1.3
        else:
            vmin, vmax = 0.0, 1.5
        return float(vmin), float(vmax)

    mask = df[col_op] == op

    if mask.any():
        v = df.loc[mask, col_val].astype(float)
        vmin = float(v.quantile(q_low))
        vmax = float(v.quantile(q_high))
        if vmin == vmax:
            vmin -= abs(vmin) * 0.1 + 1e-3
            vmax += abs(vmax) * 0.1 + 1e-3
    else:
        if op == "gamma":
            vmin, vmax = 0.3, 1.5
        elif op == "equalization":
            vmin, vmax = 5.0, 50.0
        elif op == "brightness":
            vmin, vmax = -50, 50
        elif op == "contrast":
            vmin, vmax = 0.5, 2.0
        else:
            vmin, vmax = 0.0, 3.0

    if op == "gamma":
        vmin = max(vmin, 0.7); vmax = min(vmax, 1.3)
    elif op == "contrast":
        vmin = max(vmin, 0.7); vmax = min(vmax, 1.3)
    elif op == "sharpness":
        vmin = max(vmin, 0.0); vmax = min(vmax, 1.5)
    elif op == "brightness":
        vmin = max(vmin, -80.0); vmax = min(vmax, 80.0)
    elif op == "equalization":
        vmin = max(vmin, 5.0); vmax = min(vmax, 40.0)

    return float(vmin), float(vmax)


# ==========================
# モデル・グリッドサーチ関連
# ==========================
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


def create_base_regressor(model_type: str, params: dict):
    if model_type == "RandomForest":
        base_params = {
            "n_estimators": 300,
            "random_state": 42,
            "n_jobs": -1,
        }
        base_params.update(params)
        return RandomForestRegressor(**base_params)
    else:
        base_params = {
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
        }
        base_params.update(params)
        return XGBRegressor(**base_params)


def grid_search_stage1(X, y, sample_weights, groups, model_type: str):
    if model_type == "RandomForest":
        param_grid = RF_PARAM_GRID_STAGE1
    else:
        param_grid = XGB_PARAM_GRID_STAGE1

    total_combo = 1
    for v in param_grid.values():
        total_combo *= len(v)

    prog = st.progress(0.0, text=f"1段目 GridSearch 実行中... (0/{total_combo})")

    if groups is not None:
        splitter = GroupKFold(n_splits=5)
        def split_iter():
            return splitter.split(X, y, groups)
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        def split_iter():
            return splitter.split(X, y)

    best_score = -np.inf
    best_params = None
    best_train_scores = None
    best_test_scores = None

    done = 0
    for params in iter_param_grid(param_grid):
        train_scores = []
        test_scores = []
        for tr_idx, te_idx in split_iter():
            X_tr = X.iloc[tr_idx]
            X_te = X.iloc[te_idx]
            y_tr = y.iloc[tr_idx]
            y_te = y.iloc[te_idx]
            w_tr = sample_weights.iloc[tr_idx]

            model = create_base_regressor(model_type, params)
            model.fit(X_tr, y_tr, sample_weight=w_tr)

            y_tr_pred = model.predict(X_tr)
            y_te_pred = model.predict(X_te)
            train_scores.append(r2_score(y_tr, y_tr_pred))
            test_scores.append(r2_score(y_te, y_te_pred))

        mean_test = float(np.mean(test_scores))
        if mean_test > best_score:
            best_score = mean_test
            best_params = params
            best_train_scores = train_scores
            best_test_scores = test_scores

        done += 1
        prog.progress(done / total_combo,
                      text=f"1段目 GridSearch 実行中... ({done}/{total_combo})")

    final_model = create_base_regressor(model_type, best_params)
    final_model.fit(X, y, sample_weight=sample_weights)

    prog.progress(1.0, text="1段目 GridSearch 完了 ✅")

    cv_summary = {
        "mean_train": float(np.mean(best_train_scores)),
        "std_train": float(np.std(best_train_scores)),
        "mean_test": float(np.mean(best_test_scores)),
        "std_test": float(np.std(best_test_scores)),
    }
    return final_model, best_params, cv_summary


def grid_search_stage2(X2, Y2, sample_weights, groups, model_type: str):
    if model_type == "RandomForest":
        param_grid = RF_PARAM_GRID_STAGE2
    else:
        param_grid = XGB_PARAM_GRID_STAGE2

    total_combo = 1
    for v in param_grid.values():
        total_combo *= len(v)

    prog2 = st.progress(0.0, text=f"2段目 GridSearch 実行中... (0/{total_combo})")

    if groups is not None:
        splitter = GroupKFold(n_splits=5)
        def split_iter():
            return splitter.split(X2, Y2, groups)
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        def split_iter():
            return splitter.split(X2, Y2)

    best_score2 = -np.inf
    best_params2 = None
    best_Y2_te_all = None
    best_pred2 = None

    done2 = 0
    for params in iter_param_grid(param_grid):
        cv_scores = []
        Y2_te_list = []
        Y2_pred_list = []
        for tr_idx, te_idx in split_iter():
            X2_tr = X2.iloc[tr_idx]
            X2_te = X2.iloc[te_idx]
            Y2_tr = Y2.iloc[tr_idx]
            Y2_te = Y2.iloc[te_idx]
            w2_tr = sample_weights.iloc[tr_idx]

            base_est = create_base_regressor(model_type, params)
            mo = MultiOutputRegressor(base_est)
            mo.fit(X2_tr, Y2_tr, sample_weight=w2_tr)

            Y2_pred = mo.predict(X2_te)
            score = r2_score(Y2_te, Y2_pred, multioutput="uniform_average")
            cv_scores.append(score)

            Y2_te_list.append(Y2_te)
            Y2_pred_list.append(Y2_pred)

        mean_cv = float(np.mean(cv_scores))
        if mean_cv > best_score2:
            best_score2 = mean_cv
            best_params2 = params
            best_Y2_te_all = pd.concat(Y2_te_list, axis=0)
            best_pred2 = np.vstack(Y2_pred_list)

        done2 += 1
        prog2.progress(done2 / total_combo,
                       text=f"2段目 GridSearch 実行中... ({done2}/{total_combo})")

    base_est_final = create_base_regressor(model_type, best_params2)
    mo2 = MultiOutputRegressor(base_est_final)
    mo2.fit(X2, Y2, sample_weight=sample_weights)

    r2_each2 = r2_score(best_Y2_te_all, best_pred2, multioutput="raw_values")
    r2_mean2 = best_score2

    prog2.progress(1.0, text="2段目 GridSearch 完了 ✅")

    return mo2, best_params2, r2_each2, r2_mean2


# ==== GridSearch OFF 用の簡易トレーナ ============================
def train_stage1_fixed_params(X, y, sample_weights, groups, model_type: str, params: dict):
    prog = st.progress(0.0, text="1段目 モデル学習中...（GridSearch なし）")

    if groups is not None:
        splitter = GroupKFold(n_splits=5)
        splits = list(splitter.split(X, y, groups))
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(splitter.split(X, y))

    train_scores = []
    test_scores = []

    for i, (tr_idx, te_idx) in enumerate(splits):
        X_tr = X.iloc[tr_idx]
        X_te = X.iloc[te_idx]
        y_tr = y.iloc[tr_idx]
        y_te = y.iloc[te_idx]
        w_tr = sample_weights.iloc[tr_idx]

        model = create_base_regressor(model_type, params or {})
        model.fit(X_tr, y_tr, sample_weight=w_tr)

        y_tr_pred = model.predict(X_tr)
        y_te_pred = model.predict(X_te)
        train_scores.append(r2_score(y_tr, y_tr_pred))
        test_scores.append(r2_score(y_te, y_te_pred))

        prog.progress((i + 1) / len(splits),
                      text=f"1段目 モデル学習中...（{i+1}/{len(splits)} fold）")

    final_model = create_base_regressor(model_type, params or {})
    final_model.fit(X, y, sample_weight=sample_weights)

    prog.progress(1.0, text="1段目 モデル学習 完了 ✅")

    cv_summary = {
        "mean_train": float(np.mean(train_scores)),
        "std_train": float(np.std(train_scores)),
        "mean_test": float(np.mean(test_scores)),
        "std_test": float(np.std(test_scores)),
    }
    return final_model, cv_summary


def train_stage2_simple(X2, Y2, sample_weights, groups, model_type: str):
    prog2 = st.progress(0.0, text="2段目 モデル学習中...（GridSearch なし）")

    if groups is not None:
        splitter = GroupKFold(n_splits=5)
        splits = list(splitter.split(X2, Y2, groups))
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(splitter.split(X2, Y2))

    cv_scores = []
    Y2_te_list = []
    Y2_pred_list = []

    for i, (tr_idx, te_idx) in enumerate(splits):
        X2_tr = X2.iloc[tr_idx]
        X2_te = X2.iloc[te_idx]
        Y2_tr = Y2.iloc[tr_idx]
        Y2_te = Y2.iloc[te_idx]
        w2_tr = sample_weights.iloc[tr_idx]

        base_est = create_base_regressor(model_type, {})
        mo = MultiOutputRegressor(base_est)
        mo.fit(X2_tr, Y2_tr, sample_weight=w2_tr)

        Y2_pred = mo.predict(X2_te)
        score = r2_score(Y2_te, Y2_pred, multioutput="uniform_average")
        cv_scores.append(score)

        Y2_te_list.append(Y2_te)
        Y2_pred_list.append(Y2_pred)

        prog2.progress((i + 1) / len(splits),
                       text=f"2段目 モデル学習中...（{i+1}/{len(splits)} fold）")

    Y2_te_all = pd.concat(Y2_te_list, axis=0)
    Y2_pred_all = np.vstack(Y2_pred_list)
    r2_each2 = r2_score(Y2_te_all, Y2_pred_all, multioutput="raw_values")
    r2_mean2 = float(np.mean(cv_scores))

    base_est_final = create_base_regressor(model_type, {})
    mo2 = MultiOutputRegressor(base_est_final)
    mo2.fit(X2, Y2, sample_weight=sample_weights)

    prog2.progress(1.0, text="2段目 モデル学習 完了 ✅")

    return mo2, {}, r2_each2, r2_mean2


# ==== ROI → all_area / all_pupil（1枚分）の集約 ====================
def make_weighted_globals_for_single(roi_feats: dict):
    metric_map = {}  # feat_name -> {region: value}
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
        num_area = 0.0
        den_area = 0.0
        for r, val in region_vals.items():
            w = ROI_AREA_WEIGHTS.get(r, 0.0)
            if w == 0:
                continue
            num_area += float(val) * w
            den_area += w
        out[f"all_area_{feat_name}"] = num_area / den_area if den_area > 0 else np.nan

        # pupil
        num_pupil = 0.0
        den_pupil = 0.0
        for r, val in region_vals.items():
            w = ROI_PUPIL_WEIGHTS.get(r, 0.0)
            if w == 0:
                continue
            num_pupil += float(val) * w
            den_pupil += w
        out[f"all_pupil_{feat_name}"] = num_pupil / den_pupil if den_pupil > 0 else np.nan

    return out


# ==========================
# パイプライン実行関数
# ==========================
def run_pipeline(
    df_full,
    sample_weights,
    groups,
    pupil_col,
    sign_dir,
    feat_group,
    candidate_cols,
    top_k,
    n_trials_per_pattern,
    model1_type,
    model2_type,
    use_grid1,
    use_grid2,
    new_image_file,
    fallback_idx,
):
    results = {}

    # -------------------------
    # 1段目: 画像特徴 → 縮瞳 + z の重み
    # -------------------------
    with st.spinner("1段目: 画像特徴 → 縮瞳 モデルを学習中 ..."):
        X_img_all = df_full[candidate_cols].copy()
        y_pupil   = df_full[pupil_col]

        if use_grid1:
            rf_pupil_full, best_params1, cv_summary1_full = grid_search_stage1(
                X_img_all, y_pupil, sample_weights, groups, model1_type
            )
        else:
            best_params1 = {}
            rf_pupil_full, cv_summary1_full = train_stage1_fixed_params(
                X_img_all, y_pupil, sample_weights, groups, model1_type, best_params1
            )

        imp_all = rf_pupil_full.feature_importances_
        imp_df_all = (
            pd.DataFrame({"feature": candidate_cols, "importance": imp_all})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        # 上位kだけ使用
        selected_features = imp_df_all["feature"].head(top_k).tolist()

        X_img_sel = df_full[selected_features].copy()
        rf_pupil, cv_summary1_sel = train_stage1_fixed_params(
            X_img_sel, y_pupil, sample_weights, groups, model1_type, best_params1
        )

        # z 用の重み
        imp_sel = rf_pupil.feature_importances_
        signs = []
        for f in selected_features:
            r = df_full[f].corr(y_pupil)
            if pd.isna(r) or r == 0:
                s = 0.0
            else:
                s = sign_dir * np.sign(r)
            signs.append(s)
        signs = np.array(signs)

        w_raw = imp_sel * signs
        if np.any(w_raw != 0):
            thresh = 0.01 * np.max(np.abs(w_raw))
            w_raw[np.abs(w_raw) < thresh] = 0.0
        if np.sum(np.abs(w_raw)) > 0:
            w_raw = w_raw / np.sum(np.abs(w_raw))

        feature_weights = pd.Series(w_raw, index=selected_features)

        feat_mean = df_full[selected_features].mean()
        feat_std  = df_full[selected_features].std().replace(0, 1.0)
        img_feature_means = df_full[selected_features].mean()

    # -------------------------
    # 2段目: (param + *_orig) → selected特徴量
    # -------------------------
    with st.spinner("2段目: 加工 + 元画像特徴 → 重要特徴量 モデルを学習中 ..."):
        X_param = create_interaction_features(df_full)

        orig_cols = [
            c for c in df_full.columns
            if c.endswith("_orig") or c.endswith("_orig_area") or c.endswith("_orig_pupil")
        ]
        if orig_cols:
            X_orig = df_full[orig_cols].copy()
            X2 = pd.concat([X_param, X_orig], axis=1)
        else:
            X_orig = pd.DataFrame(index=df_full.index)
            X2 = X_param.copy()

        if X2.empty:
            raise RuntimeError("param系・_orig系の説明変数がありません。2段目モデルを構築できません。")

        Y2 = df_full[selected_features].copy()

        if use_grid2:
            mo2, best_params2, r2_each2, r2_mean2 = grid_search_stage2(
                X2, Y2, sample_weights, groups, model2_type
            )
        else:
            mo2, best_params2, r2_each2, r2_mean2 = train_stage2_simple(
                X2, Y2, sample_weights, groups, model2_type
            )

        X2_means = X2.mean()

    # -------------------------
    # 新しい画像の特徴量
    # -------------------------
    with st.spinner("新しい画像の特徴量計算中..."):
        new_image_for_display = None
        feats_before = {}

        if new_image_file is not None:
            pil_img = Image.open(new_image_file).convert("RGB")
            new_image_for_display = pil_img
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            h, w = img_bgr.shape[:2]

            roi_masks = fp.make_masks(
                h, w, SCREEN_W_MM, DIST_MM, RES_X, CENTER_DEG, PARAFOVEA_DEG
            )
            feats_roi = fp.compute_features_for_image(
                img_bgr, roi_masks,
                screen_w_mm=SCREEN_W_MM, dist_mm=DIST_MM, res_x=RES_X
            )

            all_masks = fp.make_all_masks()
            feats_all = fp.compute_features_for_image(
                img_bgr, all_masks,
                screen_w_mm=SCREEN_W_MM, dist_mm=DIST_MM, res_x=RES_X
            )

            feats_area_pupil = make_weighted_globals_for_single(feats_roi)

            feats_before = {**feats_roi, **feats_all, **feats_area_pupil}
        else:
            feats_before = {
                c: df_full.loc[fallback_idx, c]
                for c in selected_features
                if c in df_full.columns
            }

        x_before = pd.Series(index=selected_features, dtype=float)
        missing_feats = []
        for f in selected_features:
            if f in feats_before:
                x_before[f] = feats_before[f]
            else:
                missing_feats.append(f)
                x_before[f] = np.nan

        if missing_feats:
            missing_feats_list = missing_feats  # keep

        x_before = x_before.fillna(img_feature_means)
        pupil_before = float(
            rf_pupil.predict(x_before.values.reshape(1, -1))[0]
        )
        z_before = float(
            sum(
                feature_weights[f] * ((x_before[f] - feat_mean[f]) / feat_std[f])
                for f in selected_features
            )
        )

        # z の途中式
        z_details = []
        for f in selected_features:
            x_val = float(x_before[f])
            mu = float(feat_mean[f])
            sd = float(feat_std[f]) if feat_std[f] != 0 else 1.0
            z_i = (x_val - mu) / sd
            w_i = float(feature_weights[f])
            contrib = w_i * z_i
            z_details.append({
                "特徴量": f,
                "値 x_i": x_val,
                "平均 μ_i": mu,
                "標準偏差 σ_i": sd,
                "標準化 (x_i-μ_i)/σ_i": z_i,
                "重み w_i": w_i,
                "寄与 w_i * z_i": contrib,
            })

        df_z = pd.DataFrame(z_details)
        df_z["寄与 w_i * z_i 累積"] = df_z["寄与 w_i * z_i"].cumsum()

    # -------------------------
    # 18パターン × ランダムサーチ
    # -------------------------
    with st.spinner("18パターン × ランダムサーチでベスト加工を探索中..."):
        allowed_patterns = generate_allowed_patterns()
        sim_records = []

        # orig 特徴
        orig_cols = [
            c for c in df_full.columns
            if c.endswith("_orig") or c.endswith("_orig_area") or c.endswith("_orig_pupil")
        ]
        orig_vec = pd.Series(index=orig_cols, dtype=float)

        if new_image_file is not None:
            for c in orig_cols:
                orig_vec[c] = df_full.loc[fallback_idx, c] if c in df_full.columns else np.nan
        else:
            if orig_cols:
                orig_vec = df_full.loc[fallback_idx, orig_cols].astype(float)

        for pat in allowed_patterns:
            op1, op2, op3 = pat.split("_")
            v1min, v1max = get_param_range(df_full, 1, op1)
            v2min, v2max = get_param_range(df_full, 2, op2)
            v3min, v3max = get_param_range(df_full, 3, op3)

            vals1 = np.random.uniform(v1min, v1max, n_trials_per_pattern)
            vals2 = np.random.uniform(v2min, v2max, n_trials_per_pattern)
            vals3 = np.random.uniform(v3min, v3max, n_trials_per_pattern)

            sim_X2 = pd.DataFrame(0.0, index=range(n_trials_per_pattern), columns=X2.columns)

            for c in orig_cols:
                sim_X2[c] = orig_vec.get(c, 0.0)
            sim_X2 = sim_X2.fillna(X2_means)

            c1 = f"step1_{op1}"
            c2 = f"step2_{op2}"
            c3 = f"step3_{op3}"
            if c1 in sim_X2.columns:
                sim_X2[c1] = vals1
            if c2 in sim_X2.columns:
                sim_X2[c2] = vals2
            if c3 in sim_X2.columns:
                sim_X2[c3] = vals3

            Y_pred_feats = mo2.predict(sim_X2)
            pupil_preds = rf_pupil.predict(Y_pred_feats)

            scores = []
            for row_idx in range(n_trials_per_pattern):
                feat_vec = pd.Series(Y_pred_feats[row_idx, :], index=selected_features)
                val_z = 0.0
                for f in selected_features:
                    x_norm = (feat_vec[f] - feat_mean[f]) / feat_std[f]
                    val_z += feature_weights[f] * x_norm
                scores.append(val_z)
            scores = np.array(scores)

            df_pat = pd.DataFrame(
                {
                    "pattern": pat,
                    "Score": scores,
                    "Pupil": pupil_preds,
                    "step1_op": op1,
                    "step2_op": op2,
                    "step3_op": op3,
                    "step1_val": vals1,
                    "step2_val": vals2,
                    "step3_val": vals3,
                }
            )
            for i, feat in enumerate(selected_features):
                df_pat[feat] = Y_pred_feats[:, i]

            sim_records.append(df_pat)

        sim_all = pd.concat(sim_records, ignore_index=True)

        def top5_mean(x):
            k = max(1, int(len(x) * 0.05))
            return x.nlargest(k).mean()

        summary = (
            sim_all.groupby("pattern")["Score"]
            .agg(max_score="max", top5_mean=top5_mean)
            .reset_index()
            .sort_values(["top5_mean", "max_score"], ascending=False)
            .reset_index(drop=True)
        )

        best_row = sim_all.loc[sim_all["Score"].idxmax()].copy()

        pupil_after = float(best_row["Pupil"])
        delta_pupil = pupil_after - pupil_before
        ratio_pupil = np.nan if pupil_before == 0 else delta_pupil / pupil_before * 100.0

        feat_after_vec = pd.Series(
            [best_row[f] for f in selected_features],
            index=selected_features,
        )
        z_after = float(
            sum(
                feature_weights[f] * ((feat_after_vec[f] - feat_mean[f]) / feat_std[f])
                for f in selected_features
            )
        )
        delta_z = z_after - z_before

    # 結果をまとめて返す
    results.update(
        dict(
            df_full=df_full,
            pupil_col=pupil_col,
            sign_dir=sign_dir,
            feat_group=feat_group,
            candidate_cols=candidate_cols,
            top_k=top_k,
            n_trials_per_pattern=n_trials_per_pattern,
            model1_type=model1_type,
            model2_type=model2_type,
            use_grid1=use_grid1,
            use_grid2=use_grid2,
            rf_pupil_full=rf_pupil_full,
            best_params1=best_params1,
            cv_summary1_full=cv_summary1_full,
            imp_df_all=imp_df_all,
            selected_features=selected_features,
            rf_pupil=rf_pupil,
            cv_summary1_sel=cv_summary1_sel,
            feature_weights=feature_weights,
            feat_mean=feat_mean,
            feat_std=feat_std,
            img_feature_means=img_feature_means,
            X2=X2,
            Y2=Y2,
            mo2=mo2,
            best_params2=best_params2,
            r2_each2=r2_each2,
            r2_mean2=r2_mean2,
            X2_means=X2_means,
            orig_cols=orig_cols,
            new_image_for_display=new_image_for_display,
            x_before=x_before,
            pupil_before=pupil_before,
            z_before=z_before,
            df_z=df_z,
            sim_all=sim_all,
            summary=summary,
            best_row=best_row,
            pupil_after=pupil_after,
            delta_pupil=delta_pupil,
            ratio_pupil=ratio_pupil,
            feat_after_vec=feat_after_vec,
            z_after=z_after,
            delta_z=delta_z,
        )
    )
    return results


# ==========================
# 可視化レンダリング
# ==========================
def render_results(results):
    df_full = results["df_full"]
    pupil_col = results["pupil_col"]

    st.markdown("### 1️⃣ モデル1: 画像特徴 → 瞳孔径（縮瞳指標 z の作成）")

    st.markdown(
        """
        - まず、選択した特徴量グループから候補特徴量をすべて使って **瞳孔径の回帰モデル（モデル1）** を学習します。  
        - その後、特徴量重要度と瞳孔径との相関の向きを掛け合わせて **符号付きの重み** を作り、  
          標準化した特徴量の重み付き和として **z 指標** を定義します。
        """
    )

    # 全部入りの重要度
    st.subheader("候補特徴量の重要度（全部入りモデル）")
    st.dataframe(results["imp_df_all"].head(30), use_container_width=True)

    st.subheader("1段目 CV 結果（全部入りモデル）")
    cv1 = results["cv_summary1_full"]
    st.write(f"Train R²: **{cv1['mean_train']:.3f} ± {cv1['std_train']:.3f}**")
    st.write(f"Test  R²: **{cv1['mean_test']:.3f} ± {cv1['std_test']:.3f}**")
    st.write("ベストパラメータ:", results["best_params1"] if results["use_grid1"] else "GridSearch OFF（デフォルト）")

    st.markdown("---")
    st.subheader(f"上位 {results['top_k']} 特徴のみで学習し直したモデル")
    st.write("選択された特徴量:", results["selected_features"])
    cv1_sel = results["cv_summary1_sel"]
    st.write(f"Train R²: **{cv1_sel['mean_train']:.3f} ± {cv1_sel['std_train']:.3f}**")
    st.write(f"Test  R²: **{cv1_sel['mean_test']:.3f} ± {cv1_sel['std_test']:.3f}**")

    st.subheader("z の重み（符号付き・正規化済）")
    fw = results["feature_weights"]
    df_w = pd.DataFrame({"feature": fw.index, "weight": fw.values})
    st.dataframe(df_w, use_container_width=True)

    fig, ax = plt.subplots(figsize=(6, max(4, len(fw) * 0.25)))
    ax.barh(df_w["feature"], df_w["weight"])
    ax.set_xlabel("weight")
    ax.set_title("z を構成する特徴量の重み")
    ax.axvline(0, color="black", linewidth=1)
    st.pyplot(fig)

    st.markdown("### 2️⃣ モデル2: 加工 + 元画像特徴 → 重要特徴量")

    st.markdown(
        """
        - 2段目モデルでは、各ステップの加工量（step1\_gamma など）と元画像の \*_orig 特徴を入力として、  
          モデル1で選ばれた **重要特徴量の変化** をまとめて予測します。  
        - これにより、任意の加工パターンを指定したときに、z がどの方向へ動くかをシミュレートできます。
        """
    )

    r2_each2 = results["r2_each2"]
    r2_mean2 = results["r2_mean2"]
    selected_features = results["selected_features"]

    r2_df2 = pd.DataFrame({"feature": selected_features, "Test_R2": r2_each2})
    st.subheader("2段目: 特徴量ごとの当てはまり（CV ベース）")
    st.dataframe(r2_df2, use_container_width=True)
    st.caption(f"平均 Test R²: **{r2_mean2:.3f}**")

    fig_h = max(4, len(selected_features) * 0.25)
    fig2, ax2 = plt.subplots(figsize=(8, fig_h))
    ax2.barh(r2_df2["feature"], r2_df2["Test_R2"])
    ax2.set_xlabel("Test R²")
    ax2.set_title("2段目: 特徴量ごとの当てはまり")
    ax2.grid(axis="x", linestyle="--", alpha=0.6)
    st.pyplot(fig2)

    # ---- モデル2: 特徴量別の重要度 & 相関（変数切替しても再学習しない） ----
    st.markdown("#### モデル2における「加工パラメータ → 特徴量」の重要度と相関")

    mo2 = results["mo2"]
    X2 = results["X2"]
    Y2 = results["Y2"]

    param_cols = [c for c in X2.columns if c.startswith("step")]
    target_feat = st.selectbox(
        "解析対象とする特徴量（Y）",
        options=selected_features,
        key="target_feat_for_corr",
        help="ここを切り替えてもモデルの再学習は行いません。"
    )

    # 対象特徴量のインデックスを取得
    j = selected_features.index(target_feat)
    est_j = mo2.estimators_[j]
    fi = pd.Series(est_j.feature_importances_, index=X2.columns)
    fi_param = fi[param_cols].sort_values(ascending=False)

    # 相関（X と対象 Y のピアソン相関）
    y_col = Y2[target_feat]
    corr_param = X2[fi_param.index].corrwith(y_col)

    imp_corr_df = pd.DataFrame(
        {
            "feature": fi_param.index,
            "importance": fi_param.values,
            "correlation": corr_param.values,
        }
    ).sort_values("importance", ascending=False)

    st.write("選択した特徴量に対して、どの加工パラメータがどれくらい効いているかを可視化します。")
    st.dataframe(imp_corr_df.head(20), use_container_width=True)

    fig3, ax3 = plt.subplots(figsize=(8, max(4, len(imp_corr_df.head(20)) * 0.3)))
    ax3.barh(imp_corr_df["feature"].head(20), imp_corr_df["importance"].head(20))
    ax3.set_xlabel("Feature importance")
    ax3.set_title(f"モデル2: {target_feat} に対する加工パラメータの重要度")
    ax3.invert_yaxis()
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots(figsize=(8, max(4, len(imp_corr_df.head(20)) * 0.3)))
    ax4.barh(imp_corr_df["feature"].head(20), imp_corr_df["correlation"].head(20))
    ax4.set_xlabel("Pearson correlation")
    ax4.set_title(f"モデル2: {target_feat} と加工パラメータの相関")
    ax4.axvline(0, color="black", linewidth=1)
    ax4.invert_yaxis()
    st.pyplot(fig4)

    # ---- 新画像に対する z の計算過程 ----
    st.markdown("### 3️⃣ 新しい画像（または代表行）に対する z の計算")

    st.subheader("加工前の予測値と z の内訳")
    st.write(f"{pupil_col} の予測値（加工前）: **{results['pupil_before']:.3f}**")
    st.write(f"z（縮瞳に効く特徴の合成指標, 加工前）: **{results['z_before']:.3f}**")

    st.markdown(
        r"""
        定義:  
        \[
        z = \sum_i w_i \cdot \frac{x_i - \mu_i}{\sigma_i}
        \]
        """
    )
    st.dataframe(results["df_z"], use_container_width=True)

    # ---- 18パターン × ランダムサーチの結果 ----
    st.markdown("### 4️⃣ 18パターン × ランダムサーチによるベスト加工探索")

    st.markdown(
        """
        - brightness / contrast / gamma / sharpness / equalization から 3 つを選び，順番付きで並べた18パターンを用意。  
        - 各パターンについて、実験データから推定した範囲内でパラメータをランダムサンプリングし、  
          モデル2で特徴量 → モデル1で瞳孔径・z を予測してスコアリングしました。
        """
    )

    summary = results["summary"]
    st.subheader("18パターンの評価結果（Score = z, 大きいほど良）")
    st.dataframe(summary.style.format({"max_score": "{:.3f}", "top5_mean": "{:.3f}"}), use_container_width=True)

    # 散布図: pattern ごとに Score vs Pupil
    st.subheader("パターン別シミュレーション結果の散布図")
    sim_all = results["sim_all"]
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    for pat, g in sim_all.groupby("pattern"):
        ax5.scatter(g["Score"], g["Pupil"], s=8, alpha=0.35, label=pat)
    ax5.set_xlabel("Score (z)")
    ax5.set_ylabel("Predicted pupil")
    ax5.set_title("18パターン × ランダムサーチの結果")
    ax5.grid(True, linestyle="--", alpha=0.4)
    ax5.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    st.pyplot(fig5)

    # ---- ベスト加工案 ----
    st.markdown("### 5️⃣ この画像に対するベスト加工案と Before/After")

    best_row = results["best_row"]
    pupil_after = results["pupil_after"]
    delta_pupil = results["delta_pupil"]
    ratio_pupil = results["ratio_pupil"]
    z_before = results["z_before"]
    z_after = results["z_after"]
    delta_z = results["delta_z"]

    st.markdown(
        f"- **ベストパターン**: `{best_row['pattern'].replace('_', ' → ')}`  \n"
        f"- Step1: **{best_row['step1_op']}** = `{best_row['step1_val']:.3f}`  \n"
        f"- Step2: **{best_row['step2_op']}** = `{best_row['step2_val']:.3f}`  \n"
        f"- Step3: **{best_row['step3_op']}** = `{best_row['step3_val']:.3f}`"
    )

    df_pupil = pd.DataFrame(
        {
            "状態": ["加工前", "ベスト加工後"],
            f"予測 {pupil_col}": [results["pupil_before"], pupil_after],
            "変化量": [np.nan, delta_pupil],
            "変化率[%]": [np.nan, ratio_pupil],
            "z": [z_before, z_after],
            "z変化量": [np.nan, delta_z],
        }
    )
    st.subheader("縮瞳指標・z の予測値（加工前 vs ベスト加工後）")
    st.dataframe(df_pupil, use_container_width=True)

    df_feats = pd.DataFrame(
        {
            "特徴量": selected_features,
            "加工前": [results["x_before"][f] for f in selected_features],
            "ベスト加工後": [results["feat_after_vec"][f] for f in selected_features],
            "重みw": [results["feature_weights"][f] for f in selected_features],
        }
    )
    st.subheader("重要な特徴量の変化（中間の特徴量）")
    st.dataframe(df_feats, use_container_width=True)

    # 画像 Before / After
    if results["new_image_for_display"] is not None:
        st.subheader("画像の Before / After")
        new_image_for_display = results["new_image_for_display"]
        ops_best = [best_row["step1_op"], best_row["step2_op"], best_row["step3_op"]]
        vals_best = [best_row["step1_val"], best_row["step2_val"], best_row["step3_val"]]
        img_after = apply_processing_sequence(new_image_for_display, ops_best, vals_best)

        c1, c2 = st.columns(2)
        with c1:
            st.image(new_image_for_display, caption="加工前", use_container_width=True)
        with c2:
            cap = (
                f"ベスト加工後\n"
                f"{best_row['pattern'].replace('_', ' → ')}\n"
                f"予測 {pupil_col} = {pupil_after:.3f}\n"
                f"z = {z_after:.3f}"
            )
            st.image(img_after, caption=cap, use_container_width=True)


# ==========================
# main
# ==========================
def main():
    st.set_page_config(page_title="縮瞳モデル付き 加工推薦ツール（z最適化, RF/XGB）", layout="wide")

    # フォント大きめ
    st.markdown("""
        <style>
        html, body, [class*="css"]  {
            font-size: 18px !important;
        }
        h1, h2, h3, h4 {
            font-size: 1.3em !important;
        }
        .stDataFrame div, .stMetric, .stButton>button, .stSelectbox, .stRadio, .stSlider {
            font-size: 18px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🧪 画像特徴 → 縮瞳 → 加工推薦 ツール（2段モデル + z最適化, RF/XGB + GridSearch）")

    st.sidebar.header("📁 データ入力")
    uploaded_file = st.sidebar.file_uploader("実験データ(CSV/Excel)", type=["csv", "xlsx", "xls"])

    if uploaded_file is None:
        st.info("👈 左のサイドバーからデータをアップロードしてください。")
        return

    try:
        df_full = load_and_parse_data(uploaded_file)
    except Exception as e:
        st.error(f"データ読み込みエラー: {e}")
        return

    # ============================
    # 🔧 folder_name の除外指定
    # ============================
    if "folder_name" in df_full.columns:
        all_subjects = sorted(df_full["folder_name"].dropna().unique().tolist())
        excluded_subjects = st.sidebar.multiselect(
            "GroupKFold / 学習に使わない folder_name",
            options=all_subjects,
            help="ここで選んだ被験者IDは、1段目・2段目の学習とCVから完全に除外されます。"
        )
        if excluded_subjects:
            df_full = df_full[~df_full["folder_name"].isin(excluded_subjects)].copy()
            st.sidebar.write(f"有効な被験者数: {df_full['folder_name'].nunique()} / 行数: {len(df_full)}")
    else:
        st.sidebar.warning("folder_name 列が無いので、被験者除外は使えません。")

    # ===== GroupKFold 用 group 列の設定 =====
    st.sidebar.subheader("🧪 クロスバリデーション設定")

    use_groupkfold = st.sidebar.checkbox(
        "GroupKFold を使う（同じgroupを同一foldに入れない）",
        value=("folder_name" in df_full.columns),
        help="OFFにすると通常のKFoldになります。"
    )

    group_col = None
    groups = None

    if use_groupkfold:
        candidate_group_cols = []
        for c in df_full.columns:
            nunique = df_full[c].nunique(dropna=True)
            if 1 < nunique < len(df_full):
                candidate_group_cols.append(c)

        if not candidate_group_cols:
            st.sidebar.warning("GroupKFold に使えそうな列が見つからなかったので、通常の KFold を使います。")
            use_groupkfold = False
            groups = None
        else:
            default_idx = 0
            if "folder_name" in candidate_group_cols:
                default_idx = candidate_group_cols.index("folder_name")

            group_col = st.sidebar.selectbox(
                "GroupKFold に使う列",
                options=candidate_group_cols,
                index=default_idx,
                help="例：folder_name（被験者ID）など"
            )
            groups = df_full[group_col]
    else:
        groups = None

    sample_weights = compute_sample_weights(df_full)

    tab1, tab2 = st.tabs(["📊 データ概要", "🧬 縮瞳に効く加工推薦（プロセス可視化付き）"])

    # ===========================
    # Tab1
    # ===========================
    with tab1:
        st.subheader("データセット概要")
        st.write(f"総データ数: **{len(df_full)}** 行")
        st.dataframe(df_full.head(), use_container_width=True)

        st.divider()
        st.subheader("加工パターンの分布")

        if "pattern_id" in df_full.columns:
            pattern_counts = df_full["pattern_id"].value_counts().sort_values(ascending=False)
        else:
            pattern_counts = pd.Series([], dtype=int)

        if not pattern_counts.empty:
            fig_h = max(5, len(pattern_counts) * 0.4)
            fig, ax = plt.subplots(figsize=(10, fig_h))
            bars = ax.barh(pattern_counts.index, pattern_counts.values)
            ax.set_xlabel("件数")
            ax.grid(axis="x", linestyle="--", alpha=0.7)
            ax.set_title("pattern_id ごとの件数")
            for b in bars:
                w = b.get_width()
                ax.text(w + 1, b.get_y() + b.get_height()/2, f"{int(w)}",
                        ha="left", va="center")
            st.pyplot(fig)
        else:
            st.warning("pattern_id がありません。")

        if {"param1", "param2", "param3"}.issubset(df_full.columns):
            st.divider()
            st.subheader("param 出現頻度")
            op_counts = pd.concat([
                df_full["param1"], df_full["param2"], df_full["param3"]
            ]).value_counts().rename("count")
            st.dataframe(op_counts.to_frame(), use_container_width=True)

        st.markdown("""
        **🔧 サンプル重み**  

        - 各行に `pattern_id` を付与し、その出現回数の逆数を学習時の重みとして使用。  
        - 頻出パターンだけでなくレアなパターンも、できるだけ公平に寄与させています。
        """)

    # ===========================
    # Tab2
    # ===========================
    with tab2:
        st.header("🧬 2段モデルでの縮瞳向き加工推薦（プロセス可視化付き）")

        num_cols_all = df_full.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols_all:
            st.error("数値列がありません。縮瞳列が入ったファイルを指定してください。")
            st.stop()

        default_pupil = "corrected_pupil" if "corrected_pupil" in num_cols_all else num_cols_all[0]
        pupil_col = st.selectbox(
            "縮瞳を表す列（ターゲット）",
            options=num_cols_all,
            index=num_cols_all.index(default_pupil),
        )

        direction = st.radio(
            "どちらの方向が『良い』？",
            ["値が小さいほど良い（縮瞳）", "値が大きいほど良い（散瞳）"],
            index=0,
            horizontal=True,
        )
        sign_dir = -1.0 if "小さい" in direction else 1.0

        feat_group = st.radio(
            "1段目で使用する特徴量グループ",
            ["all", "all_area", "all_pupil", "ROI"],
            index=0,
            horizontal=True,
        )

        if feat_group == "all":
            candidate_cols = [
                c for c in num_cols_all
                if c.startswith("all_")
                and not c.startswith("all_area_")
                and not c.startswith("all_pupil_")
                and not c.endswith("_orig")
                and c not in NON_FEATURE_COLS
                and c != pupil_col
            ]
        elif feat_group == "all_area":
            candidate_cols = [
                c for c in num_cols_all
                if c.startswith("all_area_")
                and not c.endswith("_orig")
                and c not in NON_FEATURE_COLS
                and c != pupil_col
            ]
        elif feat_group == "all_pupil":
            candidate_cols = [
                c for c in num_cols_all
                if c.startswith("all_pupil_")
                and not c.endswith("_orig")
                and c not in NON_FEATURE_COLS
                and c != pupil_col
            ]
        else:  # ROI
            candidate_cols = [
                c for c in num_cols_all
                if (
                    c.startswith("center_")
                    or c.startswith("parafovea_")
                    or c.startswith("periphery_")
                )
                and "_orig" not in c
                and c not in NON_FEATURE_COLS
                and c != pupil_col
            ]

        if not candidate_cols:
            st.error("候補となる特徴量列が見つかりません。列名規則や NON_FEATURE_COLS を確認してください。")
            st.stop()

        st.markdown(f"**グループ: {feat_group}** / 候補特徴量の数: **{len(candidate_cols)}** 列")

        top_k = st.slider(
            "縮瞳モデルで使う特徴量の数（重要度上位）",
            min_value=3, max_value=min(30, len(candidate_cols)),
            value=min(10, len(candidate_cols)),
        )

        n_trials_per_pattern = st.slider(
            "1パターンあたりのランダムサーチ試行数",
            min_value=200, max_value=5000, value=1000, step=200
        )

        model1_type = st.radio(
            "1段目のモデル（画像特徴 → 縮瞳）",
            ["RandomForest", "XGBoost"],
            index=0,
            horizontal=True,
        )
        model2_type = st.radio(
            "2段目のモデル（加工+*_orig → 画像特徴）",
            ["RandomForest", "XGBoost"],
            index=0,
            horizontal=True,
        )

        use_grid1 = st.checkbox("1段目で GridSearch を使う", value=True)
        use_grid2 = st.checkbox("2段目で GridSearch を使う", value=True)

        # 新画像入力
        st.subheader("新しい画像の入力")
        new_image_file = st.file_uploader("新しい画像 (JPEG/PNG)", type=["jpg", "jpeg", "png"], key="new_img")

        def _fmt_idx(i):
            if "image_name" in df_full.columns:
                return f"{i}: {df_full.loc[i, 'image_name']}"
            elif "file_name" in df_full.columns:
                return f"{i}: {df_full.loc[i, 'file_name']}"
            else:
                return str(i)

        st.markdown("画像をアップしない場合は、訓練データの1行を『仮の新画像』として使えます。")
        fallback_idx = st.selectbox(
            "fallback 用の行",
            options=df_full.index,
            format_func=_fmt_idx,
        )

        # ----------------------
        # 実行ボタン & セッション管理
        # ----------------------
        run_clicked = st.button("🚀 モデル学習 & 推薦加工探索を実行 / 更新")

        key_name = "app_comb_allROI_results"

        if run_clicked:
            try:
                results = run_pipeline(
                    df_full=df_full,
                    sample_weights=sample_weights,
                    groups=groups,
                    pupil_col=pupil_col,
                    sign_dir=sign_dir,
                    feat_group=feat_group,
                    candidate_cols=candidate_cols,
                    top_k=top_k,
                    n_trials_per_pattern=n_trials_per_pattern,
                    model1_type=model1_type,
                    model2_type=model2_type,
                    use_grid1=use_grid1,
                    use_grid2=use_grid2,
                    new_image_file=new_image_file,
                    fallback_idx=fallback_idx,
                )
                st.session_state[key_name] = results
            except Exception as e:
                st.error(f"処理中にエラーが発生しました: {e}")
                return

        if key_name in st.session_state:
            st.info("▶ 右側のセレクタを切り替えても再学習は行わず、保存済みの結果を使って可視化しています。")
            render_results(st.session_state[key_name])
        else:
            st.warning("まず『モデル学習 & 推薦加工探索を実行 / 更新』ボタンを押して結果を生成してください。")


if __name__ == "__main__":
    main()
