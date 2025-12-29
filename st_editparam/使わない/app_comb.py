# app_recommend_processing_two_stage.py

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import cv2
from PIL import Image

# ==== features_pupil / GPU 対応 =====================================
try:
    import features_pupil as fp
    if cv2.cuda.getCudaEnabledDeviceCount() == 0:
        raise ImportError("CUDA device not found")
except Exception:
    import features_pupil as fp

# ==== 画面・観察距離など（features_pupil 用） =======================
SCREEN_W_MM = 260
DIST_MM     = 450
RES_X       = 6000
CENTER_DEG  = 2
PARAFOVEA_DEG = 5


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

    parsed_list = [parse_params_ordered(n) for n in df["image_name"]]
    params_df = pd.DataFrame(parsed_list)

    params_df["pattern_id"] = (
        params_df["param1"] + " → " + params_df["param2"] + " → " + params_df["param3"]
    )

    cols_to_use = params_df.columns.tolist()
    df = df.drop(columns=[c for c in cols_to_use if c in df.columns], errors="ignore")
    df_full = pd.concat([df, params_df], axis=1)
    return df_full


def create_interaction_features(df):
    valid_ops = ["brightness", "contrast", "gamma", "sharpness", "equalization"]
    X_dict = {}
    for i in range(1, 4):
        col_op = f"param{i}"
        col_val = f"param{i}_val"
        for op in valid_ops:
            mask = (df[col_op] == op).astype(float)
            X_dict[f"step{i}_{op}"] = mask * df[col_val]
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


def main():
    st.set_page_config(page_title="縮瞳モデル付き 加工推薦ツール", layout="wide")
    st.title("🧪 画像特徴 → 縮瞳 → 加工推薦 ツール（2段モデル）")

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

    sample_weights = compute_sample_weights(df_full)

    tab1, tab2 = st.tabs(["📊 データ概要", "🧬 縮瞳に効く加工推薦"])

    # ===========================
    # Tab1
    # ===========================
    with tab1:
        st.subheader("データセット概要")
        st.write(f"総データ数: **{len(df_full)}** 行")
        st.dataframe(df_full.head())

        st.divider()
        st.subheader("加工パターンの分布")

        pattern_counts = df_full["pattern_id"].value_counts().sort_values(ascending=False)
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
    # Tab2 : 2段モデル
    # ===========================
    with tab2:
        st.header("🧬 2段モデルでの縮瞳向き加工推薦")

        # --- 0. pupil列の選択 ---
        num_cols = df_full.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            st.error("数値列がありません。縮瞳列が入ったファイルを指定してください。")
            st.stop()

        default_pupil = "corrected_pupil" if "corrected_pupil" in num_cols else num_cols[0]
        pupil_col = st.selectbox(
            "縮瞳を表す列（ターゲット）",
            options=num_cols,
            index=num_cols.index(default_pupil),
        )

        direction = st.radio(
            "どちらの方向が『良い』？",
            ["値が小さいほど良い（縮瞳）", "値が大きいほど良い（散瞳）"],
            index=0,
            horizontal=True,
        )
        sign = -1.0 if "小さい" in direction else 1.0

        # --- 1. all_* 特徴量の候補 ---
        all_cols = [c for c in df_full.columns
                    if c.startswith("all_")
                    and not c.endswith("_orig")
                    and not c.endswith("_orig_area")]
        if not all_cols:
            st.error("all_* という名前の特徴量列が見つかりません。")
            st.stop()

        st.markdown(f"候補となる all系特徴量の数: **{len(all_cols)}** 列")

        top_k = st.slider("縮瞳モデルで使う all特徴量の数（重要度上位）",
                          min_value=3, max_value=min(30, len(all_cols)),
                          value=min(10, len(all_cols)))

        n_trials_per_pattern = st.slider(
            "1パターンあたりのランダムサーチ試行数",
            min_value=200, max_value=5000, value=1000, step=200
        )

        # --- 新しい画像 / fallback 行の選択 ---
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

        if st.button("🚀 モデル学習 & 推薦加工探索"):
            # -------------------------
            # 1段目: all_* → pupil
            # -------------------------
            with st.spinner("1段目: all特徴量 → 縮瞳 モデルを学習中..."):
                X_img_all = df_full[all_cols].copy()
                y_pupil   = df_full[pupil_col]

                X_tr1, X_te1, y_tr1, y_te1, w_tr1, w_te1 = train_test_split(
                    X_img_all, y_pupil, sample_weights, test_size=0.2, random_state=42
                )

                rf_tmp = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
                rf_tmp.fit(X_tr1, y_tr1, sample_weight=w_tr1)

                imp_all = rf_tmp.feature_importances_
                imp_df_all = (
                    pd.DataFrame({"feature": all_cols, "importance": imp_all})
                    .sort_values("importance", ascending=False)
                    .reset_index(drop=True)
                )

                st.subheader("all系特徴量の重要度（縮瞳モデル・予備学習）")
                st.dataframe(imp_df_all.head(30), use_container_width=True)

                selected_features = imp_df_all["feature"].head(top_k).tolist()
                st.markdown("**このうち上位 k 個だけを使って、縮瞳モデルを作り直します。**")
                st.write("選択された特徴量:", selected_features)

                # 上位kで改めて学習
                X_img_sel = df_full[selected_features].copy()
                X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(
                    X_img_sel, y_pupil, sample_weights, test_size=0.2, random_state=42
                )
                rf_pupil = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
                rf_pupil.fit(X_tr, y_tr, sample_weight=w_tr)

                r2_train1 = rf_pupil.score(X_tr, y_tr)
                r2_test1  = rf_pupil.score(X_te, y_te)

                st.subheader("1段目: 縮瞳モデル（選ばれた特徴量のみ）の当てはまり")
                st.write(f"Train R²: **{r2_train1:.3f}**,  Test R²: **{r2_test1:.3f}**")

                img_feature_means = X_img_sel.mean()

            # -------------------------
            # 2段目: (param + *_orig) → selected all特徴量
            # -------------------------
            with st.spinner("2段目: 加工 + 元画像特徴 → all特徴量 モデルを学習中..."):
                X_param = create_interaction_features(df_full)

                orig_cols = [c for c in df_full.columns
                             if c.endswith("_orig") or c.endswith("_orig_area")]
                if orig_cols:
                    X_orig = df_full[orig_cols].copy()
                    X2 = pd.concat([X_param, X_orig], axis=1)
                else:
                    X_orig = pd.DataFrame(index=df_full.index)
                    X2 = X_param.copy()

                Y2 = df_full[selected_features].copy()

                X2_tr, X2_te, Y2_tr, Y2_te, w2_tr, w2_te = train_test_split(
                    X2, Y2, sample_weights, test_size=0.2, random_state=42
                )

                base_rf2 = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
                mo2 = MultiOutputRegressor(base_rf2)
                mo2.fit(X2_tr, Y2_tr, sample_weight=w2_tr)

                Y2_pred_te = mo2.predict(X2_te)
                r2_each2 = r2_score(Y2_te, Y2_pred_te, multioutput="raw_values")
                r2_mean2 = r2_score(Y2_te, Y2_pred_te, multioutput="uniform_average")

                st.subheader("2段目: all特徴量モデルの当てはまり")
                r2_df2 = pd.DataFrame({"feature": selected_features, "Test_R2": r2_each2})
                st.dataframe(r2_df2, use_container_width=True)
                st.caption(f"平均 Test R²: **{r2_mean2:.3f}**")

                X2_means = X2.mean()

            # -------------------------
            # 新しい画像の特徴量 & 加工前の縮瞳値
            # -------------------------
            with st.spinner("新しい画像の特徴量計算中..."):
                new_image_for_display = None
                feats_before = {}

                if new_image_file is not None:
                    pil_img = Image.open(new_image_file).convert("RGB")
                    new_image_for_display = pil_img
                    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    h, w = img_bgr.shape[:2]

                    roi_masks = fp.make_masks(h, w, SCREEN_W_MM, DIST_MM, RES_X, CENTER_DEG, PARAFOVEA_DEG)
                    feats_roi = fp.compute_features_for_image(
                        img_bgr, roi_masks,
                        screen_w_mm=SCREEN_W_MM, dist_mm=DIST_MM, res_x=RES_X
                    )
                    all_masks = fp.make_all_masks()
                    feats_all = fp.compute_features_for_image(
                        img_bgr, all_masks,
                        screen_w_mm=SCREEN_W_MM, dist_mm=DIST_MM, res_x=RES_X
                    )
                    feats_before = {**feats_roi, **feats_all}

                    total_px = float(h * w)
                    area_map = {name: float(mask.sum()) / total_px for name, mask in roi_masks.items()}
                    area_map["all"] = 1.0
                else:
                    # fallback: 訓練データ1行
                    feats_before = {
                        c: df_full.loc[fallback_idx, c]
                        for c in selected_features
                        if c in df_full.columns
                    }

                # ベースの all特徴量ベクトル
                x_before = pd.Series(index=selected_features, dtype=float)
                missing_feats = []
                for f in selected_features:
                    if f in feats_before:
                        x_before[f] = feats_before[f]
                    else:
                        missing_feats.append(f)
                        x_before[f] = np.nan

                if missing_feats:
                    st.warning(f"新画像から取得できなかった特徴量: {missing_feats} "
                               f"→ データセットの平均値で補完します。")

                x_before = x_before.fillna(img_feature_means)
                pupil_before = float(rf_pupil.predict(x_before.values.reshape(1, -1))[0])

                st.subheader("加工前の予測縮瞳値")
                st.write(f"{pupil_col} の予測値（加工前）: **{pupil_before:.3f}**")

            # -------------------------
            # 18パターン × ランダムサーチ
            # -------------------------
            with st.spinner("18パターン × ランダムサーチでベスト加工を探索中..."):
                allowed_patterns = generate_allowed_patterns()
                sim_records = []

                # 新画像の orig ベクトル（X2 用）
                orig_vec = pd.Series(index=orig_cols, dtype=float)
                if new_image_file is not None:
                    # 既に roi_masks 等は計算済み
                    # feats_before から *_orig を再計算するのが難しい場合は
                    # とりあえず fallback 行の *_orig を使う
                    for c in orig_cols:
                        orig_vec[c] = df_full.loc[fallback_idx, c]
                else:
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

                    # *_orig 部分は新画像（or fallback）の値で固定
                    for c in orig_cols:
                        sim_X2[c] = orig_vec.get(c, np.nan)
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

                    # 2段目モデルで all特徴量を予測
                    Y_pred_feats = mo2.predict(sim_X2)  # shape: (n_trials, top_k)

                    # 1段目モデルで縮瞳を予測
                    pupil_preds = rf_pupil.predict(Y_pred_feats)

                    # Score: direction に応じて符号反転
                    scores = sign * pupil_preds

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
                    # all特徴量も保存（後で before/after に使う）
                    for i, feat in enumerate(selected_features):
                        df_pat[feat] = Y_pred_feats[:, i]

                    sim_records.append(df_pat)

                sim_all = pd.concat(sim_records, ignore_index=True)

                # pattern単位のまとめ
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

                st.subheader("18パターンの評価結果（Score高いほど良）")
                st.dataframe(summary.style.format({"max_score": "{:.3f}", "top5_mean": "{:.3f}"}),
                             use_container_width=True)

                # ベスト1点
                best_row = sim_all.loc[sim_all["Score"].idxmax()].copy()

                pupil_after = float(best_row["Pupil"])
                delta = pupil_after - pupil_before
                ratio = np.nan if pupil_before == 0 else delta / pupil_before * 100.0

                st.divider()
                st.subheader("👑 この画像に対するベスト加工案（Score 最大）")

                st.markdown(
                    f"- パターン: **{best_row['pattern'].replace('_', ' → ')}**  \n"
                    f"- Step1: **{best_row['step1_op']}** = `{best_row['step1_val']:.3f}`  \n"
                    f"- Step2: **{best_row['step2_op']}** = `{best_row['step2_val']:.3f}`  \n"
                    f"- Step3: **{best_row['step3_op']}** = `{best_row['step3_val']:.3f}`"
                )

                st.subheader("縮瞳指標の予測値（加工前 vs ベスト加工後）")
                df_pupil = pd.DataFrame(
                    {
                        "状態": ["加工前", "ベスト加工後"],
                        f"予測 {pupil_col}": [pupil_before, pupil_after],
                        "変化量": [np.nan, delta],
                        "変化率[%]": [np.nan, ratio],
                    }
                )
                st.dataframe(df_pupil, use_container_width=True)

                # 選ばれた all特徴量の before / after
                feat_after = [best_row[f] for f in selected_features]
                df_feats = pd.DataFrame(
                    {
                        "特徴量": selected_features,
                        "加工前": [x_before[f] for f in selected_features],
                        "ベスト加工後": feat_after,
                    }
                )
                st.subheader("重要な all特徴量の変化（中間の特徴量）")
                st.dataframe(df_feats, use_container_width=True)

                # 画像をアップしていれば Before/After も表示
                if new_image_for_display is not None:
                    st.subheader("画像の Before / After")
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
                            f"予測 {pupil_col} = {pupil_after:.3f}"
                        )
                        st.image(img_after, caption=cap, use_container_width=True)


if __name__ == "__main__":
    main()
