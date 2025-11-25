# st_poly.py
# 多項式モデル（Lasso / AIC による自動変数選択付き）

import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LassoCV, LogisticRegression

import statsmodels.api as sm

import xai  # 既存の XAI モジュール（st_tree.py と同じものを想定）


# ============================================================
# ユーティリティ
# ============================================================

def make_train_test(
    df_train,
    df_test,
    mode,
    target,
    feature_cols,
    group_col=None,
    test_size=0.2,
    random_state=42
):
    """
    単一ファイル or 学習/評価ファイル を受け取り、X_train, X_test, y_train, y_test を返す。
    単一ファイルの場合は GroupShuffleSplit（group_col があれば）で分割。
    """
    if mode == "single":
        data = df_train.copy()
        y = data[target]
        X = data[feature_cols]

        if group_col and group_col in data.columns:
            groups = data[group_col]
            gss = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            )
            tr_idx, te_idx = next(gss.split(X, y, groups))
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
    else:
        # train / test ファイル別
        tr = df_train.copy()
        te = df_test.copy()
        y_tr = tr[target]
        y_te = te[target]
        X_tr = tr[feature_cols]
        X_te = te[feature_cols]

    return X_tr, X_te, y_tr, y_te


def bin_target_if_needed(y, n_bins=None):
    """
    連続目的変数を分位でビン化（分類用）。
    n_bins が None の時はそのまま返す。
    """
    if n_bins is None:
        return y

    y_cont = pd.to_numeric(y, errors="coerce")
    valid_idx = y_cont.notna()
    if not valid_idx.all():
        raise ValueError("目的変数に数値化できない値が含まれています。")

    try:
        y_bins = pd.qcut(y_cont, q=n_bins, labels=False, duplicates="drop")
    except ValueError as e:
        raise ValueError(f"分位ビン作成に失敗しました: {e}")

    y_bins = y_bins.astype(int)
    y_bins.index = y.index
    return y_bins


def coef_table_from_lasso_poly(pipe, feature_cols, task_type):
    """
    Pipeline(poly + scaler + Lasso または LogisticRegression) から
    多項式項の係数表を作る。
    """
    poly = pipe.named_steps["poly"]
    if task_type == "回帰":
        reg = pipe.named_steps["model"]
        coefs = reg.coef_
        intercept = reg.intercept_
    else:
        # 2 クラス想定
        clf = pipe.named_steps["model"]
        coefs = clf.coef_[0]
        intercept = clf.intercept_[0]

    names = poly.get_feature_names_out(feature_cols)
    df = pd.DataFrame(
        {"term": names, "coef": coefs}
    ).sort_values("coef", key=np.abs, ascending=False)

    return df, intercept


def stepwise_aic_ols(X, y, max_steps=50, verbose=False):
    """
    前進ステップワイズ AIC 最小化（回帰：OLS）。
    X: DataFrame (constant 列はまだ加えないで渡す)
    """
    remaining = list(X.columns)
    selected = []
    current_score = np.inf

    for _ in range(max_steps):
        scores_with_candidates = []
        for cand in remaining:
            cols = selected + [cand]
            X_c = sm.add_constant(X[cols])
            model = sm.OLS(y, X_c).fit()
            scores_with_candidates.append((model.aic, cand))
        scores_with_candidates.sort(key=lambda x: x[0])
        best_new_score, best_cand = scores_with_candidates[0]

        if verbose:
            print("Trying", best_cand, "AIC=", best_new_score)

        if best_new_score < current_score - 1e-4:
            remaining.remove(best_cand)
            selected.append(best_cand)
            current_score = best_new_score
        else:
            break

    X_sel = sm.add_constant(X[selected])
    best_model = sm.OLS(y, X_sel).fit()
    return best_model, selected


def stepwise_aic_logit(X, y, max_steps=50, verbose=False):
    """
    前進ステップワイズ AIC 最小化（2値ロジスティック回帰）。
    y は 0/1 を想定。
    """
    remaining = list(X.columns)
    selected = []
    current_score = np.inf

    for _ in range(max_steps):
        scores_with_candidates = []
        for cand in remaining:
            cols = selected + [cand]
            X_c = sm.add_constant(X[cols])
            try:
                model = sm.Logit(y, X_c).fit(disp=0)
            except Exception:
                continue
            scores_with_candidates.append((model.aic, cand))
        if not scores_with_candidates:
            break
        scores_with_candidates.sort(key=lambda x: x[0])
        best_new_score, best_cand = scores_with_candidates[0]

        if verbose:
            print("Trying", best_cand, "AIC=", best_new_score)

        if best_new_score < current_score - 1e-4:
            remaining.remove(best_cand)
            selected.append(best_cand)
            current_score = best_new_score
        else:
            break

    X_sel = sm.add_constant(X[selected])
    best_model = sm.Logit(y, X_sel).fit(disp=0)
    return best_model, selected


# ============================================================
# Streamlit UI
# ============================================================

st.title("多項式モデル（Lasso/AIC による自動変数選択付き）")

# ---------- データ指定 ----------
mode_choice = st.radio(
    "データの指定方法",
    ("ランダム分割（単一ファイル）", "学習/評価ファイルを分ける")
)

df = train_df = test_df = None
mode_flag = "single" if mode_choice.startswith("ランダム") else "split"

if mode_flag == "single":
    up = st.file_uploader("データファイル（CSV / XLSX）", type=["csv", "xls", "xlsx"])
    if up is not None:
        df = pd.read_excel(up) if up.name.endswith((".xls", ".xlsx")) else pd.read_csv(up)
else:
    up_tr = st.file_uploader("学習用データ（CSV / XLSX）", type=["csv", "xls", "xlsx"], key="up_tr")
    up_te = st.file_uploader("評価用データ（CSV / XLSX）", type=["csv", "xls", "xlsx"], key="up_te")
    if up_tr is not None and up_te is not None:
        train_df = pd.read_excel(up_tr) if up_tr.name.endswith((".xls", ".xlsx")) else pd.read_csv(up_tr)
        test_df  = pd.read_excel(up_te) if up_te.name.endswith((".xls", ".xlsx")) else pd.read_csv(up_te)

base_df = df if df is not None else train_df

# ---------- サイドバー：モデリング設定 ----------
st.sidebar.header("モデリング設定")
task_type = st.sidebar.radio("タスク種別", ["回帰", "分類"])
model_type = st.sidebar.selectbox("変数選択の方法", ["Lasso（正則化）", "AIC（ステップワイズ）","なし（全変数使用）"])

deg = st.sidebar.slider("多項式の次数", 1, 3, 2)
interaction_only = st.sidebar.checkbox("交互作用項のみ追加（同じ変数の2乗などは含めない）", value=False)
test_size = st.sidebar.slider("テストデータの割合（単一ファイル時）", 0.1, 0.5, 0.2)
random_state = st.sidebar.number_input("random_state", 0, 9999, 42)

if task_type == "分類":
    bin_choice = st.sidebar.selectbox(
        "目的変数をビン化",
        ["そのまま使う", "二分位", "三分位", "四分位"]
    )
    bin_map = {"二分位": 2, "三分位": 3, "四分位": 4}
else:
    bin_choice = "そのまま使う"
    bin_map = {}

# ---------- 目的変数・説明変数の指定 ----------
target = None
feature_cols = []
group_col = None

if base_df is not None:
    st.markdown("---")
    st.subheader("目的変数と説明変数の指定")

    all_cols = list(base_df.columns)
    target = st.selectbox("目的変数（予測したい列）", all_cols, key="target_col")

    group_col = st.text_input(
        "グループID列名（任意：被験者IDなど。同一グループが train/test に跨らないように分割）",
        value="folder_name"
    )

    # 数値列だけを候補に
    numeric_cols = base_df.select_dtypes(include=[np.number]).columns.tolist()

    # 説明変数候補（target, group は除く）
    expl_candidates = [c for c in numeric_cols if c not in [target, group_col]]

    feature_cols = st.multiselect(
        "多項式展開に使う説明変数（追加方式）",
        expl_candidates,
        default=expl_candidates,
        key="feature_cols"
    )

    st.caption("※ 説明変数を 0 個にすると学習できません。")


# ============================================================
# モデル学習・評価 実行
# ============================================================
run = st.button("モデル学習・評価を実行")

if run:
    if base_df is None:
        st.error("まずデータファイルを読み込んでください。")
        st.stop()
    if target is None or len(feature_cols) == 0:
        st.error("目的変数と説明変数を指定してください。")
        st.stop()

    # ---------- train / test 分割 ----------
    try:
        X_tr, X_te, y_tr, y_te = make_train_test(
            df if df is not None else train_df,
            None if df is not None else test_df,
            mode_flag,
            target,
            feature_cols,
            group_col=group_col if group_col in base_df.columns else None,
            test_size=test_size,
            random_state=random_state
        )
    except Exception as e:
        st.error(f"train/test 分割でエラー: {e}")
        st.stop()

    # ---------- 分類ならビン化 ----------
    n_bins = None
    if task_type == "分類" and bin_choice != "そのまま使う":
        n_bins = bin_map[bin_choice]

    if task_type == "分類" and n_bins is not None:
        try:
            y_tr = bin_target_if_needed(y_tr, n_bins=n_bins)
            y_te = bin_target_if_needed(y_te, n_bins=n_bins)
        except Exception as e:
            st.error(str(e))
            st.stop()

    # ========================================================
    # Lasso（sklearn Pipeline）
    # ========================================================
    if model_type.startswith("Lasso"):
        st.subheader("Lasso 多項式モデルの結果")

        if task_type == "回帰":
            pipe = Pipeline([
                ("poly", PolynomialFeatures(
                    degree=deg, include_bias=False, interaction_only=interaction_only
                )),
                ("scaler", StandardScaler()),
                ("model", LassoCV(cv=5, random_state=random_state))
            ])
        else:
            pipe = Pipeline([
                ("poly", PolynomialFeatures(
                    degree=deg, include_bias=False, interaction_only=interaction_only
                )),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    max_iter=5000,
                    random_state=random_state,
                    class_weight="balanced"
                ))
            ])

        pipe.fit(X_tr, y_tr)

        # ---------- 予測 ----------
        yhat_tr = pipe.predict(X_tr)
        yhat_te = pipe.predict(X_te)

        if task_type == "回帰":
            rmse_tr = np.sqrt(mean_squared_error(y_tr, yhat_tr))
            mae_tr = mean_absolute_error(y_tr, yhat_tr)
            r2_tr = r2_score(y_tr, yhat_tr)

            rmse_te = np.sqrt(mean_squared_error(y_te, yhat_te))
            mae_te = mean_absolute_error(y_te, yhat_te)
            r2_te = r2_score(y_te, yhat_te)

            st.write(
                f"**Train**: R²={r2_tr:.3f}, RMSE={rmse_tr:.3f}, MAE={mae_tr:.3f}"
            )
            st.write(
                f"**Test** : R²={r2_te:.3f}, RMSE={rmse_te:.3f}, MAE={mae_te:.3f}"
            )

            # 散布図
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(y_tr, yhat_tr, alpha=0.6, label="Train", marker="o")
            ax.scatter(y_te, yhat_te, alpha=0.6, label="Test", marker="^")
            lo = min(y_tr.min(), y_te.min(), yhat_tr.min(), yhat_te.min())
            hi = max(y_tr.max(), y_te.max(), yhat_tr.max(), yhat_te.max())
            ax.plot([lo, hi], [lo, hi], "--")
            ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted (Lasso-Poly)")
            ax.legend()
            st.pyplot(fig)

        else:
            # 分類指標
            yhat_te_cls = yhat_te
            acc = accuracy_score(y_te, yhat_te_cls)
            prec = precision_score(y_te, yhat_te_cls, average="macro", zero_division=0)
            rec = recall_score(y_te, yhat_te_cls, average="macro", zero_division=0)
            f1 = f1_score(y_te, yhat_te_cls, average="macro", zero_division=0)
            st.write(
                f"Accuracy={acc:.3f}, Precision(macro)={prec:.3f}, "
                f"Recall(macro)={rec:.3f}, F1(macro)={f1:.3f}"
            )

            labs = sorted(pd.Series(y_te).unique().tolist())
            cm = confusion_matrix(y_te, yhat_te_cls, labels=labs)
            cm_df = pd.DataFrame(
                cm,
                index=[f"T{i}" for i in labs],
                columns=[f"P{i}" for i in labs]
            )
            st.write("混同行列（件数）")
            st.dataframe(cm_df)

            cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            im = ax_cm.imshow(cm_norm, interpolation="nearest")
            ax_cm.set_xticks(range(len(labs))); ax_cm.set_xticklabels(labs)
            ax_cm.set_yticks(range(len(labs))); ax_cm.set_yticklabels(labs)
            ax_cm.set_title("Confusion Matrix (row-normalized)")
            for i in range(cm_norm.shape[0]):
                for j in range(cm_norm.shape[1]):
                    ax_cm.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")
            fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
            st.pyplot(fig_cm)

            rep = classification_report(
                y_te, yhat_te_cls, labels=labs, output_dict=True, zero_division=0
            )
            rep_df = pd.DataFrame(rep).T
            st.write("分類レポート")
            st.dataframe(rep_df)

        # 係数表
        coef_df, intercept = coef_table_from_lasso_poly(
            pipe, feature_cols, task_type
        )
        st.subheader("多項式項の係数（Lasso により自動選択）")
        st.write(f"切片（intercept）: {intercept:.4f}")
        st.dataframe(coef_df.head(50))

        # === XAI ===
        st.subheader("XAI（SHAP / LIME：Lasso-Poly）")
        try:
            # 背景は train のサンプル
            bg = X_tr if len(X_tr) <= 200 else X_tr.sample(200, random_state=42)
            xai.explain_shap(pipe, bg, X_te, task_type, "Poly-Lasso")
            xai.explain_lime(pipe, bg, X_te, task_type)
        except Exception as e:
            st.info(f"XAI 計算でエラーが発生しました: {e}")

    # ========================================================
    # AIC（statsmodels によるステップワイズ）
    # ========================================================
    elif model_type.startswith("AIC"):
        st.subheader("AIC ステップワイズ多項式モデルの結果")

        # まず多項式特徴を明示的に作る
        poly = PolynomialFeatures(
            degree=deg, include_bias=False, interaction_only=interaction_only
        )
        X_tr_poly = poly.fit_transform(X_tr)
        X_te_poly = poly.transform(X_te)
        poly_names = poly.get_feature_names_out(feature_cols)

        X_tr_poly_df = pd.DataFrame(X_tr_poly, columns=poly_names, index=X_tr.index)
        X_te_poly_df = pd.DataFrame(X_te_poly, columns=poly_names, index=X_te.index)

        if task_type == "回帰":
            model_aic, selected = stepwise_aic_ols(X_tr_poly_df, y_tr)
            st.write(f"選択された項の数: {len(selected)}")
            st.write("選択された項（一部）：")
            st.write(selected[:30])

            # 係数表
            params = model_aic.params
            pvals = model_aic.pvalues
            coef_df = pd.DataFrame(
                {"term": params.index, "coef": params.values, "p": pvals.values}
            ).sort_values("coef", key=np.abs, ascending=False)
            st.subheader("係数・p値（AIC 選択後 OLS）")
            st.dataframe(coef_df.head(50))

            # 予測と評価
            X_tr_sel = sm.add_constant(X_tr_poly_df[selected])
            X_te_sel = sm.add_constant(X_te_poly_df[selected], has_constant="add")
            yhat_tr = model_aic.predict(X_tr_sel)
            yhat_te = model_aic.predict(X_te_sel)

            rmse_tr = np.sqrt(mean_squared_error(y_tr, yhat_tr))
            mae_tr = mean_absolute_error(y_tr, yhat_tr)
            r2_tr = r2_score(y_tr, yhat_tr)

            rmse_te = np.sqrt(mean_squared_error(y_te, yhat_te))
            mae_te = mean_absolute_error(y_te, yhat_te)
            r2_te = r2_score(y_te, yhat_te)

            st.write(
                f"**Train**: R²={r2_tr:.3f}, RMSE={rmse_tr:.3f}, MAE={mae_tr:.3f}"
            )
            st.write(
                f"**Test** : R²={r2_te:.3f}, RMSE={rmse_te:.3f}, MAE={mae_te:.3f}"
            )

            # 散布図
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(y_tr, yhat_tr, alpha=0.6, label="Train", marker="o")
            ax.scatter(y_te, yhat_te, alpha=0.6, label="Test", marker="^")
            lo = min(y_tr.min(), y_te.min(), yhat_tr.min(), yhat_te.min())
            hi = max(y_tr.max(), y_te.max(), yhat_tr.max(), yhat_te.max())
            ax.plot([lo, hi], [lo, hi], "--")
            ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted (AIC-Poly)")
            ax.legend()
            st.pyplot(fig)

        else:
            # 2値ロジットのみ想定
            if y_tr.nunique() != 2:
                st.error("AIC ロジットは 2 クラス分類のみ対応です。目的変数を 0/1 の2値にしてください。")
                st.stop()

            model_aic, selected = stepwise_aic_logit(X_tr_poly_df, y_tr)
            st.write(f"選択された項の数: {len(selected)}")
            st.write("選択された項（一部）：")
            st.write(selected[:30])

            params = model_aic.params
            pvals = model_aic.pvalues
            coef_df = pd.DataFrame(
                {"term": params.index, "coef": params.values, "p": pvals.values}
            ).sort_values("coef", key=np.abs, ascending=False)
            st.subheader("係数・p値（AIC 選択後 Logit）")
            st.dataframe(coef_df.head(50))

            X_tr_sel = sm.add_constant(X_tr_poly_df[selected])
            X_te_sel = sm.add_constant(X_te_poly_df[selected], has_constant="add")
            yhat_tr_prob = model_aic.predict(X_tr_sel)
            yhat_te_prob = model_aic.predict(X_te_sel)

            # 0.5 を閾値にクラス化
            yhat_tr = (yhat_tr_prob >= 0.5).astype(int)
            yhat_te = (yhat_te_prob >= 0.5).astype(int)

            acc = accuracy_score(y_te, yhat_te)
            prec = precision_score(y_te, yhat_te, average="macro", zero_division=0)
            rec = recall_score(y_te, yhat_te, average="macro", zero_division=0)
            f1 = f1_score(y_te, yhat_te, average="macro", zero_division=0)
            st.write(
                f"Accuracy={acc:.3f}, Precision(macro)={prec:.3f}, "
                f"Recall(macro)={rec:.3f}, F1(macro)={f1:.3f}"
            )

            labs = sorted(pd.Series(y_te).unique().tolist())
            cm = confusion_matrix(y_te, yhat_te, labels=labs)
            cm_df = pd.DataFrame(
                cm,
                index=[f"T{i}" for i in labs],
                columns=[f"P{i}" for i in labs]
            )
            st.write("混同行列（件数）")
            st.dataframe(cm_df)

            cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            im = ax_cm.imshow(cm_norm, interpolation="nearest")
            ax_cm.set_xticks(range(len(labs))); ax_cm.set_xticklabels(labs)
            ax_cm.set_yticks(range(len(labs))); ax_cm.set_yticklabels(labs)
            ax_cm.set_title("Confusion Matrix (row-normalized)")
            for i in range(cm_norm.shape[0]):
                for j in range(cm_norm.shape[1]):
                    ax_cm.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")
            fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
            st.pyplot(fig_cm)


        st.info("AIC モデルは statsmodels を使っているため、SHAP/LIME はここでは実行していません。")
    else:
        # 全変数でそのまま
        st.write("全ての多項式項を使用します。")   
        st.subheader("全変数多項式モデルの結果")

        # まず多項式特徴を明示的に作る
        poly = PolynomialFeatures(
            degree=deg, include_bias=False, interaction_only=interaction_only
        )
        X_tr_poly = poly.fit_transform(X_tr)
        X_te_poly = poly.transform(X_te)
        poly_names = poly.get_feature_names_out(feature_cols)

        X_tr_poly_df = pd.DataFrame(X_tr_poly, columns=poly_names, index=X_tr.index)
        X_te_poly_df = pd.DataFrame(X_te_poly, columns=poly_names, index=X_te.index)

        if task_type == "回帰":
            # ---- OLS を全項で当てる ----
            X_tr_design = sm.add_constant(X_tr_poly_df)
            X_te_design = sm.add_constant(X_te_poly_df, has_constant="add")

            model = sm.OLS(y_tr, X_tr_design).fit()

            st.write(f"使用した項の数: {len(poly_names)}")

            # 係数表
            params = model.params
            pvals = model.pvalues
            coef_df = pd.DataFrame(
                {"term": params.index, "coef": params.values, "p": pvals.values}
            ).sort_values("coef", key=np.abs, ascending=False)
            st.subheader("係数・p値（全項 OLS）")
            st.dataframe(coef_df.head(50))

            # 予測と評価
            yhat_tr = model.predict(X_tr_design)
            yhat_te = model.predict(X_te_design)

            rmse_tr = np.sqrt(mean_squared_error(y_tr, yhat_tr))
            mae_tr  = mean_absolute_error(y_tr, yhat_tr)
            r2_tr   = r2_score(y_tr, yhat_tr)

            rmse_te = np.sqrt(mean_squared_error(y_te, yhat_te))
            mae_te  = mean_absolute_error(y_te, yhat_te)
            r2_te   = r2_score(y_te, yhat_te)

            st.write(
                f"**Train**: R²={r2_tr:.3f}, RMSE={rmse_tr:.3f}, MAE={mae_tr:.3f}"
            )
            st.write(
                f"**Test** : R²={r2_te:.3f}, RMSE={rmse_te:.3f}, MAE={mae_te:.3f}"
            )

            # 散布図
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(y_tr, yhat_tr, alpha=0.6, label="Train", marker="o")
            ax.scatter(y_te, yhat_te, alpha=0.6, label="Test",  marker="^")
            lo = min(y_tr.min(), y_te.min(), yhat_tr.min(), yhat_te.min())
            hi = max(y_tr.max(), y_te.max(), yhat_tr.max(), yhat_te.max())
            ax.plot([lo, hi], [lo, hi], "--")
            ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted (Full Poly OLS)")
            ax.legend()
            st.pyplot(fig)

        else:
            # ---- 2値ロジットを全項で当てる ----
            if y_tr.nunique() != 2:
                st.error("全変数ロジットは 2 クラス分類のみ対応です。目的変数を 0/1 の2値にしてください。")
                st.stop()

            X_tr_design = sm.add_constant(X_tr_poly_df)
            X_te_design = sm.add_constant(X_te_poly_df, has_constant="add")

            model = sm.L
