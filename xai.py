import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import streamlit.components.v1 as components

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor  # XGBoost も Tree 系として扱う


# ============================================================
# ユーティリティ
# ============================================================

def _is_tree_classifier(m):
    return isinstance(m, (RandomForestClassifier, DecisionTreeClassifier, XGBClassifier))

def _is_tree_regressor(m):
    return isinstance(m, (RandomForestRegressor, DecisionTreeRegressor, XGBRegressor))

def _align_df(df: pd.DataFrame, columns):
    """列順と dtype を揃える"""
    df = df.copy()
    for c in columns:
        if c not in df.columns:
            df[c] = 0
    df = df[columns]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df.columns = df.columns.astype(str)
    return df

def _ensure_2d(X):
    """Series / 1D array を必ず 2D にする"""
    if isinstance(X, pd.Series):
        return X.to_frame().T
    if isinstance(X, np.ndarray) and X.ndim == 1:
        return pd.DataFrame([X])
    return X

def _summarize_background(X_bg: pd.DataFrame):
    """KernelExplainer 用に背景を要約"""
    try:
        k = 400 if len(X_bg) > 2000 else max(100, min(len(X_bg), 800))
        return shap.kmeans(X_bg, k)
    except Exception:
        return X_bg

def _slice_output_to_single(expl: shap.Explanation, out_idx: int) -> shap.Explanation:
    """(n, f, out) → (n, f) に切り出し（多クラス用）"""
    vals = expl.values
    base = expl.base_values

    if vals.ndim == 3:
        vals = vals[:, :, out_idx]

    if isinstance(base, np.ndarray) and np.ndim(base) == 2:
        base = base[:, out_idx]

    return shap.Explanation(
        values=vals,
        base_values=base,
        data=expl.data,
        feature_names=expl.feature_names,
    )


# ============================================================
# SHAP 計算（モデルはキャッシュ対象外にしてエラー回避）
# ============================================================

@st.cache_resource
def _compute_shap_values(_model, X_bg: pd.DataFrame, X_te: pd.DataFrame, task_type: str, columns):
    """
    SHAP値を計算する（Tree → TreeExplainer、その他 → shap.Explainer）。
    _model はキャッシュのハッシュ対象外。
    """
    model = _model

    X_bg = _align_df(X_bg, columns)
    X_te = _align_df(X_te, columns)

    masker = _summarize_background(X_bg)

    # --- Tree 系 ---
    if _is_tree_classifier(model):
        explainer = shap.TreeExplainer(
            model,
            data=masker,
            model_output="probability",
            feature_perturbation="interventional",
        )
        try:
            sv = explainer(X_te, check_additivity=False)
        except TypeError:
            sv = explainer(X_te)

    elif _is_tree_regressor(model):
        explainer = shap.TreeExplainer(
            model,
            data=masker,
            feature_perturbation="interventional",
        )
        try:
            sv = explainer(X_te, check_additivity=False)
        except TypeError:
            sv = explainer(X_te)

    # --- 非 Tree 系（Kernel / その他）---
    else:
        if hasattr(model, "predict_proba") and task_type == "分類":
            def f(a):
                df_tmp = pd.DataFrame(a, columns=columns)
                return model.predict_proba(df_tmp)
        else:
            def f(a):
                df_tmp = pd.DataFrame(a, columns=columns)
                return model.predict(df_tmp)

        explainer = shap.Explainer(f, masker)
        sv = explainer(X_te)

    if not hasattr(sv, "feature_names") or sv.feature_names is None:
        sv.feature_names = list(columns)

    return sv


# ============================================================
# SHAP 可視化
# ============================================================
def explain_shap(model, X_train: pd.DataFrame, X_test: pd.DataFrame, task_type: str, ml_type: str):

    st.markdown("## SHAP による説明")

    if X_train is None or X_test is None or len(X_test) == 0:
        st.info("SHAP: データが空です。")
        return

    columns = list(X_train.columns.astype(str))
    X_bg = X_train[columns]
    X_te = X_test[columns]

    # ---- SHAP 計算 ----
    sv = _compute_shap_values(model, X_bg, X_te, task_type, columns)

    # ---- 出力次元処理 ----
    out_dim = sv.values.shape[2] if sv.values.ndim == 3 else 1
    if out_dim > 1:
        out_idx = st.selectbox(
            "可視化する出力（クラス）",
            list(range(out_dim)),
            format_func=lambda i: f"output {i}"
        )
        sv_plot = _slice_output_to_single(sv, out_idx)
    else:
        sv_plot = sv

    # ============================================================
    # ① Summary plot
    # ============================================================
    st.subheader("Summary（global importance）")

    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.bar(sv_plot, show=False)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.warning(f"bar summary failed: {e}")

    try:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        shap.summary_plot(sv_plot, X_te, show=False)
        st.pyplot(fig2)
        plt.close(fig2)
    except Exception as e:
        st.warning(f"dot summary failed: {e}")

    # ============================================================
    # ② Dependence plot は削除（なし）
    # ============================================================

    # ============================================================
    # ③ Interaction values（交互作用）
    # ============================================================
    st.subheader("Interaction values（交互作用）")

    inter_df = None

    if _is_tree_classifier(model) or _is_tree_regressor(model):
        try:
            explainer_int = shap.TreeExplainer(model)

            # 必ず 2D ndarray に変換
            X_int_df = _align_df(_ensure_2d(X_te), columns)

            # 計算コスト削減したければサンプリング
            max_samples_for_inter = 300
            if len(X_int_df) > max_samples_for_inter:
                X_int_df = X_int_df.sample(max_samples_for_inter, random_state=0)

            if isinstance(X_int_df, pd.DataFrame):
                X_int = X_int_df.values
            else:
                X_int = np.asarray(X_int_df, dtype=float)

            if X_int.ndim == 1:
                X_int = X_int.reshape(1, -1)

            inter_vals = explainer_int.shap_interaction_values(X_int)

            if isinstance(inter_vals, list):
                # 多クラス → クラスごとの行列を絶対値平均
                mats = [np.abs(v).mean(axis=0) for v in inter_vals]
                inter_mat = np.mean(mats, axis=0)
            else:
                inter_mat = np.abs(inter_vals).mean(axis=0)

            # 上三角だけ → ペアに変換
            tri = np.triu_indices_from(inter_mat, k=1)
            pairs = [
                (columns[i], columns[j], float(inter_mat[i, j]))
                for i, j in zip(tri[0], tri[1])
            ]
            inter_df = pd.DataFrame(pairs, columns=["feat1", "feat2", "interaction"])
            inter_df = inter_df.sort_values("interaction", ascending=False)

            if len(inter_df) > 0:
                st.dataframe(inter_df.head(20))
                st.success(
                    f"最大交互作用：{inter_df.iloc[0]['feat1']} × {inter_df.iloc[0]['feat2']}"
                )
            else:
                st.info("交互作用が計算できませんでした。")

        except Exception as e:
            st.warning(f"interaction SHAP failed: {e}")
    else:
        st.info("交互作用 SHAP は Tree 系モデルのみ対応しています。")
    # ============================================================
    # ④ Interaction dependence plot（2変数の関係）
    # ============================================================
    if inter_df is not None and len(inter_df) > 0:
        st.subheader("Interaction dependence plot（2変数の関係）")

        # 特徴量1は全特徴量から選択（x軸）
        f1 = st.selectbox("特徴量1（x軸）", columns, index=0)

        # 特徴量2のモード
        feat2_options = ["選ばない", "最大交互作用"] + [c for c in columns if c != f1]
        feat2_choice = st.selectbox("特徴量2（色）", feat2_options, index=1)  # デフォルトは「最大交互作用」

        try:
            if feat2_choice == "選ばない":
                # ---- 単色散布図（交互作用なし・純粋な単変量形）----
                X2 = _ensure_2d(X_te)
                x = X2[f1].values
                feat_idx = columns.index(f1)
                y = sv_plot.values[:, feat_idx]

                fig4, ax4 = plt.subplots(figsize=(8, 5))
                ax4.scatter(x, y, s=10)  # 単色
                ax4.set_xlabel(f1)
                ax4.set_ylabel(f"SHAP value for {f1}")
                ax4.grid(True, alpha=0.3)
                st.pyplot(fig4)
                plt.close(fig4)

            else:
                # ---- interaction_index を決めて shap.dependence_plot で描画 ----
                if feat2_choice == "最大交互作用":
                    # inter_df から f1 と最大交互作用の相手を探す
                    cand = inter_df[
                        (inter_df["feat1"] == f1) | (inter_df["feat2"] == f1)
                    ]

                    if len(cand) > 0:
                        best = cand.iloc[0]  # inter_df は interaction 降順ソート済み
                        if best["feat1"] == f1:
                            interaction_index = best["feat2"]
                        else:
                            interaction_index = best["feat1"]
                        st.info(f"特徴量2: {interaction_index}（{f1} と最大交互作用）")
                    else:
                        interaction_index = f1
                        st.info("この特徴量との交互作用情報がないため、特徴量1自身で色付けしました。")
                else:
                    # 任意に選んだ特徴量
                    interaction_index = feat2_choice

                fig4, ax4 = plt.subplots(figsize=(8, 5))
                shap.dependence_plot(
                    f1,
                    sv_plot.values,
                    _ensure_2d(X_te),
                    interaction_index=interaction_index,
                    feature_names=columns,
                    ax=ax4,
                    show=False
                )
                st.pyplot(fig4)
                plt.close(fig4)

        except Exception as e:
            st.warning(f"interaction dependence plot failed: {e}")


    # ============================================================
    # ⑤ Local waterfall
    # ============================================================
    st.subheader("Local explanation（waterfall）")

    if len(X_te) > 0:
        idx = st.slider("インデックス", 0, len(X_te) - 1, 0)

        try:
            exp = shap.Explanation(
                values=sv_plot.values[idx],
                base_values=(
                    sv_plot.base_values[idx]
                    if np.ndim(sv_plot.base_values) > 0 else sv_plot.base_values
                ),
                data=X_te.iloc[idx].values,
                feature_names=columns
            )
            fig5, ax5 = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(exp, show=False)
            st.pyplot(fig5)
            plt.close(fig5)
        except Exception as e:
            st.warning(f"waterfall failed: {e}")


# ============================================================
# LIME
# ============================================================

@st.cache_resource
def _lime_explainer(_X, feature_names, class_names, mode):
    """_X の先頭に _ をつけてキャッシュのハッシュ対象から外す"""
    return LimeTabularExplainer(
        training_data=_X,
        feature_names=feature_names,
        class_names=class_names,
        mode=mode
    )

def explain_lime(model, X_train: pd.DataFrame, X_test: pd.DataFrame, task_type: str):
    st.markdown("## LIME による説明")

    if X_train is None or X_test is None or len(X_test) == 0:
        st.info("LIME: データが空です。")
        return

    feature_names = X_train.columns.tolist()
    class_names = model.classes_.tolist() if task_type == "分類" and hasattr(model, "classes_") else None
    mode = "classification" if task_type == "分類" else "regression"

    explainer = _lime_explainer(X_train.values, feature_names, class_names, mode)

    idx = st.slider("LIME インデックス", 0, len(X_test) - 1, 0)
    predict_fn = model.predict_proba if task_type == "分類" and hasattr(model, "predict_proba") \
                 else model.predict

    exp = explainer.explain_instance(X_test.values[idx], predict_fn)
    components.html(exp.as_html(), height=380)
