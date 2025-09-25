# import shap
# from lime.lime_tabular import LimeTabularExplainer
# import streamlit.components.v1 as components
# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# def explain_shap(clf_model, X_train: pd.DataFrame, X_test: pd.DataFrame, task_type: str, ml_type: str):
#     """
#     Display SHAP summary and waterfall plots in Streamlit.
#     """
#     st.markdown("## SHAP による説明")
    
#     # Explainer の選択
#     if ml_type in ("決定木", "ランダムフォレスト"):
#         explainer = shap.TreeExplainer(clf_model)
#     else:
#         explainer = shap.KernelExplainer(
#             clf_model.predict_proba if task_type == "分類" else clf_model.predict,
#             X_train.sample(min(50, len(X_train)), random_state=0)
#         )

#     # サンプルを絞って計算
#     sample_idx = X_test.sample(min(50, len(X_test)), random_state=0).index
#     sample_X = X_test.loc[sample_idx]
#     shap_values = explainer.shap_values(sample_X)

#     # Global summary plot
#     st.text("SHAP summary plot (global)")
    
#     # プロット作成
#     fig = plt.figure(figsize=(8, 6))
#     shap.summary_plot(shap_values, sample_X, show=False)
#     st.pyplot(fig)

#     fig2 = plt.figure(figsize=(8, 6))
#     st.text("SHAP summary plot (bar)")
#     shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)
#     st.pyplot(fig2)
    
    

#     # Local explanation: waterfall plot
#     idx = st.slider("🔎 SHAP waterfall plot で見るテストデータのインデックス", 0, len(X_test)-1, 0)
#     st.text(f"Index={idx} のローカル説明 (waterfall plot)")
#     sv_local = explainer.shap_values(X_test.iloc[idx:idx+1])

#     # 多次元出力対応: 値と base を取捨選択
#     if isinstance(sv_local, list):
#         # TreeExplainer の分類: list of arrays
#         values = sv_local[0][0]
#         base_val = explainer.expected_value[0]
#     elif isinstance(sv_local, np.ndarray) and sv_local.ndim == 3:
#         # KernelExplainer の多出力: shape=(1, n_features, n_outputs)
#         values = sv_local[0, :, 0]
#         base_val = (explainer.expected_value[0]
#                     if isinstance(explainer.expected_value, (list, np.ndarray))
#                     else explainer.expected_value)
#     else:
#         # Regression or single-output
#         values = sv_local[0] if (isinstance(sv_local, np.ndarray) and sv_local.ndim == 2) else sv_local
#         base_val = explainer.expected_value

#     # Waterfall plot
#     fig, ax = plt.subplots(figsize=(8, 3 + len(values)*0.1))
#     shap.plots.waterfall(
#         shap.Explanation(
#             values=values,
#             base_values=base_val,
#             data=X_test.iloc[idx]
#         ),
#         show=False
#     )
#     st.pyplot(fig)

# def explain_lime(clf_model, X_train: pd.DataFrame, X_test: pd.DataFrame, task_type: str):
#     """
#     Display LIME explanation in Streamlit.
#     """
#     st.markdown("## LIME による説明")
#     class_names = (clf_model.classes_.tolist() if task_type == "分類" else None)
#     explainer = LimeTabularExplainer(
#         training_data=X_train.values,
#         feature_names=X_train.columns.tolist(),
#         class_names=class_names,
#         mode=("classification" if task_type == "分類" else "regression")
#     )
#     idx = st.slider("LIME 説明のサンプル インデックス", 0, len(X_test)-1, 0)
#     exp = explainer.explain_instance(
#         X_test.values[idx],
#         (clf_model.predict_proba if task_type == "分類" else clf_model.predict)
#     )
#     components.html(exp.as_html(), height=350)


# xai.py
import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import streamlit.components.v1 as components

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ---- ユーティリティ ----
def _is_tree_classifier(m):
    return isinstance(m, (RandomForestClassifier, DecisionTreeClassifier))

def _is_tree_regressor(m):
    return isinstance(m, (RandomForestRegressor, DecisionTreeRegressor))

def _align_df(df: pd.DataFrame, columns):
    df = df.copy()
    for c in columns:
        if c not in df.columns:
            df[c] = 0
    df = df[columns]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df.columns = df.columns.astype(str)
    return df

def _summarize_background(X_bg: pd.DataFrame):
    try:
        k = 500 if len(X_bg) > 2000 else max(200, min(len(X_bg), 1000))
        return shap.kmeans(X_bg, k)
    except Exception:
        return X_bg

def _slice_output_to_single(expl: shap.Explanation, out_idx: int) -> shap.Explanation:
    """(n_samples, n_features, n_outputs) -> (n_samples, n_features) に安全にスライス"""
    vals = expl.values
    base = expl.base_values
    data = expl.data
    # values: (..., n_outputs)
    if vals.ndim == 3:
        vals = vals[:, :, out_idx]
    # base_values: (n_samples, n_outputs) or (n_samples,)
    if isinstance(base, (np.ndarray, list)) and np.ndim(base) == 2 and base.shape[1] > 1:
        base = np.asarray(base)[:, out_idx]
    return shap.Explanation(
        values=vals,
        base_values=base,
        data=data,
        feature_names=expl.feature_names,
    )

@st.cache_resource
def _compute_shap_values(_model, X_bg: pd.DataFrame, X_te: pd.DataFrame, task_type: str, columns):
    # 列そろえ
    X_bg = _align_df(X_bg, columns)
    X_te = _align_df(X_te, columns)

    # 背景要約
    masker = _summarize_background(X_bg)

    # モデル別に Explainer
    if _is_tree_classifier(_model):
        explainer = shap.TreeExplainer(
            _model,
            data=masker,
            model_output="probability",
            feature_perturbation="interventional",
        )
        sv = explainer(X_te, check_additivity=False)

    elif _is_tree_regressor(_model):
        explainer = shap.TreeExplainer(
            _model,
            data=masker,
            feature_perturbation="interventional",
        )
        sv = explainer(X_te, check_additivity=False)

    else:
        # 非木系: DataFrameを渡すpredict_fnでラップ
        if hasattr(_model, "predict_proba") and task_type == "分類":
            def f(a):
                df = pd.DataFrame(a, columns=columns)
                return _model.predict_proba(df)
        else:
            def f(a):
                df = pd.DataFrame(a, columns=columns)
                return _model.predict(df)

        explainer = shap.Explainer(f, masker)
        sv = explainer(X_te, check_additivity=False)

    # feature_names補完
    if not hasattr(sv, "feature_names") or sv.feature_names is None:
        sv.feature_names = list(columns)

    return sv

def explain_shap(clf_model, X_train: pd.DataFrame, X_test: pd.DataFrame, task_type: str, ml_type: str):
    st.markdown("## SHAP による説明")

    if X_train is None or X_test is None or len(X_test) == 0:
        st.info("SHAP: データが空のためスキップ")
        return

    columns = list(X_train.columns.astype(str))
    X_bg = X_train[columns]
    X_te = X_test[columns]

    # 計算（キャッシュ）
    sv = _compute_shap_values(clf_model, X_bg, X_te, task_type, columns)

    # ---- 出力次元を判定し、1出力に絞る（分類の多クラス/回帰多出力に対応）----
    # 形状が (n_samples, n_features, n_outputs) なら出力選択UI
    out_dim = sv.values.shape[2] if (hasattr(sv, "values") and sv.values.ndim == 3) else 1
    if out_dim > 1:
        out_idx = st.selectbox("可視化する出力（クラス/ターゲット）", list(range(out_dim)), format_func=lambda i: f"output {i}")
        sv_plot = _slice_output_to_single(sv, out_idx)
    else:
        sv_plot = sv

    # ---- グローバル（bar）----
    st.text("SHAP summary (bar)")
    try:
        fig = plt.figure(figsize=(8, 5))
        shap.plots.bar(sv_plot, show=False, max_display=30)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"summary(bar) の描画に失敗: {e}")

    # ---- サマリー（dot）----
    st.text("SHAP summary (dot)")
    try:
        fig2 = plt.figure(figsize=(8, 5))
        shap.summary_plot(sv_plot, X_te, show=False, max_display=30)
        st.pyplot(fig2)
    except Exception as e:
        st.warning(f"summary(dot) の描画に失敗: {e}")

    # ---- ローカル（waterfall）----
    if len(X_te) > 0:
        idx = st.slider("🔎 Waterfallで見るテストインデックス", 0, len(X_te) - 1, 0)
        try:
            # 単一サンプルにスライス
            if hasattr(sv_plot, "values") and sv_plot.values.ndim == 2:
                exp_one = shap.Explanation(
                    values=sv_plot.values[idx],
                    base_values=sv_plot.base_values[idx] if np.ndim(sv_plot.base_values) > 0 else sv_plot.base_values,
                    data=X_te.iloc[idx].values,
                    feature_names=columns,
                )
            else:
                # 既に Explanation の list 等を防ぐためのfallback
                exp_one = sv_plot[idx]

            fig3 = plt.figure(figsize=(8, 5))
            shap.plots.waterfall(exp_one, show=False)
            st.pyplot(fig3)
        except Exception as e:
            st.warning(f"waterfall の描画に失敗: {e}")



# --------------------
# 外部公開：LIME
# --------------------
@st.cache_resource
def _lime_explainer(_X, feature_names, class_names, mode):
    return LimeTabularExplainer(
        training_data=_X,
        feature_names=feature_names,
        class_names=class_names,
        mode=mode
    )

def explain_lime(clf_model, X_train: pd.DataFrame, X_test: pd.DataFrame, task_type: str):
    st.markdown("## LIME による説明")

    if X_train is None or X_test is None or len(X_test) == 0:
        st.info("LIME: データが空のためスキップ")
        return

    feature_names = X_train.columns.tolist()
    class_names = clf_model.classes_.tolist() if task_type == "分類" and hasattr(clf_model, "classes_") else None
    mode = "classification" if task_type == "分類" else "regression"

    explainer = _lime_explainer(X_train.values, feature_names, class_names, mode)

    idx = st.slider("LIME のサンプルインデックス", 0, len(X_test) - 1, 0)
    predict_fn = clf_model.predict_proba if task_type == "分類" and hasattr(clf_model, "predict_proba") \
                 else clf_model.predict

    exp = explainer.explain_instance(
        X_test.values[idx],
        predict_fn
    )
    components.html(exp.as_html(), height=380)
