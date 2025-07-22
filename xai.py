import shap
from lime.lime_tabular import LimeTabularExplainer
import streamlit.components.v1 as components
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def explain_shap(clf_model, X_train: pd.DataFrame, X_test: pd.DataFrame, task_type: str, ml_type: str):
    """
    Display SHAP summary and waterfall plots in Streamlit.
    """
    st.markdown("## SHAP による説明")
    
    # Explainer の選択
    if ml_type in ("決定木", "ランダムフォレスト"):
        explainer = shap.TreeExplainer(clf_model)
    else:
        explainer = shap.KernelExplainer(
            clf_model.predict_proba if task_type == "分類" else clf_model.predict,
            X_train.sample(min(50, len(X_train)), random_state=0)
        )

    # サンプルを絞って計算
    sample_idx = X_test.sample(min(50, len(X_test)), random_state=0).index
    sample_X = X_test.loc[sample_idx]
    shap_values = explainer.shap_values(sample_X)

    # Global summary plot
    st.text("SHAP summary plot (global)")
    
    # プロット作成
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, sample_X, show=False)
    st.pyplot(fig)

    fig2 = plt.figure(figsize=(8, 6))
    st.text("SHAP summary plot (bar)")
    shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)
    st.pyplot(fig2)
    
    

    # Local explanation: waterfall plot
    idx = st.slider("🔎 SHAP waterfall plot で見るテストデータのインデックス", 0, len(X_test)-1, 0)
    st.text(f"Index={idx} のローカル説明 (waterfall plot)")
    sv_local = explainer.shap_values(X_test.iloc[idx:idx+1])

    # 多次元出力対応: 値と base を取捨選択
    if isinstance(sv_local, list):
        # TreeExplainer の分類: list of arrays
        values = sv_local[0][0]
        base_val = explainer.expected_value[0]
    elif isinstance(sv_local, np.ndarray) and sv_local.ndim == 3:
        # KernelExplainer の多出力: shape=(1, n_features, n_outputs)
        values = sv_local[0, :, 0]
        base_val = (explainer.expected_value[0]
                    if isinstance(explainer.expected_value, (list, np.ndarray))
                    else explainer.expected_value)
    else:
        # Regression or single-output
        values = sv_local[0] if (isinstance(sv_local, np.ndarray) and sv_local.ndim == 2) else sv_local
        base_val = explainer.expected_value

    # Waterfall plot
    fig, ax = plt.subplots(figsize=(8, 3 + len(values)*0.1))
    shap.plots.waterfall(
        shap.Explanation(
            values=values,
            base_values=base_val,
            data=X_test.iloc[idx]
        ),
        show=False
    )
    st.pyplot(fig)

def explain_lime(clf_model, X_train: pd.DataFrame, X_test: pd.DataFrame, task_type: str):
    """
    Display LIME explanation in Streamlit.
    """
    st.markdown("## LIME による説明")
    class_names = (clf_model.classes_.tolist() if task_type == "分類" else None)
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=class_names,
        mode=("classification" if task_type == "分類" else "regression")
    )
    idx = st.slider("LIME 説明のサンプル インデックス", 0, len(X_test)-1, 0)
    exp = explainer.explain_instance(
        X_test.values[idx],
        (clf_model.predict_proba if task_type == "分類" else clf_model.predict)
    )
    components.html(exp.as_html(), height=350)
