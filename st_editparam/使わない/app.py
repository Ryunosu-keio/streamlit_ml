# app.py
import streamlit as st
from core import load_and_parse_data, compute_sample_weights
from tabs import (
    show_tab1_overview,
    show_tab2_filter,
    show_tab3_sensitivity,
    show_tab4_multiopt_single,
    show_tab5_multiopt_18patterns,
)

def main():
    st.set_page_config(page_title="画像加工分析ツール", layout="wide")
    st.title("🧪 画像加工分析 & 最適化ツール")

    st.markdown("""
    このアプリでは、以下の4つ(+1)のアプローチで  
    **縮瞳に有利な画像加工レシピ** を探索します。

    1. **実績分析 (フィルタリング)** … 過去の成功データから「勝率の高いパターン」を見つける  
    2. **個別ML分析 (感度分析)** … 輝度・エントロピー・コントラストそれぞれに効く因子を特定  
    3. **ML最適化 (シミュレーター, 1パターン)** … 3指標を同時に満たす「未知の最強設定」を探す  
    4. **18パターン一括評価** … brightness/equalization 制約付き18パターンを一気に比較
    """)

    # サイドバー: データ入力
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

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 データ概要",
        "🔍 アプローチA: フィルタリング",
        "🤖 アプローチB: 感度分析",
        "🚀 アプローチC: 単一パターン最適化",
        "🧮 アプローチD: 18パターン評価",
    ])

    with tab1:
        show_tab1_overview(df_full, sample_weights)

    with tab2:
        show_tab2_filter(df_full)

    with tab3:
        show_tab3_sensitivity(df_full, sample_weights)

    with tab4:
        show_tab4_multiopt_single(df_full, sample_weights)

    with tab5:
        show_tab5_multiopt_18patterns(df_full, sample_weights)


if __name__ == "__main__":
    main()
