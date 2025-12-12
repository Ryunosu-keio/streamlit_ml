import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# ==========================================
# 1. データ読み込み & パース処理 (共通)
# ==========================================
@st.cache_data
def load_and_parse_data(uploaded_file):
    """ファイルを読み込み、ファイル名から順序と値を抽出して構造化する"""
    # 拡張子判別
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    def parse_params_ordered(name):
        if pd.isna(name):
            return {
                'param1': 'None', 'param1_val': 0.0,
                'param2': 'None', 'param2_val': 0.0,
                'param3': 'None', 'param3_val': 0.0
            }

        clean_name = str(name).replace('.jpg', '').replace('.JPG', '')
        parts = clean_name.split('_')
        valid_ops = ['brightness', 'contrast', 'gamma', 'sharpness', 'equalization']

        params = []
        for part in parts:
            for op in valid_ops:
                if part.startswith(op):
                    try:
                        val_str = part.replace(op, '')
                        val = float(val_str)
                        params.append((op, val))
                    except ValueError:
                        continue
                    break

        # 3ステップ未満を埋める
        while len(params) < 3:
            params.append(('None', 0.0))

        return {
            'param1': params[0][0], 'param1_val': params[0][1],
            'param2': params[1][0], 'param2_val': params[1][1],
            'param3': params[2][0], 'param3_val': params[2][1]
        }

    parsed_list = [parse_params_ordered(n) for n in df['image_name']]
    params_df = pd.DataFrame(parsed_list)

    # 順序パターンIDを作成 (例: gamma -> sharpness -> equalization)
    params_df['pattern_id'] = (
        params_df['param1'] + " → " + params_df['param2'] + " → " + params_df['param3']
    )

    # 重複列削除
    cols_to_use = params_df.columns.tolist()
    df = df.drop(columns=[c for c in cols_to_use if c in df.columns], errors='ignore')

    df_full = pd.concat([df, params_df], axis=1)
    return df_full


# ==========================================
# 2. 特徴量エンジニアリング & 重み計算
# ==========================================
def create_interaction_features(df):
    """
    'step1_gamma' のように、「場所×種類」で値を格納する特徴量を作成。
    これによりモデルは「1手目のGamma」と「2手目のGamma」を区別できる。
    """
    valid_ops = ['brightness', 'contrast', 'gamma', 'sharpness', 'equalization']
    X_dict = {}

    for i in range(1, 4):
        col_op = f'param{i}'
        col_val = f'param{i}_val'

        for op in valid_ops:
            # 該当する操作の場合のみ値を入れ、それ以外は0
            mask = (df[col_op] == op).astype(float)
            X_dict[f"step{i}_{op}"] = mask * df[col_val]

    return pd.DataFrame(X_dict, index=df.index)


def compute_sample_weights(df):
    """
    pattern_id ごとに件数を数え、その逆数を重みとする。
    -> 多く含まれる加工パターンに学習が偏らないようにする。
    """
    key = df['pattern_id']
    freq = key.value_counts()
    w = key.map(freq).astype(float)
    w = 1.0 / w
    # 平均1にスケーリング（任意）
    w *= (len(w) / w.sum())
    return w


# ---- 18パターン生成 & 値の範囲取得ユーティリティ --------------------------
def generate_allowed_patterns():
    """
    brightnessは最初 / equalizationは最後 / brightnessとequalizationは同居しない / 重複なし
    を満たす全パターン（想定18通り）を列挙。
    文字列表現は 'op1_op2_op3'
    """
    ops = ["brightness", "contrast", "gamma", "sharpness", "equalization"]
    patterns = []

    for p1 in ops:
        for p2 in ops:
            for p3 in ops:
                pat = [p1, p2, p3]

                # 重複禁止
                if len(set(pat)) < 3:
                    continue

                # brightness はあっても Step1 のみ
                if "brightness" in pat and p1 != "brightness":
                    continue

                # equalization はあっても Step3 のみ
                if "equalization" in pat and p3 != "equalization":
                    continue

                # brightness と equalization は同居しない
                if "brightness" in pat and "equalization" in pat:
                    continue

                patterns.append(f"{p1}_{p2}_{p3}")

    return patterns  # 18個になるはず


def get_param_range(df, step, op):
    """
    学習データから step×op ごとの値の min/max を取り、
    存在しない場合はフォールバックの範囲を返す。
    """
    col_op = f'param{step}'
    col_val = f'param{step}_val'
    mask = df[col_op] == op

    if mask.any():
        vmin = df.loc[mask, col_val].min()
        vmax = df.loc[mask, col_val].max()
        if vmin == vmax:
            vmin -= abs(vmin) * 0.1 + 1e-3
            vmax += abs(vmax) * 0.1 + 1e-3
    else:
        # フォールバック範囲
        if op == 'gamma':
            vmin, vmax = 0.3, 1.5
        elif op == 'equalization':
            vmin, vmax = 5.0, 50.0
        elif op == 'brightness':
            vmin, vmax = -50, 50
        elif op == 'contrast':
            vmin, vmax = 0.5, 2.0
        else:  # sharpness など
            vmin, vmax = 0.0, 3.0

    return float(vmin), float(vmax)


def create_full_features_with_orig(df):
    """
    加工パラメータ(step×op) + 元画像特徴(*_orig, *_orig_area) をまとめた特徴量行列を作る。
    戻り値: X_all, orig_cols
      - X_all: 全特徴 DataFrame
      - orig_cols: 「元画像特徴」の列名リスト（新しい画像にコピーする対象）
    """
    # 既存の param 特徴
    X_param = create_interaction_features(df)

    # 元画像特徴（*_orig, *_orig_area）を探す
    orig_cols = [
        c for c in df.columns
        if c.endswith("_orig") or c.endswith("_orig_area")
    ]
    if orig_cols:
        X_orig = df[orig_cols].copy()
    else:
        X_orig = pd.DataFrame(index=df.index)

    X_all = pd.concat([X_param, X_orig], axis=1)
    return X_all, orig_cols


# ==========================================
# 3. メインアプリ
# ==========================================
def main():
    st.set_page_config(page_title="画像加工分析ツール", layout="wide")
    st.title("🧪 画像加工分析 & 最適化ツール")

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

    # サンプル重み（pattern の偏り補正用）
    sample_weights = compute_sample_weights(df_full)

    # タブ定義 (6つ)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 データ概要",
        "🔍 アプローチA: フィルタリング分析",
        "🤖 アプローチB: 個別ML分析",
        "🚀 アプローチC: ML同時最適化",
        "🏆 アプローチD: 18パターン比較",
        "🧬 アプローチE: 画像特徴に応じた加工推薦"
    ])

    # ---------------------------------------------------------------------
    # Tab 1: データ概要
    # ---------------------------------------------------------------------
    with tab1:
        st.subheader("データセットの概要")
        st.write(f"総データ数: **{len(df_full)}** 行")
        st.dataframe(df_full.head())

        st.divider()
        st.subheader("実験された加工パターンの組み合わせ件数")

        pattern_counts = df_full['pattern_id'].value_counts().sort_values(ascending=False)

        if not pattern_counts.empty:
            fig_height = max(5, len(pattern_counts) * 0.4)
            fig, ax = plt.subplots(figsize=(10, fig_height))
            bars = ax.barh(pattern_counts.index, pattern_counts.values)
            ax.set_xlabel("件数")
            ax.set_title("加工パターンの分布")
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height() / 2,
                        f'{int(width)}', ha='left', va='center', fontweight='bold')
            st.pyplot(fig)
        else:
            st.warning("パターンデータがありません。")

        st.divider()
        st.subheader("param / param_val の分布チェック（簡易）")

        op_counts = pd.concat([
            df_full['param1'],
            df_full['param2'],
            df_full['param3']
        ]).value_counts().rename("count")

        st.markdown("各加工種別の出現回数（3ステップ合計）")
        st.dataframe(op_counts.to_frame(), use_container_width=True)

        st.markdown("""
        **🔧 偏り補正の考え方（Tab2〜6 で共通）**  

        - 各画像に `pattern_id = param1 → param2 → param3` を付与。  
        - `pattern_id` ごとの出現回数を数え、その **逆数 (1 / 回数)** をサンプル重みとして学習に使用。  
        - これにより、頻出パターンだけでなくレアなパターンも、  
          できるだけ **公平に学習へ寄与** するようにしている。
        """)

    # ---------------------------------------------------------------------
    # Tab 2: フィルタリング分析
    # ---------------------------------------------------------------------
    with tab2:
        st.header("🔍 アプローチA: 実績からの成功パターン抽出")
        st.markdown("画質指標の閾値で『成功画像』を定義し、リフト値で加工の勝率をみる。")

        c1, c2, c3 = st.columns(3)
        q_mean = c1.slider("輝度(Mean) 上位%", 0, 100, 30, 5) / 100.0
        q_entr = c2.slider("エントロピー 上位%", 0, 100, 30, 5) / 100.0
        q_cont = c3.slider("コントラスト(RMS) 下位%", 0, 100, 30, 5) / 100.0

        th_mean = df_full['all_bL_mean'].quantile(1.0 - q_mean)
        th_entr = df_full['all_sh_grad_entropy'].quantile(1.0 - q_entr)
        th_cont = df_full['all_c_rms_contrast'].quantile(q_cont)

        success_df = df_full[
            (df_full['all_bL_mean'] >= th_mean) &
            (df_full['all_sh_grad_entropy'] >= th_entr) &
            (df_full['all_c_rms_contrast'] <= th_cont)
        ]

        st.metric("条件を満たす成功画像数", f"{len(success_df)} / {len(df_full)}")

        if len(success_df) > 0:
            st.divider()
            st.subheader("ステップごとの勝率分析 (リフト値)")
            st.caption("リフト値 > 1.0 → 成功率が平均より高い")

            best_recipe_filter = {}
            cols = st.columns(3)

            for i in range(1, 4):
                with cols[i - 1]:
                    step_col = f'param{i}'
                    base = df_full[step_col].value_counts(normalize=True)
                    succ = success_df[step_col].value_counts(normalize=True)
                    lift = (succ / base).sort_values(ascending=False)

                    counts = df_full[step_col].value_counts()
                    valid_ops = counts[counts > 10].index
                    lift = lift.loc[lift.index.intersection(valid_ops)]

                    st.markdown(f"#### Step {i}")
                    if not lift.empty:
                        best_op = lift.idxmax()
                        best_recipe_filter[f'Step{i}'] = best_op
                        st.dataframe(
                            lift.to_frame(name="リフト値")
                            .style.format("{:.2f}")
                            .background_gradient(cmap="Reds"),
                            use_container_width=True
                        )
                        st.success(f"推奨: **{best_op}**")
                    else:
                        st.warning("データ不足")

            st.divider()
            st.subheader("📊 フィルタリングによる推奨レシピ")

            rec_vals = []
            for i in range(1, 4):
                op = best_recipe_filter.get(f'Step{i}')
                if op:
                    vals = success_df.loc[success_df[f'param{i}'] == op, f'param{i}_val']
                    avg = vals.mean() if not vals.empty else 0
                    rec_vals.append(f"{op} ({avg:.2f})")
                else:
                    rec_vals.append("N/A")

            st.info(f"👉 **Step1:** {rec_vals[0]}  →  **Step2:** {rec_vals[1]}  →  **Step3:** {rec_vals[2]}")
        else:
            st.warning("成功画像が0枚です。閾値を緩めてください。")

    # ---------------------------------------------------------------------
    # Tab 3: 個別ML分析 (感度分析)
    # ---------------------------------------------------------------------
    with tab3:
        st.header("🤖 アプローチB: 個別因子の感度分析")
        st.markdown("""
        3つの目的変数それぞれについて、独立したランダムフォレスト回帰を構築。  
        **pattern 出現頻度の逆数でサンプル重み付け**して学習します。
        """)

        if st.button("モデル学習を開始する (感度分析)"):
            with st.spinner("特徴量生成 & モデル学習中..."):
                X = create_interaction_features(df_full)

                targets = {
                    '輝度 (bL_mean)': 'all_bL_mean',
                    'エントロピー (Entropy)': 'all_sh_grad_entropy',
                    'コントラスト (RMS)': 'all_c_rms_contrast'
                }

                for label, col_name in targets.items():
                    y = df_full[col_name]

                    X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(
                        X, y, sample_weights,
                        test_size=0.2, random_state=42
                    )

                    rf = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf.fit(X_tr, y_tr, sample_weight=w_tr)

                    r2_train = rf.score(X_tr, y_tr)
                    r2_test = rf.score(X_te, y_te)

                    st.divider()
                    st.markdown(f"#### 🎯 {label}")
                    st.caption(
                        f"[重み付き] Train $R^2$: **{r2_train:.3f}** / "
                        f"Test $R^2$: **{r2_test:.3f}**"
                    )

                    imps = rf.feature_importances_
                    feat_imp = (
                        pd.DataFrame({'feature': X.columns, 'importance': imps})
                        .sort_values('importance', ascending=False)
                        .head(5)
                    )

                    corrs = []
                    dirs = []
                    for f in feat_imp['feature']:
                        c = df_full[col_name].corr(X[f])
                        corrs.append(c)
                        dirs.append("➕ 増加" if c > 0 else "➖ 減少")

                    feat_imp['Correlation'] = corrs
                    feat_imp['Direction'] = dirs

                    st.dataframe(
                        feat_imp[['feature', 'importance', 'Correlation', 'Direction']]
                        .style.background_gradient(subset=['importance'], cmap='Greens'),
                        use_container_width=True
                    )
                    st.caption("※ importance: 寄与度 / Correlation: 値を上げたときの変化方向")

    # ---------------------------------------------------------------------
    # Tab 4: ML同時最適化 (Multi-Output RF 1パターン用)
    # ---------------------------------------------------------------------
    with tab4:
        st.header("🚀 アプローチC: MLによる多目的最適化 (1パターンを深掘り)")
        st.markdown("""
        3つの指標（輝度・エントロピー・コントラスト）を**1つのRFで同時に学習**し，  
        指定した 1 パターンのパラメータ空間をランダムサーチします。
        """)

        c1, c2, c3 = st.columns(3)
        w_mean_c = c1.slider("輝度(Mean) 重視度", 0.0, 5.0, 1.0, key="w1_c")
        w_entr_c = c2.slider("Entropy 重視度", 0.0, 5.0, 2.0, key="w2_c")
        w_cont_c = c3.slider("Contrast抑制 重視度", 0.0, 5.0, 1.0, key="w3_c")

        df_full['internal_pattern_id'] = df_full['pattern_id'].str.replace(' → ', '_', regex=False)
        unique_patterns = sorted(df_full['internal_pattern_id'].unique().tolist())
        default_pattern = "gamma_sharpness_equalization"
        idx = unique_patterns.index(default_pattern) if default_pattern in unique_patterns else 0

        target_pattern = st.selectbox(
            "最適化したい加工順序を選択してください",
            unique_patterns,
            index=idx,
            key="target_pattern_c"
        )
        st.markdown(f"選択中のパターン: **{target_pattern.replace('_', ' → ')}**")

        if st.button("🚀 シミュレーション開始 (アプローチC)"):
            with st.spinner("Multi-Output RF 学習 & ランダムサーチ中..."):
                X_all = create_interaction_features(df_full)
                y_cols = ['all_bL_mean', 'all_sh_grad_entropy', 'all_c_rms_contrast']
                Y_all = df_full[y_cols]

                X_tr, X_te, Y_tr, Y_te, w_tr, w_te = train_test_split(
                    X_all, Y_all, sample_weights,
                    test_size=0.2, random_state=42
                )

                rf_mo = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_mo.fit(X_tr, Y_tr, sample_weight=w_tr)

                Y_pred_te = rf_mo.predict(X_te)
                r2_each = r2_score(Y_te, Y_pred_te, multioutput='raw_values')
                r2_mean = r2_score(Y_te, Y_pred_te, multioutput='uniform_average')

                labels = ['Mean', 'Entropy', 'Contrast']
                r2_df = pd.DataFrame({"Metric": labels, "Test_R2": r2_each})
                st.subheader("Multi-Output RF の当てはまり (C)")
                st.dataframe(r2_df.style.format({"Test_R2": "{:.3f}"}), use_container_width=True)
                st.caption(f"平均 Test R²: **{r2_mean:.3f}**")

                # ランダムサーチ（選択パターンのみ）
                n_trials = 10000
                sim_X = pd.DataFrame(0, index=range(n_trials), columns=X_all.columns)
                ops = target_pattern.split('_')
                sim_params = {}

                for i, op in enumerate(ops, 1):
                    vmin, vmax = get_param_range(df_full, i, op)
                    vals = np.random.uniform(vmin, vmax, n_trials)
                    col_name = f"step{i}_{op}"
                    if col_name in sim_X.columns:
                        sim_X[col_name] = vals
                    sim_params[f"Step{i} ({op})"] = vals

                preds_matrix = rf_mo.predict(sim_X)
                scaler = StandardScaler()
                metrics_norm = scaler.fit_transform(preds_matrix)

                scores = (
                    w_mean_c * metrics_norm[:, 0] +
                    w_entr_c * metrics_norm[:, 1] -
                    w_cont_c * metrics_norm[:, 2]
                )
                best_idx = np.argmax(scores)

                preds = {
                    'Mean': preds_matrix[:, 0],
                    'Entropy': preds_matrix[:, 1],
                    'Contrast': preds_matrix[:, 2]
                }

                st.divider()
                st.subheader("👑 アプローチCで得られた最適パラメータ")
                for key, vals in sim_params.items():
                    st.write(f"**{key}:** `{vals[best_idx]:.3f}`")

                st.subheader("📈 そのときの予測画質指標")
                st.write(f"輝度 (Mean): **{preds['Mean'][best_idx]:.3f}**")
                st.write(f"エントロピー: **{preds['Entropy'][best_idx]:.3f}**")
                st.write(f"コントラスト: **{preds['Contrast'][best_idx]:.3f}**")

                # 散布図
                st.subheader("訓練分布とシミュレーション分布の比較")
                chart_df = pd.DataFrame({
                    'Mean': preds['Mean'],
                    'Entropy': preds['Entropy'],
                    'Contrast': preds['Contrast'],
                    'Score': scores
                })
                top_k = chart_df.nlargest(100, 'Score')

                fig, ax = plt.subplots(1, 2, figsize=(12, 5))

                # Train (暗めグレー)
                ax[0].scatter(df_full['all_bL_mean'], df_full['all_c_rms_contrast'],
                              c='dimgray', alpha=0.25, s=3, label="Train (実データ)")
                ax[1].scatter(df_full['all_sh_grad_entropy'], df_full['all_c_rms_contrast'],
                              c='dimgray', alpha=0.25, s=3, label="Train (実データ)")

                # Sim all（薄グレー）
                ax[0].scatter(chart_df['Mean'], chart_df['Contrast'],
                              c='lightgray', alpha=0.2, s=2, label="Sim (all)")
                ax[1].scatter(chart_df['Entropy'], chart_df['Contrast'],
                              c='lightgray', alpha=0.2, s=2, label="Sim (all)")

                # Top score（赤）
                ax[0].scatter(top_k['Mean'], top_k['Contrast'],
                              c='red', alpha=0.9, s=15, label="Sim (Top Score)")
                ax[1].scatter(top_k['Entropy'], top_k['Contrast'],
                              c='red', alpha=0.9, s=15, label="Sim (Top Score)")

                ax[0].set_title("Mean vs Contrast")
                ax[0].set_xlabel("Mean")
                ax[0].set_ylabel("Contrast")
                ax[1].set_title("Entropy vs Contrast")
                ax[1].set_xlabel("Entropy")
                ax[1].set_ylabel("Contrast")

                for a in ax:
                    a.legend(loc="upper right", fontsize=8)

                st.pyplot(fig)

    # ---------------------------------------------------------------------
    # Tab 5: 18パターンを同条件で比較（アプローチD）
    # ---------------------------------------------------------------------
    with tab5:
        st.header("🏆 アプローチD: 18パターンを同条件で比較")
        st.markdown("""
        - brightness / equalization の制約付きで取りうる **18通りの加工順序** を列挙。  
        - 各パターンで **同じ試行回数** ランダムサンプリングし，  
          MultiOutputRegressor が予測した画質指標からスコアを計算。  
        - 各パターンについて  
          - `max_score` : 最大スコア  
          - `top5_mean` : 上位5%サンプルのスコア平均  
          を指標としてランク付けします。
        """)

        c1, c2, c3 = st.columns(3)
        w_mean_d = c1.slider("輝度(Mean) 重視度", 0.0, 5.0, 1.0, key="w1_d")
        w_entr_d = c2.slider("Entropy 重視度", 0.0, 5.0, 2.0, key="w2_d")
        w_cont_d = c3.slider("Contrast抑制 重視度", 0.0, 5.0, 1.0, key="w3_d")

        n_trials_per_pattern = st.slider(
            "パターンごとのシミュレーション試行数",
            min_value=200, max_value=5000, value=1000, step=200
        )

        if st.button("🏁 18パターン一括シミュレーション開始"):
            with st.spinner("MultiOutputRegressor 学習 & 18パターン一括シミュレーション中..."):
                # ---- モデル学習（順問題） ----
                X_all = create_interaction_features(df_full)
                y_cols = ['all_bL_mean', 'all_sh_grad_entropy', 'all_c_rms_contrast']
                Y_all = df_full[y_cols]

                X_tr, X_te, Y_tr, Y_te, w_tr, w_te = train_test_split(
                    X_all, Y_all, sample_weights,
                    test_size=0.2, random_state=42
                )

                base_rf = RandomForestRegressor(n_estimators=150, random_state=42)
                mo = MultiOutputRegressor(base_rf)
                mo.fit(X_tr, Y_tr, sample_weight=w_tr)

                Y_pred_te = mo.predict(X_te)
                r2_each = r2_score(Y_te, Y_pred_te, multioutput='raw_values')
                r2_mean = r2_score(Y_te, Y_pred_te, multioutput='uniform_average')

                labels = ['Mean', 'Entropy', 'Contrast']
                r2_df = pd.DataFrame({"Metric": labels, "Test_R2": r2_each})

                st.subheader("MultiOutputRegressor の当てはまり (D)")
                st.dataframe(r2_df.style.format({"Test_R2": "{:.3f}"}), use_container_width=True)
                st.caption(f"3指標平均 Test R²: **{r2_mean:.3f}**")

                # ---- 18パターン一括シミュレーション ----
                allowed_patterns = generate_allowed_patterns()
                st.markdown(f"探索対象のパターン数: **{len(allowed_patterns)}** 通り")

                sim_dfs = []

                for pat in allowed_patterns:
                    op1, op2, op3 = pat.split('_')
                    # それぞれの値の範囲（学習データに合わせる）
                    v1min, v1max = get_param_range(df_full, 1, op1)
                    v2min, v2max = get_param_range(df_full, 2, op2)
                    v3min, v3max = get_param_range(df_full, 3, op3)

                    vals1 = np.random.uniform(v1min, v1max, n_trials_per_pattern)
                    vals2 = np.random.uniform(v2min, v2max, n_trials_per_pattern)
                    vals3 = np.random.uniform(v3min, v3max, n_trials_per_pattern)

                    sim_X = pd.DataFrame(0, index=range(n_trials_per_pattern), columns=X_all.columns)
                    col1 = f"step1_{op1}"
                    col2 = f"step2_{op2}"
                    col3 = f"step3_{op3}"
                    if col1 in sim_X.columns:
                        sim_X[col1] = vals1
                    if col2 in sim_X.columns:
                        sim_X[col2] = vals2
                    if col3 in sim_X.columns:
                        sim_X[col3] = vals3

                    preds = mo.predict(sim_X)

                    df_pat = pd.DataFrame({
                        "pattern": pat,
                        "Mean": preds[:, 0],
                        "Entropy": preds[:, 1],
                        "Contrast": preds[:, 2],
                        "step1_op": op1,
                        "step2_op": op2,
                        "step3_op": op3,
                        "step1_val": vals1,
                        "step2_val": vals2,
                        "step3_val": vals3,
                    })
                    sim_dfs.append(df_pat)

                sim_all = pd.concat(sim_dfs, ignore_index=True)

                # ---- スコア計算（全パターンまとめて標準化） ----
                metrics_mat = sim_all[["Mean", "Entropy", "Contrast"]].values
                scaler = StandardScaler()
                metrics_norm = scaler.fit_transform(metrics_mat)

                sim_all["Score"] = (
                    w_mean_d * metrics_norm[:, 0] +
                    w_entr_d * metrics_norm[:, 1] -
                    w_cont_d * metrics_norm[:, 2]
                )

                # ---- パターンごとのランキング ----
                def top5_mean(x):
                    k = max(1, int(len(x) * 0.05))
                    return x.nlargest(k).mean()

                summary = (sim_all
                           .groupby("pattern")["Score"]
                           .agg(max_score="max", top5_mean=top5_mean)
                           .reset_index())

                summary = summary.sort_values(
                    ["top5_mean", "max_score"], ascending=False
                ).reset_index(drop=True)

                st.subheader("18パターンのランキング")
                st.dataframe(
                    summary.style.format({"max_score": "{:.3f}", "top5_mean": "{:.3f}"}),
                    use_container_width=True
                )

                # ---- 全体での「理想解候補」（スコア最大） ----
                best_idx = sim_all["Score"].idxmax()
                best_row = sim_all.loc[best_idx]

                st.divider()
                st.subheader("👑 全パターン中の理想解候補（Score 最大）")

                st.markdown(
                    f"- パターン: **{best_row['pattern'].replace('_', ' → ')}**  \n"
                    f"- Step1: **{best_row['step1_op']}** = `{best_row['step1_val']:.3f}`  \n"
                    f"- Step2: **{best_row['step2_op']}** = `{best_row['step2_val']:.3f}`  \n"
                    f"- Step3: **{best_row['step3_op']}** = `{best_row['step3_val']:.3f}`"
                )

                st.markdown("**そのときの予測画質指標**")
                st.write(f"輝度 (Mean): **{best_row['Mean']:.3f}**")
                st.write(f"エントロピー: **{best_row['Entropy']:.3f}**")
                st.write(f"コントラスト: **{best_row['Contrast']:.3f}**")
                st.write(f"Score: **{best_row['Score']:.3f}**")

                # ---- ざっくり散布図（Train vs Sim）----
                st.subheader("訓練分布と18パターン・シミュレーション分布の比較")

                fig, ax = plt.subplots(1, 2, figsize=(12, 5))

                # Sim (all)
                ax[0].scatter(sim_all['Mean'], sim_all['Contrast'],
                              c='lightgray', alpha=0.15, s=2, label="Sim (18patterns all)")
                ax[1].scatter(sim_all['Entropy'], sim_all['Contrast'],
                              c='lightgray', alpha=0.15, s=2, label="Sim (18patterns all)")

                # Train
                ax[0].scatter(df_full['all_bL_mean'], df_full['all_c_rms_contrast'],
                              c='dimgray', alpha=0.25, s=3, label="Train (real data)")
                ax[1].scatter(df_full['all_sh_grad_entropy'], df_full['all_c_rms_contrast'],
                              c='dimgray', alpha=0.25, s=3, label="Train (real data)")

                # Best 1点
                ax[0].scatter(best_row['Mean'], best_row['Contrast'],
                              c='red', s=40, label="Best Score")
                ax[1].scatter(best_row['Entropy'], best_row['Contrast'],
                              c='red', s=40, label="Best Score")

                ax[0].set_title("Mean vs Contrast")
                ax[0].set_xlabel("Mean")
                ax[0].set_ylabel("Contrast")
                ax[1].set_title("Entropy vs Contrast")
                ax[1].set_xlabel("Entropy")
                ax[1].set_ylabel("Contrast")

                for a in ax:
                    a.legend(loc="upper right", fontsize=8)

                st.pyplot(fig)

        # ---------------------------------------------------------------------
    # Tab 6: 画像特徴に応じた加工推薦（アプローチE）
    # ---------------------------------------------------------------------
    with tab6:
        st.header("🧬 アプローチE: 画像特徴に応じた加工推薦")
        st.markdown("""
        **目的**  
        もともとの画像特徴量（`*_orig` / `*_orig_area`）に応じて、  
        「どの加工パターン・パラメータが良さそうか」を MultiOutputRegressor で推薦します。

        手順:
        1. 順問題モデルを学習  
           - 入力: 加工パラメータ(step×op) + 元画像特徴(`*_orig`, `*_orig_area`)  
           - 出力: Mean / Entropy / Contrast  
        2. 新しい画像の特徴量を 1 行だけ入力  
        3. 18パターン × ランダムサーチでパラメータを振ってスコアを計算  
        4. スコア最大の加工パターン & パラメータを推薦
        """)

        # --- 元画像特徴があるかチェック & 候補列を取得 -------------------
        X_all_tmp, orig_cols_tmp = create_full_features_with_orig(df_full)
        if not orig_cols_tmp:
            st.warning("このデータには *_orig / *_orig_area 列が見つからないため、元画像特徴に応じた推薦はできません。")
            st.stop()

        st.subheader("1. 入力に使う元画像特徴量の選択")
        st.caption("意味がありそうな列だけを選んで学習に使えます（デフォルトは全て）。")

        # デフォルトは全列
        selected_orig_cols = st.multiselect(
            "元画像特徴量（入力に使用）",
            options=orig_cols_tmp,
            default=orig_cols_tmp,
            help="チェックを外した列はモデルの入力から除外されます。"
        )

        if len(selected_orig_cols) == 0:
            st.error("少なくとも1列は選んでください。")
            st.stop()

        st.subheader("2. スコアの重み設定")
        c1, c2, c3 = st.columns(3)
        w_mean_e = c1.slider("輝度(Mean) 重視度", 0.0, 5.0, 1.0, key="w1_e")
        w_entr_e = c2.slider("Entropy 重視度", 0.0, 5.0, 2.0, key="w2_e")
        w_cont_e = c3.slider("Contrast抑制 重視度", 0.0, 5.0, 1.0, key="w3_e")

        st.subheader("3. 新しい画像の特徴量（任意）")
        st.markdown("original-only の特徴量を 1 行だけ持つ CSV / Excel をアップロードしてください（列名は `*_orig`, `*_orig_area` に対応）。")

        new_orig_file = st.file_uploader(
            "新しい画像の特徴量 (optional)",
            type=["csv", "xlsx", "xls"],
            key="new_orig_file"
        )

        # 表示用に image_name があれば使う
        def _fmt_idx(i):
            if "image_name" in df_full.columns:
                return f"{i}: {df_full.loc[i, 'image_name']}"
            elif "file_name" in df_full.columns:
                return f"{i}: {df_full.loc[i, 'file_name']}"
            else:
                return str(i)

        st.markdown("**ファイルを用意していない場合**は、訓練データの中から 1 行選んで \"新しい画像\" とみなしてテストできます。")
        fallback_idx = st.selectbox(
            "訓練データからテスト用の1行を選ぶ（新しい画像ファイルが無い場合用）",
            options=df_full.index,
            format_func=_fmt_idx
        )

        n_trials_per_pattern_e = st.slider(
            "1パターンあたりのランダムサーチ試行数",
            min_value=200, max_value=5000, value=1000, step=200
        )

        if st.button("🚀 学習 & 新しい画像に対する推薦加工を計算"):
            with st.spinner("順問題モデルの学習 & 推薦加工の探索中..."):

                # ---- 順問題モデルの学習 ----
                # 選ばれた元画像特徴のみを使って特徴量を構成
                X_param = create_interaction_features(df_full)
                X_orig = df_full[selected_orig_cols].copy()
                X_all = pd.concat([X_param, X_orig], axis=1)
                orig_cols = selected_orig_cols  # 後でコピーに使う列

                y_cols = ['all_bL_mean', 'all_sh_grad_entropy', 'all_c_rms_contrast']
                Y_all = df_full[y_cols]

                X_tr, X_te, Y_tr, Y_te, w_tr, w_te = train_test_split(
                    X_all, Y_all, sample_weights,
                    test_size=0.2, random_state=42
                )

                base_rf = RandomForestRegressor(
                    n_estimators=150,
                    random_state=42,
                    n_jobs=-1
                )
                mo = MultiOutputRegressor(base_rf)
                mo.fit(X_tr, Y_tr, sample_weight=w_tr)

                Y_pred_te = mo.predict(X_te)
                r2_each = r2_score(Y_te, Y_pred_te, multioutput='raw_values')
                r2_mean = r2_score(Y_te, Y_pred_te, multioutput='uniform_average')

                labels = ['Mean', 'Entropy', 'Contrast']
                r2_df = pd.DataFrame({"Metric": labels, "Test_R2": r2_each})
                st.subheader("順問題モデルの当てはまり (E)")
                st.dataframe(
                    r2_df.style.format({"Test_R2": "{:.3f}"}),
                    use_container_width=True
                )
                st.caption(f"3指標平均 Test R²: **{r2_mean:.3f}**")

                # ---- 特徴量重要度（3出力の平均） -----------------------
                importances_list = []
                for est in mo.estimators_:
                    importances_list.append(est.feature_importances_)
                mean_importance = np.mean(importances_list, axis=0)

                imp_df = pd.DataFrame({
                    "feature": X_all.columns,
                    "importance": mean_importance,
                    "kind": ["orig_feature" if f in orig_cols else "param_feature"
                             for f in X_all.columns]
                }).sort_values("importance", ascending=False)

                st.subheader("特徴量重要度（3出力の平均）")
                st.dataframe(
                    imp_df.head(30)
                    .style.background_gradient(subset=['importance'], cmap='Greens'),
                    use_container_width=True
                )

                # グラフ（上位20特徴）
                top_imp = imp_df.head(20).sort_values("importance")
                fig_imp, ax_imp = plt.subplots(figsize=(8, max(4, len(top_imp) * 0.3)))
                ax_imp.barh(top_imp["feature"], top_imp["importance"])
                ax_imp.set_xlabel("importance (avg over 3 outputs)")
                ax_imp.grid(axis="x", linestyle="--", alpha=0.6)
                st.pyplot(fig_imp)

                # ---- 新しい画像の特徴ベクトル orig_vec を作る ------------
                def _load_table(f):
                    if f.name.endswith(".csv"):
                        return pd.read_csv(f)
                    elif f.name.endswith((".xlsx", ".xls")):
                        return pd.read_excel(f)
                    else:
                        return pd.read_csv(f)

                if new_orig_file is not None:
                    new_df = _load_table(new_orig_file)
                    # orig_cols に対応する列だけ取り出し、足りない列は訓練データの平均で補完
                    orig_vec = pd.Series(index=orig_cols, dtype=float)
                    for c in orig_cols:
                        if c in new_df.columns:
                            orig_vec[c] = float(new_df.loc[0, c])
                        else:
                            orig_vec[c] = float(df_full[c].mean())
                else:
                    # 訓練データから fallback_idx 行を使う
                    orig_vec = df_full.loc[fallback_idx, orig_cols].astype(float)

                st.markdown("**使用する元画像特徴の一部（先頭10列）**")
                st.write(orig_vec.head(10))

                # ---- 18パターン × ランダムサーチで推薦加工を探索 ----
                allowed_patterns = generate_allowed_patterns()
                sim_dfs = []

                for pat in allowed_patterns:
                    op1, op2, op3 = pat.split('_')

                    v1min, v1max = get_param_range(df_full, 1, op1)
                    v2min, v2max = get_param_range(df_full, 2, op2)
                    v3min, v3max = get_param_range(df_full, 3, op3)

                    vals1 = np.random.uniform(v1min, v1max, n_trials_per_pattern_e)
                    vals2 = np.random.uniform(v2min, v2max, n_trials_per_pattern_e)
                    vals3 = np.random.uniform(v3min, v3max, n_trials_per_pattern_e)

                    # X_all と同じ列構造の DataFrame を作る
                    sim_X = pd.DataFrame(0.0, index=range(n_trials_per_pattern_e), columns=X_all.columns)

                    # 元画像特徴は全行同じ値にする
                    for c in orig_cols:
                        sim_X[c] = orig_vec[c]

                    # 加工パラメータをセット
                    c1_name = f"step1_{op1}"
                    c2_name = f"step2_{op2}"
                    c3_name = f"step3_{op3}"
                    if c1_name in sim_X.columns:
                        sim_X[c1_name] = vals1
                    if c2_name in sim_X.columns:
                        sim_X[c2_name] = vals2
                    if c3_name in sim_X.columns:
                        sim_X[c3_name] = vals3

                    preds = mo.predict(sim_X)

                    df_pat = pd.DataFrame({
                        "pattern": pat,
                        "Mean": preds[:, 0],
                        "Entropy": preds[:, 1],
                        "Contrast": preds[:, 2],
                        "step1_op": op1,
                        "step2_op": op2,
                        "step3_op": op3,
                        "step1_val": vals1,
                        "step2_val": vals2,
                        "step3_val": vals3,
                    })
                    sim_dfs.append(df_pat)

                sim_all = pd.concat(sim_dfs, ignore_index=True)

                # ---- スコア計算（全パターンまとめて標準化） ----
                metrics_mat = sim_all[["Mean", "Entropy", "Contrast"]].values
                scaler = StandardScaler()
                metrics_norm = scaler.fit_transform(metrics_mat)

                sim_all["Score"] = (
                    w_mean_e * metrics_norm[:, 0] +
                    w_entr_e * metrics_norm[:, 1] -
                    w_cont_e * metrics_norm[:, 2]
                )

                # パターンごとの評価（max_score / top5_mean）
                def top5_mean(x):
                    k = max(1, int(len(x) * 0.05))
                    return x.nlargest(k).mean()

                summary_e = (sim_all
                             .groupby("pattern")["Score"]
                             .agg(max_score="max", top5_mean=top5_mean)
                             .reset_index())

                summary_e = summary_e.sort_values(
                    ["top5_mean", "max_score"], ascending=False
                ).reset_index(drop=True)

                st.subheader("この画像に対する 18パターンの評価（アプローチE）")
                st.dataframe(
                    summary_e.style.format({"max_score": "{:.3f}", "top5_mean": "{:.3f}"}),
                    use_container_width=True
                )

                # パターン別評価のグラフ（Top 10）
                st.subheader("Top 10 パターン（top5_mean 順）のグラフ")
                top10 = summary_e.head(10).copy()
                top10["pattern_disp"] = top10["pattern"].str.replace("_", " → ")

                fig_pat, ax_pat = plt.subplots(figsize=(8, max(4, len(top10) * 0.4)))
                ax_pat.barh(top10["pattern_disp"], top10["top5_mean"])
                ax_pat.set_xlabel("top5_mean Score")
                ax_pat.invert_yaxis()
                ax_pat.grid(axis="x", linestyle="--", alpha=0.6)
                st.pyplot(fig_pat)

                # 全体でのベスト1候補
                best_idx_e = sim_all["Score"].idxmax()
                best_row_e = sim_all.loc[best_idx_e]

                st.divider()
                st.subheader("👑 この画像に対する理想解候補（Score 最大）")

                st.markdown(
                    f"- パターン: **{best_row_e['pattern'].replace('_', ' → ')}**  \n"
                    f"- Step1: **{best_row_e['step1_op']}** = `{best_row_e['step1_val']:.3f}`  \n"
                    f"- Step2: **{best_row_e['step2_op']}** = `{best_row_e['step2_val']:.3f}`  \n"
                    f"- Step3: **{best_row_e['step3_op']}** = `{best_row_e['step3_val']:.3f}`"
                )
                st.markdown("**そのときの予測画質指標**")
                st.write(f"輝度 (Mean): **{best_row_e['Mean']:.3f}**")
                st.write(f"エントロピー: **{best_row_e['Entropy']:.3f}**")
                st.write(f"コントラスト: **{best_row_e['Contrast']:.3f}**")
                st.write(f"Score: **{best_row_e['Score']:.3f}**")

                # ---- 訓練分布 vs シミュレーション分布のプロット ---------
                st.subheader("訓練分布とシミュレーション分布の比較（この画像向け）")

                fig_sc, ax_sc = plt.subplots(1, 2, figsize=(12, 5))

                # Train（実データ）
                ax_sc[0].scatter(df_full['all_bL_mean'], df_full['all_c_rms_contrast'],
                                 c='dimgray', alpha=0.25, s=3, label="Train (real)")
                ax_sc[1].scatter(df_full['all_sh_grad_entropy'], df_full['all_c_rms_contrast'],
                                 c='dimgray', alpha=0.25, s=3, label="Train (real)")

                # Sim（全サンプル）
                ax_sc[0].scatter(sim_all['Mean'], sim_all['Contrast'],
                                 c='lightgray', alpha=0.15, s=2, label="Sim (all)")
                ax_sc[1].scatter(sim_all['Entropy'], sim_all['Contrast'],
                                 c='lightgray', alpha=0.15, s=2, label="Sim (all)")

                # Best 1点
                ax_sc[0].scatter(best_row_e['Mean'], best_row_e['Contrast'],
                                 c='red', s=40, label="Best Score")
                ax_sc[1].scatter(best_row_e['Entropy'], best_row_e['Contrast'],
                                 c='red', s=40, label="Best Score")

                ax_sc[0].set_title("Mean vs Contrast")
                ax_sc[0].set_xlabel("Mean")
                ax_sc[0].set_ylabel("Contrast")
                ax_sc[1].set_title("Entropy vs Contrast")
                ax_sc[1].set_xlabel("Entropy")
                ax_sc[1].set_ylabel("Contrast")

                for a in ax_sc:
                    a.legend(loc="upper right", fontsize=8)

                st.pyplot(fig_sc)

if __name__ == "__main__":
    main()