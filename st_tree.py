import streamlit as st
import tree as tr
from sklearn.tree import DecisionTreeClassifier as DTC, DecisionTreeRegressor as DTR
from sklearn.ensemble import RandomForestClassifier as RFC, RandomForestRegressor as RFR
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error, confusion_matrix, classification_report, precision_score, recall_score, f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE  
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import xai


st.title("機械学習")

warnings.simplefilter('ignore')

# モード選択
mode = st.radio("訓練データとテストデータの決め方", ('ランダム', '自分で決める'))

# ファイルアップロード
if mode == 'ランダム':
    uploaded_file = st.file_uploader("CSVファイルをアップロード", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        features = df.columns
        target = st.selectbox("目的変数を選択", features)
        removal = st.multiselect("説明変数として除外する列", features)
        name = st.text_input("ファイル名")
else:
    uploaded_file_train = st.file_uploader("訓練用CSV", type="csv")
    uploaded_file_test = st.file_uploader("テスト用CSV", type="csv")
    if uploaded_file_train and uploaded_file_test:
        train_df = pd.read_csv(uploaded_file_train)
        test_df = pd.read_csv(uploaded_file_test)
        features = train_df.columns
        target = st.selectbox("目的変数を選択", features)
        removal = st.multiselect("説明変数として除外する列", features)
        name = st.text_input("出力ファイル名")

# モデルと分類 or 回帰選択
ml_type = st.sidebar.selectbox("モデルタイプ", ["DecisionTree", "RandomForest", "SVM","NN"])
task_type = st.sidebar.radio("学習タイプ", ["分類", "回帰"])

# ハイパーパラメータ
if ml_type == "DecisionTree":
    depth = st.sidebar.slider("max_depth", 1, 20, (2, 5))
    min_split = st.sidebar.slider("min_samples_split", 2, 10, (2, 4))
    leaf = st.sidebar.slider("min_samples_leaf", 1, 10, (1, 2))
    random_state = st.sidebar.slider("random_state", 0, 30, (0, 3))
    params = {
        "max_depth": list(range(depth[0], depth[1])),
        "min_samples_split": list(range(min_split[0], min_split[1])),
        "min_samples_leaf": list(range(leaf[0], leaf[1])),
        "random_state": list(range(random_state[0], random_state[1]))
    }
elif ml_type == "RandomForest":
    estimators = st.sidebar.slider("n_estimators", 10, 100, (10, 30))
    depth = st.sidebar.slider("max_depth", 1, 20, (2, 5))
    params = {
        "n_estimators": list(range(estimators[0], estimators[1])),
        "max_depth": list(range(depth[0], depth[1]))
    }
elif ml_type == "SVM":
    C = st.sidebar.slider("C", 1, 100, (1, 10))
    gamma = st.sidebar.slider("gamma", 0.001, 1.0, (0.01, 0.1))
    params = {
        "C": list(range(C[0], C[1])),
        "gamma": [round(i, 4) for i in np.arange(gamma[0], gamma[1], 0.01)]
    }
elif ml_type == "NN":
    # MLP の層構成と正則化パラメータ
    n_layers = st.sidebar.slider("隠れ層の数",      1, 3, 2)
    size     = st.sidebar.slider("各層のユニット数", 10, 200, 50)
    alpha    = st.sidebar.slider("L2 正則化 (alpha)",     1e-5, 1e-1, 1e-4, format="%.5f")
    lr       = st.sidebar.slider("学習率 init",          1e-4, 1e-1, 1e-3, format="%.5f")
    hidden   = tuple([size] * n_layers)
    params = {
        "hidden_layer_sizes": [hidden],
        "alpha": [alpha],                  # now a float, not a tuple
        "learning_rate_init": [lr],        # now a float, not a tuple
        # 必要なら solver 等も追加可能
    }


# 分類タスク時のみ現れるオーバーサンプリング選択
oversample_option = None
if task_type == "分類":
    oversample_option = st.sidebar.selectbox(
        "オーバーサンプリング方法",
        ["なし", "SMOTE", "Resample"]
    )

if st.button("モデル構築"):
    # ── データ準備 ──────────────────────────────
    if mode == 'ランダム':
        X, Y, features = tr.dataset(df, target, removal)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=0,
            stratify=Y if task_type=="分類" else None
        )
    else:
        X_train, Y_train, features = tr.dataset(train_df, target, removal)
        X_test,  Y_test,  _        = tr.dataset(test_df,  target, removal)

    # ── 分類タスク：オーバーサンプリング適用 ────────────
    if task_type == "分類" and oversample_option != "なし":
        if oversample_option == "SMOTE":
            smote = SMOTE(random_state=42)
            X_train, Y_train = smote.fit_resample(X_train, Y_train)
            st.info(f"SMOTE 適用後の学習サンプル数：{X_train.shape[0]}")

        elif oversample_option == "Resample":
            df_tr = pd.concat([X_train, Y_train.rename("target")], axis=1)
            max_count = df_tr["target"].value_counts().max()
            parts = []
            for cls, grp in df_tr.groupby("target"):
                up = resample(
                    grp, replace=True,
                    n_samples=max_count,
                    random_state=42
                )
                parts.append(up)
            df_bal = pd.concat(parts)
            Y_train = df_bal["target"]
            X_train = df_bal.drop(columns=["target"])
            st.info(f"Resample 適用後の学習サンプル数：{X_train.shape[0]}")


    # モデル選択
    if ml_type == "DecisionTree":
        model = DTC(class_weight="balanced") if task_type == "分類" else DTR()
    elif ml_type == "RandomForest":
        model = RFC(class_weight='balanced') if task_type == "分類" else RFR()
    elif ml_type == "SVM":
        # model = SVC() if task_type == "分類" else SVR()
        model = SVC(probability=True) if task_type == "分類" else SVR()
    elif ml_type == "NN":
         model = MLPClassifier(max_iter=1000, random_state=0) if task_type == "分類" \
                else MLPRegressor(max_iter=1000, random_state=0)


    clf = tr.grid_search(model, X_train, Y_train, params)
    clf_model = model.set_params(**clf.best_params_)
    clf_model.fit(X_train, Y_train)
    pred_test = clf_model.predict(X_test)

    # 評価出力
    if task_type == "分類":
        st.write("Accuracy (Train):", clf_model.score(X_train, Y_train))
        st.write("Accuracy (Test):", clf_model.score(X_test, Y_test))
        st.dataframe(confusion_matrix(Y_test, pred_test))
        st.write("Classification Report:")

        st.subheader("Classification Report")
        report = classification_report(Y_test, pred_test, output_dict=True)
        st.dataframe(pd.DataFrame(report).T)  # 指標を表形式で表示

        # 個別指標で出したい場合
        prec = precision_score(Y_test, pred_test, average="macro")
        rec  = recall_score   (Y_test, pred_test, average="macro")
        f1   = f1_score       (Y_test, pred_test, average="macro")
        st.write(f"Precision (macro): {prec:.3f}")
        st.write(f"Recall    (macro): {rec:.3f}")
        st.write(f"F1-score  (macro): {f1:.3f}")
        
    else:
        ## ── 予測 ───────────────────────────────
        pred_test  = clf_model.predict(X_test)
        pred_train = clf_model.predict(X_train)

        # ── 基本指標 ───────────────────────────
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        import numpy as np

        # Test
        rmse_test = np.sqrt(mean_squared_error(Y_test,  pred_test))
        mae_test  = mean_absolute_error     (Y_test,  pred_test)

        # Train
        rmse_train = np.sqrt(mean_squared_error(Y_train, pred_train))
        mae_train  = mean_absolute_error     (Y_train, pred_train)

        st.subheader("回帰指標 (Train / Test)")
        st.write(f"RMSE  : {rmse_train:.3f}  /  {rmse_test:.3f}")
        st.write(f"MAE   : {mae_train:.3f}  /  {mae_test:.3f}")

        # ── k‑fold CV でばらつきチェック ────────
        from sklearn.model_selection import cross_val_score

        cv_rmse = -cross_val_score(
            clf_model, X_train, Y_train,
            scoring="neg_root_mean_squared_error",
            cv=5
        )
        st.write(f"CV‐RMSE (5‑fold) 平均±SD : {cv_rmse.mean():.3f} ± {cv_rmse.std():.3f}")

        # ── 実測 vs 予測　散布図 ──────────────────
        
        fig_scatter, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(Y_train, pred_train,  alpha=0.6, label="Train",  marker="o")
        ax.scatter(Y_test,  pred_test,   alpha=0.6, label="Test",   marker="^")
        ax.plot([Y_train.min(), Y_train.max()],
                [Y_train.min(), Y_train.max()],
                "--", color="gray")           # 45° 完全一致ライン
        ax.set_xlabel("Actual value")
        ax.set_ylabel("Predicted value")
        ax.set_title(f"Actual vs Predicted (Model:{ml_type})")
        #決定係数
        r2_train = r2_score(Y_train, pred_train)
        r2_test  = r2_score(Y_test,  pred_test)
        ax.text(0.05, 0.95, f"R² (Train): {r2_train:.3f}\nR² (Test): {r2_test:.3f}",
                transform=ax.transAxes, fontsize=12, verticalalignment='top')
        ax.legend()
        st.pyplot(fig_scatter)

        

    st.json(clf.best_params_)

        
    if isinstance(clf_model, (DTC, DTR)):
        graph = tr.visualize(clf_model, features)
        if isinstance(graph, str):
            st.warning(graph)
        else:
            st.graphviz_chart(graph)
    else:
        st.info("可視化はDecisionTreeの分類器のみ対応しています。")

    st.markdown("Feature Importance(特徴量重要度)")
    # ── 特徴量重要度（feature_importances_ 属性があるモデルのみ） ──
    if hasattr(clf_model, "feature_importances_"):
        fig = tr.importance(clf_model, features)
        st.pyplot(fig)
    else:
        st.info("このモデルでは特徴量重要度を表示できません。")

    st.session_state.model_ready = True
    st.session_state.clf_model   = clf_model
    st.session_state.X_train     = X_train
    st.session_state.X_test      = X_test
    st.session_state.task_type   = task_type
    st.session_state.ml_type     = ml_type

if st.session_state.get("model_ready", False):
    clf_model = st.session_state.clf_model
    X_train   = st.session_state.X_train
    X_test    = st.session_state.X_test
    task_type = st.session_state.task_type
    ml_type   = st.session_state.ml_type

    # SHAP 説明
    xai.explain_shap(clf_model, X_train, X_test, task_type, ml_type)

    # LIME 説明
    xai.explain_lime(clf_model, X_train, X_test, task_type)

