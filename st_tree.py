import streamlit as st
import tree as tr
from sklearn.tree import DecisionTreeClassifier as DTC, DecisionTreeRegressor as DTR
from sklearn.ensemble import RandomForestClassifier as RFC, RandomForestRegressor as RFR
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, classification_report
)
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
import pandas as pd
import numpy as np
import warnings

import matplotlib.pyplot as plt
import xai as xai

# =========================
# 初期設定
# =========================
st.title("機械学習（被験者リーク無し GroupKFold｜再集計＋ベストfold可視化＋XAI）")
warnings.simplefilter('ignore')

# ---- セッション初期化（再実行でもモデルを保持）----
if "cv_ready" not in st.session_state:
    st.session_state.cv_ready = False
    st.session_state.cv_payload = None      # CV結果一式（学習済モデル含む）
    st.session_state.shap_cache_key = None  # SHAP計算のキャッシュキー
    st.session_state.shap_values = None     # shap.Explanation（または list[Explanation]）
    st.session_state.shap_task = None       # "分類" / "回帰"
    st.session_state.shap_model_id = None   # id(mdl_b) でモデルが変わったら再計算

# =========================
# 入力UI
# =========================
mode = st.radio("データの指定方法", ('ランダム（単一ファイルからCV）', '自分で決める（学習/評価を別ファイル）'))

df = train_df = test_df = None
if mode == 'ランダム（単一ファイルからCV）':
    up = st.file_uploader("データファイル（CSV / XLSX）", type=["csv", "xls", "xlsx"])
    if up:
        df = pd.read_excel(up) if up.name.endswith((".xls", ".xlsx")) else pd.read_csv(up)
        features_all = df.columns
        target = st.selectbox("目的変数を選択", features_all)
        group_col = "folder_name"  # 被験者ID列（固定）
        removal = st.multiselect("説明変数から除外する列", features_all, default=[group_col])
        name = st.text_input("実験名（任意）")
else:
    up_tr = st.file_uploader("学習用（CSV / XLSX）", type=["csv", "xls", "xlsx"], key="up_tr")
    up_te = st.file_uploader("評価用（CSV / XLSX）", type=["csv", "xls", "xlsx"], key="up_te")
    if up_tr and up_te:
        train_df = pd.read_excel(up_tr) if up_tr.name.endswith((".xls", ".xlsx")) else pd.read_csv(up_tr)
        test_df  = pd.read_excel(up_te) if up_te.name.endswith((".xls", ".xlsx")) else pd.read_csv(up_te)
        features_all = train_df.columns
        target = st.selectbox("目的変数を選択", features_all)
        group_col = "folder_name"
        removal = st.multiselect("説明変数から除外する列", features_all, default=[group_col])
        name = st.text_input("実験名（任意）")

ml_type = st.sidebar.selectbox("モデル", ["DecisionTree", "RandomForest", "SVM", "NN", "XGBoost"])
task_type = st.sidebar.radio("タスク", ["分類", "回帰"])

# 分類タスクなら、連続目的変数の分位ラベリングオプション
if task_type == "分類":
    bin_choice = st.sidebar.selectbox("クラス分割（目的変数をビン化）", ["二分位", "三分位", "四分位"])
    n_bins = {"二分位": 2, "三分位": 3, "四分位": 4}[bin_choice]

# ハイパーパラメータ（探索範囲）
if ml_type == "DecisionTree":
    depth = st.sidebar.slider("max_depth", 1, 30, (3, 8))
    min_split = st.sidebar.slider("min_samples_split", 2, 20, (2, 6))
    leaf = st.sidebar.slider("min_samples_leaf", 1, 20, (1, 3))
    params = {
        "max_depth": list(range(depth[0], depth[1] + 1)),
        "min_samples_split": list(range(min_split[0], min_split[1] + 1)),
        "min_samples_leaf": list(range(leaf[0], leaf[1] + 1)),
    }
elif ml_type == "RandomForest":
    estimators = st.sidebar.slider("n_estimators", 10, 300, (50, 150))
    depth = st.sidebar.slider("max_depth", 2, 30, (5, 15))
    params = {
        "n_estimators": list(range(estimators[0], estimators[1] + 1, 25)),
        "max_depth": list(range(depth[0], depth[1] + 1, 5)),
    }
elif ml_type == "SVM":
    C = st.sidebar.slider("C", 1, 200, (1, 50))
    gamma = st.sidebar.slider("gamma", 1e-4, 1.0, (0.001, 0.1))
    params = {
        "C": list(range(C[0], C[1] + 1, 5)),
        "gamma": [round(v, 5) for v in np.geomspace(gamma[0], gamma[1], 6)],
    }
elif ml_type == "NN":
    n_layers = st.sidebar.slider("隠れ層数", 1, 3, 2)
    size     = st.sidebar.slider("各層ユニット数", 10, 300, 100)
    alpha    = st.sidebar.select_slider("L2(alpha)", options=[1e-5, 1e-4, 1e-3, 1e-2], value=1e-4)
    hidden   = tuple([size] * n_layers)
    params = {"hidden_layer_sizes": [hidden], "alpha": [alpha]}

elif ml_type == "XGBoost":
    lr = st.sidebar.slider("learning_rate", 0.01, 0.5, (0.05, 0.2))
    depth = st.sidebar.slider("max_depth", 2, 15, (3, 8))
    estimators = st.sidebar.slider("n_estimators", 50, 500, (100, 300))
    subsample = st.sidebar.slider("subsample", 0.5, 1.0, (0.8, 1.0))
    colsample = st.sidebar.slider("colsample_bytree", 0.5, 1.0, (0.8, 1.0))
    params = {
        "learning_rate": [round(v, 3) for v in np.linspace(lr[0], lr[1], 3)],
        "max_depth": list(range(depth[0], depth[1] + 1, 2)),
        "n_estimators": list(range(estimators[0], estimators[1] + 1, 50)),
        "subsample": [round(v, 2) for v in np.linspace(subsample[0], subsample[1], 3)],
        "colsample_bytree": [round(v, 2) for v in np.linspace(colsample[0], colsample[1], 3)],
    }

# 分類のみ：オーバーサンプリング
oversample_option = None
if task_type == "分類":
    oversample_option = st.sidebar.selectbox("オーバーサンプリング", ["なし", "SMOTE", "Resample"])

# CV設定
n_splits = st.sidebar.slider("CV分割数（GroupKFold）", 2, 10, 5)
random_state = st.sidebar.number_input("random_state", 0, 9999, 42)

# =========================
# 学習・CV 実行（ボタン押下のときだけ）
# =========================
if st.button("クロスバリデーション実行"):
    # データセット生成
    if mode == 'ランダム（単一ファイルからCV）':
        if df is None:
            st.error("ファイルをアップロードしてください。"); st.stop()
        X, Y, features = tr.dataset(df, target, removal)
        groups = df[group_col]
    else:
        if train_df is None:
            st.error("学習/評価ファイルをアップロードしてください。"); st.stop()
        X, Y, features = tr.dataset(train_df, target, removal)
        groups = train_df[group_col]

    # 分類：連続目的変数を分位ラベリング
    if task_type == "分類":
        y_cont = pd.to_numeric(Y, errors="coerce")
        if y_cont.isna().any():
            st.warning("目的変数に数値化できない値が含まれています。欠損は除外して学習します。")
        valid_idx = y_cont.notna()
        X, y_cont, groups = X.loc[valid_idx], y_cont.loc[valid_idx], groups.loc[valid_idx]
        try:
            y_bins = pd.qcut(y_cont, q=n_bins, labels=False, duplicates="drop")
        except ValueError as e:
            st.error(f"分位ビン作成に失敗しました: {e}")
            st.stop()
        if pd.Series(y_bins).nunique() < n_bins:
            st.warning("分位点が重複し、クラス数が減少しました。")
        Y = pd.Series(y_bins.astype(int), index=y_cont.index, name=target)

    # モデル生成（ベース器）
    if ml_type == "DecisionTree":
        model = DTC(class_weight="balanced", random_state=random_state) if task_type == "分類" else DTR(random_state=random_state)
    elif ml_type == "RandomForest":
        model = RFC(class_weight="balanced", random_state=random_state) if task_type == "分類" else RFR(random_state=random_state)
    elif ml_type == "SVM":
        model = SVC(probability=True, random_state=random_state) if task_type == "分類" else SVR()
    elif ml_type == "NN":
        model = MLPClassifier(max_iter=1000, random_state=random_state) if task_type == "分類" else MLPRegressor(max_iter=1000, random_state=random_state)

    elif ml_type == "XGBoost":
        model = XGBClassifier(eval_metric="mlogloss", random_state=random_state) if task_type == "分類" else XGBRegressor(random_state=random_state)

    # elif ml_type == "SVM":
    #     st.write("SVMは標準化を強く推奨します。")
    #     if task_type == "分類":
    #         model = Pipeline(steps=[
    #             ("scaler", StandardScaler()),
    #             ("model", SVC(probability=True, random_state=random_state))
    #         ])
    #         # パラメータ名は "ステップ名__ハイパラ"
    #         C_lo, C_hi = 1, 50
    #         gammas = [round(v, 5) for v in np.geomspace(0.001, 0.1, 6)]
    #         params = {
    #             "model__C": list(range(C_lo, C_hi + 1, 5)),
    #             "model__gamma": gammas,
    #             # 必要なら "model__kernel": ["rbf", "linear"] なども
    #         }
    #     else:
    #         model = Pipeline(steps=[
    #             ("scaler", StandardScaler()),
    #             ("model", SVR())
    #         ])
    #         C_lo, C_hi = 1, 50
    #         gammas = [round(v, 5) for v in np.geomspace(0.001, 0.1, 6)]
    #         params = {
    #             "model__C": list(range(C_lo, C_hi + 1, 5)),
    #             "model__gamma": gammas,
    #             # "model__epsilon": [0.1, 0.2, 0.3] なども可
    #         }

    # elif ml_type == "NN":
    #     # NN も標準化推奨（収束・安定性のため）
    #     model = Pipeline(steps=[
    #         ("scaler", StandardScaler()),
    #         ("model", MLPClassifier(max_iter=1000, random_state=random_state) if task_type=="分類"
    #                 else MLPRegressor(max_iter=1000, random_state=random_state))
    #     ])
    #     # 既存の params を "model__" プレフィックスに
    #     params = {f"model__{k}": v for k, v in params.items()}








    gkf = GroupKFold(n_splits=n_splits)

    # 集計器
    best_params_list = []
    best_fold = None
    best_score = -np.inf
    best_pack = None  # ("cls"/"reg", mdl, X_tr, X_te, Y_tr, Y_te, features, meta)

    if task_type == "分類":
        acc_list, prec_list, rec_list, f1_list = [], [], [], []
        cm_sum = None
        labels_seen = set()
    else:
        y_tr_all, yhat_tr_all = [], []
        y_te_all, yhat_te_all = [], []
        rmse_te_list, mae_te_list, r2_te_list = [], [], []
        rmse_tr_list, mae_tr_list, r2_tr_list = [], [], []

    # ===== CV ループ =====
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, Y, groups=groups), 1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        Y_tr, Y_te = Y.iloc[tr_idx], Y.iloc[te_idx]

        # 分類：オーバーサンプリング
        if task_type == "分類" and oversample_option != "なし":
            if oversample_option == "SMOTE":
                sm = SMOTE(random_state=random_state)
                X_tr, Y_tr = sm.fit_resample(X_tr, Y_tr)
            elif oversample_option == "Resample":
                tmp = pd.concat([X_tr, Y_tr.rename("target")], axis=1)
                max_count = tmp["target"].value_counts().max()
                parts = [resample(g, replace=True, n_samples=max_count, random_state=random_state)
                         for _, g in tmp.groupby("target")]
                tmp_up = pd.concat(parts)
                Y_tr = tmp_up["target"]
                X_tr = tmp_up.drop(columns=["target"])

        # グリッドサーチ → 学習
        clf = tr.grid_search(model, X_tr, Y_tr, params)
        best_params = clf.best_params_
        best_params_list.append({"fold": fold, **best_params})

        mdl = model.set_params(**best_params)
        mdl.fit(X_tr, Y_tr)

        # 予測・集計・ベストfold更新
        if task_type == "分類":
            pred_te = mdl.predict(X_te)
            acc  = mdl.score(X_te, Y_te)
            prec = precision_score(Y_te, pred_te, average="macro", zero_division=0)
            rec  = recall_score   (Y_te, pred_te, average="macro", zero_division=0)
            f1   = f1_score       (Y_te, pred_te, average="macro", zero_division=0)
            acc_list.append(acc); prec_list.append(prec); rec_list.append(rec); f1_list.append(f1)

            # 総和混同行列
            labels_seen.update(pd.Series(Y_te).unique().tolist())
            labs_seen_sorted = sorted(list(labels_seen))
            cm = confusion_matrix(Y_te, pred_te, labels=labs_seen_sorted)
            if cm_sum is None:
                cm_sum = cm
            else:
                # ラベル数拡張に対応
                if cm_sum.shape != cm.shape:
                    new_cm = np.zeros((len(labs_seen_sorted), len(labs_seen_sorted)), dtype=int)
                    new_cm[:cm_sum.shape[0], :cm_sum.shape[1]] = cm_sum
                    cm_sum = new_cm
                cm_sum += cm

            # ベストfold（F1）
            if f1 > best_score:
                best_score = f1
                best_fold = fold
                best_pack = ("cls", mdl, X_tr, X_te, Y_tr, Y_te, features,
                             {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "cm": cm})

        else:
            pred_tr = mdl.predict(X_tr)
            pred_te = mdl.predict(X_te)

            rmse_tr = float(np.sqrt(mean_squared_error(Y_tr, pred_tr)))
            mae_tr  = float(mean_absolute_error(Y_tr, pred_tr))
            r2_tr   = float(r2_score(Y_tr, pred_tr))
            rmse_te = float(np.sqrt(mean_squared_error(Y_te, pred_te)))
            mae_te  = float(mean_absolute_error(Y_te, pred_te))
            r2_te   = float(r2_score(Y_te, pred_te))
            rmse_tr_list.append(rmse_tr); mae_tr_list.append(mae_tr); r2_tr_list.append(r2_tr)
            rmse_te_list.append(rmse_te); mae_te_list.append(mae_te); r2_te_list.append(r2_te)

            y_tr_all.extend(Y_tr.tolist());    yhat_tr_all.extend(pred_tr.tolist())
            y_te_all.extend(Y_te.tolist());    yhat_te_all.extend(pred_te.tolist())

            # ベストfold（R2）
            if r2_te > best_score:
                best_score = r2_te
                best_fold = fold
                best_pack = ("reg", mdl, X_tr, X_te, Y_tr, Y_te, features,
                             {"rmse_tr": rmse_tr, "mae_tr": mae_tr, "r2_tr": r2_tr,
                              "rmse_te": rmse_te, "mae_te": mae_te, "r2_te": r2_te,
                              "yhat_tr": pred_tr, "yhat_te": pred_te})

    # ====== 再集計表示 ======
    st.subheader("クロスバリデーション再集計")

    if task_type == "分類":
        df_scores = pd.DataFrame({
            "Accuracy": acc_list,
            "Precision(macro)": prec_list,
            "Recall(macro)": rec_list,
            "F1(macro)": f1_list
        }, index=[f"fold{i}" for i in range(1, len(acc_list)+1)])
        st.dataframe(df_scores)
        st.write("平均 ± SD"); st.write(df_scores.agg(["mean", "std"]))

        if cm_sum is not None:
            st.write("総和混同行列（全fold合算）")
            labs = sorted(list(labels_seen))
            cm_df = pd.DataFrame(cm_sum, index=[f"T{i}" for i in labs], columns=[f"P{i}" for i in labs])
            st.dataframe(cm_df)

    else:
        df_scores_test = pd.DataFrame({
            "RMSE(test)": rmse_te_list,
            "MAE(test)" : mae_te_list,
            "R2(test)"  : r2_te_list
        }, index=[f"fold{i}" for i in range(1, len(rmse_te_list)+1)])
        df_scores_train = pd.DataFrame({
            "RMSE(train)": rmse_tr_list,
            "MAE(train)" : mae_tr_list,
            "R2(train)"  : r2_tr_list
        }, index=[f"fold{i}" for i in range(1, len(rmse_tr_list)+1)])
        st.write("Test 指標（平均 ± SD）");  st.dataframe(df_scores_test);  st.write(df_scores_test.agg(["mean", "std"]))
        st.write("Train 指標（平均 ± SD）"); st.dataframe(df_scores_train); st.write(df_scores_train.agg(["mean", "std"]))

        # 全fold統合のR²（参考）
        y_tr_all = pd.Series(y_tr_all); yhat_tr_all = pd.Series(yhat_tr_all)
        y_te_all = pd.Series(y_te_all); yhat_te_all = pd.Series(yhat_te_all)
        r2_train_all = r2_score(y_tr_all, yhat_tr_all) if len(y_tr_all) > 0 else np.nan
        r2_test_all  = r2_score(y_te_all, yhat_te_all) if len(y_te_all) > 0 else np.nan

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_tr_all, yhat_tr_all, alpha=0.6, label="Train", marker="o", color="blue")
        ax.scatter(y_te_all, yhat_te_all, alpha=0.6, label="Test",  marker="^", color="red")
        lo = min(y_tr_all.min(), y_te_all.min(), yhat_tr_all.min(), yhat_te_all.min())
        hi = max(y_tr_all.max(), y_te_all.max(), yhat_tr_all.max(), yhat_te_all.max())
        ax.plot([lo, hi], [lo, hi], "--", color="gray")
        ax.set_xlabel("Actual value"); ax.set_ylabel("Predicted value")
        ax.set_title(f"Actual vs Predicted (All folds, {ml_type})")
        ax.text(0.05, 0.95, f"R² (Train): {r2_train_all:.3f}\nR² (Test): {r2_test_all:.3f}",
                transform=ax.transAxes, fontsize=12, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
        ax.legend()
        st.pyplot(fig)

    st.subheader("各Foldの最適ハイパーパラメータ（参考）")
    st.dataframe(pd.DataFrame(best_params_list))

    # ===== ベストfoldの結果（学習時に一度だけ表示できる） =====
    st.subheader(f"ベストFoldの結果（fold={best_fold}）")
    if best_pack is not None:
        kind, mdl_b, X_tr_b, X_te_b, Y_tr_b, Y_te_b, feats_b, meta = best_pack

        if kind == "cls":
            pred_te_b = mdl_b.predict(X_te_b)
            labs = sorted(pd.Series(Y_te_b).unique().tolist())
            class_names = [f"C{i}" for i in labs]
            cm_raw = confusion_matrix(Y_te_b, pred_te_b, labels=labs)
            cm_df = pd.DataFrame(cm_raw, index=[f"True_{c}" for c in class_names],
                                        columns=[f"Pred_{c}" for c in class_names])


            # pred_te_b はすでに計算済み
            acc_b  = accuracy_score(Y_te_b, pred_te_b)
            prec_b = precision_score(Y_te_b, pred_te_b, average="macro", zero_division=0)
            rec_b  = recall_score   (Y_te_b, pred_te_b, average="macro", zero_division=0)
            f1_b   = f1_score       (Y_te_b, pred_te_b, average="macro", zero_division=0)

            st.write(f"Accuracy: {acc_b:.3f} / Precision(macro): {prec_b:.3f} "
                    f"/ Recall(macro): {rec_b:.3f} / F1(macro): {f1_b:.3f}")
            st.caption(f"Baseline(1/num_classes) ≈ {1/len(labs):.3f}")
            st.write("混同行列（Best fold, 件数）"); st.dataframe(cm_df)

            cm_norm = cm_raw.astype(float) / (cm_raw.sum(axis=1, keepdims=True) + 1e-12)
            fig1, ax1 = plt.subplots(figsize=(5.5, 5))
            im = ax1.imshow(cm_norm, interpolation="nearest")
            ax1.set_title("Confusion Matrix (row-normalized)")
            ax1.set_xticks(range(len(class_names))); ax1.set_xticklabels(class_names)
            ax1.set_yticks(range(len(class_names))); ax1.set_yticklabels(class_names)
            for i in range(cm_norm.shape[0]):
                for j in range(cm_norm.shape[1]):
                    ax1.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")
            fig1.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
            st.pyplot(fig1)

            rep = classification_report(Y_te_b, pred_te_b, labels=labs, output_dict=True, zero_division=0)
            rep_df = pd.DataFrame(rep).T
            rep_df = rep_df.rename(index={str(k): class_names[i] for i, k in enumerate(labs)})
            st.write("クラス別 Precision/Recall/F1（Best fold）")
            st.dataframe(rep_df[['precision','recall','f1-score','support']])

        else:
            st.write(f"Train: R²={meta['r2_tr']:.3f}, RMSE={meta['rmse_tr']:.3f}, MAE={meta['mae_tr']:.3f}")
            st.write (f"Test : R²={meta['r2_te']:.3f}, RMSE={meta['rmse_te']:.3f}, MAE={meta['mae_te']:.3f}")
            fig, ax = plt.subplots(figsize=(6,6))
            ax.scatter(Y_tr_b, meta["yhat_tr"], alpha=0.6, label="Train", marker="o", color="blue")
            ax.scatter(Y_te_b, meta["yhat_te"], alpha=0.6, label="Test",  marker="^", color="red")
            lo = min(Y_tr_b.min(), Y_te_b.min(), np.min(meta["yhat_tr"]), np.min(meta["yhat_te"]))
            hi = max(Y_tr_b.max(), Y_te_b.max(), np.max(meta["yhat_tr"]), np.max(meta["yhat_te"]))
            ax.plot([lo, hi], [lo, hi], "--", color="gray")
            ax.set_xlabel("Actual value"); ax.set_ylabel("Predicted value")
            ax.set_title(f"Best Fold Actual vs Predicted ({ml_type})")
            ax.text(0.05, 0.95, f"R² (Train): {meta['r2_tr']:.3f}\nR² (Test): {meta['r2_te']:.3f}",
                    transform=ax.transAxes, fontsize=12, va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
            ax.legend()
            st.pyplot(fig)

        # 特徴量重要度
        st.subheader("特徴量重要度（Best fold, 対応モデルのみ）")
        if hasattr(mdl_b, "feature_importances_"):
            try:
                fig_imp = tr.importance(mdl_b, feats_b)
                st.pyplot(fig_imp)
            except Exception:
                importances = pd.Series(mdl_b.feature_importances_, index=feats_b).sort_values(ascending=False).head(30)
                fig2, ax2 = plt.subplots(figsize=(6, min(10, 0.3*len(importances))))
                importances.iloc[::-1].plot(kind="barh", ax=ax2)
                ax2.set_title("Top Feature Importances (Best fold)")
                st.pyplot(fig2)
        else:
            st.info("このモデルでは特徴量重要度を表示できません。")

        # === ここで「結果をセッションへ保存」：以降の再描画で再学習しない ===
        bg = X_tr_b if len(X_tr_b) <= 200 else X_tr_b.sample(200, random_state=42)
        st.session_state.cv_ready = True
        st.session_state.cv_payload = {
            "task_type": task_type,
            "ml_type": ml_type,
            "best_pack": best_pack,     # 学習済みモデル含む
            "bg": bg,                   # SHAP背景
            "X_te": X_te_b,             # Best fold の X_test
            "features": feats_b,
            "best_fold": best_fold,
        }
        # SHAPキャッシュは新しいモデルなので無効化
        st.session_state.shap_cache_key = None
        st.session_state.shap_values = None
        st.session_state.shap_task = None
        st.session_state.shap_model_id = id(mdl_b)

# =========================
# SHAP（Best fold）描画：保存済み結果だけを使うので再学習しない
# =========================
st.subheader("XAI（Best fold）")

if st.session_state.get("cv_ready", False) and st.session_state.get("cv_payload"):
    payload = st.session_state.cv_payload
    task_type = payload["task_type"]
    ml_type   = payload["ml_type"]
    best_fold = payload["best_fold"]

    # best_pack: ("cls"/"reg", mdl, X_tr, X_te, Y_tr, Y_te, features, meta)
    kind, mdl_b, X_tr_b, X_te_b, Y_tr_b, Y_te_b, feats_b, meta = payload["best_pack"]

    # 背景＆テスト（列順を best fold 学習時の features で揃える）
    X_bg = payload["bg"][feats_b]
    X_te = payload["X_te"][feats_b]

    # --- SHAP ---
    xai.explain_shap(mdl_b, X_bg, X_te, task_type, ml_type)

    # --- LIME ---
    xai.explain_lime(mdl_b, X_bg, X_te, task_type)
else:
    st.info("上の『クロスバリデーション実行』でモデルを作成すると、ここにXAIが表示されます。")


    #パスが通ってないのでこれでやる　
    # python -m streamlit run st_tree.py
