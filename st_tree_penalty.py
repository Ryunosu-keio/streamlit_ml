# st_consistent_cv.py
# -*- coding: utf-8 -*-
import warnings
warnings.simplefilter("ignore")

import inspect
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import tree as tr                  # あなたのユーティリティ（dataset, importance など）
import xai as xai                  # あなたのXAIモジュール

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

from sklearn.tree import DecisionTreeClassifier as DTC, DecisionTreeRegressor as DTR
from sklearn.ensemble import RandomForestClassifier as RFC, RandomForestRegressor as RFR
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor


# =====================
# グループ重み & 一貫性関連
# =====================
def group_sample_weight(groups: pd.Series) -> np.ndarray:
    """各グループの合計寄与が均等になるよう w_i = 1/|g|（合計はサンプル数に正規化）"""
    counts = groups.value_counts()
    w = groups.map(lambda g: 1.0 / counts[g]).astype(float).values
    w *= (len(w) / w.sum())
    return w

def consistency_penalty(yhat: np.ndarray, groups_val: pd.Series) -> float:
    """Σ_g Var(ŷ_i | i∈g)"""
    s = 0.0
    gsr = groups_val.reset_index(drop=True)
    for g, subidx in gsr.groupby(gsr):
        arr = yhat[subidx.index]
        if len(arr) >= 2:
            s += float(np.var(arr))
    return s

def combined_loss(y_true, y_pred, groups_val: pd.Series, task_type: str, lam: float) -> float:
    """L_total = L_task + λ * consistency"""
    if task_type == "分類":
        task = 1.0 - accuracy_score(y_true, y_pred)
    else:
        task = float(np.mean((y_true - y_pred) ** 2))
    cons = consistency_penalty(np.asarray(y_pred), groups_val)
    return task + lam * cons

def model_supports_sample_weight(model) -> bool:
    """estimator.fit が sample_weight を受けるか"""
    try:
        est = model.steps[-1][1] if hasattr(model, "steps") else model
        return "sample_weight" in inspect.signature(est.fit).parameters
    except Exception:
        return False

class ConsistencyScorer:
    """GridSearchCV の scoring に渡す callable（大きいほど良い → -L_total を返す）"""
    def __init__(self, task_type: str, groups_full: pd.Series, lam: float):
        self.task_type = task_type
        self.groups_full = groups_full
        self.lam = float(lam)
    def __call__(self, estimator, X_val, y_val):
        gv = self.groups_full.loc[X_val.index] if hasattr(X_val, "index") else self.groups_full.iloc[:len(y_val)]
        y_pred = estimator.predict(X_val)
        L = combined_loss(np.asarray(y_val), np.asarray(y_pred), gv, self.task_type, self.lam)
        return -L

def fit_with_consistent_grid_search(
    base_model,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    groups_tr_cv: pd.Series,      # ← CV分割用（被験者IDなど）
    groups_tr_cons: pd.Series,    # ← 一貫性＆重み用（あなたの27×7グループなど）
    param_grid: dict,
    task_type: str,
    lambda_cons: float = 0.0,
    inner_splits: int = 3,
    random_state: int = 42
):
    model = deepcopy(base_model)

    # scoring は λ=0 なら通常, >0 なら一貫性込み
    scoring = ("accuracy" if task_type == "分類" else "neg_mean_squared_error") if lambda_cons == 0.0 \
              else ConsistencyScorer(task_type, groups_full=groups_tr_cons, lam=lambda_cons)

    inner_cv = GroupKFold(n_splits=inner_splits)

    fit_params = {}
    if model_supports_sample_weight(model):
        fit_params["sample_weight"] = group_sample_weight(groups_tr_cons)

    gscv = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=inner_cv,
        refit=True,
        n_jobs=-1,
        verbose=0,
        error_score="raise"
    )
    gscv.fit(X_tr, y_tr, groups=groups_tr_cv, **fit_params)
    return gscv.best_estimator_, gscv.best_params_


# =====================
# アプリ本体
# =====================
st.title("GroupKFold + Consistency Constraint + XAI")
st.caption("CV用グループ（被験者ID等）と、一貫性用グループ（加工グループ等）を分離")

# --- 入力 ---
mode = st.radio("データの指定方法", ("ランダム（単一ファイルからCV）", "自分で決める（学習/評価を別ファイル）"))
df = train_df = test_df = None
features_all = []
target = None

if mode == "ランダム（単一ファイルからCV）":
    up = st.file_uploader("データファイル（CSV / XLSX）", type=["csv", "xls", "xlsx"])
    if up:
        df = pd.read_excel(up) if up.name.endswith((".xls", ".xlsx")) else pd.read_csv(up)
        features_all = list(df.columns)
        target = st.selectbox("目的変数", features_all)
        removal = st.multiselect("説明変数から除外", features_all, [])
else:
    up_tr = st.file_uploader("学習用（CSV / XLSX）", type=["csv", "xls", "xlsx"], key="up_tr")
    up_te = st.file_uploader("評価用（CSV / XLSX）", type=["csv", "xls", "xlsx"], key="up_te")
    if up_tr and up_te:
        train_df = pd.read_excel(up_tr) if up_tr.name.endswith((".xls", ".xlsx")) else pd.read_csv(up_tr)
        test_df  = pd.read_excel(up_te) if up_te.name.endswith((".xls", ".xlsx")) else pd.read_csv(up_te)
        features_all = list(train_df.columns)
        target = st.selectbox("目的変数", features_all)
        removal = st.multiselect("説明変数から除外", features_all, [])

# --- モデル選択 ---
ml_type = st.sidebar.selectbox("モデル", ["DecisionTree", "RandomForest", "SVM", "NN", "XGBoost"])
task_type = st.sidebar.radio("タスク", ["分類", "回帰"])

# クラス分割（任意）
if task_type == "分類":
    n_bins = {"二分位": 2, "三分位": 3, "四分位": 4}[st.sidebar.selectbox("目的変数の分位ビン化", ["二分位", "三分位", "四分位"])]

# ハイパラグリッド
if ml_type == "DecisionTree":
    depth = st.sidebar.slider("max_depth", 1, 30, (3, 9, 2))  # (start, stop, step) 風に利用
    min_split = st.sidebar.slider("min_samples_split (max)", 2, 20, 10)
    leaf = st.sidebar.slider("min_samples_leaf (max)", 1, 20, 5)
    params = {
        "max_depth": list(range(3, depth + 1, 2)),
        "min_samples_split": list(range(2, min_split + 1, 2)),
        "min_samples_leaf": list(range(1, leaf + 1))
    }
elif ml_type == "RandomForest":
    n_est = st.sidebar.slider("n_estimators (max)", 50, 400, 200)
    depth = st.sidebar.slider("max_depth (max)", 4, 30, 16)
    params = {
        "n_estimators": list(range(50, n_est + 1, 50)),
        "max_depth": list(range(4, depth + 1, 4)),
    }
elif ml_type == "SVM":
    params = {"C": [1, 10, 50, 100], "gamma": [0.001, 0.01, 0.1]}
elif ml_type == "NN":
    size = st.sidebar.slider("hidden size", 10, 300, 100)
    layers = st.sidebar.slider("layers", 1, 3, 2)
    alpha = st.sidebar.select_slider("alpha", options=[1e-5, 1e-4, 1e-3, 1e-2], value=1e-4)
    params = {"hidden_layer_sizes": [tuple([size]*layers)], "alpha": [alpha]}
else:  # XGBoost
    lr = st.sidebar.slider("learning_rate (max)", 0.05, 0.3, 0.2)
    depth = st.sidebar.slider("max_depth (max)", 3, 12, 8)
    n_est = st.sidebar.slider("n_estimators (max)", 100, 500, 300)
    params = {
        "learning_rate": [0.05, 0.1, lr],
        "max_depth": list(range(3, depth + 1, 2)),
        "n_estimators": list(range(100, n_est + 1, 100)),
    }

# オーバーサンプリング（分類のみ）
oversample = st.sidebar.selectbox("オーバーサンプリング（分類）", ["なし", "SMOTE", "Resample"]) if task_type == "分類" else "なし"

# CV・λ
n_splits = st.sidebar.slider("GroupKFold 分割", 2, 10, 5)
lambda_cons = st.sidebar.slider("Consistency λ（0で無効）", 0.0, 1.0, 0.0, step=0.05)
inner_splits = st.sidebar.slider("内側CV分割（GridSearch）", 2, 10, 3)

# ============= ここがポイント：2つのグループ列を選ばせる ============
if features_all:
    cv_group_col = st.sidebar.selectbox("CV用グループ列（リーク防止）", features_all,
                                        index=features_all.index("folder_name") if "folder_name" in features_all else 0)
    cons_group_col = st.sidebar.selectbox("一貫性用グループ列（27×7など）", features_all,
                                          index=features_all.index("group_label") if "group_label" in features_all else 0)

# =====================
# 実行
# =====================
if st.button("クロスバリデーション実行"):
    # データの準備
    if mode == "ランダム（単一ファイルからCV）":
        if df is None:
            st.error("データをアップロードしてください。"); st.stop()
        X, Y, features = tr.dataset(df, target, removal)
        groups_cv   = df[cv_group_col]
        groups_cons = df[cons_group_col]
    else:
        if train_df is None:
            st.error("学習/評価ファイルをアップロードしてください。"); st.stop()
        X, Y, features = tr.dataset(train_df, target, removal)
        groups_cv   = train_df[cv_group_col]
        groups_cons = train_df[cons_group_col]

    # 分類なら分位ビン化
    if task_type == "分類":
        y_cont = pd.to_numeric(Y, errors="coerce")
        valid = y_cont.notna()
        X, y_cont = X.loc[valid], y_cont.loc[valid]
        groups_cv = groups_cv.loc[valid]
        groups_cons = groups_cons.loc[valid]
        Y = pd.Series(pd.qcut(y_cont, q=n_bins, labels=False, duplicates="drop").astype(int),
                      index=y_cont.index, name=target)

    # ベースモデル
    if ml_type == "DecisionTree":
        model = DTC(class_weight="balanced") if task_type == "分類" else DTR()
    elif ml_type == "RandomForest":
        model = RFC(class_weight="balanced", n_jobs=-1) if task_type == "分類" else RFR(n_jobs=-1)
    elif ml_type == "SVM":
        model = SVC(probability=True) if task_type == "分類" else SVR()
    elif ml_type == "NN":
        model = MLPClassifier(max_iter=1000) if task_type == "分類" else MLPRegressor(max_iter=1000)
    else:
        model = XGBClassifier(eval_metric="mlogloss", n_jobs=-1) if task_type == "分類" else XGBRegressor(n_jobs=-1)

    # CV
    gkf = GroupKFold(n_splits=n_splits)

    best_params_list = []
    best_fold = None
    best_score = -np.inf
    best_pack = None

    if task_type == "分類":
        accs, precs, recs, f1s = [], [], [], []
        cm_sum = None
        labels_seen = set()
    else:
        y_tr_all, yhat_tr_all, y_te_all, yhat_te_all = [], [], [], []
        rmse_trs, mae_trs, r2_trs = [], [], []
        rmse_tes, mae_tes, r2_tes = [], [], []

    progress = st.progress(0.0)

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, Y, groups=groups_cv), 1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        Y_tr, Y_te = Y.iloc[tr_idx], Y.iloc[te_idx]
        groups_tr_cv   = groups_cv.iloc[tr_idx]
        groups_tr_cons = groups_cons.iloc[tr_idx]

        # 分類オーバーサンプリング
        lambda_fold = lambda_cons
        if task_type == "分類" and oversample != "なし":
            if oversample == "SMOTE":
                sm = SMOTE()
                X_tr, Y_tr = sm.fit_resample(X_tr, Y_tr)
                # groups_tr_cons が壊れるので λ=0 にフォールバック
                st.warning(f"Fold {fold}: SMOTE と Consistency は併用不可のため λ=0 にします。")
                lambda_fold = 0.0
            else:  # Resample
                tmp = pd.concat([X_tr, Y_tr.rename("target"), groups_tr_cons.rename("grp")], axis=1)
                maxc = tmp["target"].value_counts().max()
                parts = [resample(g, replace=True, n_samples=maxc, random_state=42)
                         for _, g in tmp.groupby("target")]
                up = pd.concat(parts)
                Y_tr = up["target"]
                groups_tr_cons = up["grp"]
                X_tr = up.drop(columns=["target", "grp"])

        # 一貫性込みグリッドサーチ
        mdl, best_params = fit_with_consistent_grid_search(
            base_model=model,
            X_tr=X_tr, y_tr=Y_tr,
            groups_tr_cv=groups_tr_cv,
            groups_tr_cons=groups_tr_cons,
            param_grid=params,
            task_type=task_type,
            lambda_cons=lambda_fold,
            inner_splits=inner_splits
        )
        best_params_list.append({"fold": fold, **best_params})

        # 評価
        if task_type == "分類":
            pred = mdl.predict(X_te)
            acc = accuracy_score(Y_te, pred)
            prec = precision_score(Y_te, pred, average="macro", zero_division=0)
            rec = recall_score(Y_te, pred, average="macro", zero_division=0)
            f1 = f1_score(Y_te, pred, average="macro", zero_division=0)
            accs.append(acc); precs.append(prec); recs.append(rec); f1s.append(f1)

            labels_seen.update(pd.Series(Y_te).unique().tolist())
            labs = sorted(list(labels_seen))
            cm = confusion_matrix(Y_te, pred, labels=labs)
            cm_sum = cm if cm_sum is None else cm_sum + cm

            if f1 > best_score:
                best_score = f1; best_fold = fold
                best_pack = ("cls", mdl, X_tr, X_te, Y_tr, Y_te, list(X.columns),
                             {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "cm": cm})
        else:
            pred_tr = mdl.predict(X_tr); pred_te = mdl.predict(X_te)
            rmse_tr = float(np.sqrt(mean_squared_error(Y_tr, pred_tr)))
            mae_tr  = float(mean_absolute_error(Y_tr, pred_tr))
            r2_tr   = float(r2_score(Y_tr, pred_tr))
            rmse_te = float(np.sqrt(mean_squared_error(Y_te, pred_te)))
            mae_te  = float(mean_absolute_error(Y_te, pred_te))
            r2_te   = float(r2_score(Y_te, pred_te))
            rmse_trs.append(rmse_tr); mae_trs.append(mae_tr); r2_trs.append(r2_tr)
            rmse_tes.append(rmse_te); mae_tes.append(mae_te); r2_tes.append(r2_te)
            y_tr_all.extend(Y_tr.tolist()); yhat_tr_all.extend(pred_tr.tolist())
            y_te_all.extend(Y_te.tolist()); yhat_te_all.extend(pred_te.tolist())
            if r2_te > best_score:
                best_score = r2_te; best_fold = fold
                best_pack = ("reg", mdl, X_tr, X_te, Y_tr, Y_te, list(X.columns),
                             {"rmse_tr": rmse_tr, "mae_tr": mae_tr, "r2_tr": r2_tr,
                              "rmse_te": rmse_te, "mae_te": mae_te, "r2_te": r2_te,
                              "yhat_tr": pred_tr, "yhat_te": pred_te})

        progress.progress(fold / n_splits)

    st.success("CV完了")

    # --- 集計表示 ---
    if task_type == "分類":
        df_scores = pd.DataFrame({
            "Accuracy": accs, "Precision(macro)": precs,
            "Recall(macro)": recs, "F1(macro)": f1s
        }, index=[f"fold{i}" for i in range(1, len(accs)+1)])
        st.subheader("CV結果（分類）"); st.dataframe(df_scores); st.write(df_scores.agg(["mean","std"]))
        if "cm_sum" in locals() and cm_sum is not None:
            labs = sorted(list(labels_seen))
            st.write("総和混同行列"); st.dataframe(pd.DataFrame(cm_sum, index=[f"T{i}" for i in labs], columns=[f"P{i}" for i in labs]))
    else:
        df_te = pd.DataFrame({"RMSE": rmse_tes, "MAE": mae_tes, "R2": r2_tes}, index=[f"fold{i}" for i in range(1,len(rmse_tes)+1)])
        df_tr = pd.DataFrame({"RMSE": rmse_trs, "MAE": mae_trs, "R2": r2_trs}, index=[f"fold{i}" for i in range(1,len(rmse_trs)+1)])
        st.subheader("CV結果（回帰）"); st.write("Test"); st.dataframe(df_te); st.write(df_te.agg(["mean","std"]))
        st.write("Train"); st.dataframe(df_tr); st.write(df_tr.agg(["mean","std"]))
        y_tr_all, yhat_tr_all, y_te_all, yhat_te_all = map(pd.Series, (y_tr_all, yhat_tr_all, y_te_all, yhat_te_all))
        r2_tr_all = r2_score(y_tr_all, yhat_tr_all) if len(y_tr_all) else np.nan
        r2_te_all = r2_score(y_te_all, yhat_te_all) if len(y_te_all) else np.nan
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(y_tr_all, yhat_tr_all, alpha=0.6, label="Train", marker="o")
        ax.scatter(y_te_all, yhat_te_all, alpha=0.6, label="Test", marker="^")
        lo = min(y_tr_all.min(), y_te_all.min(), yhat_tr_all.min(), yhat_te_all.min())
        hi = max(y_tr_all.max(), y_te_all.max(), yhat_tr_all.max(), yhat_te_all.max())
        ax.plot([lo,hi],[lo,hi],"--",color="gray"); ax.legend()
        ax.set_title(f"Actual vs Predicted ({ml_type})\nR2(train)={r2_tr_all:.3f}, R2(test)={r2_te_all:.3f}")
        st.pyplot(fig)

    st.subheader("各Foldの最適ハイパーパラメータ")
    st.dataframe(pd.DataFrame(best_params_list))

    # --- ベストfold詳細 + XAI ---
    if best_pack is not None:
        kind, mdl_b, X_tr_b, X_te_b, Y_tr_b, Y_te_b, feats_b, meta = best_pack
        st.subheader(f"ベストFold詳細（fold={best_fold}）")
        if kind == "cls":
            pred = mdl_b.predict(X_te_b)
            labs = sorted(pd.Series(Y_te_b).unique().tolist())
            acc = accuracy_score(Y_te_b, pred); prec = precision_score(Y_te_b, pred, average="macro", zero_division=0)
            rec = recall_score(Y_te_b, pred, average="macro", zero_division=0); f1 = f1_score(Y_te_b, pred, average="macro", zero_division=0)
            st.write(f"Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}")
            cm = confusion_matrix(Y_te_b, pred, labels=labs)
            st.dataframe(pd.DataFrame(cm, index=[f"T{i}" for i in labs], columns=[f"P{i}" for i in labs]))
        else:
            st.write(f"Train: R²={meta['r2_tr']:.3f}, RMSE={meta['rmse_tr']:.3f}, MAE={meta['mae_tr']:.3f}")
            st.write (f"Test : R²={meta['r2_te']:.3f}, RMSE={meta['rmse_te']:.3f}, MAE={meta['mae_te']:.3f}")

        # 重要度
        st.subheader("特徴量重要度（対応モデルのみ）")
        if hasattr(mdl_b, "feature_importances_"):
            try:
                fig_imp = tr.importance(mdl_b, feats_b); st.pyplot(fig_imp)
            except Exception:
                imp = pd.Series(mdl_b.feature_importances_, index=feats_b).sort_values(ascending=False).head(30)
                fig2, ax2 = plt.subplots(figsize=(6, min(10, 0.3*len(imp))))
                imp.iloc[::-1].plot(kind="barh", ax=ax2); ax2.set_title("Top Feature Importances"); st.pyplot(fig2)
        else:
            st.info("このモデルは feature_importances_ を提供しません。")

        # XAI
        st.subheader("XAI（Best fold）")
        bg = X_tr_b if len(X_tr_b) <= 200 else X_tr_b.sample(200, random_state=42)
        try:
            xai.explain_shap(mdl_b, bg[feats_b], X_te_b[feats_b], task_type, ml_type)
        except Exception as e:
            st.warning(f"SHAP表示に失敗: {e}")
        try:
            xai.explain_lime(mdl_b, bg[feats_b], X_te_b[feats_b], task_type)
        except Exception as e:
            st.warning(f"LIME表示に失敗: {e}")

st.markdown("---")
st.caption("💡 CV用グループ（被験者IDなど）と一貫性用グループ（加工グループ等）を別々に指定できます。λ>0 で一貫性ペナルティを有効化。")
