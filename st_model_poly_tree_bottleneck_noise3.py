# -*- coding: utf-8 -*-
# ============================================================
# GroupKFold ツリーモデル + 多項式モデル + XAI
# + 3値分類（平均周りノイズ帯）ラベル設計
# + 過学習抑制（gamma / l1 / l2 / elastic）
# + Bayesianで当たり→近傍GridSearch
# ============================================================

import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import (
    LeaveOneGroupOut, KFold, StratifiedKFold,
    GroupKFold, GroupShuffleSplit, train_test_split,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, accuracy_score
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import Ridge, LassoCV, LogisticRegression
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier as DTC, DecisionTreeRegressor as DTR
from sklearn.ensemble import RandomForestClassifier as RFC, RandomForestRegressor as RFR
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import statsmodels.api as sm

# あなたの既存モジュール
import xai          # SHAP / LIME
import tree as tr   # dataset / grid_search / importance など

# ============================================================
# skopt があれば Bayesian search、なければ RandomizedSearch fallback
# ============================================================
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    HAS_SKOPT = True
except Exception:
    HAS_SKOPT = False

# ============================================================
# スコア/診断ユーティリティ
# ============================================================

def _metric_name(task_type: str) -> str:
    return "R2" if task_type == "回帰" else "F1(macro)"

def _score(task_type: str, y_true, y_pred) -> float:
    if task_type == "回帰":
        return float(r2_score(y_true, y_pred))
    else:
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

def cv_eval_quick(task_type, X, y, groups=None, scheme="within", n_splits=5, seed=42):
    """
    scheme:
      - within: 被験者をまたいでランダムCV（= 被験者が train/test に混ざる）
      - loso : 被験者LOSO（folder_name で LeaveOneGroupOut）
      - loio : 画像LOIO（file_name などで LeaveOneGroupOut）
    """
    if task_type == "回帰":
        mdl = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=seed)),
        ])
    else:
        mdl = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=5000, class_weight="balanced", random_state=seed
            )),
        ])

    y_true_all, y_pred_all = [], []

    if scheme == "within":
        if task_type == "分類":
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for tr_idx, te_idx in cv.split(X, y):
                mdl.fit(X.iloc[tr_idx], y.iloc[tr_idx])
                pred = mdl.predict(X.iloc[te_idx])
                y_true_all.extend(y.iloc[te_idx].tolist())
                y_pred_all.extend(pred.tolist())
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for tr_idx, te_idx in cv.split(X):
                mdl.fit(X.iloc[tr_idx], y.iloc[tr_idx])
                pred = mdl.predict(X.iloc[te_idx])
                y_true_all.extend(y.iloc[te_idx].tolist())
                y_pred_all.extend(pred.tolist())

    elif scheme in ("loso", "loio"):
        if groups is None:
            raise ValueError("LOSO/LOIO には groups が必要です。")
        logo = LeaveOneGroupOut()
        for tr_idx, te_idx in logo.split(X, y, groups=groups):
            mdl.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            pred = mdl.predict(X.iloc[te_idx])
            y_true_all.extend(y.iloc[te_idx].tolist())
            y_pred_all.extend(pred.tolist())
    else:
        raise ValueError("unknown scheme")

    return _score(task_type, y_true_all, y_pred_all)

def permutation_signal_test(task_type, X, y, groups=None, scheme="loso", n_perm=30, seed=42):
    rng = np.random.default_rng(seed)
    real = cv_eval_quick(task_type, X, y, groups=groups, scheme=scheme, seed=seed)

    perm_scores = []
    for _ in range(n_perm):
        y_perm = pd.Series(y.values.copy(), index=y.index)
        y_perm = y_perm.sample(frac=1, random_state=int(rng.integers(0, 1_000_000_000)))
        perm_scores.append(cv_eval_quick(task_type, X, y_perm, groups=groups, scheme=scheme, seed=seed))

    perm_mean = float(np.mean(perm_scores))
    perm_sd   = float(np.std(perm_scores))
    p = float((np.sum(np.array(perm_scores) >= real) + 1) / (len(perm_scores) + 1))
    return {"real": real, "perm_mean": perm_mean, "perm_sd": perm_sd, "p_like": p}

def estimate_y_reliability_split_half(df, y_col, unit_cols):
    if any(c not in df.columns for c in unit_cols):
        return None
    tmp = df[unit_cols + [y_col]].copy()
    tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")
    tmp = tmp.dropna(subset=[y_col])
    if len(tmp) == 0:
        return None

    tmp["_rn"] = tmp.groupby(unit_cols).cumcount()
    a = tmp[tmp["_rn"] % 2 == 0].groupby(unit_cols)[y_col].mean()
    b = tmp[tmp["_rn"] % 2 == 1].groupby(unit_cols)[y_col].mean()
    merged = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(merged) < 5:
        return None

    r = float(np.corrcoef(merged["a"], merged["b"])[0, 1])
    rel_full = (2 * r) / (1 + r + 1e-12)
    return {"split_half_r": r, "rel_full_est": float(rel_full), "n_units": int(len(merged))}

def estimate_var_component_reliability(df, y_col, subject_col="folder_name", unit_col="file_name"):
    if subject_col not in df.columns:
        return None
    if unit_col not in df.columns:
        unit_col = None

    cols = [subject_col] + ([unit_col] if unit_col else []) + [y_col]
    tmp = df[cols].copy()
    tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")
    tmp = tmp.dropna(subset=[y_col])
    if len(tmp) == 0:
        return None

    subj_mean = tmp.groupby(subject_col)[y_col].mean()
    var_between = float(subj_mean.var(ddof=1)) if subj_mean.nunique() > 1 else 0.0

    if unit_col:
        mu = tmp.groupby([subject_col, unit_col])[y_col].transform("mean")
    else:
        mu = tmp.groupby(subject_col)[y_col].transform("mean")

    resid = tmp[y_col] - mu
    var_within = float(resid.var(ddof=1)) if resid.nunique() > 1 else 0.0

    def rel_k(k):
        return var_between / (var_between + var_within / max(k, 1) + 1e-12)

    if unit_col:
        k_med = int(tmp.groupby([subject_col, unit_col]).size().median())
    else:
        k_med = int(tmp.groupby(subject_col).size().median())

    return {
        "var_between": var_between,
        "var_within": var_within,
        "rel_k1": float(rel_k(1)),
        "k_med": k_med,
        "rel_k_med": float(rel_k(max(k_med, 1))),
        "rel_k4": float(rel_k(4)),
        "rel_k8": float(rel_k(8)),
    }

def decide_bottleneck(diag):
    y_rel = diag.get("y_rel", None)
    rel_k1 = None if y_rel is None else y_rel.get("rel_k1", None)

    sig = diag.get("x_signal", None)
    delta = None if sig is None else (sig["real"] - sig["perm_mean"])

    within = diag.get("scores", {}).get("within", None)
    loso   = diag.get("scores", {}).get("loso", None)

    reasons = []
    if rel_k1 is not None and rel_k1 < 0.35:
        reasons.append("yの1試行信頼性が低い（ノイズ支配）→平均化・前処理・計測改善が効きやすい")

    if delta is not None and delta < 0.02:
        reasons.append("実スコアがシャッフル同程度→x側の信号が弱い/特徴量設計 or ラベル設計が不足")

    if within is not None and loso is not None and (within - loso) > 0.10:
        reasons.append("withinは良いのにLOSOが悪い→被験者差が支配的（正規化/被験者増が効く）")

    if delta is not None and delta < 0.02:
        primary = "xをよくする（特徴量・前処理・ラベル設計・モデル仮定）"
    elif rel_k1 is not None and rel_k1 < 0.35:
        primary = "yをよくする（平均化・外れ値・ベースライン・計測条件の安定化）"
    elif within is not None and loso is not None and (within - loso) > 0.10:
        primary = "被験者をよくする（被験者増/スクリーニング/個人差吸収）"
    else:
        primary = "混合（y/x/被験者）を小さく改善：学習曲線でデータ不足か確認"
    return primary, reasons

# ============================================================
# 前処理: log変換 / outlier clip
# ============================================================

def log_transform_X(X: pd.DataFrame, cols, eps: float = 1e-6) -> pd.DataFrame:
    X_new = X.copy()
    for c in cols:
        if c not in X_new.columns:
            continue
        vals = pd.to_numeric(X_new[c], errors="coerce")
        min_val = np.nanmin(vals.values)

        shift = 0.0
        if min_val <= -1.0 + eps:
            shift = -min_val + 1.0 + eps
        elif min_val < 0.0:
            shift = -min_val + eps

        X_new[c] = np.log1p(vals + shift)
    return X_new

def log_transform_train_test(X_tr: pd.DataFrame, X_te: pd.DataFrame, cols, eps: float = 1e-6):
    X_tr_new = X_tr.copy()
    X_te_new = X_te.copy()
    for c in cols:
        if c not in X_tr_new.columns:
            continue
        vals_tr = pd.to_numeric(X_tr_new[c], errors="coerce")
        min_val = np.nanmin(vals_tr.values)

        shift = 0.0
        if min_val <= -1.0 + eps:
            shift = -min_val + 1.0 + eps
        elif min_val < 0.0:
            shift = -min_val + eps

        X_tr_new[c] = np.log1p(vals_tr + shift)
        if c in X_te_new.columns:
            vals_te = pd.to_numeric(X_te_new[c], errors="coerce")
            X_te_new[c] = np.log1p(vals_te + shift)
    return X_tr_new, X_te_new

def apply_outlier_clip_df(X: pd.DataFrame, cols, method: str, params: dict) -> pd.DataFrame:
    X_new = X.copy()
    for c in cols:
        if c not in X_new.columns:
            continue
        vals = pd.to_numeric(X_new[c], errors="coerce")

        if method == "IQRでクリップ":
            k = params.get("iqr_k", 1.5)
            q1 = vals.quantile(0.25)
            q3 = vals.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - k * iqr
            upper = q3 + k * iqr
        elif method == "標準偏差でクリップ":
            k = params.get("sigma_k", 3.0)
            m = vals.mean()
            s = vals.std(ddof=0)
            lower = m - k * s
            upper = m + k * s
        elif method == "分位点でクリップ":
            q_low  = params.get("q_low", 0.01)
            q_high = params.get("q_high", 0.99)
            lower = vals.quantile(q_low)
            upper = vals.quantile(q_high)
        else:
            continue

        X_new[c] = vals.clip(lower, upper)
    return X_new

def apply_outlier_clip_train_test(X_tr: pd.DataFrame, X_te: pd.DataFrame, cols, method: str, params: dict):
    X_tr_new = X_tr.copy()
    X_te_new = X_te.copy()

    for c in cols:
        if c not in X_tr_new.columns:
            continue
        vals_tr = pd.to_numeric(X_tr_new[c], errors="coerce")

        if method == "IQRでクリップ":
            k = params.get("iqr_k", 1.5)
            q1 = vals_tr.quantile(0.25)
            q3 = vals_tr.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - k * iqr
            upper = q3 + k * iqr
        elif method == "標準偏差でクリップ":
            k = params.get("sigma_k", 3.0)
            m = vals_tr.mean()
            s = vals_tr.std(ddof=0)
            lower = m - k * s
            upper = m + k * s
        elif method == "分位点でクリップ":
            q_low  = params.get("q_low", 0.01)
            q_high = params.get("q_high", 0.99)
            lower = vals_tr.quantile(q_low)
            upper = vals_tr.quantile(q_high)
        else:
            continue

        X_tr_new[c] = vals_tr.clip(lower, upper)
        if c in X_te_new.columns:
            vals_te = pd.to_numeric(X_te_new[c], errors="coerce")
            X_te_new[c] = vals_te.clip(lower, upper)

    return X_tr_new, X_te_new

# ============================================================
# 交互作用項（任意ペア指定）追加
# ============================================================

def add_interaction_features(X: pd.DataFrame, pair_list):
    X_new = X.copy()
    for c1, c2 in pair_list:
        if (c1 not in X_new.columns) or (c2 not in X_new.columns):
            continue
        name = f"{c1}*{c2}"
        if name in X_new.columns:
            continue
        try:
            X_new[name] = pd.to_numeric(X_new[c1], errors="coerce") * pd.to_numeric(X_new[c2], errors="coerce")
        except Exception:
            continue
    return X_new, list(X_new.columns)

# ============================================================
# ★ 3値分類（平均周りノイズ帯）ラベル設計
# ============================================================

def make_ternary_noiseband_label(y, method="sigma", k=0.5):
    """
    中心帯（ノイズ帯）を class=1 とする 3値分類
      y < lower -> 0
      lower <= y <= upper -> 1
      y > upper -> 2

    method:
      - "sigma": mean ± k*std
      - "IQR"  : mean ± k*IQR（IQR=Q3-Q1）
    """
    y_num = pd.to_numeric(y, errors="coerce")
    valid = y_num.notna()
    if valid.sum() == 0:
        raise ValueError("目的変数が数値化できません。")

    mu = float(y_num[valid].mean())
    if method == "sigma":
        sd = float(y_num[valid].std(ddof=0))
        width = k * sd
    else:
        q1 = float(y_num[valid].quantile(0.25))
        q3 = float(y_num[valid].quantile(0.75))
        iqr = q3 - q1
        width = k * iqr

    lower = mu - width
    upper = mu + width

    lab = pd.Series(index=y.index, dtype="Int64")
    lab.loc[valid & (y_num < lower)] = 0
    lab.loc[valid & (y_num >= lower) & (y_num <= upper)] = 1
    lab.loc[valid & (y_num > upper)] = 2
    return lab.astype(int), {"mu": mu, "lower": lower, "upper": upper, "width": width}

# ============================================================
# Bayesian→Grid のチューニング
# ============================================================

def _narrow_grid_around_best(best_params: dict, base_grid: dict, shrink=0.5):
    """
    best_params の近傍に grid を自動生成（雑でOK）
    - float: [best*(1-shrink), best, best*(1+shrink)]
    - int  : [best-Δ, best, best+Δ] ただし base_grid からΔ推定
    - categorical: best を中心に base_grid そのまま
    """
    new_grid = {}
    for k, vals in base_grid.items():
        if k not in best_params:
            new_grid[k] = vals
            continue

        b = best_params[k]
        # categorical/list
        if isinstance(vals, list) and len(vals) > 0 and isinstance(vals[0], (str, bool)) or isinstance(b, str):
            # 文字列系は base をそのまま（安全）
            new_grid[k] = vals
            continue

        # int
        if isinstance(b, (int, np.integer)):
            # baseの幅からΔ推定
            if isinstance(vals, list) and len(vals) >= 2 and all(isinstance(v, (int, np.integer)) for v in vals):
                step = max(1, int(round((max(vals) - min(vals)) / 4)))
            else:
                step = max(1, int(round(abs(b) * 0.2)))
            cand = sorted(list(set([int(b - step), int(b), int(b + step)])))
            # base_grid の範囲に軽く収める
            if isinstance(vals, list) and len(vals) > 0 and all(isinstance(v, (int, np.integer)) for v in vals):
                lo, hi = min(vals), max(vals)
                cand = [v for v in cand if lo <= v <= hi] or [int(b)]
            new_grid[k] = cand
            continue

        # float
        if isinstance(b, (float, np.floating)):
            lo = b * (1 - shrink)
            hi = b * (1 + shrink)
            cand = [lo, b, hi]
            # base_grid 由来の範囲があればクリップ
            if isinstance(vals, list) and len(vals) > 0 and all(isinstance(v, (float, np.floating, int, np.integer)) for v in vals):
                vmin, vmax = float(min(vals)), float(max(vals))
                cand = [float(np.clip(v, vmin, vmax)) for v in cand]
            # 重複除去
            cand = sorted(list({round(v, 6) for v in cand}))
            new_grid[k] = cand
            continue

        new_grid[k] = vals
    return new_grid

def bayes_then_grid_search(
    estimator,
    base_param_grid: dict,
    scoring: str,
    cv,
    X_tr, y_tr,
    groups=None,
    use_bayes=True,
    n_iter=20,
    random_state=42,
    bayes_shrink=0.5,
):
    """
    1) Bayesian（or Randomized）で best_params を取る
    2) best_params 近傍に絞った grid で GridSearch
    """
    # ---- 1) coarse search ----
    best_params_stage1 = None

    if use_bayes:
        if HAS_SKOPT:
            # base_param_grid から skopt space を自動生成（雑）
            space = {}
            for k, vals in base_param_grid.items():
                if not isinstance(vals, list) or len(vals) == 0:
                    continue
                if all(isinstance(v, (int, np.integer)) for v in vals):
                    space[k] = Integer(int(min(vals)), int(max(vals)))
                elif all(isinstance(v, (float, np.floating, int, np.integer)) for v in vals):
                    space[k] = Real(float(min(vals)), float(max(vals)), prior="log-uniform" if min(vals) > 0 else "uniform")
                else:
                    space[k] = Categorical(vals)

            bayes = BayesSearchCV(
                estimator=estimator,
                search_spaces=space,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv,
                # n_jobs=-1,
                n_jobs=1,
                random_state=random_state,
                refit=True,
                verbose=10,
                error_score="raise",
            )
            bayes.fit(X_tr, y_tr, **({"groups": groups} if groups is not None else {}))
            best_params_stage1 = dict(bayes.best_params_)
        else:
            # skopt無いなら Randomized
            rnd = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=base_param_grid,
                n_iter=min(n_iter, 30),
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                random_state=random_state,
                refit=True,
            )
            rnd.fit(X_tr, y_tr, **({"groups": groups} if groups is not None else {}))
            best_params_stage1 = dict(rnd.best_params_)
    else:
        # いきなりGrid
        best_params_stage1 = None

    # ---- 2) narrow grid around best ----
    if best_params_stage1 is not None:
        narrow_grid = _narrow_grid_around_best(best_params_stage1, base_param_grid, shrink=bayes_shrink)
    else:
        narrow_grid = base_param_grid

    gscv = GridSearchCV(
        estimator=estimator,
        param_grid=narrow_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    gscv.fit(X_tr, y_tr, **({"groups": groups} if groups is not None else {}))

    return gscv.best_estimator_, dict(gscv.best_params_), {"stage1_best": best_params_stage1, "grid_used": narrow_grid}

# ============================================================
# 多項式モデル用ユーティリティ（既存のまま + 微修正）
# ============================================================

def make_train_test(df_train, df_test, mode_flag, target, feature_cols, group_col=None, test_size=0.2, random_state=42):
    if mode_flag == "single":
        data = df_train.copy()
        y = data[target]
        X = data[feature_cols]

        if group_col and group_col in data.columns:
            groups = data[group_col]
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            tr_idx, te_idx = next(gss.split(X, y, groups))
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)
    else:
        tr_df = df_train.copy()
        te_df = df_test.copy()
        y_tr = tr_df[target]
        y_te = te_df[target]
        X_tr = tr_df[feature_cols]
        X_te = te_df[feature_cols]
    return X_tr, X_te, y_tr, y_te

class PolynomialSelector(BaseEstimator, TransformerMixin):
    """
    mode:
      - "1_only"
      - "1_plus_interactions"
      - "2_only"
      - "2_plus_interactions"
    """
    def __init__(self, mode="1_only"):
        self.mode = mode

    def fit(self, X, y=None):
        self.poly_ = PolynomialFeatures(degree=2, include_bias=False)
        self.poly_.fit(X)
        powers = self.poly_.powers_
        degs = powers.sum(axis=1)
        nnz = (powers != 0).sum(axis=1)

        if self.mode == "1_only":
            mask = (degs == 1)
        elif self.mode == "1_plus_interactions":
            mask = (degs == 1) | ((degs >= 2) & (nnz >= 2))
        elif self.mode == "2_only":
            mask = (degs == 1) | ((degs == 2) & (nnz == 1))
        elif self.mode == "2_plus_interactions":
            mask = (degs == 1) | (degs == 2)
        else:
            mask = (degs == 1)

        self.keep_idx_ = np.where(mask)[0]
        return self

    def transform(self, X):
        Z = self.poly_.transform(X)
        return Z[:, self.keep_idx_]

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.poly_.n_features_in_)]
        names = self.poly_.get_feature_names_out(input_features)
        names = np.asarray(names)[self.keep_idx_]
        return names

def coef_table_from_lasso_poly(pipe, feature_cols, task_type):
    poly = pipe.named_steps["poly"]
    names = poly.get_feature_names_out(feature_cols)

    if task_type == "回帰":
        reg = pipe.named_steps["model"]
        coefs = reg.coef_
        intercept = reg.intercept_
    else:
        clf = pipe.named_steps["model"]
        coefs = clf.coef_[0]
        intercept = clf.intercept_[0]

    df = pd.DataFrame({"term": names, "coef": coefs}).sort_values("coef", key=np.abs, ascending=False)
    return df, float(intercept)

def stepwise_aic_ols(X, y, max_steps=50):
    remaining = list(X.columns)
    selected = []
    current_score = np.inf

    for _ in range(max_steps):
        scores = []
        for cand in remaining:
            cols = selected + [cand]
            X_c = sm.add_constant(X[cols])
            model = sm.OLS(y, X_c).fit()
            scores.append((model.aic, cand))
        scores.sort(key=lambda x: x[0])
        best_new_score, best_cand = scores[0]

        if best_new_score < current_score - 1e-4:
            remaining.remove(best_cand)
            selected.append(best_cand)
            current_score = best_new_score
        else:
            break

    best_model = sm.OLS(y, sm.add_constant(X[selected])).fit()
    return best_model, selected

def stepwise_aic_logit(X, y, max_steps=50):
    remaining = list(X.columns)
    selected = []
    current_score = np.inf

    for _ in range(max_steps):
        scores = []
        for cand in remaining:
            cols = selected + [cand]
            X_c = sm.add_constant(X[cols])
            try:
                model = sm.Logit(y, X_c).fit(disp=0)
            except Exception:
                continue
            scores.append((model.aic, cand))
        if not scores:
            break

        scores.sort(key=lambda x: x[0])
        best_new_score, best_cand = scores[0]

        if best_new_score < current_score - 1e-4:
            remaining.remove(best_cand)
            selected.append(best_cand)
            current_score = best_new_score
        else:
            break

    best_model = sm.Logit(y, sm.add_constant(X[selected])).fit(disp=0)
    return best_model, selected

# ============================================================
# Streamlit 初期設定
# ============================================================

st.title("機械学習（GroupKFold ツリー＋多項式｜3値分類ノイズ帯｜Bayes→Grid｜XAI対応）")

if "cv_ready" not in st.session_state:
    st.session_state.cv_ready = False
    st.session_state.cv_payload = None
    st.session_state.shap_cache_key = None
    st.session_state.shap_values = None
    st.session_state.shap_task = None
    st.session_state.shap_model_id = None
    st.session_state.cv_result = None

if "inter_pairs" not in st.session_state:
    st.session_state.inter_pairs = []
if "add_interactions" not in st.session_state:
    st.session_state.add_interactions = False

if "log_all" not in st.session_state:
    st.session_state.log_all = False
if "log_cols" not in st.session_state:
    st.session_state.log_cols = []

if "outlier_method" not in st.session_state:
    st.session_state.outlier_method = "しない"
if "outlier_cols" not in st.session_state:
    st.session_state.outlier_cols = []
if "iqr_k" not in st.session_state:
    st.session_state.iqr_k = 1.5
if "sigma_k" not in st.session_state:
    st.session_state.sigma_k = 3.0
if "q_low" not in st.session_state:
    st.session_state.q_low = 0.01
if "q_high" not in st.session_state:
    st.session_state.q_high = 0.99

# ============================================================
# 入力UI
# ============================================================

mode = st.radio("データの指定方法", ('ランダム（単一ファイルから分割/CV）', '自分で決める（学習/評価を別ファイル）'))

df = train_df = test_df = None
features_all = None
target = None
group_col = "folder_name"
removal = []
feature_cols = None

if mode == 'ランダム（単一ファイルから分割/CV）':
    up = st.file_uploader("データファイル（CSV / XLSX）", type=["csv", "xls", "xlsx"])
    if up:
        df = pd.read_excel(up) if up.name.endswith((".xls", ".xlsx")) else pd.read_csv(up)
        features_all = df.columns
        target = st.selectbox("目的変数を選択", features_all)
        group_col = "folder_name"

        feature_candidates = [c for c in features_all if c != target]

        sel_mode = st.radio("説明変数の指定方法", ["除外する列を選ぶ", "使う列を選ぶ"], horizontal=True)

        if sel_mode == "除外する列を選ぶ":
            default_removal = [group_col] if group_col in feature_candidates else []
            removal = st.multiselect("説明変数から除外する列", feature_candidates, default=default_removal)
        else:
            default_use = [c for c in feature_candidates if c != group_col]
            use_cols = st.multiselect("説明変数として使う列", feature_candidates, default=default_use)
            removal = [c for c in feature_candidates if c not in use_cols]

        if group_col in features_all and group_col not in removal:
            removal.append(group_col)

        feature_cols = [c for c in features_all if c not in removal and c != target]
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

        feature_candidates = [c for c in features_all if c != target]
        sel_mode = st.radio("説明変数の指定方法", ["除外する列を選ぶ", "使う列を選ぶ"], horizontal=True)

        if sel_mode == "除外する列を選ぶ":
            default_removal = [group_col] if group_col in feature_candidates else []
            removal = st.multiselect("説明変数から除外する列", feature_candidates, default=default_removal)
        else:
            default_use = [c for c in feature_candidates if c != group_col]
            use_cols = st.multiselect("説明変数として使う列", feature_candidates, default=default_use)
            removal = [c for c in feature_candidates if c not in use_cols]

        if group_col in features_all and group_col not in removal:
            removal.append(group_col)

        feature_cols = [c for c in features_all if c not in removal and c != target]
        name = st.text_input("実験名（任意）")

mode_flag = "single" if mode.startswith("ランダム") else "split"

# ============================================================
# サイドバー：モデリング設定
# ============================================================

st.sidebar.header("モデリング設定")
task_type = st.sidebar.radio("タスク", ["分類", "回帰"])

# ---- ラベル設計（分類のみ）----
label_design = None
tern_method = "sigma"
tern_k = 0.5

if task_type == "分類":
    st.sidebar.subheader("ラベル設計（分類）")
    label_design = st.sidebar.selectbox(
        "ラベルの作り方",
        ["そのまま使う", "分位ビン（2/3/4）", "3値（平均周りノイズ帯）"]
    )

    if label_design == "分位ビン（2/3/4）":
        bin_choice = st.sidebar.selectbox("分位", ["二分位", "三分位", "四分位"])
        bin_map = {"二分位": 2, "三分位": 3, "四分位": 4}
    else:
        bin_choice = "そのまま"
        bin_map = {}

    if label_design == "3値（平均周りノイズ帯）":
        tern_method = st.sidebar.selectbox("ノイズ幅の基準", ["sigma", "IQR"])
        tern_k = st.sidebar.slider("ノイズ幅係数 k", 0.05, 2.0, 0.50, 0.05)
else:
    label_design = None
    bin_choice = "そのまま"
    bin_map = {}

# ---- 前処理（log）----
st.sidebar.subheader("前処理（log 変換）")
if features_all is None or target is None:
    st.sidebar.info("ファイル読み込み後に選べます。")
else:
    base_df = df if df is not None else train_df
    if base_df is None:
        st.sidebar.info("ファイル読み込み後に選べます。")
    else:
        col_candidates = [c for c in base_df.columns if c != target and c != group_col]
        numeric_candidates = [c for c in col_candidates if pd.api.types.is_numeric_dtype(base_df[c])]
        if len(numeric_candidates) == 0:
            st.sidebar.info("log 変換できる数値列がありません。")
            st.session_state.log_cols = []
        else:
            log_all = st.sidebar.checkbox("数値の説明変数すべてに log をかける", value=st.session_state.log_all)
            st.session_state.log_all = log_all
            if log_all:
                st.session_state.log_cols = numeric_candidates
            else:
                default_logs = [c for c in st.session_state.log_cols if c in numeric_candidates]
                st.session_state.log_cols = st.sidebar.multiselect(
                    "log(1+x) をかける列",
                    options=numeric_candidates,
                    default=default_logs
                )

# ---- 前処理（外れ値）----
st.sidebar.subheader("前処理（ハズレ値処理）")
if features_all is None or target is None:
    st.sidebar.info("ファイル読み込み後に設定できます。")
else:
    base_df = df if df is not None else train_df
    if base_df is None:
        st.sidebar.info("ファイル読み込み後に設定できます。")
    else:
        col_candidates = [c for c in base_df.columns if c != group_col]
        numeric_candidates_out = [c for c in col_candidates if pd.api.types.is_numeric_dtype(base_df[c])]
        if len(numeric_candidates_out) == 0:
            st.sidebar.info("ハズレ値処理できる数値列がありません。")
            st.session_state.outlier_cols = []
            st.session_state.outlier_method = "しない"
        else:
            method_options = ["しない", "IQRでクリップ", "標準偏差でクリップ", "分位点でクリップ"]
            current_method = st.session_state.outlier_method if st.session_state.outlier_method in method_options else "しない"
            method = st.sidebar.selectbox("方法", method_options, index=method_options.index(current_method))
            st.session_state.outlier_method = method

            if method == "しない":
                st.session_state.outlier_cols = []
            else:
                default_out_cols = [c for c in st.session_state.outlier_cols if c in numeric_candidates_out]
                out_cols = st.sidebar.multiselect("処理する列", numeric_candidates_out, default=default_out_cols)
                st.session_state.outlier_cols = out_cols

                if out_cols:
                    if method == "IQRでクリップ":
                        st.session_state.iqr_k = st.sidebar.slider("IQR倍率 k", 0.5, 5.0, float(st.session_state.iqr_k))
                    elif method == "標準偏差でクリップ":
                        st.session_state.sigma_k = st.sidebar.slider("標準偏差倍率 k", 0.5, 5.0, float(st.session_state.sigma_k))
                    elif method == "分位点でクリップ":
                        st.session_state.q_low = st.sidebar.slider("下側分位点", 0.0, 0.4, float(st.session_state.q_low))
                        st.session_state.q_high = st.sidebar.slider("上側分位点", 0.6, 1.0, float(st.session_state.q_high))

# ---- モデル ----
ml_type = st.sidebar.selectbox("モデル", ["DecisionTree", "RandomForest", "SVM", "NN", "XGBoost", "LightGBM", "Polynomial"])

# ---- 過学習抑制（gamma / l1 / l2 / elastic） + Bayesian→Grid ----
st.sidebar.subheader("過学習抑制・探索（Bayes→Grid）")

use_bayes = st.sidebar.checkbox("Bayesian（or Randomized）で当たり→近傍GridSearch", value=True)
bayes_n_iter = st.sidebar.slider("Bayes/Randomized 試行回数", 5, 60, 20)
bayes_shrink = st.sidebar.slider("近傍Gridの絞り幅（±割合）", 0.1, 1.0, 0.5, 0.1)

# 正則化の選択（モデルにより実際の反映先が違う）
reg_choice = st.sidebar.selectbox("正則化タイプ（モデル対応分のみ有効）", ["なし", "L2", "L1", "ElasticNet"])
elastic_l1_ratio = 0.5
if reg_choice == "ElasticNet":
    elastic_l1_ratio = st.sidebar.slider("ElasticNet l1_ratio", 0.05, 0.95, 0.50, 0.05)

# ============================================================
# 交互作用ペア（ツリー側で任意追加）
# ============================================================
if ml_type != "Polynomial":
    add_interactions_flag = st.sidebar.checkbox("説明変数の交互作用項を明示的に追加する", value=st.session_state.get("add_interactions", False))
    st.session_state.add_interactions = add_interactions_flag

    if add_interactions_flag:
        if feature_cols is not None and len(feature_cols) > 0:
            st.sidebar.markdown("**交互作用にしたいペアを登録**")
            c1, c2 = st.sidebar.columns(2)
            with c1:
                var1 = st.selectbox("変数1", feature_cols, key="inter_var1")
            with c2:
                var2 = st.selectbox("変数2", feature_cols, key="inter_var2")

            if st.sidebar.button("このペアを追加", key="add_inter_pair_btn"):
                if var1 == var2:
                    st.sidebar.warning("同じ変数同士は交互作用にできません。")
                else:
                    pair = tuple(sorted([var1, var2]))
                    if pair not in st.session_state.inter_pairs:
                        st.session_state.inter_pairs.append(pair)

            if st.session_state.inter_pairs:
                st.sidebar.markdown("**交互作用ペア一覧**")
                for i, (p1, p2) in enumerate(st.session_state.inter_pairs, start=1):
                    st.sidebar.write(f"{i}. {p1} × {p2}")
                if st.sidebar.button("交互作用ペアをすべてクリア", key="clear_inter_pairs_btn"):
                    st.session_state.inter_pairs = []
        else:
            st.sidebar.info("ファイル読み込み後に選べます。")

# ============================================================
# CV設定
# ============================================================
n_splits = st.sidebar.slider("CV分割数（GroupKFold）", 2, 10, 5)
random_state = st.sidebar.number_input("random_state", 0, 9999, 42)

# ============================================================
# モデル別：ベース param_grid（Bayes→Grid に使う）
# ここで「gamma / L1 / L2 / ElasticNet」を反映
# ============================================================

def build_param_grid_and_model(ml_type, task_type, random_state, reg_choice, elastic_l1_ratio):
    """
    返り値: (model_instance, base_param_grid, scoring_str)
    """
    # scoring
    scoring = "r2" if task_type == "回帰" else "f1_macro"

    if ml_type == "DecisionTree":
        model = DTC(class_weight="balanced", random_state=random_state) if task_type == "分類" else DTR(random_state=random_state)
        param_grid = {
            "max_depth": list(range(2, 21, 2)),
            "min_samples_split": list(range(2, 13, 2)),
            "min_samples_leaf": list(range(1, 8, 1)),
        }
        return model, param_grid, scoring

    if ml_type == "RandomForest":
        model = RFC(class_weight="balanced", random_state=random_state) if task_type == "分類" else RFR(random_state=random_state)
        param_grid = {
            "n_estimators": list(range(50, 401, 50)),
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_leaf": [1, 2, 3, 5],
            "max_features": ["sqrt", "log2", None],
        }
        return model, param_grid, scoring

    if ml_type == "SVM":
        if task_type == "分類":
            model = SVC(probability=True, class_weight="balanced", random_state=random_state)
            # gamma で過学習制御（高いと局所的=保守的/過学習寄り）
            param_grid = {
                "C": [0.1, 0.3, 1, 3, 10, 30, 100],
                "gamma": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
                "kernel": ["rbf"],
            }
        else:
            model = SVR()
            param_grid = {
                "C": [0.1, 0.3, 1, 3, 10, 30, 100],
                "gamma": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
                "kernel": ["rbf"],
                "epsilon": [0.01, 0.05, 0.1, 0.2],
            }
        return model, param_grid, scoring

    if ml_type == "NN":
        if task_type == "分類":
            model = MLPClassifier(max_iter=1500, random_state=random_state)
        else:
            model = MLPRegressor(max_iter=1500, random_state=random_state)

        # alpha = L2（過学習抑制）
        param_grid = {
            "hidden_layer_sizes": [(64,), (128,), (256,), (128, 64), (256, 128)],
            "alpha": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
            "learning_rate_init": [1e-4, 3e-4, 1e-3, 3e-3],
        }
        return model, param_grid, scoring

    if ml_type == "XGBoost":
        if task_type == "分類":
            model = XGBClassifier(
                eval_metric="mlogloss",
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            model = XGBRegressor(random_state=random_state, n_jobs=-1)

        # gamma / reg_alpha(L1) / reg_lambda(L2) で過学習抑制
        param_grid = {
            "n_estimators": list(range(100, 601, 100)),
            "max_depth": [2, 3, 4, 6, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "gamma": [0.0, 0.1, 0.3, 1.0, 3.0],         # ★
            "reg_lambda": [0.0, 0.1, 1.0, 3.0, 10.0],   # ★ L2
            "reg_alpha":  [0.0, 0.01, 0.1, 0.3, 1.0],   # ★ L1
            "min_child_weight": [1, 3, 5, 10],
        }

        # reg_choice で “推奨重み” を少し寄せる（grid自体は残す）
        return model, param_grid, scoring

    if ml_type == "LightGBM":
        if task_type == "分類":
            model = LGBMClassifier(
                random_state=random_state,
                class_weight="balanced",
                n_jobs=-1
            )
        else:
            model = LGBMRegressor(random_state=random_state, n_jobs=-1)

        # lambda_l1 / lambda_l2 で過学習抑制（LGBM命名）
        param_grid = {
            "n_estimators": list(range(100, 701, 100)),
            "num_leaves": [15, 31, 63, 127, 255],
            "max_depth": [-1, 3, 5, 8, 12],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "lambda_l2": [0.0, 0.1, 1.0, 3.0, 10.0],    # ★ L2
            "lambda_l1": [0.0, 0.01, 0.1, 0.3, 1.0],    # ★ L1
            "min_child_samples": [5, 10, 20, 40],
        }
        return model, param_grid, scoring

    # Polynomial は別ブロックで処理
    return None, {}, scoring

# ============================================================
# ツリーモデル系 CV（ml_type != Polynomial）
# ============================================================

if ml_type != "Polynomial" and st.button("クロスバリデーション実行"):
    if mode == 'ランダム（単一ファイルから分割/CV）':
        if df is None:
            st.error("ファイルをアップロードしてください。"); st.stop()
        base_df = df.copy()
    else:
        if train_df is None:
            st.error("学習/評価ファイルをアップロードしてください。"); st.stop()
        base_df = train_df.copy()

    # dataset
    X, Y, features = tr.dataset(base_df, target, removal)
    groups = base_df[group_col] if group_col in base_df.columns else None

    # ---- X前処理（外れ値 / log / 交互作用）----
    out_method = st.session_state.get("outlier_method", "しない")
    out_cols = st.session_state.get("outlier_cols", [])
    if out_method != "しない" and out_cols:
        out_params = {
            "iqr_k": st.session_state.get("iqr_k", 1.5),
            "sigma_k": st.session_state.get("sigma_k", 3.0),
            "q_low": st.session_state.get("q_low", 0.01),
            "q_high": st.session_state.get("q_high", 0.99),
        }
        X = apply_outlier_clip_df(X, out_cols, out_method, out_params)

    log_cols = st.session_state.get("log_cols", [])
    if log_cols:
        X = log_transform_X(X, log_cols)

    inter_pairs = st.session_state.get("inter_pairs", [])
    if st.session_state.get("add_interactions", False) and inter_pairs:
        X, features = add_interaction_features(X, inter_pairs)

    # ---- 分類ラベル変換（ここが今回の主役）----
    label_meta = None
    if task_type == "分類":
        if label_design == "分位ビン（2/3/4）":
            n_bins = bin_map[bin_choice]
            y_cont = pd.to_numeric(Y, errors="coerce")
            valid = y_cont.notna()
            X = X.loc[valid]
            Y = y_cont.loc[valid]
            if groups is not None:
                groups = groups.loc[valid]
            Y = pd.Series(pd.qcut(Y, q=n_bins, labels=False, duplicates="drop").astype(int), index=Y.index, name=target)

        elif label_design == "3値（平均周りノイズ帯）":
            y_num = pd.to_numeric(Y, errors="coerce")
            valid = y_num.notna()
            X = X.loc[valid]
            Y = y_num.loc[valid]
            if groups is not None:
                groups = groups.loc[valid]
            Y, label_meta = make_ternary_noiseband_label(Y, method=tern_method, k=tern_k)
            Y = pd.Series(Y, index=X.index, name=target)

            st.info(f"3値分類ノイズ帯: method={tern_method}, k={tern_k:.2f} / "
                    f"mean={label_meta['mu']:.4f}, lower={label_meta['lower']:.4f}, upper={label_meta['upper']:.4f}")

        else:
            # そのまま
            Y = Y.copy()

    # ---- オーバーサンプル（分類のみ）----
    oversample_option = None
    if task_type == "分類":
        oversample_option = st.sidebar.selectbox("オーバーサンプリング", ["なし", "SMOTE", "Resample"], key="oversample_select")

    # ---- モデル + param grid（Bayes→Grid対応）----
    model, base_param_grid, scoring = build_param_grid_and_model(ml_type, task_type, random_state, reg_choice, elastic_l1_ratio)
    if model is None:
        st.error("モデル生成に失敗しました。"); st.stop()

    gkf = GroupKFold(n_splits=n_splits)

    fold_rows = []
    best_params_list = []
    best_fold = None
    best_pack = None

    labels_seen = set()
    cm_sum = None

    y_tr_all, yhat_tr_all, y_te_all, yhat_te_all = [], [], [], []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, Y, groups=groups), 1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        Y_tr, Y_te = Y.iloc[tr_idx], Y.iloc[te_idx]
        groups_tr = groups.iloc[tr_idx] if groups is not None else None

        # ---- oversample は train のみ ----
        if task_type == "分類" and oversample_option != "なし":
            if oversample_option == "SMOTE":
                sm_ = SMOTE(random_state=random_state)
                X_tr, Y_tr = sm_.fit_resample(X_tr, Y_tr)
                groups_tr = None  # resample後は group 整合が崩れるのでチューニングは通常CVで扱う
            elif oversample_option == "Resample":
                tmp = pd.concat([X_tr, Y_tr.rename("target")], axis=1)
                max_count = tmp["target"].value_counts().max()
                parts = [
                    tmp[tmp["target"] == cls].sample(n=max_count, replace=True, random_state=random_state)
                    for cls in tmp["target"].unique()
                ]
                tmp_up = pd.concat(parts, axis=0)
                X_tr = tmp_up.drop(columns=["target"])
                Y_tr = tmp_up["target"]
                groups_tr = None

        # ---- Bayes→Grid（このfold内で）----
        best_est, best_params, tune_meta = bayes_then_grid_search(
            estimator=model,
            base_param_grid=base_param_grid,
            scoring=scoring,
            cv=gkf,  # group-awareにする（ただし resample時は groups_tr=None）
            X_tr=X_tr, y_tr=Y_tr,
            groups=groups_tr,
            use_bayes=use_bayes,
            n_iter=bayes_n_iter,
            random_state=random_state,
            bayes_shrink=bayes_shrink,
        )
        best_params_list.append({"fold": fold, **best_params})

        mdl = best_est
        mdl.fit(X_tr, Y_tr)

        if task_type == "分類":
            pred_te = mdl.predict(X_te)
            acc  = accuracy_score(Y_te, pred_te)
            prec = precision_score(Y_te, pred_te, average="macro", zero_division=0)
            rec  = recall_score(Y_te, pred_te, average="macro", zero_division=0)
            f1   = f1_score(Y_te, pred_te, average="macro", zero_division=0)

            labels_seen.update(pd.Series(Y_te).unique().tolist())
            labs = sorted(list(labels_seen))
            cm = confusion_matrix(Y_te, pred_te, labels=labs)
            if cm_sum is None:
                cm_sum = cm
            else:
                # サイズ不一致が起きたら拡張
                if cm_sum.shape != cm.shape:
                    new_cm = np.zeros((len(labs), len(labs)), dtype=int)
                    new_cm[:cm_sum.shape[0], :cm_sum.shape[1]] = cm_sum
                    cm_sum = new_cm
                cm_sum += cm

            fold_rows.append({
                "fold": fold,
                "Accuracy": acc,
                "Precision(macro)": prec,
                "Recall(macro)": rec,
                "F1(macro)": f1,
                "mdl": mdl,
                "X_tr": X_tr, "X_te": X_te,
                "Y_tr": Y_tr, "Y_te": Y_te,
                "cm": cm,
                "yhat_te": pred_te,
            })

        else:
            pred_tr = mdl.predict(X_tr)
            pred_te = mdl.predict(X_te)

            rmse_tr = float(np.sqrt(mean_squared_error(Y_tr, pred_tr)))
            mae_tr  = float(mean_absolute_error(Y_tr, pred_tr))
            r2_tr   = float(r2_score(Y_tr, pred_tr))
            rmse_te = float(np.sqrt(mean_squared_error(Y_te, pred_te)))
            mae_te  = float(mean_absolute_error(Y_te, pred_te))
            r2_te   = float(r2_score(Y_te, pred_te))

            y_tr_all.extend(Y_tr.tolist()); yhat_tr_all.extend(pred_tr.tolist())
            y_te_all.extend(Y_te.tolist()); yhat_te_all.extend(pred_te.tolist())

            fold_rows.append({
                "fold": fold,
                "RMSE(train)": rmse_tr,
                "MAE(train)":  mae_tr,
                "R2(train)":   r2_tr,
                "RMSE(test)":  rmse_te,
                "MAE(test)":   mae_te,
                "R2(test)":    r2_te,
                "mdl": mdl,
                "X_tr": X_tr, "X_te": X_te,
                "Y_tr": Y_tr, "Y_te": Y_te,
                "yhat_tr": pred_tr,
                "yhat_te": pred_te,
            })

    # ---- CV結果保存 ----
    fold_df = pd.DataFrame(fold_rows)
    if fold_df.empty:
        st.error("CVで有効なfoldが得られませんでした。"); st.stop()

    if task_type == "分類":
        df_scores = fold_df.set_index("fold")[["Accuracy", "Precision(macro)", "Recall(macro)", "F1(macro)"]].sort_index()
        best_idx = fold_df["F1(macro)"].idxmax()
        best_row = fold_df.loc[best_idx]
        best_fold = int(best_row["fold"])
        best_pack = (
            "cls",
            best_row["mdl"],
            best_row["X_tr"], best_row["X_te"],
            best_row["Y_tr"], best_row["Y_te"],
            features,
            {
                "acc": best_row["Accuracy"],
                "prec": best_row["Precision(macro)"],
                "rec": best_row["Recall(macro)"],
                "f1": best_row["F1(macro)"],
                "cm": best_row["cm"],
                "yhat_te": best_row["yhat_te"],
            }
        )
        st.session_state.cv_result = {
            "task_type": "分類",
            "ml_type": ml_type,
            "df_scores": df_scores,
            "cm_sum": cm_sum,
            "labels_seen": sorted(list(labels_seen)),
            "best_params_list": best_params_list,
        }

    else:
        df_scores_test = fold_df.set_index("fold")[["RMSE(test)", "MAE(test)", "R2(test)"]].sort_index()
        df_scores_train = fold_df.set_index("fold")[["RMSE(train)", "MAE(train)", "R2(train)"]].sort_index()

        best_idx = fold_df["R2(test)"].idxmax()
        best_row = fold_df.loc[best_idx]
        best_fold = int(best_row["fold"])
        best_pack = (
            "reg",
            best_row["mdl"],
            best_row["X_tr"], best_row["X_te"],
            best_row["Y_tr"], best_row["Y_te"],
            features,
            {
                "rmse_tr": best_row["RMSE(train)"],
                "mae_tr":  best_row["MAE(train)"],
                "r2_tr":   best_row["R2(train)"],
                "rmse_te": best_row["RMSE(test)"],
                "mae_te":  best_row["MAE(test)"],
                "r2_te":   best_row["R2(test)"],
                "yhat_tr": best_row["yhat_tr"],
                "yhat_te": best_row["yhat_te"],
            }
        )

        st.session_state.cv_result = {
            "task_type": "回帰",
            "ml_type": ml_type,
            "df_scores_test": df_scores_test,
            "df_scores_train": df_scores_train,
            "y_tr_all": y_tr_all, "yhat_tr_all": yhat_tr_all,
            "y_te_all": y_te_all, "yhat_te_all": yhat_te_all,
        }

    # payload（Best fold）
    if best_pack is not None:
        kind, mdl_b, X_tr_b, X_te_b, Y_tr_b, Y_te_b, feats_b, meta = best_pack
        bg = X_tr_b if len(X_tr_b) <= 200 else X_tr_b.sample(200, random_state=42)

        st.session_state.cv_ready = True
        st.session_state.cv_payload = {
            "task_type": task_type,
            "ml_type": ml_type,
            "best_pack": best_pack,
            "bg": bg,
            "X_te": X_te_b,
            "features": feats_b,
            "best_fold": best_fold,
        }
        st.session_state.shap_cache_key = None
        st.session_state.shap_values = None
        st.session_state.shap_task = None
        st.session_state.shap_model_id = id(mdl_b)

# ============================================================
# ボトルネック診断
# ============================================================

st.markdown("---")
st.subheader("🔎 ボトルネック診断（y / x / 被験者のどこを優先して改善するか）")

run_diag = st.button("この設定でボトルネック診断を実行")

if run_diag:
    if mode == 'ランダム（単一ファイルから分割/CV）':
        if df is None:
            st.error("ファイルをアップロードしてください。"); st.stop()
        base_df = df.copy()
    else:
        if train_df is None:
            st.error("学習/評価ファイルをアップロードしてください。"); st.stop()
        base_df = train_df.copy()

    X, Y, features = tr.dataset(base_df, target, removal)
    groups_subj = base_df[group_col] if group_col in base_df.columns else None

    # 前処理（CVと同じ）
    out_method = st.session_state.get("outlier_method", "しない")
    out_cols   = st.session_state.get("outlier_cols", [])
    if out_method != "しない" and out_cols:
        out_params = {
            "iqr_k":  st.session_state.get("iqr_k", 1.5),
            "sigma_k": st.session_state.get("sigma_k", 3.0),
            "q_low":  st.session_state.get("q_low", 0.01),
            "q_high": st.session_state.get("q_high", 0.99),
        }
        X = apply_outlier_clip_df(X, out_cols, out_method, out_params)

    log_cols = st.session_state.get("log_cols", [])
    if log_cols:
        X = log_transform_X(X, log_cols)

    inter_pairs = st.session_state.get("inter_pairs", [])
    if st.session_state.get("add_interactions", False) and inter_pairs:
        X, features = add_interaction_features(X, inter_pairs)

    # 分類ラベル（診断側も同じ設計を反映）
    Y_use = Y.copy()
    if task_type == "分類":
        if label_design == "分位ビン（2/3/4）":
            n_bins = bin_map[bin_choice]
            y_cont = pd.to_numeric(Y_use, errors="coerce")
            valid = y_cont.notna()
            X = X.loc[valid]; y_cont = y_cont.loc[valid]
            if groups_subj is not None: groups_subj = groups_subj.loc[valid]
            Y_use = pd.Series(pd.qcut(y_cont, q=n_bins, labels=False, duplicates="drop").astype(int), index=X.index)
        elif label_design == "3値（平均周りノイズ帯）":
            y_num = pd.to_numeric(Y_use, errors="coerce")
            valid = y_num.notna()
            X = X.loc[valid]; y_num = y_num.loc[valid]
            if groups_subj is not None: groups_subj = groups_subj.loc[valid]
            Y_use, _ = make_ternary_noiseband_label(y_num, method=tern_method, k=tern_k)
            Y_use = pd.Series(Y_use, index=X.index)

    # LOIO group 推定
    loio_group = None
    if "file_name" in base_df.columns:
        loio_group = base_df.loc[X.index, "file_name"]
    elif "figure" in base_df.columns:
        loio_group = base_df.loc[X.index, "figure"]

    # y信頼性
    y_rel = estimate_var_component_reliability(base_df.loc[X.index], target, subject_col=group_col,
                                               unit_col="file_name" if "file_name" in base_df.columns else None)
    y_split = estimate_y_reliability_split_half(base_df.loc[X.index], target,
                                                unit_cols=[c for c in [group_col, "file_name"] if c in base_df.columns])

    # x信号（Permutation）
    x_sig = permutation_signal_test(task_type, X, Y_use, groups=groups_subj, scheme="loso", n_perm=20, seed=42)

    # within / loso / loio
    scores = {}
    scores["within"] = cv_eval_quick(task_type, X, Y_use, groups=None, scheme="within", n_splits=5, seed=42)
    if groups_subj is not None:
        scores["loso"] = cv_eval_quick(task_type, X, Y_use, groups=groups_subj, scheme="loso", seed=42)
    if loio_group is not None:
        scores["loio"] = cv_eval_quick(task_type, X, Y_use, groups=loio_group, scheme="loio", seed=42)

    diag = {"task_type": task_type, "y_rel": y_rel, "y_split": y_split, "x_signal": x_sig, "scores": scores}
    primary, reasons = decide_bottleneck(diag)

    st.markdown("### ① y（目的変数）のノイズ診断")
    if y_rel is None:
        st.info("y信頼性の推定ができません（必要列不足の可能性）。")
    else:
        st.dataframe(pd.DataFrame([y_rel]))
    if y_split is not None:
        st.dataframe(pd.DataFrame([y_split]))

    st.markdown("### ② x（特徴量）の信号診断（Permutation）")
    st.dataframe(pd.DataFrame([{
        _metric_name(task_type)+"(real)": x_sig["real"],
        _metric_name(task_type)+"(perm_mean)": x_sig["perm_mean"],
        "perm_sd": x_sig["perm_sd"],
        "delta(real-perm)": x_sig["real"] - x_sig["perm_mean"],
        "p_like": x_sig["p_like"],
    }]))

    st.markdown("### ③ 被験者差・一般化（within / LOSO / LOIO）")
    st.dataframe(pd.DataFrame([scores]))

    st.markdown("### ✅ 更新方針（結論）")
    st.success(primary)
    for r in reasons:
        st.write("- " + r)

# ============================================================
# CV結果表示（ツリー系）
# ============================================================

if ml_type != "Polynomial":
    st.markdown("---")
    st.subheader("クロスバリデーション再集計")

    cv_res = st.session_state.get("cv_result", None)
    if cv_res is None:
        st.info("『クロスバリデーション実行』後に表示されます。")
    else:
        if cv_res["task_type"] == "分類":
            df_scores = cv_res["df_scores"]
            st.dataframe(df_scores)
            st.write("平均 ± SD")
            st.dataframe(df_scores.agg(["mean", "std"]))

            cm_sum = cv_res["cm_sum"]
            if cm_sum is not None:
                labs = cv_res["labels_seen"]
                cm_df = pd.DataFrame(cm_sum, index=[f"T{i}" for i in labs], columns=[f"P{i}" for i in labs])
                st.write("総和混同行列（全fold合算）")
                st.dataframe(cm_df)

            st.subheader("各Foldの最適ハイパーパラメータ（Bayes→Grid後）")
            st.dataframe(pd.DataFrame(cv_res["best_params_list"]))
        else:
            st.write("Test 指標（平均 ± SD）")
            st.dataframe(cv_res["df_scores_test"])
            st.dataframe(cv_res["df_scores_test"].agg(["mean", "std"]))

            st.write("Train 指標（平均 ± SD）")
            st.dataframe(cv_res["df_scores_train"])
            st.dataframe(cv_res["df_scores_train"].agg(["mean", "std"]))

            y_tr_all = pd.Series(cv_res["y_tr_all"])
            yhat_tr_all = pd.Series(cv_res["yhat_tr_all"])
            y_te_all = pd.Series(cv_res["y_te_all"])
            yhat_te_all = pd.Series(cv_res["yhat_te_all"])

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(y_tr_all, yhat_tr_all, alpha=0.6, label="Train", marker="o")
            ax.scatter(y_te_all, yhat_te_all, alpha=0.6, label="Test", marker="^")
            lo = float(min(y_tr_all.min(), y_te_all.min(), yhat_tr_all.min(), yhat_te_all.min()))
            hi = float(max(y_tr_all.max(), y_te_all.max(), yhat_tr_all.max(), yhat_te_all.max()))
            ax.plot([lo, hi], [lo, hi], "--")
            ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
            ax.set_title(f"Actual vs Predicted (All folds, {cv_res['ml_type']})")
            ax.legend()
            st.pyplot(fig)

    st.subheader("ベストFoldの結果")
    if st.session_state.get("cv_ready", False) and st.session_state.get("cv_payload"):
        payload = st.session_state.cv_payload
        kind, mdl_b, X_tr_b, X_te_b, Y_tr_b, Y_te_b, feats_b, meta = payload["best_pack"]

        st.caption(f"ベストFold: fold={payload['best_fold']}")

        if kind == "cls":
            yhat_te_b = meta["yhat_te"]
            labs = sorted(pd.Series(Y_te_b).unique().tolist())
            cm_raw = confusion_matrix(Y_te_b, yhat_te_b, labels=labs)

            acc_b  = accuracy_score(Y_te_b, yhat_te_b)
            prec_b = precision_score(Y_te_b, yhat_te_b, average="macro", zero_division=0)
            rec_b  = recall_score(Y_te_b, yhat_te_b, average="macro", zero_division=0)
            f1_b   = f1_score(Y_te_b, yhat_te_b, average="macro", zero_division=0)

            st.write(f"Accuracy: {acc_b:.3f} / Precision(macro): {prec_b:.3f} / Recall(macro): {rec_b:.3f} / F1(macro): {f1_b:.3f}")
            st.caption(f"Baseline(1/num_classes) ≈ {1/len(labs):.3f}")

            cm_df = pd.DataFrame(cm_raw, index=[f"True_{c}" for c in labs], columns=[f"Pred_{c}" for c in labs])
            st.write("混同行列（Best fold, 件数）")
            st.dataframe(cm_df)

        else:
            st.write(f"Train: R²={meta['r2_tr']:.3f}, RMSE={meta['rmse_tr']:.3f}, MAE={meta['mae_tr']:.3f}")
            st.write(f"Test : R²={meta['r2_te']:.3f}, RMSE={meta['rmse_te']:.3f}, MAE={meta['mae_te']:.3f}")

        if ml_type == "DecisionTree":
            st.subheader("決定木の構造（Best fold）")
            fig_dt, ax_dt = plt.subplots(figsize=(16, 10))
            if kind == "cls":
                classes = [str(c) for c in sorted(pd.Series(Y_tr_b).unique())]
                plot_tree(mdl_b, feature_names=feats_b, class_names=classes, filled=True, rounded=True, impurity=False, ax=ax_dt)
            else:
                plot_tree(mdl_b, feature_names=feats_b, filled=True, rounded=True, impurity=False, ax=ax_dt)
            st.pyplot(fig_dt)

        st.subheader("特徴量重要度（Best fold, 対応モデルのみ）")
        if hasattr(mdl_b, "feature_importances_"):
            try:
                fig_imp = tr.importance(mdl_b, feats_b)
                st.pyplot(fig_imp)
            except Exception:
                importances = pd.Series(mdl_b.feature_importances_, index=feats_b).sort_values(ascending=False).head(30)
                fig2, ax2 = plt.subplots(figsize=(6, min(10, 0.3 * len(importances))))
                importances.iloc[::-1].plot(kind="barh", ax=ax2)
                ax2.set_title("Top Feature Importances (Best fold)")
                st.pyplot(fig2)
        else:
            st.info("このモデルでは特徴量重要度を表示できません。")

        st.subheader("XAI（Best fold）")
        X_bg = payload["bg"][feats_b]
        X_te = payload["X_te"][feats_b]
        try:
            xai.explain_shap(mdl_b, X_bg, X_te, payload["task_type"], payload["ml_type"])
            xai.explain_lime(mdl_b, X_bg, X_te, payload["task_type"])
        except Exception as e:
            st.info(f"XAI 計算でエラー: {e}")

    else:
        st.info("CV実行後に表示されます。")

# ============================================================
# Polynomial（あなたの元設計を大きく壊さずに残す）
# ※ 正則化(L1/L2/Elastic)は LogisticRegression/回帰側は LassoCV中心。
# ============================================================

if ml_type == "Polynomial":
    st.markdown("---")
    st.subheader("多項式モデル（Lasso / AIC / 全変数）")

    # Polynomial 用UI
    poly_model_type = st.sidebar.selectbox("多項式モデルの変数選択方法", ["Lasso（正則化）", "AIC（ステップワイズ）", "なし（全変数使用）"])
    poly_mode = st.sidebar.selectbox("多項式項のパターン（2次まで）", ["一次のみ", "一次＋交互作用のみ", "２次のみ", "２次＋交互作用のみ"])
    poly_standardize = st.sidebar.checkbox("多項式展開後に標準化する", value=True)
    poly_test_size = st.sidebar.slider("テスト割合（単一ファイル時）", 0.1, 0.5, 0.2)
    poly_random_state = st.sidebar.number_input("random_state（Polynomial）", 0, 9999, 42)

    run_poly = st.button("多項式モデルの学習・評価を実行")
    if run_poly:
        base_df = df if df is not None else train_df
        if base_df is None or target is None or feature_cols is None or len(feature_cols) == 0:
            st.error("データと目的変数・説明変数を指定してください。"); st.stop()

        X_tr, X_te, y_tr, y_te = make_train_test(
            df if df is not None else train_df,
            None if df is not None else test_df,
            mode_flag,
            target,
            feature_cols,
            group_col=group_col if group_col in base_df.columns else None,
            test_size=poly_test_size,
            random_state=poly_random_state
        )

        # outlier
        out_method = st.session_state.get("outlier_method", "しない")
        out_cols = st.session_state.get("outlier_cols", [])
        if out_method != "しない" and out_cols:
            out_params = {
                "iqr_k": st.session_state.get("iqr_k", 1.5),
                "sigma_k": st.session_state.get("sigma_k", 3.0),
                "q_low": st.session_state.get("q_low", 0.01),
                "q_high": st.session_state.get("q_high", 0.99),
            }
            X_tr, X_te = apply_outlier_clip_train_test(X_tr, X_te, out_cols, out_method, out_params)

        # log
        log_cols = st.session_state.get("log_cols", [])
        if log_cols:
            X_tr, X_te = log_transform_train_test(X_tr, X_te, log_cols)

        # poly mode
        if poly_mode == "一次のみ":
            poly_sel_mode = "1_only"
        elif poly_mode == "一次＋交互作用のみ":
            poly_sel_mode = "1_plus_interactions"
        elif poly_mode == "２次のみ":
            poly_sel_mode = "2_only"
        else:
            poly_sel_mode = "2_plus_interactions"

        if poly_model_type.startswith("Lasso"):
            st.subheader("Lasso 多項式モデル（回帰） / Logistic（分類）")

            poly_step = PolynomialSelector(mode=poly_sel_mode)
            steps = [("poly", poly_step)]
            if poly_standardize:
                steps.append(("scaler", StandardScaler()))

            if task_type == "回帰":
                steps.append(("model", LassoCV(cv=5, random_state=poly_random_state)))
            else:
                # 分類：penalty を reg_choice に合わせる（L1/L2/Elastic）
                if reg_choice == "L1":
                    penalty, solver, l1_ratio = "l1", "saga", None
                elif reg_choice == "ElasticNet":
                    penalty, solver, l1_ratio = "elasticnet", "saga", elastic_l1_ratio
                else:
                    penalty, solver, l1_ratio = "l2", "lbfgs", None

                steps.append(("model", LogisticRegression(
                    penalty=penalty,
                    solver=solver,
                    l1_ratio=l1_ratio,
                    max_iter=5000,
                    random_state=poly_random_state,
                    class_weight="balanced"
                )))

            pipe = Pipeline(steps)
            pipe.fit(X_tr, y_tr)

            yhat_tr = pipe.predict(X_tr)
            yhat_te = pipe.predict(X_te)

            if task_type == "回帰":
                rmse_tr = np.sqrt(mean_squared_error(y_tr, yhat_tr))
                mae_tr  = mean_absolute_error(y_tr, yhat_tr)
                r2_tr   = r2_score(y_tr, yhat_tr)
                rmse_te = np.sqrt(mean_squared_error(y_te, yhat_te))
                mae_te  = mean_absolute_error(y_te, yhat_te)
                r2_te   = r2_score(y_te, yhat_te)
                st.write(f"Train: R²={r2_tr:.3f}, RMSE={rmse_tr:.3f}, MAE={mae_tr:.3f}")
                st.write(f"Test : R²={r2_te:.3f}, RMSE={rmse_te:.3f}, MAE={mae_te:.3f}")
            else:
                acc = accuracy_score(y_te, yhat_te)
                prec = precision_score(y_te, yhat_te, average="macro", zero_division=0)
                rec = recall_score(y_te, yhat_te, average="macro", zero_division=0)
                f1 = f1_score(y_te, yhat_te, average="macro", zero_division=0)
                st.write(f"Accuracy={acc:.3f}, Precision(macro)={prec:.3f}, Recall(macro)={rec:.3f}, F1(macro)={f1:.3f}")

            coef_df, intercept = coef_table_from_lasso_poly(pipe, feature_cols, task_type)
            st.subheader("多項式項の係数（上位）")
            st.write(f"intercept: {intercept:.4f}")
            st.dataframe(coef_df.head(50))

            st.subheader("XAI（Poly）")
            try:
                bg = X_tr if len(X_tr) <= 200 else X_tr.sample(200, random_state=42)
                xai.explain_shap(pipe, bg, X_te, task_type, "Poly")
                xai.explain_lime(pipe, bg, X_te, task_type)
            except Exception as e:
                st.info(f"XAIでエラー: {e}")

        elif poly_model_type.startswith("AIC"):
            st.subheader("AIC ステップワイズ（statsmodels）")

            base_poly = PolynomialFeatures(degree=2, include_bias=False)
            Z_tr_full = base_poly.fit_transform(X_tr[feature_cols])
            Z_te_full = base_poly.transform(X_te[feature_cols])
            powers = base_poly.powers_
            degs = powers.sum(axis=1)
            nnz = (powers != 0).sum(axis=1)

            if poly_sel_mode == "1_only":
                mask = (degs == 1)
            elif poly_sel_mode == "1_plus_interactions":
                mask = (degs == 1) | ((degs >= 2) & (nnz >= 2))
            elif poly_sel_mode == "2_only":
                mask = (degs == 1) | ((degs == 2) & (nnz == 1))
            else:
                mask = (degs == 1) | (degs == 2)

            names = np.asarray(base_poly.get_feature_names_out(feature_cols))[mask]
            X_tr_poly = pd.DataFrame(Z_tr_full[:, mask], columns=names, index=X_tr.index)
            X_te_poly = pd.DataFrame(Z_te_full[:, mask], columns=names, index=X_te.index)

            if poly_standardize:
                scaler = StandardScaler()
                X_tr_poly[:] = scaler.fit_transform(X_tr_poly)
                X_te_poly[:] = scaler.transform(X_te_poly)

            if task_type == "回帰":
                model_aic, selected = stepwise_aic_ols(X_tr_poly, y_tr)
                st.write(f"選択項数: {len(selected)}")
                coef_df = pd.DataFrame({"term": model_aic.params.index, "coef": model_aic.params.values, "p": model_aic.pvalues.values})
                coef_df = coef_df.sort_values("coef", key=np.abs, ascending=False)
                st.dataframe(coef_df.head(50))

                yhat_tr = model_aic.predict(sm.add_constant(X_tr_poly[selected]))
                yhat_te = model_aic.predict(sm.add_constant(X_te_poly[selected], has_constant="add"))

                st.write(f"Train R²={r2_score(y_tr, yhat_tr):.3f} / Test R²={r2_score(y_te, yhat_te):.3f}")
            else:
                st.error("AIC(Logit) は2値のみ対応のため、ここでは割愛（必要なら追加します）。")

        else:
            st.subheader("全変数（statsmodels）")
            st.info("必要ならこのブロックも拡張できます。")
