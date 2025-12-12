# =========================
# 多項式モデル用ユーティリティ
# =========================
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st

from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, accuracy_score
)
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier as DTC, DecisionTreeRegressor as DTR
from sklearn.ensemble import RandomForestClassifier as RFC, RandomForestRegressor as RFR
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor  # ★ LightGBM 追加

from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import statsmodels.api as sm

import xai  # あなたの既存 XAI モジュール
import tree as tr  # あなたの既存ユーティリティ

# =========================
# 多項式モデル用ユーティリティ
# =========================

def make_train_test(
    df_train,
    df_test,
    mode_flag,
    target,
    feature_cols,
    group_col=None,
    test_size=0.2,
    random_state=42
):
    """
    単一ファイル or train/test 別ファイルから
    X_train, X_test, y_train, y_test を返す。
    mode_flag: "single" or "split"
    """
    if mode_flag == "single":
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
        tr_df = df_train.copy()
        te_df = df_test.copy()
        y_tr = tr_df[target]
        y_te = te_df[target]
        X_tr = tr_df[feature_cols]
        X_te = te_df[feature_cols]

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


class PolynomialSelector(BaseEstimator, TransformerMixin):
    """
    powers_ から項の次数と交互作用の有無を見て、以下4パターンを実現するための Transformer。

    mode:
      - "1_only"              : 一次のみ
      - "1_plus_interactions" : 一次＋交互作用のみ（２次以上の交互作用項を含む。２乗なし）
      - "2_only"              : 2次のみ（一次＋2乗。交互作用なし）
      - "2_plus_interactions" : 2次＋交互作用のみ（一次＋2乗＋交互作用）（2次まで）

    内部的には degree=2 の PolynomialFeatures を使う。
    """

    def __init__(self, mode="1_only"):
        self.mode = mode

    def fit(self, X, y=None):
        self.poly_ = PolynomialFeatures(degree=2, include_bias=False)
        self.poly_.fit(X)
        powers = self.poly_.powers_      # shape: (n_out, n_in)
        degs = powers.sum(axis=1)        # 各項の次数
        nnz = (powers != 0).sum(axis=1)  # 何変数掛けか（非ゼロ要素数）

        if self.mode == "1_only":
            mask = (degs == 1)
        elif self.mode == "1_plus_interactions":
            mask = (degs == 1) | ((degs >= 2) & (nnz >= 2))  # 一次 + 真の交互作用
        elif self.mode == "2_only":
            mask = (degs == 1) | ((degs == 2) & (nnz == 1))  # 一次 + 2乗のみ
        elif self.mode == "2_plus_interactions":
            mask = (degs == 1) | (degs == 2)                 # 一次 + 2乗 + 交互作用
        else:
            mask = (degs == 1)

        self.keep_idx_ = np.where(mask)[0]
        return self

    def transform(self, X):
        Z = self.poly_.transform(X)
        return Z[:, self.keep_idx_]

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [
                f"x{i}" for i in range(self.poly_.n_input_features_)
            ]
        names = self.poly_.get_feature_names_out(input_features)
        names = np.asarray(names)[self.keep_idx_]
        return names


def coef_table_from_lasso_poly(pipe, feature_cols, task_type):
    """
    Pipeline(poly + scaler? + Lasso or LogisticRegression) から
    多項式項の係数表を作る。
    """
    poly = pipe.named_steps["poly"]
    if hasattr(poly, "get_feature_names_out"):
        names = poly.get_feature_names_out(feature_cols)
    else:
        names = np.array([f"f{i}" for i in range(pipe.named_steps["model"].coef_.shape[0])])

    if task_type == "回帰":
        reg = pipe.named_steps["model"]
        coefs = reg.coef_
        intercept = reg.intercept_
    else:
        clf = pipe.named_steps["model"]
        coefs = clf.coef_[0]
        intercept = clf.intercept_[0]

    df = pd.DataFrame({"term": names, "coef": coefs}).sort_values(
        "coef", key=np.abs, ascending=False
    )
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
    前進ステップワイズ AIC 最小化（2値ロジスティック回帰）。 y は 0/1 を想定。
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


# =========================
# 交互作用項（任意ペア指定）用ユーティリティ
# =========================

def add_interaction_features(X: pd.DataFrame, pair_list):
    """
    pair_list: [(col1, col2), (col1, col3), ...] の形で
               交互作用として追加したい変数ペアのリスト。

    各ペアについて X[col1] * X[col2] を計算し、
    列名 "col1*col2" で X に追加する。

    返り値:
      - X_new : 交互作用列を追加した DataFrame
      - new_feature_list : X_new.columns のリスト
    """
    X_new = X.copy()
    for c1, c2 in pair_list:
        if (c1 not in X_new.columns) or (c2 not in X_new.columns):
            continue

        name = f"{c1}*{c2}"
        if name in X_new.columns:
            continue

        try:
            X_new[name] = X_new[c1] * X_new[c2]
        except Exception:
            continue

    return X_new, list(X_new.columns)


# =========================
# log 変換ユーティリティ
# =========================

def log_transform_X(X: pd.DataFrame, cols, eps: float = 1e-6) -> pd.DataFrame:
    """
    単一 DataFrame に対して、指定列に log(1 + x + shift) を適用する。
    shift は train 側の最小値から決める想定なので、
    CV のときなど「X 全体」を渡して使う。
    """
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


def log_transform_train_test(
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    cols,
    eps: float = 1e-6
):
    """
    train/test で同じ shift を使って log(1 + x + shift) を掛ける。
    shift は train 側の最小値から決める。
    """
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


# =========================
# ハズレ値処理ユーティリティ
# =========================

def apply_outlier_clip_df(
    X: pd.DataFrame,
    cols,
    method: str,
    params: dict
) -> pd.DataFrame:
    """
    1つの DataFrame に対してハズレ値をクリップする。
    method:
      - "IQRでクリップ"
      - "標準偏差でクリップ"
      - "分位点でクリップ"
    """
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
            # 未知の method → 何もしない
            continue

        X_new[c] = vals.clip(lower, upper)
    return X_new


def apply_outlier_clip_train_test(
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    cols,
    method: str,
    params: dict
):
    """
    train / test で同じしきい値になるように、
    train から閾値を算出して両方をクリップする。
    """
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


# =========================
# Streamlit 初期設定
# =========================

st.title("機械学習（GroupKFold ツリーモデル＋多項式モデル｜XAI対応）")

# ---- セッション初期化 ----
if "cv_ready" not in st.session_state:
    st.session_state.cv_ready = False
    st.session_state.cv_payload = None      # ベストfold用
    st.session_state.shap_cache_key = None  # SHAP計算のキャッシュキー
    st.session_state.shap_values = None
    st.session_state.shap_task = None       # "分類" / "回帰"
    st.session_state.shap_model_id = None   # id(mdl_b) でモデルが変わったら再計算
    st.session_state.cv_result = None       # CV 全体の集計結果

# 交互作用ペア管理
if "inter_pairs" not in st.session_state:
    st.session_state.inter_pairs = []
if "add_interactions" not in st.session_state:
    st.session_state.add_interactions = False

# log 変換の状態
if "log_all" not in st.session_state:
    st.session_state.log_all = False
if "log_cols" not in st.session_state:
    st.session_state.log_cols = []

# ★ ハズレ値処理の状態
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

# =========================
# 入力UI（共通）
# =========================

mode = st.radio(
    "データの指定方法",
    ('ランダム（単一ファイルから分割/CV）', '自分で決める（学習/評価を別ファイル）')
)

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
        group_col = "folder_name"  # 被験者ID列（固定とする）

        # 目的変数は説明変数にならないので候補から外す
        feature_candidates = [c for c in features_all if c != target]

        # 説明変数の指定方法
        sel_mode = st.radio(
            "説明変数の指定方法",
            ["除外する列を選ぶ", "使う列を選ぶ"],
            horizontal=True,
        )

        if sel_mode == "除外する列を選ぶ":
            default_removal = [group_col] if group_col in feature_candidates else []
            removal = st.multiselect(
                "説明変数から除外する列",
                feature_candidates,
                default=default_removal
            )
        else:
            default_use = [c for c in feature_candidates if c != group_col]
            use_cols = st.multiselect(
                "説明変数として使う列",
                feature_candidates,
                default=default_use
            )
            removal = [c for c in feature_candidates if c not in use_cols]

        # group_col は必ず除外
        if group_col in features_all and group_col not in removal:
            removal.append(group_col)

        # 説明変数として実際に使う列（Polynomial でもツリーモデルでも共通）
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

        sel_mode = st.radio(
            "説明変数の指定方法",
            ["除外する列を選ぶ", "使う列を選ぶ"],
            horizontal=True,
        )

        if sel_mode == "除外する列を選ぶ":
            default_removal = [group_col] if group_col in feature_candidates else []
            removal = st.multiselect(
                "説明変数から除外する列",
                feature_candidates,
                default=default_removal
            )
        else:
            default_use = [c for c in feature_candidates if c != group_col]
            use_cols = st.multiselect(
                "説明変数として使う列",
                feature_candidates,
                default=default_use
            )
            removal = [c for c in feature_candidates if c not in use_cols]

        if group_col in features_all and group_col not in removal:
            removal.append(group_col)

        feature_cols = [c for c in features_all if c not in removal and c != target]

        name = st.text_input("実験名（任意）")  # ← ここ typo に気づいたら "text_input" に戻して

# mode_flag（Polynomial 用）
mode_flag = "single" if mode.startswith("ランダム") else "split"

# =========================
# サイドバー：モデリング設定
# =========================
st.sidebar.header("モデリング設定")

# タスク種別
task_type = st.sidebar.radio("タスク", ["分類", "回帰"])

# =========================
# 前処理（log 変換）
# =========================
st.sidebar.subheader("前処理（log 変換）")

if features_all is None or target is None:
    st.sidebar.info("ファイル読み込み後に log 変換する列を選べます。")
else:
    base_df = df if df is not None else train_df
    if base_df is None:
        st.sidebar.info("ファイル読み込み後に log 変換する列を選べます。")
    else:
        # 目的変数と group_col は除外しつつ、数値列だけ候補にする
        col_candidates = [
            c for c in base_df.columns
            if c != target and c != group_col
        ]
        numeric_candidates = [
            c for c in col_candidates
            if pd.api.types.is_numeric_dtype(base_df[c])
        ]

        if len(numeric_candidates) == 0:
            st.sidebar.info("log 変換できる数値列がありません。")
            st.session_state.log_cols = []
        else:
            log_all = st.sidebar.checkbox(
                "数値の説明変数すべてに log をかける",
                value=st.session_state.log_all
            )
            st.session_state.log_all = log_all

            if log_all:
                st.sidebar.caption("現在の数値説明変数すべてに log(1+x) を適用します。")
                st.session_state.log_cols = numeric_candidates
            else:
                default_logs = [
                    c for c in st.session_state.log_cols
                    if c in numeric_candidates
                ]
                log_cols = st.sidebar.multiselect(
                    "log(1+x) をかける列を選択",
                    options=numeric_candidates,
                    default=default_logs
                )
                st.session_state.log_cols = log_cols

# =========================
# 前処理（ハズレ値処理）
# =========================
st.sidebar.subheader("前処理（ハズレ値処理）")

if features_all is None or target is None:
    st.sidebar.info("ファイル読み込み後にハズレ値処理を設定できます。")
else:
    base_df = df if df is not None else train_df
    if base_df is None:
        st.sidebar.info("ファイル読み込み後にハズレ値処理を設定できます。")
    else:
        col_candidates = [
            c for c in base_df.columns
            if c != group_col
        ]
        numeric_candidates_out = [
            c for c in col_candidates
            if pd.api.types.is_numeric_dtype(base_df[c])
        ]

        if len(numeric_candidates_out) == 0:
            st.sidebar.info("ハズレ値処理できる数値列がありません。")
            st.session_state.outlier_cols = []
            st.session_state.outlier_method = "しない"
        else:
            method_options = ["しない", "IQRでクリップ", "標準偏差でクリップ", "分位点でクリップ"]
            current_method = st.session_state.outlier_method
            if current_method not in method_options:
                current_method = "しない"

            method = st.sidebar.selectbox(
                "ハズレ値処理の方法",
                method_options,
                index=method_options.index(current_method)
            )
            st.session_state.outlier_method = method

            if method == "しない":
                st.session_state.outlier_cols = []
            else:
                default_out_cols = [
                    c for c in st.session_state.outlier_cols
                    if c in numeric_candidates_out
                ]
                out_cols = st.sidebar.multiselect(
                    "ハズレ値処理を行う列",
                    options=numeric_candidates_out,
                    default=default_out_cols
                )
                st.session_state.outlier_cols = out_cols

                if out_cols:
                    if method == "IQRでクリップ":
                        k = st.sidebar.slider(
                            "IQR倍率 k",
                            0.5, 5.0,
                            float(st.session_state.iqr_k)
                        )
                        st.session_state.iqr_k = k
                        st.sidebar.caption("Q1 - k×IQR 未満、Q3 + k×IQR を超える値をクリップします。")

                    elif method == "標準偏差でクリップ":
                        k = st.sidebar.slider(
                            "標準偏差倍率 k",
                            0.5, 5.0,
                            float(st.session_state.sigma_k)
                        )
                        st.session_state.sigma_k = k
                        st.sidebar.caption("平均 ± k×σ の外側をクリップします。")

                    elif method == "分位点でクリップ":
                        q_low = st.sidebar.slider(
                            "下側分位点 (例: 0.01)",
                            0.0, 0.4,
                            float(st.session_state.q_low)
                        )
                        q_high = st.sidebar.slider(
                            "上側分位点 (例: 0.99)",
                            0.6, 1.0,
                            float(st.session_state.q_high)
                        )
                        if q_low >= q_high:
                            st.sidebar.warning("下側分位点は上側分位点より小さくしてください。")
                        st.session_state.q_low = q_low
                        st.session_state.q_high = q_high
                else:
                    st.sidebar.info("ハズレ値処理する列を選択してください。")

# 分類タスクのときのクラス分割（ツリー系＆Polynomial 共通で使う）
if task_type == "分類":
    bin_choice = st.sidebar.selectbox(
        "クラス分割（目的変数をビン化）",
        ["そのまま使う", "二分位", "三分位", "四分位"]
    )
    bin_map = {"二分位": 2, "三分位": 3, "四分位": 4}
else:
    bin_choice = "そのまま使う"
    bin_map = {}

# モデル種類（ここに Polynomial と LightGBM を含む）
ml_type = st.sidebar.selectbox(
    "モデル",
    ["DecisionTree", "RandomForest", "SVM", "NN", "XGBoost", "LightGBM", "Polynomial"]
)

# Polynomial 用オプション
poly_model_type = None
poly_mode = None
poly_standardize = True
poly_test_size = 0.2
poly_random_state = 42

if ml_type == "Polynomial":
    poly_model_type = st.sidebar.selectbox(
        "多項式モデルの変数選択方法",
        ["Lasso（正則化）", "AIC（ステップワイズ）", "なし（全変数使用）"]
    )
    poly_mode = st.sidebar.selectbox(
        "多項式項のパターン（2次まで）",
        ["一次のみ", "一次＋交互作用のみ", "２次のみ", "２次＋交互作用のみ"]
    )
    poly_standardize = st.sidebar.checkbox(
        "多項式展開後に標準化する（StandardScaler）", value=True
    )
    poly_test_size = st.sidebar.slider(
        "テストデータの割合（単一ファイルのとき）", 0.1, 0.5, 0.2
    )
    poly_random_state = st.sidebar.number_input(
        "random_state（Polynomial用）", 0, 9999, 42
    )

else:
    # ツリーモデル系のハイパラ設定
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
    elif ml_type == "LightGBM":
        estimators = st.sidebar.slider("n_estimators", 50, 500, (100, 300))
        num_leaves = st.sidebar.slider("num_leaves", 10, 200, (31, 127))
        depth = st.sidebar.slider("max_depth", -1, 20, (-1, 10))
        lr = st.sidebar.slider("learning_rate", 0.01, 0.5, (0.05, 0.2))
        subsample = st.sidebar.slider("subsample", 0.5, 1.0, (0.8, 1.0))
        colsample = st.sidebar.slider("colsample_bytree", 0.5, 1.0, (0.8, 1.0))

        # max_depth は -1 を含む可能性があるのでちょっと工夫
        if depth[0] < 0:
            depth_list = [-1, 5, 10, 15]
        else:
            depth_list = list(range(depth[0], depth[1] + 1, 2))

        params = {
            "n_estimators": list(range(estimators[0], estimators[1] + 1, 50)),
            "num_leaves":   list(range(num_leaves[0], num_leaves[1] + 1, 20)),
            "max_depth":    depth_list,
            "learning_rate": [round(v, 3) for v in np.linspace(lr[0], lr[1], 3)],
            "subsample": [round(v, 2) for v in np.linspace(subsample[0], subsample[1], 3)],
            "colsample_bytree": [round(v, 2) for v in np.linspace(colsample[0], colsample[1], 3)],
        }

    # 分類のみ：オーバーサンプリング
    oversample_option = None
    if task_type == "分類":
        oversample_option = st.sidebar.selectbox("オーバーサンプリング", ["なし", "SMOTE", "Resample"])

    # ==== 交互作用ペアの追加設定（ツリーモデル側） ====
    add_interactions_flag = st.sidebar.checkbox(
        "説明変数の交互作用項を明示的に追加する",
        value=st.session_state.get("add_interactions", False)
    )
    st.session_state.add_interactions = add_interactions_flag

    if add_interactions_flag:
        if feature_cols is not None and len(feature_cols) > 0:
            st.sidebar.markdown("**交互作用にしたいペアを登録してください**")

            c1, c2 = st.sidebar.columns(2)
            with c1:
                var1 = st.selectbox(
                    "変数1", feature_cols,
                    key="inter_var1"
                )
            with c2:
                var2 = st.selectbox(
                    "変数2", feature_cols,
                    key="inter_var2"
                )

            if st.sidebar.button("このペアを追加", key="add_inter_pair_btn"):
                if var1 == var2:
                    st.sidebar.warning("同じ変数同士は交互作用にできません。")
                else:
                    pair = tuple(sorted([var1, var2]))
                    if pair not in st.session_state.inter_pairs:
                        st.session_state.inter_pairs.append(pair)
                    else:
                        st.sidebar.info("そのペアはすでに登録されています。")

            if st.session_state.inter_pairs:
                st.sidebar.markdown("**現在の交互作用ペア一覧**")
                for i, (p1, p2) in enumerate(st.session_state.inter_pairs, start=1):
                    st.sidebar.write(f"{i}. {p1} × {p2}")

                if st.sidebar.button("交互作用ペアをすべてクリア", key="clear_inter_pairs_btn"):
                    st.session_state.inter_pairs = []
            else:
                st.sidebar.info("まだ交互作用ペアが登録されていません。")
        else:
            st.sidebar.info("ファイル読み込み後に交互作用ペアを選択できます。")

    # CV設定
    n_splits = st.sidebar.slider("CV分割数（GroupKFold）", 2, 10, 5)
    random_state = st.sidebar.number_input("random_state", 0, 9999, 42)  # ここも text_input 同様に修正してね

# ============================================================
# ここから：ツリーモデル系（ml_type != "Polynomial"）
# ============================================================
if ml_type != "Polynomial" and st.button("クロスバリデーション実行"):
    # データチェック
    if mode == 'ランダム（単一ファイルから分割/CV）':
        if df is None:
            st.error("ファイルをアップロードしてください。")
            st.stop()
        X, Y, features = tr.dataset(df, target, removal)
        groups = df[group_col]
    else:
        if train_df is None:
            st.error("学習/評価ファイルをアップロードしてください。")
            st.stop()
        X, Y, features = tr.dataset(train_df, target, removal)
        groups = train_df[group_col]

    # ==== ハズレ値処理（説明変数側） ====
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

    # ==== log 変換（説明変数側） ====
    log_cols = st.session_state.get("log_cols", [])
    if log_cols:
        X = log_transform_X(X, log_cols)

    # 分類：連続目的変数を分位ラベリング
    if task_type == "分類":
        if bin_choice != "そのまま使う":
            n_bins = bin_map[bin_choice]
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
        else:
            Y = Y.copy()

    # ==== 交互作用ペアの反映 ====
    inter_pairs = st.session_state.get("inter_pairs", [])
    if st.session_state.get("add_interactions", False) and inter_pairs:
        X, features = add_interaction_features(X, inter_pairs)
    elif st.session_state.get("add_interactions", False) and not inter_pairs:
        st.warning("交互作用を有効にしましたが、ペアが登録されていません。交互作用項は追加されません。")

    # モデル生成
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
    elif ml_type == "LightGBM":
        model = LGBMClassifier(
            random_state=random_state,
            class_weight="balanced" if task_type == "分類" else None
        ) if task_type == "分類" else LGBMRegressor(
            random_state=random_state
        )
    else:
        st.error("未知のモデルタイプです。")
        st.stop()

    gkf = GroupKFold(n_splits=n_splits)

    # 集計器
    best_params_list = []
    fold_rows = []
    best_fold = None
    best_pack = None  # ("cls"/"reg", mdl, X_tr, X_te, Y_tr, Y_te, features, meta)

    if task_type == "分類":
        cm_sum = None
        labels_seen = set()
    else:
        y_tr_all, yhat_tr_all = [], []
        y_te_all, yhat_te_all = [], []

    # ===== CV ループ =====
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, Y, groups=groups), 1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        Y_tr, Y_te = Y.iloc[tr_idx], Y.iloc[te_idx]

        # 分類：オーバーサンプリング
        if task_type == "分類" and oversample_option != "なし":
            if oversample_option == "SMOTE":
                sm_ = SMOTE(random_state=random_state)
                X_tr, Y_tr = sm_.fit_resample(X_tr, Y_tr)
            elif oversample_option == "Resample":
                tmp = pd.concat([X_tr, Y_tr.rename("target")], axis=1)
                max_count = tmp["target"].value_counts().max()
                parts = [
                    resample(g, replace=True, n_samples=max_count, random_state=random_state)
                    for _, g in tmp.groupby("target")
                ]
                tmp_up = pd.concat(parts)
                Y_tr = tmp_up["target"]
                X_tr = tmp_up.drop(columns=["target"])

        # グリッドサーチ → 学習
        clf = tr.grid_search(model, X_tr, Y_tr, params)
        best_params = clf.best_params_
        best_params_list.append({"fold": fold, **best_params})

        mdl = model.set_params(**best_params)
        mdl.fit(X_tr, Y_tr)

        if task_type == "分類":
            # CV 時点の予測を保存
            pred_te = mdl.predict(X_te)

            acc  = accuracy_score(Y_te, pred_te)
            prec = precision_score(Y_te, pred_te, average="macro", zero_division=0)
            rec  = recall_score   (Y_te, pred_te, average="macro", zero_division=0)
            f1   = f1_score       (Y_te, pred_te, average="macro", zero_division=0)

            labels_seen.update(pd.Series(Y_te).unique().tolist())
            labs_seen_sorted = sorted(list(labels_seen))
            cm = confusion_matrix(Y_te, pred_te, labels=labs_seen_sorted)
            if cm_sum is None:
                cm_sum = cm
            else:
                if cm_sum.shape != cm.shape:
                    new_cm = np.zeros((len(labs_seen_sorted), len(labs_seen_sorted)), dtype=int)
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
                "X_tr": X_tr,
                "X_te": X_te,
                "Y_tr": Y_tr,
                "Y_te": Y_te,
                "cm": cm,
                "yhat_te": pred_te,  # ★ 予測を保存
            })

        else:  # 回帰
            pred_tr = mdl.predict(X_tr)
            pred_te = mdl.predict(X_te)

            rmse_tr = float(np.sqrt(mean_squared_error(Y_tr, pred_tr)))
            mae_tr  = float(mean_absolute_error(Y_tr, pred_tr))
            r2_tr   = float(r2_score(Y_tr, pred_tr))
            rmse_te = float(np.sqrt(mean_squared_error(Y_te, pred_te)))
            mae_te  = float(mean_absolute_error(Y_te, pred_te))
            r2_te   = float(r2_score(Y_te, pred_te))

            y_tr_all.extend(Y_tr.tolist());    yhat_tr_all.extend(pred_tr.tolist())
            y_te_all.extend(Y_te.tolist());    yhat_te_all.extend(pred_te.tolist())

            fold_rows.append({
                "fold": fold,
                "RMSE(train)": rmse_tr,
                "MAE(train)":  mae_tr,
                "R2(train)":   r2_tr,
                "RMSE(test)":  rmse_te,
                "MAE(test)":   mae_te,
                "R2(test)":    r2_te,
                "mdl": mdl,
                "X_tr": X_tr,
                "X_te": X_te,
                "Y_tr": Y_tr,
                "Y_te": Y_te,
                "yhat_tr": pred_tr,
                "yhat_te": pred_te,
            })

    # ===== CV結果をセッションに保存 =====
    if task_type == "分類":
        fold_df = pd.DataFrame(fold_rows)
        if fold_df.empty:
            st.error("CV で有効な fold が得られませんでした。")
            st.stop()

        df_scores = (
            fold_df
            .set_index("fold")[["Accuracy", "Precision(macro)", "Recall(macro)", "F1(macro)"]]
            .sort_index()
        )

        best_idx = fold_df["F1(macro)"].idxmax()
        best_row = fold_df.loc[best_idx]

        best_fold = int(best_row["fold"])
        best_pack = (
            "cls",
            best_row["mdl"],
            best_row["X_tr"],
            best_row["X_te"],
            best_row["Y_tr"],
            best_row["Y_te"],
            features,
            {
                "acc":   best_row["Accuracy"],
                "prec":  best_row["Precision(macro)"],
                "rec":   best_row["Recall(macro)"],
                "f1":    best_row["F1(macro)"],
                "cm":    best_row["cm"],
                "yhat_te": best_row["yhat_te"],  # ★ yhat_te を meta に保存
            },
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
        fold_df = pd.DataFrame(fold_rows)
        if fold_df.empty:
            st.error("CV で有効な fold が得られませんでした。")
            st.stop()

        df_scores_test = (
            fold_df
            .set_index("fold")[["RMSE(test)", "MAE(test)", "R2(test)"]]
            .sort_index()
        )
        df_scores_train = (
            fold_df
            .set_index("fold")[["RMSE(train)", "MAE(train)", "R2(train)"]]
            .sort_index()
        )

        st.session_state.cv_result = {
            "task_type": "回帰",
            "ml_type": ml_type,
            "df_scores_test": df_scores_test,
            "df_scores_train": df_scores_train,
            "y_tr_all": y_tr_all,
            "yhat_tr_all": yhat_tr_all,
            "y_te_all": y_te_all,
            "yhat_te_all": yhat_te_all,
        }

        best_idx = fold_df["R2(test)"].idxmax()
        best_row = fold_df.loc[best_idx]
        best_fold = int(best_row["fold"])
        best_pack = (
            "reg",
            best_row["mdl"],
            best_row["X_tr"],
            best_row["X_te"],
            best_row["Y_tr"],
            best_row["Y_te"],
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
            },
        )

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

# =========================
# CV結果の表示（ツリーモデル系のみ）
# =========================
if ml_type != "Polynomial":
    st.subheader("クロスバリデーション再集計")

    cv_res = st.session_state.get("cv_result", None)
    if cv_res is None:
        st.info("『クロスバリデーション実行』ボタンを押すと結果が表示されます。")
    else:
        if cv_res["task_type"] == "分類":
            df_scores = cv_res["df_scores"]
            st.dataframe(df_scores)
            st.write("平均 ± SD")
            st.write(df_scores.agg(["mean", "std"]))

            cm_sum = cv_res["cm_sum"]
            if cm_sum is not None:
                st.write("総和混同行列（全fold合算）")
                labs = cv_res["labels_seen"]
                cm_df = pd.DataFrame(
                    cm_sum,
                    index=[f"T{i}" for i in labs],
                    columns=[f"P{i}" for i in labs]
                )
                st.dataframe(cm_df)

            st.subheader("各Foldの最適ハイパーパラメータ（参考）")
            st.dataframe(pd.DataFrame(cv_res["best_params_list"]))

        else:
            df_scores_test = cv_res["df_scores_test"]
            df_scores_train = cv_res["df_scores_train"]

            st.write("Test 指標（平均 ± SD）")
            st.dataframe(df_scores_test)
            st.write(df_scores_test.agg(["mean", "std"]))

            st.write("Train 指標（平均 ± SD）")
            st.dataframe(df_scores_train)
            st.write(df_scores_train.agg(["mean", "std"]))

            y_tr_all = pd.Series(cv_res["y_tr_all"])
            yhat_tr_all = pd.Series(cv_res["yhat_tr_all"])
            y_te_all = pd.Series(cv_res["y_te_all"])
            yhat_te_all = pd.Series(cv_res["yhat_te_all"])

            if len(y_tr_all) > 0:
                r2_train_all = r2_score(y_tr_all, yhat_tr_all)
            else:
                r2_train_all = np.nan
            if len(y_te_all) > 0:
                r2_test_all  = r2_score(y_te_all, yhat_te_all)
            else:
                r2_test_all = np.nan

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(y_tr_all, yhat_tr_all, alpha=0.6, label="Train", marker="o", color="blue")
            ax.scatter(y_te_all, yhat_te_all, alpha=0.6, label="Test",  marker="^", color="red")
            lo = min(y_tr_all.min(), y_te_all.min(), yhat_tr_all.min(), yhat_te_all.min())
            hi = max(y_tr_all.max(), y_te_all.max(), yhat_tr_all.max(), yhat_te_all.max())
            ax.plot([lo, hi], [lo, hi], "--", color="gray")
            ax.set_xlabel("Actual value"); ax.set_ylabel("Predicted value")
            ax.set_title(f"Actual vs Predicted (All folds, {cv_res['ml_type']})")
            ax.text(
                0.05, 0.95,
                f"R² (Train): {r2_train_all:.3f}\nR² (Test): {r2_test_all:.3f}",
                transform=ax.transAxes, fontsize=12, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
            )
            ax.legend()
            st.pyplot(fig)

    # ベストFoldの結果
    st.subheader("ベストFoldの結果")

    if st.session_state.get("cv_ready", False) and st.session_state.get("cv_payload"):
        payload = st.session_state.cv_payload
        task_type_payload = payload["task_type"]
        ml_type_payload   = payload["ml_type"]
        best_fold         = payload["best_fold"]

        kind, mdl_b, X_tr_b, X_te_b, Y_tr_b, Y_te_b, feats_b, meta = payload["best_pack"]

        st.caption(f"ベストFold: fold={best_fold}")

        if kind == "cls":
            # CV 時の予測結果をそのまま使う
            yhat_te_b = meta["yhat_te"]

            labs = sorted(pd.Series(Y_te_b).unique().tolist())
            class_names = [f"C{i}" for i in labs]
            cm_raw = confusion_matrix(Y_te_b, yhat_te_b, labels=labs)
            cm_df = pd.DataFrame(
                cm_raw,
                index=[f"True_{c}" for c in class_names],
                columns=[f"Pred_{c}" for c in class_names]
            )

            # 指標を計算
            acc_b  = accuracy_score(Y_te_b, yhat_te_b)
            prec_b = precision_score(Y_te_b, yhat_te_b, average="macro", zero_division=0)
            rec_b  = recall_score   (Y_te_b, yhat_te_b, average="macro", zero_division=0)
            f1_b   = f1_score       (Y_te_b, yhat_te_b, average="macro", zero_division=0)

            st.write(
                f"Accuracy: {acc_b:.3f} / Precision(macro): {prec_b:.3f} "
                f"/ Recall(macro): {rec_b:.3f} / F1(macro): {f1_b:.3f}"
            )
            st.caption(f"Baseline(1/num_classes) ≈ {1/len(labs):.3f}")
            st.write("混同行列（Best fold, 件数）")
            st.dataframe(cm_df)

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

            rep = classification_report(
                Y_te_b, yhat_te_b, labels=labs, output_dict=True, zero_division=0
            )
            rep_df = pd.DataFrame(rep).T
            rep_df = rep_df.rename(index={str(k): class_names[i] for i, k in enumerate(labs)})
            st.write("クラス別 Precision/Recall/F1（Best fold）")
            st.dataframe(rep_df[['precision', 'recall', 'f1-score', 'support']])

        else:
            st.write(
                f"Train: R²={meta['r2_tr']:.3f}, RMSE={meta['rmse_tr']:.3f}, MAE={meta['mae_tr']:.3f}"
            )
            st.write(
                f"Test : R²={meta['r2_te']:.3f}, RMSE={meta['rmse_te']:.3f}, MAE={meta['mae_te']:.3f}"
            )
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(Y_tr_b, meta["yhat_tr"], alpha=0.6, label="Train", marker="o", color="blue")
            ax.scatter(Y_te_b, meta["yhat_te"], alpha=0.6, label="Test",  marker="^", color="red")
            lo = min(
                Y_tr_b.min(), Y_te_b.min(),
                np.min(meta["yhat_tr"]), np.min(meta["yhat_te"])
            )
            hi = max(
                Y_tr_b.max(), Y_te_b.max(),
                np.max(meta["yhat_tr"]), np.max(meta["yhat_te"])
            )
            ax.plot([lo, hi], [lo, hi], "--", color="gray")
            ax.set_xlabel("Actual value"); ax.set_ylabel("Predicted value")
            ax.set_title(f"Best Fold Actual vs Predicted ({ml_type_payload})")
            ax.text(
                0.05, 0.95,
                f"R² (Train): {meta['r2_tr']:.3f}\nR² (Test): {meta['r2_te']:.3f}",
                transform=ax.transAxes, fontsize=12, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
            )
            ax.legend()
            st.pyplot(fig)

        # 決定木可視化
        if ml_type_payload == "DecisionTree":
            st.subheader("決定木の構造（Best fold）")
            fig_dt, ax_dt = plt.subplots(figsize=(16, 10))
            if kind == "cls":
                classes = [str(c) for c in sorted(Y_tr_b.unique())]
                plot_tree(
                    mdl_b,
                    feature_names=feats_b,
                    class_names=classes,
                    filled=True,
                    rounded=True,
                    impurity=False,
                    ax=ax_dt,
                )
            else:
                plot_tree(
                    mdl_b,
                    feature_names=feats_b,
                    filled=True,
                    rounded=True,
                    impurity=False,
                    ax=ax_dt,
                )
            st.pyplot(fig_dt)

        # 特徴量重要度
        st.subheader("特徴量重要度（Best fold, 対応モデルのみ）")
        if hasattr(mdl_b, "feature_importances_"):
            try:
                fig_imp = tr.importance(mdl_b, feats_b)
                st.pyplot(fig_imp)
            except Exception:
                importances = pd.Series(
                    mdl_b.feature_importances_, index=feats_b
                ).sort_values(ascending=False).head(30)
                fig2, ax2 = plt.subplots(
                    figsize=(6, min(10, 0.3 * len(importances)))
                )
                importances.iloc[::-1].plot(kind="barh", ax=ax2)
                ax2.set_title("Top Feature Importances (Best fold)")
                st.pyplot(fig2)
        else:
            st.info("このモデルでは特徴量重要度を表示できません。")

    else:
        st.info("ベストFoldの結果は、CV実行後にここに表示されます。")

    # SHAP & LIME（Best fold）
    st.subheader("XAI（Best fold）")
    if st.session_state.get("cv_ready", False) and st.session_state.get("cv_payload"):
        payload = st.session_state.cv_payload
        task_type_payload = payload["task_type"]
        ml_type_payload   = payload["ml_type"]
        best_fold         = payload["best_fold"]

        kind, mdl_b, X_tr_b, X_te_b, Y_tr_b, Y_te_b, feats_b, meta = payload["best_pack"]

        X_bg = payload["bg"][feats_b]
        X_te = payload["X_te"][feats_b]

        xai.explain_shap(mdl_b, X_bg, X_te, task_type_payload, ml_type_payload)
        xai.explain_lime(mdl_b, X_bg, X_te, task_type_payload)
    else:
        st.info("上の『クロスバリデーション実行』でモデルを作成すると、ここにXAIが表示されます。")


# ここから：Polynomial モデル（ml_type == "Polynomial"）
# ============================================================

if ml_type == "Polynomial":
    st.markdown("---")
    st.subheader("多項式モデル（Lasso / AIC / 全変数）")

    run_poly = st.button("多項式モデルの学習・評価を実行")

    if run_poly:
        # データ存在チェック
        base_df = df if df is not None else train_df
        if base_df is None or target is None or feature_cols is None or len(feature_cols) == 0:
            st.error("データファイルと目的変数・説明変数を正しく指定してください。")
            st.stop()

        # train / test 分割
        try:
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
        except Exception as e:
            st.error(f"train/test 分割でエラー: {e}")
            st.stop()

        # ==== ハズレ値処理（train/test 共通しきい値） ====
        out_method = st.session_state.get("outlier_method", "しない")
        out_cols   = st.session_state.get("outlier_cols", [])
        if out_method != "しない" and out_cols:
            out_params = {
                "iqr_k":  st.session_state.get("iqr_k", 1.5),
                "sigma_k": st.session_state.get("sigma_k", 3.0),
                "q_low":  st.session_state.get("q_low", 0.01),
                "q_high": st.session_state.get("q_high", 0.99),
            }
            X_tr, X_te = apply_outlier_clip_train_test(
                X_tr, X_te, out_cols, out_method, out_params
            )

        # ==== log 変換 ====
        log_cols = st.session_state.get("log_cols", [])
        if log_cols:
            X_tr, X_te = log_transform_train_test(X_tr, X_te, log_cols)

        # 分類タスクなら、必要に応じてビン化
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

        # 多項式モード → PolynomialSelector の mode に変換
        if poly_mode == "一次のみ":
            poly_sel_mode = "1_only"
        elif poly_mode == "一次＋交互作用のみ":
            poly_sel_mode = "1_plus_interactions"
        elif poly_mode == "２次のみ":
            poly_sel_mode = "2_only"
        else:
            poly_sel_mode = "2_plus_interactions"

        # --------------------------------
        # 1) Lasso（正則化）パス
        # --------------------------------
        if poly_model_type.startswith("Lasso"):
            st.subheader("Lasso 多項式モデルの結果")

            poly_step = PolynomialSelector(mode=poly_sel_mode)

            steps = [("poly", poly_step)]
            if poly_standardize:
                steps.append(("scaler", StandardScaler()))

            if task_type == "回帰":
                steps.append(("model", LassoCV(cv=5, random_state=poly_random_state)))
            else:
                steps.append((
                    "model",
                    LogisticRegression(
                        penalty="l1",
                        solver="saga",
                        max_iter=5000,
                        random_state=poly_random_state,
                        class_weight="balanced"
                    )
                ))

            pipe = Pipeline(steps)
            pipe.fit(X_tr, y_tr)

            # 予測
            yhat_tr = pipe.predict(X_tr)
            yhat_te = pipe.predict(X_te)

            if task_type == "回帰":
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
                # 分類
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
            st.subheader("多項式項の係数（Lasso による自動選択）")
            st.write(f"切片（intercept）: {intercept:.4f}")
            st.dataframe(coef_df.head(50))

            # XAI
            st.subheader("XAI（SHAP / LIME：Lasso-Poly）")
            try:
                bg = X_tr if len(X_tr) <= 200 else X_tr.sample(200, random_state=42)
                xai.explain_shap(pipe, bg, X_te, task_type, "Poly-Lasso")
                xai.explain_lime(pipe, bg, X_te, task_type)
            except Exception as e:
                st.info(f"XAI 計算でエラーが発生しました: {e}")

        # --------------------------------
        # 2) AIC ステップワイズ（statsmodels）
        # --------------------------------
        elif poly_model_type.startswith("AIC"):
            st.subheader("AIC ステップワイズ多項式モデルの結果")

            # 明示的に多項式特徴を作成
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

            Z_tr = Z_tr_full[:, mask]
            Z_te = Z_te_full[:, mask]
            names = base_poly.get_feature_names_out(feature_cols)
            names = np.asarray(names)[mask]

            X_tr_poly_df = pd.DataFrame(Z_tr, columns=names, index=X_tr.index)
            X_te_poly_df = pd.DataFrame(Z_te, columns=names, index=X_te.index)

            # 必要なら標準化
            if poly_standardize:
                scaler = StandardScaler()
                X_tr_scaled = scaler.fit_transform(X_tr_poly_df)
                X_te_scaled = scaler.transform(X_te_poly_df)
                X_tr_poly_df = pd.DataFrame(X_tr_scaled, columns=names, index=X_tr.index)
                X_te_poly_df = pd.DataFrame(X_te_scaled, columns=names, index=X_te.index)

            if task_type == "回帰":
                model_aic, selected = stepwise_aic_ols(X_tr_poly_df, y_tr)
                st.write(f"選択された項の数: {len(selected)}")
                st.write("選択された項（一部）：")
                st.write(selected[:30])

                params = model_aic.params
                pvals = model_aic.pvalues
                coef_df = pd.DataFrame(
                    {"term": params.index, "coef": params.values, "p": pvals.values}
                ).sort_values("coef", key=np.abs, ascending=False)
                st.subheader("係数・p値（AIC 選択後 OLS）")
                st.dataframe(coef_df.head(50))

                X_tr_sel = sm.add_constant(X_tr_poly_df[selected])
                X_te_sel = sm.add_constant(X_te_poly_df[selected], has_constant="add")
                yhat_tr = model_aic.predict(X_tr_sel)
                yhat_te = model_aic.predict(X_te_sel)

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

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(y_tr, yhat_tr, alpha=0.6, label="Train", marker="o")
                ax.scatter(y_te, yhat_te, alpha=0.6, label="Test", marker="^")
                lo = min(y_tr.min(), y_te.min(), yhat_tr.min(), yhat_te.min())
                hi = max(y_tr.max(), y_te.max(), yhat_tr.max(), yhat_te.max())
                ax.plot([lo, hi], [lo, hi], "--")
                ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
                ax.setタイトル("Actual vs Predicted (AIC-Poly)")
                ax.legend()
                st.pyplot(fig)

            else:
                if pd.Series(y_tr).nunique() != 2:
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

                yhat_tr_cls = (yhat_tr_prob >= 0.5).astype(int)
                yhat_te_cls = (yhat_te_prob >= 0.5).astype(int)

                acc = accuracy_score(y_te, yhat_te_cls)
                prec = precision_score(y_te, yhat_te_cls, average="macro", zero_division=0)
                rec  = recall_score   (y_te, yhat_te_cls, average="macro", zero_division=0)
                f1   = f1_score       (y_te, yhat_te_cls, average="macro", zero_division=0)
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

            st.info("AIC モデルは statsmodels を使っているため、SHAP/LIME はここでは実行していません。")

        # --------------------------------
        # 3) 全変数使用（Full OLS / Logit）
        # --------------------------------
        else:
            st.subheader("全変数多項式モデルの結果")

            # 多項式特徴
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

            Z_tr = Z_tr_full[:, mask]
            Z_te = Z_te_full[:, mask]
            names = base_poly.get_feature_names_out(feature_cols)
            names = np.asarray(names)[mask]

            X_tr_poly_df = pd.DataFrame(Z_tr, columns=names, index=X_tr.index)
            X_te_poly_df = pd.DataFrame(Z_te, columns=names, index=X_te.index)

            # 標準化
            if poly_standardize:
                scaler = StandardScaler()
                X_tr_scaled = scaler.fit_transform(X_tr_poly_df)
                X_te_scaled = scaler.transform(X_te_poly_df)
                X_tr_poly_df = pd.DataFrame(X_tr_scaled, columns=names, index=X_tr.index)
                X_te_poly_df = pd.DataFrame(X_te_scaled, columns=names, index=X_te.index)

            if task_type == "回帰":
                X_tr_design = sm.add_constant(X_tr_poly_df)
                X_te_design = sm.add_constant(X_te_poly_df, has_constant="add")

                model = sm.OLS(y_tr, X_tr_design).fit()

                st.write(f"使用した項の数: {len(names)}")

                params = model.params
                pvals = model.pvalues
                coef_df = pd.DataFrame(
                    {"term": params.index, "coef": params.values, "p": pvals.values}
                ).sort_values("coef", key=np.abs, ascending=False)
                st.subheader("係数・p値（全項 OLS）")
                st.dataframe(coef_df.head(50))

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
                if pd.Series(y_tr).nunique() != 2:
                    st.error("全変数ロジットは 2 クラス分類のみ対応です。目的変数を 0/1 の2値にしてください。")
                    st.stop()

                X_tr_design = sm.add_constant(X_tr_poly_df)
                X_te_design = sm.add_constant(X_te_poly_df, has_constant="add")

                model = sm.Logit(y_tr, X_tr_design).fit(disp=0)

                st.write(f"使用した項の数: {len(names)}")

                params = model.params
                pvals = model.pvalues
                coef_df = pd.DataFrame(
                    {"term": params.index, "coef": params.values, "p": pvals.values}
                ).sort_values("coef", key=np.abs, ascending=False)
                st.subheader("係数・p値（全項 Logit）")
                st.dataframe(coef_df.head(50))

                yhat_tr_prob = model.predict(X_tr_design)
                yhat_te_prob = model.predict(X_te_design)

                yhat_tr_cls = (yhat_tr_prob >= 0.5).astype(int)
                yhat_te_cls = (yhat_te_prob >= 0.5).astype(int)

                acc = accuracy_score(y_te, yhat_te_cls)
                prec = precision_score(y_te, yhat_te_cls, average="macro", zero_division=0)
                rec  = recall_score   (y_te, yhat_te_cls, average="macro", zero_division=0)
                f1   = f1_score       (y_te, yhat_te_cls, average="macro", zero_division=0)
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

            st.info("全変数モデルも statsmodels を使っているため、SHAP/LIME はここでは実行していません。")
