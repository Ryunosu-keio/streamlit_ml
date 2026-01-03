import itertools
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import statsmodels.api as sm


# =========================
# 固定デフォルト（毎回ここだけ固定でOK）
#  - 相対パスなら「この .py があるフォルダ」起点で探す
# =========================
DEFAULT_DAY_PATH = r"C:\Users\naklab\Documents\kiyota\penstone\data_pupil\final_2025_bright_pupil\final_recalculated_pupil_bcss_with_roi_global_withoutNan_with_area_pupil_with_origfeats_reduced.xlsx"
DEFAULT_NIGHT_PATH = r"C:\Users\naklab\Documents\kiyota\penstone\data_pupil\final_2025_dark_pupil\darkfinal_recalculated_pupil_bcss_roi_global_area_pupil_with_origfeats_reduced.xlsx"


# =========================
# utilities: load
# =========================
def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, path)


def load_any_path(path: str) -> pd.DataFrame:
    p = _resolve_path(path)
    pl = p.lower()
    if pl.endswith((".xls", ".xlsx")):
        return pd.read_excel(p)
    # csv: try common encodings
    try:
        return pd.read_csv(p, encoding="utf-8-sig")
    except Exception:
        try:
            return pd.read_csv(p, encoding="cp932")
        except Exception:
            return pd.read_csv(p)  # last resort


def _safe_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


# =========================
# Column rules (prefix logic)
# =========================
def pick_candidate_cols(num_cols, feat_group, y_col, non_feature_cols):
    non_feature_cols = set(non_feature_cols or [])
    if feat_group == "all":
        return [c for c in num_cols
                if c.startswith("all_")
                and not c.startswith("all_area_")
                and not c.startswith("all_pupil_")
                and not c.endswith("_orig")
                and c not in non_feature_cols
                and c != y_col]
    if feat_group == "all_area":
        return [c for c in num_cols
                if c.startswith("all_area_")
                and not c.endswith("_orig")
                and c not in non_feature_cols
                and c != y_col]
    if feat_group == "all_pupil":
        return [c for c in num_cols
                if c.startswith("all_pupil_")
                and not c.endswith("_orig")
                and c not in non_feature_cols
                and c != y_col]
    # ROI
    return [c for c in num_cols
            if (c.startswith("center_") or c.startswith("parafovea_") or c.startswith("periphery_"))
            and "_orig" not in c
            and c not in non_feature_cols
            and c != y_col]


def _suggest_by_keywords(cols, keywords, k=80):
    cols_l = [(c, c.lower()) for c in cols]
    hits = []
    for c, cl in cols_l:
        score = sum(1 for kw in keywords if kw.lower() in cl)
        if score > 0:
            hits.append((score, c))
    hits.sort(key=lambda x: (-x[0], x[1]))
    return [c for _, c in hits][:k]


def make_pools(cand_cols):
    # brightness: prefer bL (exclude bnL), but keep fallback
    bright_pool = [c for c in cand_cols if ("bnl" not in c.lower()) and ("bl" in c.lower())]
    if not bright_pool:
        bright_pool = _suggest_by_keywords(cand_cols, ["bl", "bnl", "luma", "lumin", "mean", "y"], k=80)

    cont_pool = _suggest_by_keywords(cand_cols, [
        "contrast", "rms_contrast", "weber", "michelson", "glcm_contrast", "std", "rms"
    ], k=80)

    sharp_pool = _suggest_by_keywords(cand_cols, [
        "sharp", "lap", "tenengrad", "acut", "hf", "highfreq", "grad_entropy", "edge"
    ], k=80)

    def uniq(xs):
        return list(dict.fromkeys([x for x in xs if x in cand_cols]))

    return uniq(bright_pool), uniq(cont_pool), uniq(sharp_pool)


# =========================
# CV split
# =========================
def make_splits(index, mode, groups=None, test_size=0.2, random_state=42, n_splits=5):
    idx = np.asarray(index)

    if mode == "in-sample":
        return [(idx, idx)]

    if mode == "holdout":
        if groups is None:
            rs = np.random.RandomState(random_state)
            perm = rs.permutation(len(idx))
            n_te = int(np.ceil(len(idx) * test_size))
            te = perm[:n_te]
            tr = perm[n_te:]
            return [(idx[tr], idx[te])]
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        g = pd.Series(groups).loc[idx]
        tr, te = next(gss.split(np.zeros((len(idx), 1)), np.zeros(len(idx)), groups=g))
        return [(idx[tr], idx[te])]

    # LOSO / LOIO
    if groups is None:
        rs = np.random.RandomState(random_state)
        perm = rs.permutation(len(idx))
        folds = np.array_split(perm, min(n_splits, len(idx)))
        out = []
        for k in range(len(folds)):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(len(folds)) if j != k]) if len(folds) > 1 else folds[k]
            out.append((idx[tr], idx[te]))
        return out

    g = pd.Series(groups).loc[idx]
    n_g = len(g.unique())
    n_use = min(n_splits, n_g)
    gkf = GroupKFold(n_splits=n_use)
    out = []
    for tr, te in gkf.split(np.zeros((len(idx), 1)), np.zeros(len(idx)), groups=g):
        out.append((idx[tr], idx[te]))
    return out


def scale_train_test(X, tr_idx, te_idx):
    Xtr = X.loc[tr_idx].copy()
    Xte = X.loc[te_idx].copy()
    sc = StandardScaler()
    Xtr_s = pd.DataFrame(sc.fit_transform(Xtr), index=Xtr.index, columns=Xtr.columns)
    Xte_s = pd.DataFrame(sc.transform(Xte), index=Xte.index, columns=Xte.columns)
    return Xtr_s, Xte_s


# =========================
# Feature variants (raw/log/sq/sqrt)
# =========================
def signed_log1p(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def signed_sqrt(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.sqrt(np.abs(x))


def build_feature_library(Xraw: pd.DataFrame, bases, add_log: bool, add_sq: bool, add_sqrt: bool):
    """
    bases(raw col list) -> Xlib with raw (+ log__ + sq__ + sqrt__)
    base_map[feature_name] = base_col_name (raw base)
    """
    Xraw = _safe_df(Xraw[bases])

    blocks = []
    base_map = {}

    # raw
    blocks.append(Xraw)
    for c in Xraw.columns:
        base_map[c] = c

    # log
    if add_log:
        Xlog = pd.DataFrame({f"log__{c}": signed_log1p(Xraw[c].values) for c in Xraw.columns}, index=Xraw.index)
        blocks.append(Xlog)
        for c in Xraw.columns:
            base_map[f"log__{c}"] = c

    # square
    if add_sq:
        Xsq = pd.DataFrame({f"sq__{c}": (Xraw[c].values ** 2) for c in Xraw.columns}, index=Xraw.index)
        blocks.append(Xsq)
        for c in Xraw.columns:
            base_map[f"sq__{c}"] = c

    # sqrt
    if add_sqrt:
        Xrt = pd.DataFrame({f"sqrt__{c}": signed_sqrt(Xraw[c].values) for c in Xraw.columns}, index=Xraw.index)
        blocks.append(Xrt)
        for c in Xraw.columns:
            base_map[f"sqrt__{c}"] = c

    Xlib = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=Xraw.index)
    Xlib = Xlib.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    Xlib = Xlib.loc[:, ~Xlib.columns.duplicated()]
    return Xlib, base_map


def corr_rank_features(X: pd.DataFrame, y: pd.Series, feats, top_k: int):
    feats = [f for f in feats if f in X.columns]
    if len(feats) == 0:
        return []
    yy = y.astype(float).values
    out = []
    for f in feats:
        xx = X[f].astype(float).values
        if np.nanstd(xx) < 1e-12:
            continue
        c = np.corrcoef(xx, yy)[0, 1]
        if np.isnan(c):
            continue
        out.append((abs(c), f))
    out.sort(key=lambda x: (-x[0], x[1]))
    return [f for _, f in out[:top_k]]


# =========================
# OLS fit & coef/p extraction
# =========================
def fit_ols(X_s: pd.DataFrame, y: pd.Series, robust_hc3: bool):
    Xc = sm.add_constant(X_s, has_constant="add")
    res = sm.OLS(y.values.astype(float), Xc.values.astype(float)).fit()
    if robust_hc3:
        res = res.get_robustcov_results(cov_type="HC3")
    return res


def coef_p_table(res, feature_names):
    cols = ["const"] + list(feature_names)
    return pd.DataFrame({
        "coef": pd.Series(res.params, index=cols),
        "p": pd.Series(res.pvalues, index=cols),
    })


# =========================
# Interaction term (ONLY c*s) as candidate
# =========================
def interaction_name(c_feat: str, s_feat: str) -> str:
    return f"int__({c_feat})x({s_feat})"


def add_cs_interaction(Xtr_s: pd.DataFrame, Xte_s: pd.DataFrame, c_feat: str, s_feat: str):
    name = interaction_name(c_feat, s_feat)
    Xtr = Xtr_s.copy()
    Xte = Xte_s.copy()
    Xtr[name] = Xtr[c_feat] * Xtr[s_feat]
    Xte[name] = Xte[c_feat] * Xte[s_feat]
    return Xtr, Xte, name


# =========================
# Evaluate A and B with CV R2
# =========================
def cv_r2_single_feature(Xlib, y, splits, feat, robust_hc3: bool):
    r2s = []
    fail = 0
    for tr_idx, te_idx in splits:
        ytr = y.loc[tr_idx].astype(float)
        yte = y.loc[te_idx].astype(float)

        Xtr_s, Xte_s = scale_train_test(Xlib[[feat]], tr_idx, te_idx)
        try:
            res = fit_ols(Xtr_s, ytr, robust_hc3)
            Xte_c = sm.add_constant(Xte_s, has_constant="add").values
            yhat = pd.Series(res.predict(Xte_c), index=Xte_s.index)
            r2 = float(r2_score(yte, yhat)) if len(yte) > 1 else np.nan
            if np.isfinite(r2):
                r2s.append(r2)
        except Exception:
            fail += 1
            continue
    if len(r2s) == 0:
        return np.nan, fail
    return float(np.mean(r2s)), fail


def cv_r2_B_model(Xlib, y, splits, b_feat, c_feat, s_feat,
                  use_interaction: bool, robust_hc3: bool):
    feats = [b_feat, c_feat, s_feat]
    r2s = []
    fail = 0
    for tr_idx, te_idx in splits:
        ytr = y.loc[tr_idx].astype(float)
        yte = y.loc[te_idx].astype(float)

        Xtr_s, Xte_s = scale_train_test(Xlib[feats], tr_idx, te_idx)
        if use_interaction:
            Xtr_s, Xte_s, _ = add_cs_interaction(Xtr_s, Xte_s, c_feat, s_feat)

        try:
            res = fit_ols(Xtr_s, ytr, robust_hc3)
            Xte_c = sm.add_constant(Xte_s, has_constant="add").values
            yhat = pd.Series(res.predict(Xte_c), index=Xte_s.index)
            r2 = float(r2_score(yte, yhat)) if len(yte) > 1 else np.nan
            if np.isfinite(r2):
                r2s.append(r2)
        except Exception:
            fail += 1
            continue

    if len(r2s) == 0:
        return np.nan, fail
    return float(np.mean(r2s)), fail


def insample_fit_and_table(Xlib, y, feats, robust_hc3: bool,
                           use_interaction: bool = False, c_feat: str = None, s_feat: str = None):
    X_s, _ = scale_train_test(Xlib[list(feats)], Xlib.index, Xlib.index)

    feats_all = list(feats)
    int_name = None
    if use_interaction:
        if c_feat is None or s_feat is None:
            raise ValueError("use_interaction=True needs c_feat and s_feat")
        X_s, _, int_name = add_cs_interaction(X_s, X_s, c_feat, s_feat)
        feats_all = feats_all + [int_name]

    res = fit_ols(X_s, y.astype(float), robust_hc3)
    tab = coef_p_table(res, feats_all).set_index(pd.Index(["const"] + feats_all))
    return res, tab, int_name


# =========================
# Acceptance rule
#  - R2_B > R2_A
#  - brightness coef < 0
#  - sharpness coef < 0
#  - |brightness| > |sharpness|
#  - at least 2 terms have p < alpha
#    - if include_int_p_in_rule: count interaction too (when used)
# =========================
def acceptance_check(tab_B, r2A, r2B,
                     b_feat, c_feat, s_feat,
                     use_interaction: bool,
                     int_feat_name: str,
                     alpha=0.10,
                     include_int_p_in_rule: bool = False):

    if not (np.isfinite(r2A) and np.isfinite(r2B) and (r2B > r2A)):
        return False, "R2_B <= R2_A"

    for f in [b_feat, c_feat, s_feat]:
        if f not in tab_B.index:
            return False, f"missing term: {f}"

    bcoef = float(tab_B.loc[b_feat, "coef"])
    ccoef = float(tab_B.loc[c_feat, "coef"])
    scoef = float(tab_B.loc[s_feat, "coef"])

    if not (bcoef < 0):
        return False, "brightness coef not negative"
    if not (scoef < 0):
        return False, "sharpness coef not negative"
    if not (abs(bcoef) > abs(ccoef)):
        return False, "|b| <= |s|"

    terms_for_p = [b_feat, c_feat, s_feat]
    if use_interaction and include_int_p_in_rule:
        if (int_feat_name is None) or (int_feat_name not in tab_B.index):
            return False, "interaction term missing"
        terms_for_p = terms_for_p + [int_feat_name]

    pvals = [float(tab_B.loc[t, "p"]) for t in terms_for_p]
    if sum([p < alpha for p in pvals]) < 2:
        return False, "sig terms < 2 (p<alpha)"

    return True, ""


# =========================
# Core search per dataset x feature_group
# =========================
def solve_one(df, y_col, feat_group, non_feature_cols,
              mode, group_subject_col, group_image_col,
              test_size, n_splits, seed,
              add_log, add_sq, add_sqrt,
              allow_interaction_candidate,
              include_int_p_in_rule,
              robust_hc3,
              alpha, top_k_each):

    out = {
        "status": "ok",
        "reason": "",
        "feat_group": feat_group,

        "A_best_feat": "",
        "A_R2": np.nan,
        "A_const_coef": np.nan,
        "A_const_p": np.nan,
        "A_coef": np.nan,
        "A_p": np.nan,

        "B_best_feats": "",
        "B_R2": np.nan,
        "Delta(B-A)": np.nan,

        "B_const_coef": np.nan,
        "B_const_p": np.nan,

        "B_use_interaction": False,
        "B_int_feat": "",
        "B_int_coef": np.nan,
        "B_int_p": np.nan,

        "accepted": False,
        "accept_reason": "",

        "B_b_feat": "",
        "B_c_feat": "",
        "B_s_feat": "",
        "B_b_coef": np.nan, "B_b_p": np.nan,
        "B_c_coef": np.nan, "B_c_p": np.nan,
        "B_s_coef": np.nan, "B_s_p": np.nan,

        "B_terms_sigcount": np.nan,
        "B_terms_sigcount_rule": np.nan,

        # debug
        "n_bright_base": 0, "n_cont_base": 0, "n_sharp_base": 0,
        "cv_fail_A": 0, "cv_fail_B": 0,
        "n_B_tested": 0,
        "n_B_cv_valid": 0,
        "n_B_accepted": 0,
    }

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if y_col not in num_cols:
        out["status"] = "fail"
        out["reason"] = "y not numeric or missing"
        return out

    y = pd.to_numeric(df[y_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    cand = pick_candidate_cols(num_cols, feat_group, y_col, non_feature_cols)
    if len(cand) == 0:
        out["status"] = "fail"
        out["reason"] = "no candidate cols by prefix rules"
        return out

    Xraw_all = _safe_df(df[cand])

    bright_bases, cont_bases, sharp_bases = make_pools(cand)
    bright_bases = [c for c in bright_bases if c in Xraw_all.columns]
    cont_bases = [c for c in cont_bases if c in Xraw_all.columns]
    sharp_bases = [c for c in sharp_bases if c in Xraw_all.columns]

    out["n_bright_base"] = len(bright_bases)
    out["n_cont_base"] = len(cont_bases)
    out["n_sharp_base"] = len(sharp_bases)

    if len(bright_bases) == 0:
        out["status"] = "fail"
        out["reason"] = "no brightness base cols"
        return out

    # groups for split
    if mode == "LOSO":
        groups = None if group_subject_col is None else df[group_subject_col]
    elif mode == "LOIO":
        groups = None if group_image_col is None else df[group_image_col]
    elif mode == "holdout":
        groups = None if group_subject_col is None else df[group_subject_col]
    else:
        groups = None

    splits = make_splits(df.index, mode, groups=groups, test_size=test_size, random_state=seed, n_splits=n_splits)

    # ----- Build libs
    Xlib_A, _ = build_feature_library(Xraw_all, bright_bases, add_log=add_log, add_sq=add_sq, add_sqrt=add_sqrt)
    bases_B = list(dict.fromkeys(bright_bases + cont_bases + sharp_bases))
    Xlib_B, _ = build_feature_library(Xraw_all, bases_B, add_log=add_log, add_sq=add_sq, add_sqrt=add_sqrt)

    def variants_of(base):
        feats = [base]
        if add_log:
            feats.append(f"log__{base}")
        if add_sq:
            feats.append(f"sq__{base}")
        if add_sqrt:
            feats.append(f"sqrt__{base}")
        return feats

    def base_of(f):
        for pfx in ("log__", "sq__", "sqrt__"):
            if f.startswith(pfx):
                return f[len(pfx):]
        return f

    # ----- Choose best A among brightness single-term
    A_feats = []
    for b in bright_bases:
        A_feats += variants_of(b)
    A_feats = [f for f in A_feats if f in Xlib_A.columns]

    A_pref = corr_rank_features(Xlib_A, y, A_feats, top_k=max(20, min(80, len(A_feats))))
    bestA_feat, bestA_r2 = None, -np.inf
    failA_total = 0
    for f in A_pref:
        r2, failA = cv_r2_single_feature(Xlib_A, y, splits, f, robust_hc3)
        failA_total += failA
        if np.isfinite(r2) and r2 > bestA_r2:
            bestA_r2 = r2
            bestA_feat = f

    out["cv_fail_A"] = int(failA_total)

    if bestA_feat is None or (not np.isfinite(bestA_r2)):
        out["status"] = "fail"
        out["reason"] = "A model fit failed for all candidates"
        return out

    out["A_best_feat"] = bestA_feat
    out["A_R2"] = float(bestA_r2)

    # in-sample coef/p for A (incl intercept)
    try:
        _, tabA, _ = insample_fit_and_table(
            Xlib_A, y, [bestA_feat],
            robust_hc3, use_interaction=False
        )
        out["A_const_coef"] = float(tabA.loc["const", "coef"])
        out["A_const_p"] = float(tabA.loc["const", "p"])
        out["A_coef"] = float(tabA.loc[bestA_feat, "coef"])
        out["A_p"] = float(tabA.loc[bestA_feat, "p"])
    except Exception:
        pass

    # ----- If B cannot be formed, end here
    if len(cont_bases) == 0 or len(sharp_bases) == 0:
        out["status"] = "ok"
        out["accepted"] = False
        out["accept_reason"] = "B impossible (no contrast/sharp bases)"
        return out

    # Build candidate feature lists for each group (variants)
    B_bright_feats = []
    for b in bright_bases:
        B_bright_feats += variants_of(b)
    B_bright_feats = [f for f in B_bright_feats if f in Xlib_B.columns]

    B_cont_feats = []
    for c in cont_bases:
        B_cont_feats += variants_of(c)
    B_cont_feats = [f for f in B_cont_feats if f in Xlib_B.columns]

    B_sharp_feats = []
    for s in sharp_bases:
        B_sharp_feats += variants_of(s)
    B_sharp_feats = [f for f in B_sharp_feats if f in Xlib_B.columns]

    # Narrow each group by corr
    B_bright_top = corr_rank_features(Xlib_B, y, B_bright_feats, top_k=top_k_each)
    B_cont_top = corr_rank_features(Xlib_B, y, B_cont_feats, top_k=top_k_each)
    B_sharp_top = corr_rank_features(Xlib_B, y, B_sharp_feats, top_k=top_k_each)

    if len(B_bright_top) == 0 or len(B_cont_top) == 0 or len(B_sharp_top) == 0:
        out["accepted"] = False
        out["accept_reason"] = "B candidate lists empty after filtering"
        return out

    interaction_options = [False, True] if allow_interaction_candidate else [False]

    best_any = None
    best_acc = None
    failB_total = 0
    tested = 0
    cv_valid = 0
    acc_count = 0

    for b_feat, c_feat, s_feat in itertools.product(B_bright_top, B_cont_top, B_sharp_top):
        # 同一baseの raw/log/sq/sqrt を同時採用しない
        bb, cb, sb = base_of(b_feat), base_of(c_feat), base_of(s_feat)
        if len({bb, cb, sb}) < 3:
            continue

        for use_int in interaction_options:
            tested += 1

            r2B, failB = cv_r2_B_model(
                Xlib_B, y, splits,
                b_feat=b_feat, c_feat=c_feat, s_feat=s_feat,
                use_interaction=use_int,
                robust_hc3=robust_hc3
            )
            failB_total += failB
            if not np.isfinite(r2B):
                continue
            cv_valid += 1

            try:
                _, tabB, int_name = insample_fit_and_table(
                    Xlib_B, y, [b_feat, c_feat, s_feat],
                    robust_hc3,
                    use_interaction=use_int, c_feat=c_feat, s_feat=s_feat
                )
            except Exception:
                continue

            p_b = float(tabB.loc[b_feat, "p"])
            p_c = float(tabB.loc[c_feat, "p"])
            p_s = float(tabB.loc[s_feat, "p"])
            sigcnt_3 = int(sum([p_b < alpha, p_c < alpha, p_s < alpha]))

            p_int = np.nan
            sigcnt_rule = sigcnt_3
            if use_int and int_name is not None and int_name in tabB.index:
                p_int = float(tabB.loc[int_name, "p"])
                if include_int_p_in_rule:
                    sigcnt_rule = int(sum([p_b < alpha, p_c < alpha, p_s < alpha, p_int < alpha]))

            ok, reason = acceptance_check(
                tab_B=tabB,
                r2A=float(out["A_R2"]),
                r2B=float(r2B),
                b_feat=b_feat, c_feat=c_feat, s_feat=s_feat,
                use_interaction=use_int,
                int_feat_name=int_name,
                alpha=alpha,
                include_int_p_in_rule=include_int_p_in_rule
            )

            if ok:
                acc_count += 1

            rec = {
                "b_feat": b_feat, "c_feat": c_feat, "s_feat": s_feat,
                "use_int": use_int,
                "int_name": int_name,
                "R2_B": float(r2B),
                "sigcnt_3": sigcnt_3,
                "sigcnt_rule": sigcnt_rule,
                "p_int": p_int,
                "ok": ok,
                "reason": reason,
                "tabB": tabB,
            }

            if (best_any is None) or (rec["R2_B"] > best_any["R2_B"]):
                best_any = rec
            if ok:
                if (best_acc is None) or (rec["R2_B"] > best_acc["R2_B"]):
                    best_acc = rec

    out["cv_fail_B"] = int(failB_total)
    out["n_B_tested"] = int(tested)
    out["n_B_cv_valid"] = int(cv_valid)
    out["n_B_accepted"] = int(acc_count)

    chosen = best_acc if best_acc is not None else best_any
    if chosen is None:
        out["accepted"] = False
        out["accept_reason"] = "B all failed (fit/cv)"
        return out

    tabB = chosen["tabB"]
    b_feat, c_feat, s_feat = chosen["b_feat"], chosen["c_feat"], chosen["s_feat"]
    use_int = bool(chosen["use_int"])
    int_name = chosen["int_name"]

    out["B_use_interaction"] = use_int
    out["B_int_feat"] = int_name if (use_int and int_name is not None) else ""

    # intercept
    if "const" in tabB.index:
        out["B_const_coef"] = float(tabB.loc["const", "coef"])
        out["B_const_p"] = float(tabB.loc["const", "p"])

    if use_int and int_name is not None and int_name in tabB.index:
        out["B_int_coef"] = float(tabB.loc[int_name, "coef"])
        out["B_int_p"] = float(tabB.loc[int_name, "p"])

    out["B_best_feats"] = f"{b_feat}, {c_feat}, {s_feat}" + (f", {int_name}" if (use_int and int_name) else "")
    out["B_R2"] = float(chosen["R2_B"])
    out["Delta(B-A)"] = float(out["B_R2"] - out["A_R2"])

    out["B_b_feat"] = b_feat
    out["B_c_feat"] = c_feat
    out["B_s_feat"] = s_feat

    out["B_b_coef"] = float(tabB.loc[b_feat, "coef"])
    out["B_b_p"] = float(tabB.loc[b_feat, "p"])
    out["B_c_coef"] = float(tabB.loc[c_feat, "coef"])
    out["B_c_p"] = float(tabB.loc[c_feat, "p"])
    out["B_s_coef"] = float(tabB.loc[s_feat, "coef"])
    out["B_s_p"] = float(tabB.loc[s_feat, "p"])

    out["B_terms_sigcount"] = int(sum([
        out["B_b_p"] < alpha,
        out["B_c_p"] < alpha,
        out["B_s_p"] < alpha,
    ]))

    if use_int and np.isfinite(out["B_int_p"]) and include_int_p_in_rule:
        out["B_terms_sigcount_rule"] = int(sum([
            out["B_b_p"] < alpha,
            out["B_c_p"] < alpha,
            out["B_s_p"] < alpha,
            out["B_int_p"] < alpha,
        ]))
    else:
        out["B_terms_sigcount_rule"] = out["B_terms_sigcount"]

    if best_acc is not None:
        out["accepted"] = True
        out["accept_reason"] = ""
    else:
        out["accepted"] = False
        out["accept_reason"] = f"NO accepted model => fallback best-R2 (fail reason example: {chosen['reason']})"

    return out


# =========================
# パワポ用整形（日本語名・式・p値）
# =========================
ROI_MAP = {"center": "中心", "parafovea": "傍心", "periphery": "周辺"}
FEAT_MAP = {
    "bL_mean": "平均輝度",
    "c_rms_contrast": "局所コントラスト",
    "sh_grad_entropy": "勾配エントロピー",
}
DROP_PREFIXES = ("all_pupil_", "all_area_", "all_ROI_", "all_")

def _strip_group_prefix(s: str) -> str:
    s = str(s)
    for p in DROP_PREFIXES:
        if s.startswith(p):
            return s[len(p):]
    return s

def _parse_transform(feat: str):
    feat = str(feat)
    for t, pfx in [("log", "log__"), ("sqrt", "sqrt__"), ("sq", "sq__")]:
        if feat.startswith(pfx):
            return t, feat[len(pfx):]
    return "raw", feat

def _jp_base(base: str) -> str:
    base = _strip_group_prefix(base)
    for roi_key, roi_jp in ROI_MAP.items():
        if base.startswith(roi_key + "_"):
            core = base[len(roi_key) + 1:]
            return f"{roi_jp}{FEAT_MAP.get(core, core)}"
    return FEAT_MAP.get(base, base)

def jp_feat(feat: str) -> str:
    if feat is None or str(feat).strip() == "":
        return ""
    t, base = _parse_transform(str(feat))
    b = _jp_base(base)
    if t == "log":
        return f"log({b})"
    if t == "sqrt":
        return f"√({b})"
    if t == "sq":
        return f"{b}^2"
    return b

def jp_interaction(int_feat: str) -> str:
    s = "" if int_feat is None else str(int_feat).strip()
    if s == "":
        return ""
    m = re.match(r"^int__\((.+)\)x\((.+)\)$", s)
    if m:
        return f"({jp_feat(m.group(1))}×{jp_feat(m.group(2))})"
    return jp_feat(s)

def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    p = float(p)
    if p == 0.0:
        return "≈0"
    if p < 1e-3:
        return f"{p:.3e}"
    return f"{p:.4f}"

def fmt_coef(x):
    x = float(x)
    return f"{x:+.6f}"

def build_A_expr(row):
    const = row.get("A_const_coef", np.nan)
    acoef = row.get("A_coef", np.nan)
    feat  = row.get("A_best_feat", "")
    if not np.isfinite(const) or not np.isfinite(acoef):
        return ""
    return f"{const:.6f} {fmt_coef(acoef)}×{jp_feat(feat)}"

def build_B_expr(row):
    const = row.get("B_const_coef", np.nan)
    if not np.isfinite(const):
        return ""
    parts = [f"{const:.6f}"]
    parts.append(f"{fmt_coef(row['B_b_coef'])}×{jp_feat(row['B_b_feat'])}")
    parts.append(f"{fmt_coef(row['B_c_coef'])}×{jp_feat(row['B_c_feat'])}")
    parts.append(f"{fmt_coef(row['B_s_coef'])}×{jp_feat(row['B_s_feat'])}")

    use_int = bool(row.get("B_use_interaction", False))
    int_feat = row.get("B_int_feat", "")
    if use_int and str(int_feat).strip() != "" and np.isfinite(row.get("B_int_coef", np.nan)):
        parts.append(f"{fmt_coef(row['B_int_coef'])}×{jp_interaction(int_feat)}")
    return " ".join(parts)

def build_B_p_summary(row):
    p_b = fmt_p(row.get("B_b_p", np.nan))
    p_c = fmt_p(row.get("B_c_p", np.nan))
    p_s = fmt_p(row.get("B_s_p", np.nan))

    use_int = bool(row.get("B_use_interaction", False))
    int_feat = str(row.get("B_int_feat", "")).strip()
    if use_int and int_feat != "":
        p_i = fmt_p(row.get("B_int_p", np.nan))
        return f"輝度={p_b}, コントラスト={p_c}, シャープネス={p_s}, 交互作用={p_i}"
    return f"輝度={p_b}, コントラスト={p_c}, シャープネス={p_s}"


# =========================
# 保存ファイル名（ラジオボタン/チェックに準拠）
# =========================
def _slug(s: str) -> str:
    s = str(s)
    s = s.replace(" ", "")
    s = re.sub(r"[\\/:*?\"<>|]", "_", s)
    s = re.sub(r"[^0-9A-Za-zぁ-んァ-ヶ一-龠ー_+\-\.]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def make_result_filename(prefix: str,
                         y_col: str,
                         mode: str,
                         alpha: float,
                         add_log: bool,
                         add_sq: bool,
                         add_sqrt: bool,
                         allow_int: bool,
                         include_int_p: bool,
                         robust_hc3: bool,
                         top_k_each: int,
                         seed: int) -> str:
    feats = []
    if add_log: feats.append("log")
    if add_sq: feats.append("sq")
    if add_sqrt: feats.append("sqrt")
    feat_tag = "raw" if len(feats) == 0 else "+".join(feats)

    int_tag = "intCS" if allow_int else "noInt"
    intp_tag = "intP" if include_int_p else "noIntP"
    hc_tag = "HC3" if robust_hc3 else "OLS"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{prefix}__y={_slug(y_col)}__mode={mode}__a={alpha}__{feat_tag}__{int_tag}__{intp_tag}__{hc_tag}__K={top_k_each}__seed={seed}__{ts}.csv"
    return name


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Day/Night Regression (A vs B)", layout="wide")
st.title("🌞🌙 昼/夜：回帰（A=輝度1項） vs（B=輝度+コントラスト+シャープネス（+ 交互作用候補））")

day_path = _resolve_path(DEFAULT_DAY_PATH)
night_path = _resolve_path(DEFAULT_NIGHT_PATH)

st.caption(f"固定読み込み: day={day_path} / night={night_path}")

if not os.path.exists(day_path) or not os.path.exists(night_path):
    st.error("デフォルトファイルが見つかりません。DEFAULT_DAY_PATH / DEFAULT_NIGHT_PATH を確認して。")
    st.stop()

df_day = load_any_path(DEFAULT_DAY_PATH)
df_night = load_any_path(DEFAULT_NIGHT_PATH)

st.write("昼 shape:", df_day.shape, " / 夜 shape:", df_night.shape)

common_cols = sorted(list(set(df_day.columns).intersection(set(df_night.columns))))
num_day = set(df_day.select_dtypes(include=[np.number]).columns.tolist())
num_night = set(df_night.select_dtypes(include=[np.number]).columns.tolist())
common_num = sorted(list(num_day.intersection(num_night)))
if len(common_num) == 0:
    st.error("昼と夜で共通の数値列がありません。")
    st.stop()

default_y = "corrected_pupil" if "corrected_pupil" in common_num else common_num[0]
y_col = st.selectbox("目的変数（昼/夜共通の数値列）", options=common_num, index=common_num.index(default_y))

st.subheader("特徴量工学（探索候補に追加）")
o1, o2, o3, o4, o5 = st.columns(5)
with o1:
    add_log = st.checkbox("log(符号付きlog1p) を候補に追加", value=False)
with o2:
    add_sq = st.checkbox("二乗(x^2) を候補に追加", value=False)
with o3:
    add_sqrt = st.checkbox("√(符号付きsqrt) を候補に追加", value=False)
with o4:
    allow_interaction_candidate = st.checkbox("交互作用(コントラスト×シャープネス)も探索候補に入れる", value=False)
with o5:
    robust_hc3 = st.checkbox("p値をHC3ロバストで計算（昼の不安定対策）", value=True)

st.subheader("OLS評価（R²比較）")
mode = st.radio("評価法", options=["holdout", "LOSO", "LOIO", "in-sample"], index=0, horizontal=True)

group_subject_col = st.selectbox("被験者ID列（LOSO / holdout-group用）", options=["(なし)"] + common_cols, index=0)
group_image_col = st.selectbox("画像ID列（LOIO用：file_name/figure等）", options=["(なし)"] + common_cols, index=0)

test_size = st.slider("holdout test_size", 0.1, 0.5, 0.2, 0.05) if mode == "holdout" else 0.2
n_splits = st.slider("LOSO/LOIO の分割数（上限）", 2, 30, 5) if mode in ["LOSO", "LOIO"] else 5
seed = st.number_input("random_state", min_value=0, max_value=999999, value=42, step=1)

st.subheader("採用条件（Bの合格判定）")
cA, cB, cC = st.columns(3)
with cA:
    alpha = st.selectbox("有意水準（p<α）", options=[0.10, 0.05], index=0)
with cB:
    top_k_each = st.slider("探索に使う候補数（各グループ上位K）", 3, 30, 10)
with cC:
    include_int_p_in_rule = st.checkbox("採用条件のp値カウントに交互作用項も含める", value=False)

non_feature_text = st.text_input("除外列（カンマ区切り。ID列など）", value="")
NON_FEATURE_COLS = [s.strip() for s in non_feature_text.split(",") if s.strip()]

show_debug = st.checkbox("デバッグ情報（失敗理由/候補数）も表示", value=True)

feat_groups = [
    ("all", "全体（all）"),
    ("all_area", "領域重み（all_area）"),
    ("all_pupil", "瞳孔重み（all_pupil）"),
    ("ROI", "ROI別（center/parafovea/periphery）"),
]


def col_or_none(name):
    return None if name == "(なし)" else name


if st.button("実行（昼/夜 × 4グループ）"):
    rows = []
    for tag, df in [("day", df_day), ("night", df_night)]:
        for fg, fg_name in feat_groups:
            res = solve_one(
                df=df,
                y_col=y_col,
                feat_group=fg,
                non_feature_cols=NON_FEATURE_COLS,
                mode=mode,
                group_subject_col=col_or_none(group_subject_col),
                group_image_col=col_or_none(group_image_col),
                test_size=float(test_size),
                n_splits=int(n_splits),
                seed=int(seed),
                add_log=bool(add_log),
                add_sq=bool(add_sq),
                add_sqrt=bool(add_sqrt),
                allow_interaction_candidate=bool(allow_interaction_candidate),
                include_int_p_in_rule=bool(include_int_p_in_rule),
                robust_hc3=bool(robust_hc3),
                alpha=float(alpha),
                top_k_each=int(top_k_each),
            )
            res["dataset"] = tag
            res["feature_group_name"] = fg_name
            rows.append(res)

    df_out = pd.DataFrame(rows)

    st.subheader("✅ 結果（A=輝度1項 / B=b+c+s（+ c×s候補））")
    show_cols = [
        "dataset", "feature_group_name",
        "A_best_feat", "A_R2", "A_const_coef", "A_const_p", "A_coef", "A_p",
        "B_best_feats", "B_R2", "Delta(B-A)",
        "B_const_coef", "B_const_p",
        "B_use_interaction",
        "B_int_feat", "B_int_coef", "B_int_p",
        "accepted", "accept_reason",
        "B_b_feat", "B_b_coef", "B_b_p",
        "B_c_feat", "B_c_coef", "B_c_p",
        "B_s_feat", "B_s_coef", "B_s_p",
        "B_terms_sigcount",
        "B_terms_sigcount_rule",
    ]
    if show_debug:
        show_cols += [
            "status", "reason",
            "n_bright_base", "n_cont_base", "n_sharp_base",
            "n_B_tested", "n_B_cv_valid", "n_B_accepted",
            "cv_fail_A", "cv_fail_B",
        ]
    for c in show_cols:
        if c not in df_out.columns:
            df_out[c] = np.nan

    st.dataframe(df_out[show_cols].sort_values(["dataset", "feature_group_name"]), use_container_width=True)

    # =========================
    # R^2だけ（比較用）
    # =========================
    st.subheader("📌 R^2だけ（比較用）")
    slim = df_out[["dataset", "feature_group_name", "A_R2", "B_R2", "Delta(B-A)", "accepted", "B_use_interaction"]].copy()
    st.dataframe(slim.sort_values(["dataset", "feature_group_name"]), use_container_width=True)

    # =========================
    # NEW: R^2表の次に「model_table_with_pvals.csv形式 + 3/4項p値まとめ列」表示
    # =========================
    st.subheader("📌 パワポ用（R^2 + 式 + p値列）")

    df_ppt = df_out.copy()

    # A/B R^2
    df_ppt["A R^2"] = pd.to_numeric(df_ppt["A_R2"], errors="coerce")
    df_ppt["B R^2"] = pd.to_numeric(df_ppt["B_R2"], errors="coerce")

    # 日本語の項名
    df_ppt["A_輝度項"] = df_ppt["A_best_feat"].map(jp_feat)
    df_ppt["B_輝度項"] = df_ppt["B_b_feat"].map(jp_feat)
    df_ppt["B_コントラスト項"] = df_ppt["B_c_feat"].map(jp_feat)
    df_ppt["B_シャープネス項"] = df_ppt["B_s_feat"].map(jp_feat)
    df_ppt["B_交互作用項"] = df_ppt["B_int_feat"].apply(jp_interaction)

    # 式（定数項を先頭）
    df_ppt["A式"] = df_ppt.apply(build_A_expr, axis=1)
    df_ppt["B式"] = df_ppt.apply(build_B_expr, axis=1)

    # p値列（分割）
    df_ppt["A_切片p"] = df_ppt["A_const_p"].map(fmt_p)
    df_ppt["A_輝度p"] = df_ppt["A_p"].map(fmt_p)

    df_ppt["B_切片p"] = df_ppt["B_const_p"].map(fmt_p)
    df_ppt["B_輝度p"] = df_ppt["B_b_p"].map(fmt_p)
    df_ppt["B_コントラストp"] = df_ppt["B_c_p"].map(fmt_p)
    df_ppt["B_シャープネスp"] = df_ppt["B_s_p"].map(fmt_p)
    df_ppt["B_交互作用p"] = df_ppt["B_int_p"].map(fmt_p)

    # ★NEW：3項（or 4項）p値を1セルにまとめた列
    df_ppt["B_p値（3項/4項）"] = df_ppt.apply(build_B_p_summary, axis=1)

    show_cols_ppt = [
        "dataset", "feature_group_name",
        "A R^2", "B R^2",
        "A_切片p", "A_輝度項", "A_輝度p",
        "B_切片p", "B_輝度項", "B_輝度p",
        "B_コントラスト項", "B_コントラストp",
        "B_シャープネス項", "B_シャープネスp",
        "B_交互作用項", "B_交互作用p",
        "B_p値（3項/4項）",
        "A式", "B式",
    ]
    for c in show_cols_ppt:
        if c not in df_ppt.columns:
            df_ppt[c] = ""

    df_ppt_view = df_ppt[show_cols_ppt].sort_values(["dataset", "feature_group_name"])
    st.dataframe(df_ppt_view, use_container_width=True)

    # =========================
    # ダウンロード（ファイル名はUI設定準拠）
    # =========================
    fn_raw = make_result_filename(
        prefix="day_night_reg_results_RAW",
        y_col=y_col, mode=mode, alpha=float(alpha),
        add_log=bool(add_log), add_sq=bool(add_sq), add_sqrt=bool(add_sqrt),
        allow_int=bool(allow_interaction_candidate),
        include_int_p=bool(include_int_p_in_rule),
        robust_hc3=bool(robust_hc3),
        top_k_each=int(top_k_each),
        seed=int(seed),
    )
    fn_ppt = make_result_filename(
        prefix="day_night_reg_results_PPTTABLE",
        y_col=y_col, mode=mode, alpha=float(alpha),
        add_log=bool(add_log), add_sq=bool(add_sq), add_sqrt=bool(add_sqrt),
        allow_int=bool(allow_interaction_candidate),
        include_int_p=bool(include_int_p_in_rule),
        robust_hc3=bool(robust_hc3),
        top_k_each=int(top_k_each),
        seed=int(seed),
    ).replace(".csv", ".csv")  # 念のため

    raw_csv = df_out.to_csv(index=False).encode("utf-8-sig")
    ppt_csv = df_ppt_view.to_csv(index=False).encode("utf-8-sig")

    cdl1, cdl2 = st.columns(2)
    with cdl1:
        st.download_button("CSVでダウンロード（生結果）", data=raw_csv, file_name=fn_raw, mime="text/csv")
    with cdl2:
        st.download_button("CSVでダウンロード（パワポ用表）", data=ppt_csv, file_name=fn_ppt, mime="text/csv")

    with st.expander("説明"):
        st.markdown(
            "- R^2だけの表の直後に、**パワポ用（式＋p値）表**を追加表示します。\n"
            "- パワポ用表には、**Bの3項（交互作用がある場合は4項）のp値を1セルにまとめた列**も入ります。\n"
            "- ダウンロードのファイル名は、ラジオ/チェック等の設定を埋め込んだ名前になります。"
        )
