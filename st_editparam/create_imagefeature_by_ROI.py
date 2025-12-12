# -*- coding: utf-8 -*-
import os, sys, warnings
import pandas as pd
import cv2
from tqdm import tqdm

# ===== GPU / CPU 両対応 feature モジュールを自動選択 =====
try:
    import features_pupil as fp
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("[INFO] Using GPU version (features_pupil_gpu)")
    else:
        print("[WARN] No CUDA device found — fallback to CPU version (features_pupil)")
        import features_pupil as fp
except ImportError:
    print("[WARN] features_pupil_gpu not available — using CPU version (features_pupil)")
    import features_pupil as fp

# ===== 設定パラメータ =====
SCREEN_W_MM = 260
SCREEN_H_MM = 70
DIST_MM     = 450
RES_X, RES_Y = 6000, 1787
CENTER_DEG = 2
PARAFOVEA_DEG = 5

#desktop
# DIFF_ROOT = "F:/pictures/difference_images"
# TRANS_ROOT = "F:/pictures/transformed" 
# EXP_ROOT  = "F:/experiment_images"

#laptop
DIFF_ROOT = "D:/pictures/difference_images"
EXP_ROOT  = "D:/experiment_images"
TRANS_ROOT = "D:/pictures/transformed"

_INDEX_CACHE = {}

# ------------------------------------------------------------
# パス解決・ユーティリティ
# ------------------------------------------------------------
def _build_index(root):
    idx = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            idx[f.lower()] = os.path.join(dirpath, f)
    return idx

def resolve_image_path_by_basename(cell_value, colname):
    if pd.isna(cell_value): 
        return None
    bn = os.path.basename(str(cell_value)).lower()
    if colname in ("image_difference_abs","image_difference_signed","image_difference_norm"):
        root = DIFF_ROOT
    elif colname == "image_name":
        root = EXP_ROOT
    elif colname == "filename":
        root = TRANS_ROOT
    else:
        return None
    if root not in _INDEX_CACHE:
        _INDEX_CACHE[root] = _build_index(root)
    return _INDEX_CACHE[root].get(bn, None)

def overwrite_columns(df_base: pd.DataFrame, feat_df: pd.DataFrame, allow_nan_overwrite=False):
    out = df_base.copy()
    for col in feat_df.columns:
        if col in out.columns:
            if allow_nan_overwrite:
                out[col] = feat_df[col].values
            else:
                mask = feat_df[col].notna()
                if mask.any():
                    out.loc[mask, col] = feat_df.loc[mask, col]
        else:
            out[col] = feat_df[col]
    return out

# ------------------------------------------------------------
# 画像1枚の特徴量抽出
# ------------------------------------------------------------
def process_image(path, mode):  # mode = "roi" or "all"
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]

    if mode == "all":
        masks = {"all": None}
    else:
        masks = fp.make_masks(h, w, SCREEN_W_MM, DIST_MM, RES_X, CENTER_DEG, PARAFOVEA_DEG)

    try:
        feats = fp.compute_features_for_image(
            img_bgr=img,
            masks=masks,
            screen_w_mm=SCREEN_W_MM,
            dist_mm=DIST_MM,
            res_x=RES_X,
        )
        return feats
    except Exception as e:
        warnings.warn(f"[error] {path}: {e}")
        return None

# ------------------------------------------------------------
# Excelを一括処理
# ------------------------------------------------------------
def main(excel_path, path_col, out_xlsx, mode):  # mode = "roi" or "all"
    df = pd.read_excel(excel_path)
    if path_col not in df.columns:
        sys.exit(f"[ERROR] Column '{path_col}' not found in Excel.")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = resolve_image_path_by_basename(row[path_col], path_col)
        if img_path is None or not os.path.exists(img_path):
            warnings.warn(f"[skip] not found: {row[path_col]}")
            results.append({})
            continue
        feats = process_image(img_path, mode=mode)
        results.append(feats or {})

    feat_df = pd.DataFrame(results, index=df.index)
    df_out = overwrite_columns(df, feat_df, allow_nan_overwrite=False)
    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)
    df_out.to_excel(out_xlsx, index=False)
    print(f"[saved] {out_xlsx}")

# ------------------------------------------------------------
# メイン処理
# ------------------------------------------------------------
if __name__ == "__main__":
    # excel_path = "../../data_pupil/final_2025_dark_pupil/darkfinal_recalculated_pupil_originalonly.xlsx"
    # # path_col   = "image_name"
    # path_col ="filename"
    # out_xlsx   = "../../data_pupil/final_2025_dark_pupil/darkfinal_recalculated_pupil_originalonly_roi.xlsx"
    # main(excel_path, path_col, out_xlsx, mode="roi")

    # out_xlsx_global = "../../data_pupil/final_2025_dark_pupil/darkfinal_recalculated_pupil_originalonly_global.xlsx"
    # main(excel_path, path_col, out_xlsx_global, mode="all")

    # もし明条件も処理したい場合は下をアンコメント
    excel_path = "../../data_pupil/final_2025_bright_pupil/final_recalculated_pupil_originalonly.xlsx"
    # path_col   = "image_name"
    path_col ="filename"
    out_xlsx   = "../../data_pupil/final_2025_bright_pupil/final_recalculated_pupil_originalonly_roi.xlsx"
    main(excel_path, path_col, out_xlsx, mode="roi")

    out_xlsx_global = "../../data_pupil/final_2025_bright_pupil/final_recalculated_pupil_originalonly_global.xlsx"
    main(excel_path, path_col, out_xlsx_global, mode="all")