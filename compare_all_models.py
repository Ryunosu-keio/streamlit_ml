# -*- coding: utf-8 -*-
# ============================================================
# 複数モデル比較レポート自動生成
# 説明変数：center/parafovea/peripheryの bL_mean, rms_contrst, sh_gradentropy（9変数）
# 目的変数：corrected_pupil と 両眼.注視Z座標[mm]の逆数（1000倍）
# ============================================================

import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, GridSearchCV
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    f1_score, accuracy_score, classification_report
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# ============================================================
# モデル定義
# ============================================================

def get_model_configs(seed=42):
    """各モデルの設定とパラメータグリッドを返す"""
    configs = []
    
    # ========== 線形モデル（重回帰系統）をコメントアウト ==========
    # # 線形モデル（多項式次数2次）
    # configs.append({
    #     "name": "Ridge_Poly2",
    #     "model": Pipeline([
    #         ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    #         ("scaler", StandardScaler()),
    #         ("model", Ridge(random_state=seed))
    #     ]),
    #     "param_grid": {
    #         "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
    #     }
    # })
    
    # configs.append({
    #     "name": "Lasso_Poly2",
    #     "model": Pipeline([
    #         ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    #         ("scaler", StandardScaler()),
    #         ("model", Lasso(random_state=seed, max_iter=5000))
    #     ]),
    #     "param_grid": {
    #         "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]
    #     }
    # })
    
    # configs.append({
    #     "name": "ElasticNet_Poly2",
    #     "model": Pipeline([
    #         ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    #         ("scaler", StandardScaler()),
    #         ("model", ElasticNet(random_state=seed, max_iter=5000))
    #     ]),
    #     "param_grid": {
    #         "model__alpha": [0.01, 0.1, 1.0, 10.0],
    #         "model__l1_ratio": [0.3, 0.5, 0.7, 0.9]
    #     }
    # })
    
    # # 線形モデル（多項式次数3次）
    # configs.append({
    #     "name": "Ridge_Poly3",
    #     "model": Pipeline([
    #         ("poly", PolynomialFeatures(degree=3, include_bias=False)),
    #         ("scaler", StandardScaler()),
    #         ("model", Ridge(random_state=seed))
    #     ]),
    #     "param_grid": {
    #         "model__alpha": [1.0, 10.0, 100.0, 1000.0]
    #     }
    # })
    
    # ツリーモデル
    configs.append({
        "name": "DecisionTree",
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("model", DecisionTreeRegressor(random_state=seed))
        ]),
        "param_grid": {
            "model__max_depth": [3, 5, 7, 10],
            "model__min_samples_split": [5, 10, 20]
        }
    })
    
    configs.append({
        "name": "RandomForest",
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(random_state=seed))
        ]),
        "param_grid": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 5, 7, 10],
            "model__min_samples_split": [5, 10, 15],
            "model__min_samples_leaf": [1, 2]
        }
    })
    
    configs.append({
        "name": "GradientBoosting",
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(random_state=seed))
        ]),
        "param_grid": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.1, 0.2]
        }
    })
    
    # XGBoost
    if HAS_XGB:
        configs.append({
            "name": "XGBoost",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("model", XGBRegressor(random_state=seed, verbosity=0))
            ]),
            "param_grid": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.1, 0.2]
            }
        })
    
    # LightGBM
    if HAS_LGBM:
        configs.append({
            "name": "LightGBM",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LGBMRegressor(random_state=seed, verbose=-1))
            ]),
            "param_grid": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.1, 0.2]
            }
        })
    
    # SVR
    configs.append({
        "name": "SVR_RBF",
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf"))
        ]),
        "param_grid": {
            "model__C": [0.1, 1.0, 10.0, 100.0],
            "model__gamma": ["scale", "auto", 0.01, 0.1]
        }
    })
    
    configs.append({
        "name": "SVR_Linear",
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="linear"))
        ]),
        "param_grid": {
            "model__C": [0.1, 1.0, 10.0, 100.0]
        }
    })
    
    # KNN
    configs.append({
        "name": "KNN",
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor())
        ]),
        "param_grid": {
            "model__n_neighbors": [3, 5, 7, 9],
            "model__weights": ["uniform", "distance"]
        }
    })
    
    return configs

# ============================================================
# 外れ値除去
# ============================================================

def remove_outliers(X, y, groups, method="iqr", threshold=1.5):
    """
    外れ値を除去
    method: "iqr" (四分位範囲), "zscore" (標準偏差)
    threshold: IQRの場合は倍率（デフォルト1.5）、zscoreの場合は標準偏差数（デフォルト3）
    """
    X_clean = X.copy()
    y_clean = y.copy()
    groups_clean = groups.copy()
    
    if method == "iqr":
        # 目的変数のIQRで外れ値検出
        Q1 = y.quantile(0.25)
        Q3 = y.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (y >= lower_bound) & (y <= upper_bound)
        
    elif method == "zscore":
        # 目的変数のzスコアで外れ値検出
        z_scores = np.abs((y - y.mean()) / y.std())
        mask = z_scores < threshold
    else:
        mask = pd.Series([True] * len(y), index=y.index)
    
    X_clean = X_clean[mask]
    y_clean = y_clean[mask]
    groups_clean = groups_clean[mask]
    
    n_removed = len(y) - len(y_clean)
    print(f"  外れ値除去: {n_removed}サンプル削除 ({n_removed/len(y)*100:.1f}%)")
    print(f"  残りサンプル数: {len(y_clean)}")
    
    return X_clean, y_clean, groups_clean

# ============================================================
# 評価関数
# ============================================================

def grid_search_model(model, param_grid, X, y, groups, cv_method="groupkfold", n_splits=3):
    """グリッドサーチでベストパラメータを探索"""
    
    # 欠損値補完
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    
    # CV設定（GroupKFoldのみ使用）
    cv = GroupKFold(n_splits=min(n_splits, groups.nunique()))
    
    # グリッドサーチ実行
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_imp, y, groups=groups)
    
    return {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "best_estimator": grid_search.best_estimator_,
        "cv_results": grid_search.cv_results_
    }

def evaluate_model_cv(model, X, y, groups, n_splits=5, model_name="Model", param_grid=None):
    """GroupKFold CVで評価（グリッドサーチ付き）"""
    
    # グループ数に応じてn_splitsを調整
    n_groups = groups.nunique()
    n_splits = min(n_splits, n_groups)
    
    if n_splits < 2:
        raise ValueError(f"グループ数が少なすぎます（{n_groups}グループ）。最低2グループ必要です。")
    
    # グリッドサーチ実行（必須）
    best_params = None
    search_space = None
    if param_grid is not None and len(param_grid) > 0:
        search_space = param_grid.copy()
        n_combinations = int(np.prod([len(v) for v in param_grid.values()]))
        print(f"  グリッドサーチ実行中... (探索数: {n_combinations}通り × {n_splits}分割 = {n_combinations * n_splits}回学習)")
        
        gs_result = grid_search_model(model, param_grid, X, y, groups, 
                                     cv_method="groupkfold", n_splits=n_splits)
        best_params = gs_result["best_params"]
        model = gs_result["best_estimator"]
        print(f"  ベストパラメータ: {best_params}")
        print(f"  グリッドサーチベストスコア(R2): {gs_result['best_score']:.4f}")
    
    # 最終評価用のCV
    print(f"  最終評価: {n_splits}分割GroupKFold CV実行中...")
    gkf = GroupKFold(n_splits=n_splits)
    
    y_true_all = []
    y_pred_all = []
    fold_scores = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 欠損値補完
        imputer = SimpleImputer(strategy="median")
        X_train = pd.DataFrame(imputer.fit_transform(X_train), 
                              columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(imputer.transform(X_test), 
                             columns=X_test.columns, index=X_test.index)
        
        # 学習・予測
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
        
        # Fold毎のスコア
        fold_r2 = r2_score(y_test, y_pred)
        fold_scores.append(fold_r2)
        print(f"    Fold {fold_idx}/{n_splits}: R2={fold_r2:.4f}")
    
    # 全体のスコア
    r2 = r2_score(y_true_all, y_pred_all)
    rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    mae = mean_absolute_error(y_true_all, y_pred_all)
    
    result = {
        "model_name": model_name,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "fold_r2_mean": np.mean(fold_scores),
        "fold_r2_std": np.std(fold_scores),
        "n_folds": len(fold_scores),
        "y_true": y_true_all,  # 実測値
        "y_pred": y_pred_all   # 予測値
    }
    
    if best_params is not None:
        result["best_params"] = str(best_params)
        result["search_space"] = str(search_space)
    else:
        result["best_params"] = "N/A"
        result["search_space"] = "N/A"
    
    return result

# ============================================================
# レポート生成
# ============================================================

def generate_comparison_report(results_df, target_name, output_dir):
    """比較レポートを生成"""
    
    # 結果をR2でソート
    results_df = results_df.sort_values("r2", ascending=False).reset_index(drop=True)
    
    # レポート作成
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"モデル比較レポート: {target_name}")
    report_lines.append(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # サマリーテーブル
    report_lines.append("### 結果サマリー（R2スコア順）")
    report_lines.append("")
    report_lines.append(results_df.to_string(index=False))
    report_lines.append("")
    report_lines.append("")
    
    # ベストモデル
    best_model = results_df.iloc[0]
    report_lines.append("### ベストモデル")
    report_lines.append(f"モデル名: {best_model['model_name']}")
    report_lines.append(f"R2スコア: {best_model['r2']:.4f}")
    report_lines.append(f"RMSE: {best_model['rmse']:.4f}")
    report_lines.append(f"MAE: {best_model['mae']:.4f}")
    report_lines.append(f"Fold平均R2: {best_model['fold_r2_mean']:.4f} ± {best_model['fold_r2_std']:.4f}")
    
    if 'best_params' in best_model and best_model['best_params'] != 'N/A':
        report_lines.append("")
        report_lines.append("### ハイパーパラメータ最適化結果")
        report_lines.append(f"探索範囲: {best_model['search_space']}")
        report_lines.append(f"決定値: {best_model['best_params']}")
    
    report_lines.append("")
    report_lines.append("")
    
    # トップ3モデル
    report_lines.append("### トップ3モデル")
    for idx in range(min(3, len(results_df))):
        model = results_df.iloc[idx]
        report_lines.append(f"{idx+1}. {model['model_name']}: R2={model['r2']:.4f}, RMSE={model['rmse']:.4f}")
        if 'best_params' in model and model['best_params'] != 'N/A':
            report_lines.append(f"   パラメータ: {model['best_params']}")
    report_lines.append("")
    report_lines.append("")
    
    # 分析コメント
    report_lines.append("### 分析コメント")
    best_r2 = best_model['r2']
    if best_r2 > 0.7:
        report_lines.append("- 非常に良好な予測性能が得られています")
    elif best_r2 > 0.5:
        report_lines.append("- 中程度の予測性能が得られています")
    elif best_r2 > 0.3:
        report_lines.append("- 弱い予測性能です。特徴量や前処理の見直しを推奨します")
    else:
        report_lines.append("- 予測性能が低いです。データや特徴量の再検討が必要です")
    
    if best_model['fold_r2_std'] > 0.2:
        report_lines.append("- Fold間のばらつきが大きく、一般化性能に課題があります")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # ファイル保存
    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, f"report_{target_name}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"\nレポート保存: {report_path}")
    print(report_text)
    
    return report_text

def plot_prediction_scatter(results_df, target_name, output_dir):
    """予測値vs実測値の散布図を作成（トップ3モデル）"""
    
    # トップ3モデルを取得
    top3 = results_df.nsmallest(3, 'rmse').head(3)  # RMSEが小さい順
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (_, model_data) in enumerate(top3.iterrows()):
        ax = axes[idx]
        
        y_true = model_data['y_true']
        y_pred = model_data['y_pred']
        model_name = model_data['model_name']
        r2 = model_data['r2']
        rmse = model_data['rmse']
        
        # 散布図
        ax.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
        
        # 理想線（y=x）
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        
        # ラベルとタイトル
        ax.set_xlabel('Actual', fontsize=12)
        ax.set_ylabel('Predicted', fontsize=12)
        ax.set_title(f'{model_name}\nR²={r2:.4f}, RMSE={rmse:.4f}', fontsize=12)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    plt.suptitle(f'Prediction vs Actual: {target_name}', fontsize=16, y=1.02)
    plt.tight_layout()
    
    scatter_path = os.path.join(output_dir, f"scatter_{target_name}.png")
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    print(f"散布図保存: {scatter_path}")
    plt.close()

def plot_comparison(results_df, target_name, output_dir):
    """比較グラフを作成"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # R2スコア比較
    ax = axes[0]
    results_sorted = results_df.sort_values("r2", ascending=True)
    ax.barh(results_sorted['model_name'], results_sorted['r2'], color='skyblue')
    ax.set_xlabel("R2 Score", fontsize=12)
    ax.set_title(f"R2 Score Comparison\n{target_name}", fontsize=14)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    # RMSE比較
    ax = axes[1]
    results_sorted = results_df.sort_values("rmse", ascending=False)
    ax.barh(results_sorted['model_name'], results_sorted['rmse'], color='lightcoral')
    ax.set_xlabel("RMSE", fontsize=12)
    ax.set_title(f"RMSE Comparison\n{target_name}", fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    # Fold R2のばらつき
    ax = axes[2]
    results_sorted = results_df.sort_values("fold_r2_mean", ascending=True)
    ax.barh(results_sorted['model_name'], results_sorted['fold_r2_mean'], 
            xerr=results_sorted['fold_r2_std'], color='lightgreen', capsize=5)
    ax.set_xlabel("Fold Mean R2 ± SD", fontsize=12)
    ax.set_title(f"Cross-Validation Stability\n{target_name}", fontsize=14)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"comparison_{target_name}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"グラフ保存: {plot_path}")
    plt.close()

# ============================================================
# メイン処理
# ============================================================

def main():
    print("=" * 80)
    print("複数モデル比較分析 開始")
    print("=" * 80)
    
    # データパスの定義（brightとdark）
    data_configs = [
        {
            "name": "dark",
            "path": "../../penstone/data_pupil/final_2025_dark_pupil/darkfinal_recalculated_pupil_bcss_roi_global_area_pupil_with_origfeats_reduced.xlsx"
        },
        {
            "name": "bright",
            "path": "../../penstone/data_pupil/final_2025_bright_pupil/final_recalculated_pupil_bcss_with_roi_global_withoutNan_with_area_pupil_with_origfeats_reduced.xlsx"
        }
    ]
    
    # 各データセットについて処理
    for data_config in data_configs:
        condition_name = data_config["name"]
        data_path = data_config["path"]
        
        print("\n" + "=" * 80)
        print(f"条件: {condition_name.upper()}")
        print("=" * 80)
        
        # 出力ディレクトリ作成（条件名を含める）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"output/model_comparison_{condition_name}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nデータ読み込み: {data_path}")
        
        if not os.path.exists(data_path):
            print(f"\n警告: {data_path} が見つかりません")
            print(f"スキップします...")
            continue
        if not os.path.exists(data_path):
            print(f"\n警告: {data_path} が見つかりません")
            print(f"スキップします...")
            continue
        
        # ファイル拡張子を確認
        file_ext = os.path.splitext(data_path)[1].lower()
        
        if file_ext == '.xlsx' or file_ext == '.xls':
            # Excelファイル
            print(f"Excelファイルを読み込み中...")
            df = pd.read_excel(data_path)
            print(f"データ読み込み成功")
        elif file_ext == '.csv':
            # CSVファイル（エンコーディング自動判定）
            encodings = ['utf-8', 'shift-jis', 'cp932', 'latin-1']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(data_path, encoding=encoding)
                    print(f"データ読み込み成功（エンコーディング: {encoding}）")
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if df is None:
                raise ValueError(f"ファイルを読み込めませんでした: {data_path}\n試したエンコーディング: {encodings}")
        else:
            raise ValueError(f"サポートされていないファイル形式: {file_ext}\nCSVまたはExcel(.xlsx, .xls)ファイルを指定してください")
        
        print(f"データ形状: {df.shape}")
        print(f"\nカラム一覧:\n{df.columns.tolist()}")
        print(f"データ形状: {df.shape}")
        print(f"\nカラム一覧:\n{df.columns.tolist()}")
        
        # 説明変数の定義（9変数）
        feature_cols = [
            "center_bL_mean", "center_rms_contrst", "center_sh_gradentropy",
            "parafovea_bL_mean", "parafovea_rms_contrst", "parafovea_sh_gradentropy",
            "periphery_bL_mean", "periphery_rms_contrst", "periphery_sh_gradentropy"
        ]
        
        # 実際のデータに存在するカラムのみを使用
        available_features = [col for col in feature_cols if col in df.columns]
        
        if len(available_features) == 0:
            print("\n警告: 指定された特徴量がデータに存在しません")
            print("利用可能なカラムから数値カラムを自動選択します...")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # 目的変数候補を除外
            available_features = [col for col in numeric_cols 
                                if col not in ['corrected_pupil', 'gaze_z_inv_1000', 
                                             'folder_name', 'file_name']][:9]
        
        print(f"\n使用する説明変数（{len(available_features)}個）:")
        for i, col in enumerate(available_features, 1):
            print(f"  {i}. {col}")
        
        # 目的変数の定義
        target_configs = []
        
        if 'corrected_pupil' in df.columns:
            target_configs.append({
                "name": "corrected_pupil",
                "column": "corrected_pupil"
            })
        
        if '両眼.注視Z座標[mm]' in df.columns:
            # 逆数を計算（1000倍）
            df['gaze_z_inv_1000'] = 1000.0 / (df['両眼.注視Z座標[mm]'] + 1e-6)
            target_configs.append({
                "name": "gaze_z_inv_1000",
                "column": "gaze_z_inv_1000"
            })
        elif 'gaze_z_inv_1000' in df.columns:
            target_configs.append({
                "name": "gaze_z_inv_1000",
                "column": "gaze_z_inv_1000"
            })
        
        if len(target_configs) == 0:
            print("\n警告: 目的変数が見つかりません。サンプルデータの目的変数を使用します")
            target_configs = [
                {"name": "corrected_pupil", "column": "corrected_pupil"},
                {"name": "gaze_z_inv_1000", "column": "gaze_z_inv_1000"}
            ]
        
        print(f"\n目的変数（{len(target_configs)}個）:")
        for i, tgt in enumerate(target_configs, 1):
            print(f"  {i}. {tgt['name']}")
        
        # 被験者情報（GroupKFold用）
        if 'folder_name' not in df.columns:
            raise ValueError("データに'folder_name'カラムが必要です（被験者IDとして使用）")
        
        groups = df['folder_name']
        print(f"\nCV方法: GroupKFold (folder_nameでグループ分割)")
        print(f"グループ数: {groups.nunique()}")
        print(f"総サンプル数: {len(df)}")
        print(f"グループ当たり平均サンプル数: {len(df) / groups.nunique():.1f}")
        
        # モデル設定取得
        model_configs = get_model_configs(seed=42)
        print(f"\n評価モデル数: {len(model_configs)}")
        
        # 各目的変数について評価
        all_results = {}
        
        for target_config in target_configs:
            target_name = target_config['name']
            target_col = target_config['column']
            
            print("\n" + "=" * 80)
            print(f"目的変数: {target_name} の評価開始")
            print("=" * 80)
            
            # データ準備
            X = df[available_features].copy()
            y = df[target_col].copy()
            
            # 欠損値削除
            valid_idx = y.notna() & X.notna().all(axis=1)
            X = X[valid_idx]
            y = y[valid_idx]
            groups_valid = groups[valid_idx]
            
            print(f"欠損値除去後サンプル数: {len(X)}")
            
            # 外れ値除去（3σ法）
            print(f"\n外れ値除去実行中（3σ法）...")
            X, y, groups_valid = remove_outliers(X, y, groups_valid, method="zscore", threshold=3)
            
            print(f"\n最終有効サンプル数: {len(X)}")
            print(f"最終グループ数: {groups_valid.nunique()}")
            
            # CV分割数を決定（グループ数の8割、最小3、最大10）
            n_splits = max(3, min(10, int(groups_valid.nunique() * 0.8)))
            print(f"CV分割数: {n_splits}")
            
            # 各モデルで評価
            results = []
            
            for i, config in enumerate(model_configs, 1):
                model_name = config['name']
                model = config['model']
                param_grid = config.get('param_grid', None)
                
                print(f"\n{'='*60}")
                print(f"[{i}/{len(model_configs)}] {model_name} を評価中...")
                print(f"{'='*60}")
                
                try:
                    result = evaluate_model_cv(
                        model, X, y, groups_valid, 
                        n_splits=n_splits, 
                        model_name=model_name, 
                        param_grid=param_grid
                    )
                    
                    results.append(result)
                    print(f"\n  最終結果: R2={result['r2']:.4f}, RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}")
                    print(f"  CV安定性: R2_mean={result['fold_r2_mean']:.4f} ± {result['fold_r2_std']:.4f}")
                    
                except Exception as e:
                    print(f"  エラー: {str(e)}")
                    continue
            
            # 結果をDataFrameに変換
            results_df = pd.DataFrame(results)
            
            # CSVに保存（予測値と実測値は除外）
            csv_columns = [col for col in results_df.columns if col not in ['y_true', 'y_pred']]
            csv_path = os.path.join(output_dir, f"results_{target_name}.csv")
            results_df[csv_columns].to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"\n結果CSV保存: {csv_path}")
            
            # レポート生成
            generate_comparison_report(results_df, target_name, output_dir)
            
            # グラフ作成
            plot_comparison(results_df, target_name, output_dir)
            
            # 散布図作成（予測vs実測）
            plot_prediction_scatter(results_df, target_name, output_dir)
            
            all_results[target_name] = results_df
        
        # 総合レポート作成
        print("\n" + "=" * 80)
        print("総合レポート作成")
        print("=" * 80)
        
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append(f"総合比較レポート [{condition_name.upper()}]")
        summary_lines.append(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        
        for target_name, results_df in all_results.items():
            summary_lines.append(f"\n### {target_name}")
            summary_lines.append("-" * 80)
            best_model = results_df.sort_values("r2", ascending=False).iloc[0]
            summary_lines.append(f"ベストモデル: {best_model['model_name']}")
            summary_lines.append(f"R2スコア: {best_model['r2']:.4f}")
            summary_lines.append(f"RMSE: {best_model['rmse']:.4f}")
            summary_lines.append(f"MAE: {best_model['mae']:.4f}")
            summary_lines.append("")
            
            # トップ3
            summary_lines.append("トップ3:")
            for idx in range(min(3, len(results_df))):
                model = results_df.sort_values("r2", ascending=False).iloc[idx]
                summary_lines.append(f"  {idx+1}. {model['model_name']}: R2={model['r2']:.4f}")
                if 'best_params' in model and model['best_params'] != 'N/A':
                    summary_lines.append(f"     パラメータ: {model['best_params']}")
            summary_lines.append("")
        
        summary_lines.append("=" * 80)
        
        summary_text = "\n".join(summary_lines)
        summary_path = os.path.join(output_dir, "summary_report.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        
        print(summary_text)
        print(f"\n総合レポート保存: {summary_path}")
        
        print("\n" + "=" * 80)
        print(f"{condition_name.upper()} の分析が完了しました")
        print(f"結果保存先: {output_dir}")
        print("=" * 80)

if __name__ == "__main__":
    main()
