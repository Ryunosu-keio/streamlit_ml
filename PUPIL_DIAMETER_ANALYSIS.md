# 瞳孔径の変化分を計算しているコードの調査結果

## 調査結果サマリー
本リポジトリ内を徹底的に調査した結果、**瞳孔径（pupil diameter）の変化分を直接計算しているコードは見つかりませんでした**。

## 調査内容

### 1. 検索対象
- すべてのPythonファイル (.py)
- CSVデータファイル
- 日本語キーワード: 瞳孔、径、変化
- 英語キーワード: pupil, diameter, diff, difference, change

### 2. 確認したファイル一覧
- `classify.py` - 分類ルールのコード
- `tree.py` - 決定木モデルのユーティリティ
- `tree_only.py` - 決定木の基本実装
- `st_tree.py` - StreamlitベースのML GUI アプリ（GroupKFold CV、XAI対応）
- `st_tree_penalty.py` - 一貫性ペナルティ付きML GUI
- `xai.py` - SHAP/LIME による説明可能AI
- `sanpuzu.py` - 散布図の可視化

### 3. データファイルの列名確認
`input/all_riku_one.csv` の列:
- section, online, back car, small car add, big car add, same car add
- num of add alart, flame size, adjacent car, front adjacent, rear adjacent, Water

`input/dataset.csv` の列:
- section, online, rear car, num of add alart, flame size, adjacent car, Water

**結論**: データセットに瞳孔径に関する列は含まれていません。

### 4. コード内の変化量計算
以下のコードで「差分」や「変化」に関連する処理を確認:

#### `st_tree_penalty.py` (行43-51)
```python
def consistency_penalty(yhat: np.ndarray, groups_val: pd.Series) -> float:
    """Σ_g Var(ŷ_i | i∈g)"""
    s = 0.0
    gsr = groups_val.reset_index(drop=True)
    for g, subidx in gsr.groupby(gsr):
        arr = yhat[subidx.index]
        if len(arr) >= 2:
            s += float(np.var(arr))
    return s
```
- **目的**: グループ内の予測値の分散（一貫性ペナルティ）を計算
- **瞳孔径との関連**: なし

#### `st_tree_penalty.py` (行53-60)
```python
def combined_loss(y_true, y_pred, groups_val: pd.Series, task_type: str, lam: float) -> float:
    """L_total = L_task + λ * consistency"""
    if task_type == "分類":
        task = 1.0 - accuracy_score(y_true, y_pred)
    else:
        task = float(np.mean((y_true - y_pred) ** 2))
    cons = consistency_penalty(np.asarray(y_pred), groups_val)
    return task + lam * cons
```
- **目的**: 予測値と真値の差（二乗誤差）を計算
- **瞳孔径との関連**: なし（一般的な回帰損失）

#### `sanpuzu.py` (行12-75)
- **目的**: 説明変数と目的変数（diopter：屈折度）の散布図作成
- **注意**: コード内に `diopter` という変数名がありますが、これは視力の屈折度であり、瞳孔径ではありません
- **瞳孔径との関連**: 間接的な関連の可能性はあるが、瞳孔径の変化分の計算ではない

## 結論と推奨事項

### 現状
本リポジトリには**瞳孔径の変化分を計算するコードは存在しません**。

### 考えられる状況
1. **誤認識**: 他のコードベースや別のブランチに該当コードがある可能性
2. **未実装**: 瞳孔径の変化分計算機能の追加が必要
3. **命名の問題**: 変数名が瞳孔径を示唆していないため見落としている可能性

### 推奨される対応
もし瞳孔径の変化分計算機能が必要な場合：

```python
# 瞳孔径の変化分を計算する例
def calculate_pupil_diameter_change(baseline_diameter, current_diameter):
    """
    瞳孔径の変化分を計算
    
    Parameters:
    -----------
    baseline_diameter : float or array-like
        基準時点の瞳孔径 [mm]
    current_diameter : float or array-like
        現在の瞳孔径 [mm]
    
    Returns:
    --------
    change : float or array-like
        瞳孔径の変化分 [mm] (正の値は拡大、負の値は縮小)
    """
    return current_diameter - baseline_diameter

# 相対変化率（パーセンテージ）も計算する場合
def calculate_pupil_diameter_change_rate(baseline_diameter, current_diameter):
    """
    瞳孔径の変化率を計算
    
    Returns:
    --------
    change_rate : float or array-like
        瞳孔径の変化率 [%]
    """
    return ((current_diameter - baseline_diameter) / baseline_diameter) * 100
```

## 追加情報が必要な場合
- 瞳孔径データはどのファイルに含まれていますか？
- データの列名は何ですか？
- 基準時点（baseline）はどのように定義されますか？
- 変化分の計算は絶対値ですか、それとも相対値（％）ですか？

---
**調査日時**: 2025-11-20
**調査者**: GitHub Copilot Coding Agent
