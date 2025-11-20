# クイックリファレンス / Quick Reference

## 質問 / Question
**"瞳孔径の変化分を計算しているコードはどれ"**
"Which code is calculating the change in pupil diameter?"

## 回答 / Answer
**このリポジトリには瞳孔径の変化分を計算するコードは存在しません。**
**No code calculating pupil diameter changes exists in this repository.**

---

## 詳細な調査結果 / Detailed Investigation Results

### 日本語 / Japanese
📄 詳細は [PUPIL_DIAMETER_ANALYSIS.md](./PUPIL_DIAMETER_ANALYSIS.md) をご覧ください。

#### 主な発見事項:
- ✅ 全Pythonファイルを検索
- ✅ 全CSVデータファイルを確認
- ❌ 瞳孔径に関する列やコードは見つかりませんでした
- ℹ️ `sanpuzu.py` に "diopter"（屈折度）の記述がありますが、瞳孔径ではありません

#### このリポジトリの内容:
- 機械学習プロジェクト（Streamlit GUI）
- 決定木、ランダムフォレスト、SVM、NN、XGBoost
- GroupKFold クロスバリデーション
- SHAP/LIME による説明可能AI

---

### English
📄 See [INVESTIGATION_SUMMARY_EN.md](./INVESTIGATION_SUMMARY_EN.md) for details.

#### Key Findings:
- ✅ Searched all Python files
- ✅ Checked all CSV data files
- ❌ No pupil diameter columns or code found
- ℹ️ `sanpuzu.py` mentions "diopter" (refractive power), but NOT pupil diameter

#### What This Repository Contains:
- Machine learning project (Streamlit GUI)
- Decision Trees, Random Forest, SVM, NN, XGBoost
- GroupKFold cross-validation
- SHAP/LIME for explainable AI

---

## 瞳孔径計算が必要な場合 / If Pupil Diameter Calculation Needed

### サンプルコード / Sample Code:

```python
def calculate_pupil_diameter_change(baseline, current):
    """瞳孔径の変化分を計算 / Calculate pupil diameter change"""
    return current - baseline

def calculate_pupil_diameter_change_rate(baseline, current):
    """瞳孔径の変化率を計算 / Calculate pupil diameter change rate (%)"""
    return ((current - baseline) / baseline) * 100
```

### 必要な情報 / Information Needed:
1. 瞳孔径データはどこにありますか？ / Where is the pupil diameter data?
2. 列名は何ですか？ / What are the column names?
3. 基準時点の定義は？ / How is baseline defined?
4. 絶対値 or 相対値？ / Absolute or relative change?

---

## ファイル構成 / File Structure

```
streamlit_ml/
├── PUPIL_DIAMETER_ANALYSIS.md     ← 日本語の詳細調査結果
├── INVESTIGATION_SUMMARY_EN.md    ← English detailed summary
├── README_INVESTIGATION.md         ← このファイル (Quick reference)
├── classify.py                     ← 分類ルール
├── tree.py                         ← 決定木ユーティリティ
├── tree_only.py                    ← 基本決定木
├── st_tree.py                      ← Streamlit ML GUI
├── st_tree_penalty.py              ← 一貫性ペナルティ付きML
├── xai.py                          ← SHAP/LIME
├── sanpuzu.py                      ← 散布図可視化
└── input/                          ← データファイル (瞳孔径データなし)
```

---

**調査日時 / Investigation Date**: 2025-11-20  
**調査者 / Investigator**: GitHub Copilot Coding Agent  
**リポジトリ / Repository**: Ryunosu-keio/streamlit_ml
