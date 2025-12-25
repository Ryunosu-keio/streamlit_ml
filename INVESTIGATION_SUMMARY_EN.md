# Investigation Summary: Pupil Diameter Change Calculation Code

## Question Asked (Japanese)
**"瞳孔径の変化分を計算しているコードはどれ"**
Translation: "Which code is calculating the change in pupil diameter?"

## Investigation Result
**No code calculating pupil diameter changes was found in this repository.**

## Investigation Details

### Search Methodology
1. **Keyword search** in all Python files:
   - Japanese: 瞳孔 (pupil), 径 (diameter), 変化 (change)
   - English: pupil, diameter, diff, difference, change
   
2. **Manual code review** of all Python files:
   - `classify.py` - Classification rules
   - `tree.py` - Decision tree utilities
   - `tree_only.py` - Basic decision tree implementation
   - `st_tree.py` - Streamlit ML GUI (GroupKFold CV, XAI)
   - `st_tree_penalty.py` - ML with consistency penalty
   - `xai.py` - Explainable AI (SHAP/LIME)
   - `sanpuzu.py` - Scatter plot visualization

3. **Data file examination**:
   - Checked CSV files in `input/` directory
   - No columns related to pupil diameter found
   - Data contains: section, online, car-related metrics, alerts, flame size, Water

### Difference Calculations Found
The repository contains general difference calculations but none for pupil diameter:

1. **Consistency penalty** (`st_tree_penalty.py`): Calculates variance of predictions within groups
2. **Combined loss** (`st_tree_penalty.py`): Calculates difference between predictions and true values (MSE)
3. **Scatter plots** (`sanpuzu.py`): Mentions "diopter" (refractive power), NOT pupil diameter

## Conclusion
**Answer: There is no code calculating pupil diameter changes in this repository.**

This is a machine learning project for classification/regression tasks using various algorithms (Decision Trees, Random Forests, SVM, Neural Networks, XGBoost). The data and code do not involve pupil diameter measurements.

## Recommendations

### If pupil diameter calculation is needed:
Implement a function like this:

```python
import pandas as pd
import numpy as np

def calculate_pupil_diameter_change(baseline_diameter, current_diameter):
    """
    Calculate the change in pupil diameter.
    
    Parameters:
    -----------
    baseline_diameter : float or array-like
        Pupil diameter at baseline [mm]
    current_diameter : float or array-like
        Current pupil diameter [mm]
    
    Returns:
    --------
    change : float or array-like
        Change in pupil diameter [mm] (positive = dilation, negative = constriction)
    """
    return current_diameter - baseline_diameter

def calculate_pupil_diameter_change_rate(baseline_diameter, current_diameter):
    """
    Calculate the relative change rate in pupil diameter.
    
    Returns:
    --------
    change_rate : float or array-like
        Percentage change in pupil diameter [%]
    """
    return ((current_diameter - baseline_diameter) / baseline_diameter) * 100

# Example usage with pandas DataFrame
def add_pupil_change_features(df, baseline_col='pupil_baseline', current_col='pupil_current'):
    """
    Add pupil diameter change features to a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    baseline_col : str
        Name of baseline pupil diameter column
    current_col : str
        Name of current pupil diameter column
    
    Returns:
    --------
    df : pd.DataFrame
        DataFrame with added features: 'pupil_change' and 'pupil_change_rate'
    """
    df = df.copy()
    df['pupil_change'] = calculate_pupil_diameter_change(
        df[baseline_col], df[current_col]
    )
    df['pupil_change_rate'] = calculate_pupil_diameter_change_rate(
        df[baseline_col], df[current_col]
    )
    return df
```

### Questions for Clarification:
1. Does pupil diameter data exist in a different repository or branch?
2. What file contains the pupil diameter measurements?
3. What are the column names for baseline and current pupil diameter?
4. Should the change be calculated as absolute difference or relative percentage?
5. Are there multiple time points requiring temporal difference calculations?

---
**Investigation Date**: November 20, 2025
**Investigator**: GitHub Copilot Coding Agent
**Repository**: Ryunosu-keio/streamlit_ml
