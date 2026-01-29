# core.py
import streamlit as st
import pandas as pd
import numpy as np

# ==========================================
# 1. データ読み込み & パース処理 (共通)
# ==========================================
@st.cache_data
def load_and_parse_data(uploaded_file):
    """ファイルを読み込み、ファイル名から順序と値を抽出して構造化する"""
    # 拡張子判別
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    def parse_params_ordered(name):
        if pd.isna(name):
            return {
                'param1': 'None', 'param1_val': 0.0,
                'param2': 'None', 'param2_val': 0.0,
                'param3': 'None', 'param3_val': 0.0
            }

        clean_name = str(name).replace('.jpg', '').replace('.JPG', '')
        parts = clean_name.split('_')
        valid_ops = ['brightness', 'contrast', 'gamma', 'sharpness', 'equalization']

        params = []
        for part in parts:
            for op in valid_ops:
                if part.startswith(op):
                    try:
                        val_str = part.replace(op, '')
                        val = float(val_str)
                        params.append((op, val))
                    except ValueError:
                        continue
                    break

        # 3ステップ未満を埋める
        while len(params) < 3:
            params.append(('None', 0.0))

        return {
            'param1': params[0][0], 'param1_val': params[0][1],
            'param2': params[1][0], 'param2_val': params[1][1],
            'param3': params[2][0], 'param3_val': params[2][1]
        }

    parsed_list = [parse_params_ordered(n) for n in df['image_name']]
    params_df = pd.DataFrame(parsed_list)

    # 順序パターンIDを作成 (例: gamma -> sharpness -> equalization)
    params_df['pattern_id'] = (
        params_df['param1'] + " → " + params_df['param2'] + " → " + params_df['param3']
    )

    # 重複列削除
    cols_to_use = params_df.columns.tolist()
    df = df.drop(columns=[c for c in cols_to_use if c in df.columns], errors='ignore')

    df_full = pd.concat([df, params_df], axis=1)
    return df_full


# ==========================================
# 2. 特徴量エンジニアリング & 重み計算
# ==========================================
def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    'step1_gamma' のように、「場所×種類」で値を格納する特徴量を作成。
    これによりモデルは「1手目のGamma」と「2手目のGamma」を区別できる。
    """
    valid_ops = ['brightness', 'contrast', 'gamma', 'sharpness', 'equalization']
    X_dict = {}

    for i in range(1, 4):
        col_op = f'param{i}'
        col_val = f'param{i}_val'

        for op in valid_ops:
            # 該当する操作の場合のみ値を入れ、それ以外は0
            mask = (df[col_op] == op).astype(float)
            X_dict[f"step{i}_{op}"] = mask * df[col_val]

    # df と index を揃えておく（相関計算などで安心）
    return pd.DataFrame(X_dict, index=df.index)


def compute_sample_weights(df: pd.DataFrame) -> pd.Series:
    """
    pattern_id ごとに件数を数え、その逆数を重みとする。
    -> 多く含まれる加工パターンに学習が偏らないようにする。
    """
    key = df['pattern_id']
    freq = key.value_counts()
    w = key.map(freq).astype(float)
    w = 1.0 / w
    # 平均が 1 になるようにスケーリング（任意）
    w *= (len(w) / w.sum())
    return w


def generate_valid_patterns_18():
    """
    brightness は最初 / equalization は最後 / brightness と equalization は同居しない
    / 重複禁止、という条件で 18 パターンを生成
    戻り値: [("brightness","contrast","gamma"), ...] の list
    """
    ops = ['brightness', 'contrast', 'gamma', 'sharpness', 'equalization']
    patterns = []

    for s1 in ops:
        for s2 in ops:
            for s3 in ops:
                steps = [s1, s2, s3]

                # 重複禁止
                if len(set(steps)) < 3:
                    continue

                # brightness があるなら必ず step1
                if 'brightness' in steps and steps[0] != 'brightness':
                    continue

                # equalization があるなら必ず step3
                if 'equalization' in steps and steps[2] != 'equalization':
                    continue

                # brightness & equalization の同居禁止
                if 'brightness' in steps and 'equalization' in steps:
                    continue

                patterns.append(tuple(steps))

    return patterns
