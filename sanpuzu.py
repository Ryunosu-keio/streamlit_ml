import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Excelファイルのパス
excel_path = "C:/Users/ryuno/Desktop/final_dark_ml.csv"

# データ読み込み
df = pd.read_csv(excel_path)
y_column = 'diopter'

# 前処理オプションを選択
print("前処理オプションを選んでください:")
print("1: None (生の散布図)")
print("2: Transformation (log / sqrt)")
print("3: Binning (連続値→カテゴリ化)")
print("4: Clustering (KMeans でクラスタ色分け)")
opt = input(">> ")

# 各オプションのパラメータ入力
if opt == "2":
    print("変換方法を選択: log, sqrt")
    trans = input(">> ")
elif opt == "3":
    n_bins = int(input("ビン数を入力してください (例: 5): "))
elif opt == "4":
    n_clusters = int(input("クラスタ数を入力してください (例: 4): "))

# 全説明変数に対してループ
for col in df.columns:
    # if col == y_column or col in ("diocategory", "figure"):
    #     continue

    X = df[col].copy()
    Y = df[y_column]

    # 2: Transformation
    if opt == "2":
        if trans == "log":
            X = np.log1p(X)
        elif trans == "sqrt":
            X = np.sqrt(X)
        label = f"{col} ({trans})"

    # 3: Binning
    elif opt == "3":
        binned = pd.cut(X, bins=n_bins, labels=False)
        X = binned
        label = f"{col} (binned into {n_bins})"

    # 4: Clustering
    elif opt == "4":
        km = KMeans(n_clusters=n_clusters, random_state=0)
        # KMeans は 2D array を要求
        clusters = km.fit_predict(X.values.reshape(-1, 1))
        plt.scatter(X, Y, c=clusters, cmap='tab10', s=20)
        plt.colorbar(label='cluster')
        plt.xlabel(col)
        plt.ylabel(y_column)
        plt.title(f"{y_column} vs {col} (KMeans: {n_clusters} clusters)")
        plt.show()
        continue  # 次の変数へ

    else:
        # 1: None
        label = col

    # オプション 1,2,3 の散布図
    plt.scatter(X, Y, s=10, alpha=0.6)
    plt.xlabel(label)
    plt.ylabel(y_column)
    plt.title(f"{y_column} vs {label}")
    plt.tight_layout()
    plt.show()
