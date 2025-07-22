from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz
import streamlit as st

@st.cache_data
def dataset(df, target, removal): 
    X = df.drop(target, axis=1)
    for col in removal:
        X = X.drop(col, axis=1)
    Y = df[target]
    features = X.columns
    return X, Y, features

@st.cache_data(hash_funcs={BaseEstimator: lambda _: None})
def grid_search(_model: BaseEstimator, train_X, train_Y, params):
    clf = GridSearchCV(_model, params, cv=5)
    clf.fit(train_X, train_Y)
    return clf

def visualize(clf_model, features):
    # 回帰モデルは class_names を指定できないため表示制限
    if isinstance(clf_model, DecisionTreeRegressor):
        return "回帰モデルはグラフ可視化に対応していません。"
    else:
        class_names = [str(i) for i in np.unique(clf_model.classes_)]
        dot_data = export_graphviz(clf_model, feature_names=features,
                                   class_names=class_names,
                                   filled=True, rounded=True,
                                   special_characters=True)
        return graphviz.Source(dot_data)

def importance(clf_model, features):
    f_importance = np.array(clf_model.feature_importances_)
    f_importance = f_importance / np.sum(f_importance)
    df_importance = pd.DataFrame({'feature': features, 'importance': f_importance})
    df_importance = df_importance.sort_values('importance', ascending=False)
    fig = plot_feature_importance(df_importance)
    return fig


def plot_feature_importance(df, top_n=10): 
    # 上位 top_n 件を取得
    df_top = df.nlargest(top_n, 'importance')
    n_features = len(df_top)
    fig_height = max(1, n_features * 0.5)
    fig = plt.figure(figsize=(10, fig_height))

    # 小さい順に並べ替えて水平棒グラフ
    df_plot = df_top.sort_values('importance')
    plt.barh(range(n_features), df_plot['importance'].values, align='center')
    plt.yticks(np.arange(n_features), df_plot['feature'].values, fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel('Feature importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()

    return fig


# テスト実行時
if __name__ == "__main__":
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    X, Y, features = dataset(df, 'target', [])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

    model = DecisionTreeRegressor()
    params = {
        "max_depth": [2, 3, 4],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
        "random_state": [0]
    }

    clf = grid_search(model, X_train, Y_train, params)
    clf_model = model.set_params(**clf.best_params_)
    clf_model.fit(X_train, Y_train)
    pred = clf_model.predict(X_test)

    # 回帰用スコア出力
    rmse = mean_squared_error(Y_test, pred) ** 0.5
    mae = mean_absolute_error(Y_test, pred)
    print("RMSE:", rmse)
    print("MAE:", mae)

    print("特徴量重要度：")
    f_importance = clf_model.feature_importances_
    for f, v in zip(features, f_importance):
        print(f"{f}: {v:.4f}")
