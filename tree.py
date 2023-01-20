from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from matplotlib.colors import ListedColormap
import graphviz
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np


# features = ["section","online", "back car", "small car", 
#             "big car add", "same car add", "num of add alart",
#             "flame size", "adjacent car", "front adjacent", "rear adjacent"]

# params = {
#         "criterion":["gini", "entropy"],
#         # "splitter":"best",
#         "max_depth":[i for i in range(1, 5)],
#         "min_samples_split":[i for i in range(2, 5)],
#         "min_samples_leaf":[i for i in range(1, 10)],
#         # "min_weight_fraction_leaf":0.0,
#         # "max_features":4,
#         "random_state":[i for i in range(0, 30)],
#         # "max_leaf_nodes":8,
#         # "class_weight":"balanced"
#         }
params = {
        "criterion":["gini", "entropy"],
        # "splitter":"best",
        "max_depth":[i for i in range(2, 3)],
        "min_samples_split":[i for i in range(2, 3)],
        "min_samples_leaf":[i for i in range(1, 2)],
        # "min_weight_fraction_leaf":0.0,
        # "max_features":4,
        "random_state":[i for i in range(0, 1)],
        # "max_leaf_nodes":8,
        # "class_weight":"balanced"
        }

def dataset(path, target):
    df = pd.read_csv(path) #csv取得
    X = df.drop(target,axis=1) #説明変数だけ
    Y = df[target] #目的変数
    features = X.columns #特徴量の名前
    return X, Y, features 

# plt.figure(figsize=(12, 8))
# mglearn.discrete_scatter(X[:, 0], X[:, 1], Y)
# plt.show()

# def model_val(X_train, Y_train):
#     clf_model = DecisionTreeClassifier(max_depth=3)
#     clf_model.fit(X_train, Y_train)
#     return clf_model

def visualize(clf_model, features): #枝分かれの可視化図
    dot_data = export_graphviz(clf_model, feature_names=features, class_names=["0","1","2","3","4","5"]) 
    graph = graphviz.Source(dot_data)
    return dot_data
    # graph.render("output/" + name, format='png')


def grid_search(train_X, train_Y, params):
    clf = GridSearchCV(DecisionTreeClassifier(),   # グリッドサーチで決定木を定義
                   params, cv=5)
    clf.fit(train_X, train_Y)
    # best_scores = clf.cv_results_['mean_test_score']
    # best_params = clf.cv_results_['params']
    # best_clf = clf.best_estimator_
    return clf

# def importance_viz(clf):
#     feature = clf.feature_importances_
#     label = features
#     indices = np.argsort(feature)

#     # 特徴量の重要度の棒グラフ
#     fig =plt.figure (figsize = (10,10))

#     plt.barh(range(len(feature)), feature[indices])

#     plt.yticks(range(len(feature)), label, fontsize=14)
#     plt.xticks(fontsize=14)
#     plt.ylabel("Feature", fontsize=18)
#     plt.xlabel("Feature Importance", fontsize=18)

def importance(clf_model, features):
    f_importance = np.array(clf_model.feature_importances_) # 特徴量重要度の算出
    f_importance = f_importance / np.sum(f_importance)  # 正規化(必要ない場合はコメントアウト)
    df_importance = pd.DataFrame({'feature':features, 'importance':f_importance})
    df_importance = df_importance.sort_values('importance', ascending=False) # 降順ソート
    fig = plot_feature_importance(df_importance)
    return fig

def plot_feature_importance(df): 
    fig = plt.figure()
    n_features = len(df)                              # 特徴量数(説明変数の個数) 
    df_plot = df.sort_values('importance')            # df_importanceをプロット用に特徴量重要度を昇順ソート 
    f_importance_plot = df_plot['importance'].values  # 特徴量重要度の取得 
    plt.barh(range(n_features), f_importance_plot, align='center') 
    cols_plot = df_plot['feature'].values             # 特徴量の取得 
    plt.yticks(np.arange(n_features), cols_plot)      # x軸,y軸の値の設定
    plt.xlabel('Feature importance')                  # x軸のタイトル
    plt.ylabel('Feature')  
    # plt.savefig("importance.png")
    return fig

if __name__ == "__main__":
    path = "Sec2_riku.csv"
    target = "Action"
    name = path.split(".")[0]
    X, Y, features = dataset(path, target)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
    clf = grid_search(X_train, Y_train)
    max_depth_ = clf.best_params_["max_depth"]
    criterion_ = clf.best_params_["criterion"]
    min_samples_split_ = clf.best_params_["min_samples_split"]
    min_samples_leaf_ = clf.best_params_["min_samples_leaf"]
    random_state_ = clf.best_params_["random_state"]
    clf_model = DecisionTreeClassifier(
        criterion=criterion_, 
        max_depth=max_depth_, 
        min_samples_split=min_samples_split_,
        min_samples_leaf=min_samples_leaf_,
        random_state=random_state_
        )
    clf_model.fit(X_train, Y_train)
    pred_train = clf_model.predict(X_train)
    print(len(X_train))
    print(len(Y_train))
    print(clf_model.score(X_train, Y_train))
    pred_test = clf_model.predict(X_test)
    print(clf_model.score(X_test, Y_test))
    test_conf = confusion_matrix(Y_test, pred_test)
    print(test_conf)
    print(clf.best_params_)
    visualize(clf_model, features, name)


    f_importance = np.array(clf_model.feature_importances_) # 特徴量重要度の算出
    f_importance = f_importance / np.sum(f_importance)  # 正規化(必要ない場合はコメントアウト)
    df_importance = pd.DataFrame({'feature':features, 'importance':f_importance})
    df_importance = df_importance.sort_values('importance', ascending=False) # 降順ソート
    print(df_importance)
    plot_feature_importance(df_importance)

