from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from matplotlib.colors import ListedColormap
import graphviz
import pandas as pd 
import os



def dataset(path):
    df = pd.read_csv(path)
    X = df.drop("Water",axis=1)
    Y = df["Water"]
    print(X)
    print(Y)
    return X, Y

# plt.figure(figsize=(12, 8))
# mglearn.discrete_scatter(X[:, 0], X[:, 1], Y)
# plt.show()

def model_val(X_train, X_test, Y_train, Y_test):
    # clf_model = DecisionTreeClassifier(criterion="gini",

    #                               splitter="best",

    #                               max_depth=3,

    #                               min_samples_split=3,

    #                               min_samples_leaf=1,

    #                               min_weight_fraction_leaf=0.0,

    #                               max_features=4,

    #                               random_state=None,

    #                               max_leaf_nodes=8,

    #                               class_weight="balanced")
    clf_model = DecisionTreeClassifier(max_depth=3)
    clf_model.fit(X_train, Y_train)
    return clf_model


def visualize(clf_model):
    dot_data = export_graphviz(clf_model)
    graph = graphviz.Source(dot_data)
    graph.render('all_riku_one', format='png')


if __name__ == "__main__":
    path = "all_riku_one.csv"  #0107_forRIKU_hunmu_1kukan   0107_forRIKU_hunmu_2kukan
    X, Y = dataset(path)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
    clf_model = model_val(X_train, X_test, Y_train, Y_test)
    print(clf_model.score(X_test, Y_test))
    visualize(clf_model)
