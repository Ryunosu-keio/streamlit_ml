import streamlit as st
import tree as tr
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
import warnings
from PIL import Image

st.title("機械学習")
st.sidebar.title("ハイパーパラメータ")

tab1, tab2, tab3, tab4 = st.tabs(["決定木", "ランダムフォレスト", "SVM", "NN"])



warnings.simplefilter('ignore')

@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')


# param_edit = st.sidebar.checkbox("ハイパーパラメータの設定", False)

depth = st.sidebar.slider('木の深さ', 1, 10, (2, 4))
# max_depth = [i for i in depth]
min_split = st.sidebar.slider('木の分け方', 1, 10, (2, 3))

leaf = st.sidebar.slider('葉の深さ', 1, 10, (1, 2))

random_state = st.sidebar.slider('randomstate', 0, 30, (0, 3))



params = {
        "criterion":["gini", "entropy"],
        # "splitter":"best",
        "max_depth":[i for i in range(depth[0], depth[1])],
        "min_samples_split":[i for i in range(min_split[0], min_split[1])],
        "min_samples_leaf":[i for i in range(leaf[0], leaf[1])],
        # "min_weight_fraction_leaf":0.0,
        # "max_features":4,
        "random_state":[i for i in range(random_state[0], random_state[1])],
        # "max_leaf_nodes":8,
        # "class_weight":"balanced"
        }

with tab1:
    st.header("決定木")
    # file1 = st.checkbox("ファイルをアップロード",False)
    uploaded_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False)
    if uploaded_file :
        # df2 = pd.read_csv(uploaded_file)
        # df2
        df = pd.read_csv(uploaded_file)
        features = df.columns
        target = st.selectbox("目的変数を選択してください", features)
        removal_feature = st.multiselect("説明変数として使わない変数を選択してください", features)
        name = st.text_input("ファイル名を入力してください")
        if st.button("モデル構築"):
            # name = uploaded_file.split(".")[0]
            X, Y, features = tr.dataset(df, target, removal_feature)
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
            clf = tr.grid_search(X_train, Y_train, params)
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
            st.write(clf_model.score(X_train, Y_train))
            pred_test = clf_model.predict(X_test)
            st.write(clf_model.score(X_test, Y_test))
            test_conf = confusion_matrix(Y_test, pred_test)
            test_conf
            df = pd.DataFrame(test_conf)
            csv = convert_df(df)
            st.download_button(
            label="Download data",
            data=csv,
            file_name="test.csv",
            mime="text/csv",
            )
            clf.best_params_
            graph = tr.visualize(clf_model, features)
            # image = Image.open('output/test.png')
            # st.image(image, caption='サンプル',use_column_width=True)
            st.graphviz_chart(graph)
            # if st.checkbox("重要度"):
            fig = tr.importance(clf_model, features)
            st.pyplot(fig)

        
