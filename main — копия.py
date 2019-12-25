# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import export_graphviz
from sklearn import tree, metrics

import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import mglearn
import os
import matplotlib.pyplot as plt
import six
import pydot

from IPython import display

import warnings
warnings.filterwarnings('ignore')


def read_genres_from_df(data_frame: pd.DataFrame) -> list:
    raw_genres = []
    for genre in data_frame['genre'].to_string().split():
        if not genre.isnumeric():
            raw_genres.append(genre)

    res_genres = []
    for i in range(len(raw_genres)):
        temp_genres = raw_genres[i].split('|')
        res_genres.extend(temp_genres)

    res_genres = list(set(res_genres))
    return res_genres


def clustering_KMeans(df: pd.DataFrame, n_clusters:int, x_label: str, y_label: str):
    fig, axes = plt.subplots()
    
    x = df[x_label]
    y = df[y_label]
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # kmeans = KMeans(n_clusters=5).fit(df[['genre', 'metascore']])
    kmeans = KMeans(n_clusters=n_clusters).fit(df[[x_label, y_label]])
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_.astype(float)
    
    plt.scatter(x, y, c=labels, s=df.awards_nominations)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='v', c='r')


def visualize_classifier(model, X, y, ax=None, cmap='gist_rainbow'):
    ax = ax or plt.gca()
    
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    print('xx', type(xx), xx.shape)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()),
                           zorder=1)

    ax.set(xlim=xlim, ylim=ylim)


def random_forest(df: pd.DataFrame):
    # fig, axes = plt.subplots()
    
    x = df[['rate',
            'duration', 
            'awards_nominations', 
            'awards_wins', 
            'gross', 
            'genre'
            # 'metascore'
            ]].to_numpy()
    y = df['metascore'].to_numpy()
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)
    regressor = RandomForestRegressor(n_estimators=250, random_state=0)
    # classifier = RandomForestClassifier(n_estimators=250, random_state=0)
    regressor.fit(x_train, y_train)
    # classifier.fit(x_train, y_train)
    
    y_pred_r = regressor.predict(x_test)
    # y_pred_c = classifier.predict(x_test)
    # y_proba_c = classifier.predict_proba(x_test)
    
    # for i in range(len(x_test)):
    #     print(y_test[i], y_pred_r[i], y_pred_c[i])
    # print(len(y_test), len(y_pred_r), len(y_pred_c))
    
    print("R_Правильность на обучающем наборе: {:.3f}".format(regressor.score(x_train, y_train)))
    print("R_Правильность на тестовом наборе: {:.3f}".format(regressor.score(x_test, y_test)))
    
    # print("C_Правильность на обучающем наборе: {:.3f}".format(classifier.score(x_train, y_train)))
    # print("C_Правильность на тестовом наборе: {:.3f}".format(classifier.score(x_test, y_test)))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_r))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_r))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_r)))
    # print(confusion_matrix(y_test,y_pred_c))
    # print(classification_report(y_test,y_pred_c))
    # print(accuracy_score(y_test, y_pred_c))
    
    # visualize_classifier(forest, x, y)
    
    # dotfile = six.StringIO()
    # i_tree = 0
    # for tree_in_forest in forest.estimators_:
    #     export_graphviz(tree_in_forest,out_file='tree.dot',
    #                     # feature_names=col,
    #                     filled=True,
    #                     rounded=True)
    #     (graph,) = pydot.graph_from_dot_file('tree.dot')
    #     name = 'tree' + str(i_tree)
    #     graph.write_png(name+  '.png')
    #     os.system(f'C:\\Program Files (x86)\\Graphviz2.38\\bin\\dot.exe -Tpng tree.dot -o tree.png')
    #     i_tree +=1
        
    # fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    # for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    #     ax.set_title("Дерево {}".format(i))
    #     mglearn.plots.plot_tree_partition(x_train, y_train, tree, ax=ax)
    # mglearn.plots.plot_2d_separator(forest, x, fill=True, ax=axes[-1, -1], alpha=0.4)
    # axes[-1, -1].set_title("Случайный лес")
    # mglearn.discrete_scatter(x_train[:, 0], x_train[:, 1], y_train)
    pass


def fuzzy_logic(df: pd.DataFrame):
    fig, _ = plt.subplots()
    
    pass


def collaborative_filtering(df: pd.DataFrame):
    fig, _ = plt.subplots()
    
    pass


def main():
    df = pd.read_csv('dataset.csv')
    df = df[['certificate', 'duration', 'gross', 
             'genre', 'rate', 'metascore', 'popularity', 
             'awards_nominations', 'awards_wins', 
             'votes']]
    genres_list = read_genres_from_df(df)

    for i in range(len(df.loc[:, 'genre'])):
        temp_genres_list = df.loc[:, 'genre'][i].split('|')
        temp_value = 0
        for genre in temp_genres_list:
            temp_value += genres_list.index(genre)
        df.loc[:, 'genre'][i] = temp_value
    df.loc[:, 'genre'] = pd.to_numeric(df.loc[:, 'genre'])
    df.dropna(inplace=True)
    
    # df_copy = df[['year', 'duration', 'rate', 'metascore', 
    #               'votes', 'awards_nominations']]
    # df_copy = df[['genre','duration', 'rate', 'metascore', 'awards_nominations']]
    
    # clustering_KMeans(df, 5, 'genre', 'rate')
    # clustering_KMeans(df, 5, 'genre', 'metascore')
    # clustering_PCA(df_copy)
    
    random_forest(df)
    
    # fuzzy_logic(df)

if __name__ == '__main__':
    main()
