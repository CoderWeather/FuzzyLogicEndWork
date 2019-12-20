# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from sklearn import tree

import pandas as pd
import skfuzzy as fuzz
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
    
    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()),
                           zorder=1)

    ax.set(xlim=xlim, ylim=ylim)


def random_forest(df: pd.DataFrame, x1_label: str, x2_label: str, y_label: str):
    # fig, axes = plt.subplots()
    # x = df[[x1_label, x2_label]].to_numpy()
    # y = df[y_label].to_numpy()
    x = df[['metascore', 'awards_nominations', 'awards_wins', 'votes']].to_numpy()
    print(x)
    y = df['rate'].to_numpy()
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    print(len(x_train), len(x_test))
    
    forest = RandomForestClassifier(n_estimators=250, random_state=1, n_jobs=-1)
    # forest.fit(x_train, y_train)
    
    # y_predict = forest.predict(x_test)
    # print('Accuracy score: ', accuracy_score(y_test, y_predict))
    
    # print("Правильность на обучающем наборе: {:.3f}".format(forest.score(x_train, y_train)))
    # print("Правильность на тестовом наборе: {:.3f}".format(forest.score(x_test, y_test)))
    
    
    visualize_classifier(forest, x, y)
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
    #     os.system('C:\\Program Files (x86)\\Graphviz2.38\\bin\\dot.exe -Tpng tree.dot -o tree.png')
    #     i_tree +=1
    
    
    # fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    # for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    #     ax.set_title("Дерево {}".format(i))
    #     mglearn.plots.plot_tree_partition(x_train, y_train, tree, ax=ax)
    # mglearn.plots.plot_2d_separator(forest, x, fill=True, ax=axes[-1, -1], alpha=0.4)
    # axes[-1, -1].set_title("Случайный лес")
    # mglearn.discrete_scatter(x_train[:, 0], x_train[:, 1], y_train)
    


def fuzzy_logic(df: pd.DataFrame):
    pass


def main():
    df = pd.read_csv('dataset.csv')
    df = df[['year', 'movie', 'certificate', 'duration',
             'genre', 'rate', 'metascore', 
             'awards_nominations', 'awards_wins', 'votes', 'release_date']]
    genres_list = read_genres_from_df(df)

    for i in range(len(df.genre)):
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
    random_forest(df, 'genre', 'duration', 'metascore')

if __name__ == '__main__':
    main()
