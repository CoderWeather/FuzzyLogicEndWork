# -*- coding: utf-8 -*-

import pandas as pd
import mglearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import numpy as np

from sklearn.tree import export_graphviz
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
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='r')


def random_forest(df: pd.DataFrame, x1_label: str, x2_label: str, y_label: str):
    # fig, axes = plt.subplots()
    x = df[[x1_label, x2_label]].to_numpy()
    y = df[y_label].to_numpy()
    
    # plt.xlabel(x_label)
    # plt.ylabel(y_label)
    
    
    forest = RandomForestClassifier(n_estimators=100, random_state=2, 
                                    n_jobs=-1, max_features='sqrt')
    forest.fit(x, y)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
        ax.set_title("Дерево {}".format(i))
        mglearn.plots.plot_tree_partition(x, y, tree, ax=ax)
    mglearn.plots.plot_2d_separator(forest, x, fill=True, ax=axes[-1, -1], alpha=0.4)
    axes[-1, -1].set_title("Случайный лес")
    mglearn.discrete_scatter(x[:, 0], x[:, 1], y)


def fuzzy_logic(df: pd:DataFrame):
    pass


def main():
    df = pd.read_csv('dataset.csv')
    df = df[['year', 'movie', 'certificate', 'duration',
             'genre', 'rate', 'metascore', 
             'awards_nominations', 'votes', 'release_date']]
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
    # clustering_KMeans(df, 4, 'duration', 'metascore')
    # clustering_PCA(df_copy)
    # random_forest(df, 'genre', 'duration', 'metascore')

if __name__ == '__main__':
    main()
