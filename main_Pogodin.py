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
import random

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

def set_genre_ranking(df: pd.DataFrame):
    row_count = len(df.loc[:, 'genre'])
    genre_ranks = {}
    for row in df.loc[:, 'genre']:
        genre_list = row.split('|')
        for genre in genre_list:
            if genre not in genre_ranks.keys(): 
                genre_ranks[genre] = 0
            genre_ranks[genre] += 1
    for genre in genre_ranks.keys():
        genre_ranks[genre] = genre_ranks[genre] / row_count
    genre_rank_series = pd.Series([])
    for i in range(len(df)):
        genres_list = df.loc[:, 'genre'][i].split('|')
        temp_val = 0
        for genre in genres_list:
            temp_val += genre_ranks[genre]
        genre_rank_series[i] = temp_val
    df.insert(len(df.columns), 'genre_rank', genre_rank_series)


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
    
    plt.scatter(x, y, c=labels, s=df.loc[:, 'awards_nominations'])
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
            'genre_rank'
            # 'metascore'
            ]].to_numpy()
    y = df['metascore'].to_numpy()
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)
    regressor = RandomForestRegressor(n_estimators=250, random_state=0)
    
    regressor.fit(x_train, y_train)
    
    y_pred_r = regressor.predict(x_test)
    
    print("R_Правильность на обучающем наборе: {:.3f}".format(regressor.score(x_train, y_train)))
    print("R_Правильность на тестовом наборе: {:.3f}".format(regressor.score(x_test, y_test)))
    
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_r))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_r))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_r)))


def fuzzy_logic(df: pd.DataFrame):
    
    pass


def get_collaborative_recommendation(fav_films, film_corr, film_titles):
    film_similarities = np.zeros(film_corr.shape[0])
    for fav_film in fav_films:
        film_index = film_titles.index(fav_film)
        film_similarities += film_corr[film_index]
    film_preferences = []
    for i in range(len(film_titles)):
        film_preferences.append((film_titles[i], film_similarities[i]))
    return sorted(film_preferences, key=lambda x: x[1], reverse = True)


def collaborative_filtering(df: pd.DataFrame):
    pivot_table = df.pivot_table(index=['movie_id'], 
                    columns=['movie'], 
                    values='genre_rank', fill_value=0)
    film_corr = np.corrcoef(pivot_table.T)
    film_list = list(pivot_table)
    film_titles = []
    
    for i in range(len(film_list)):
        film_titles.append(film_list[i])
    
    my_fav_films = []
    for _ in range(10):
        rand_index = random.randrange(0, len(film_titles))
        my_fav_films.append(film_titles[rand_index])
        
    print('-'*50)
    print('Your favorite films')
    for film in my_fav_films:
        film_genres = df.loc[df['movie'] == film].loc[:, 'genre'].to_string(index=False)
        print(f'{film} | {film_genres}')
        
    print('-'*50)
    film_recommendations = get_collaborative_recommendation(my_fav_films, film_corr, film_titles)
    film_recommendations = list(filter(lambda film: film[0] not in my_fav_films, film_recommendations))[:10]
    print('-'*50)
    print('Films you should like: ')
    for pref_film in film_recommendations:
        film_genres = df.loc[df['movie'] == pref_film[0]].loc[:, 'genre'].to_string(index=False)
        print(f'{pref_film[0]} | {film_genres}')
    print('-'*50)

def main():
    df = pd.read_csv('dataset.csv')
    df = df[['movie_id', 'movie', 'certificate', 'duration', 'gross', 
             'genre', 'rate', 'metascore', 'popularity', 
             'awards_nominations', 'awards_wins', 
             'votes']]
    set_genre_ranking(df)
    # genres_list = read_genres_from_df(df)
    # df_with_genres = df.copy()
    # print(list(df.loc[:, 'genre'])[25])
    
    # for i in range(len(df.loc[:, 'genre'])):
    #     temp_genres_list = df.loc[:, 'genre'][i].split('|')
    #     temp_value = 0
    #     temp_genres = []
    #     for genre in temp_genres_list:
    #         temp_value += genres_list.index(genre)
    #         temp_genres.append(genre)
    #     df            .loc[:, 'genre'][i] = temp_value
    #     df_with_genres.loc[:, 'genre'][i] = temp_genres
        
    # df.loc[:, 'genre'] = pd.to_numeric(df.loc[:, 'genre'])
    df.dropna(inplace=True)
    
    # clustering_KMeans(df, 5, 'genre_rank', 'rate')
    clustering_KMeans(df, 3, 'gross', 'genre_rank')
    
    # random_forest(df)
    
    # collaborative_filtering(df)
    
    # fuzzy_logic(df)

if __name__ == '__main__':
    main()
