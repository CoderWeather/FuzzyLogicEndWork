# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics

import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import mglearn
import matplotlib.pyplot as plt
import random

import warnings
warnings.filterwarnings('ignore')


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
    
    kmeans = KMeans(n_clusters=n_clusters).fit(df[[x_label, y_label]])
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_.astype(float)
    
    # plt.scatter(x, y, c=labels, s=df.loc[:, 'duration'])
    plt.scatter(x, y, c=labels, s=10)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='r', s=100)


def random_forest(df: pd.DataFrame):
    # fig, axes = plt.subplots()
    
    x = df[['rate',
            'awards_nominations', 
            'awards_wins', 
            'gross', 
            'genre_rank'
            ]].to_numpy()
    y = df['metascore'].to_numpy()
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)
    regressor = RandomForestRegressor(n_estimators=250, random_state=0)
    classifier = RandomForestClassifier(n_estimators=250, random_state=0)
    
    regressor.fit(x_train, y_train)
    classifier.fit(x_train, y_train)
    
    y_pred_r = regressor.predict(x_test)
    y_pred_c = classifier.predict(x_test)
    
    print("R_Правильность на обучающем наборе: {:.3f}".format(regressor.score(x_train, y_train)))
    print("R_Правильность на тестовом наборе: {:.3f}".format(regressor.score(x_test, y_test)))
    
    print("C_Правильность на обучающем наборе: {:.3f}".format(classifier.score(x_train, y_train)))
    print("C_Правильность на тестовом наборе: {:.3f}".format(classifier.score(x_test, y_test)))
    
    # print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_r))
    # print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_r))
    # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_r)))


def fuzzy_logic(df: pd.DataFrame, labels: list):
    # fuzzy_objects = {}
    metascore_list = df['metascore'].to_list()
    rate_list = df['rate'].to_list()
    popularity_list = df['popularity'].to_list()
    votes_list = df['votes'].to_list()
    awards_n_list = df['awards_nominations'].to_list()
    awards_w_list = df['awards_wins'].to_list()
    
    metascore =  ctrl.Antecedent(metascore_list,  'metascore')
    rate =       ctrl.Antecedent(rate_list,       'rate')
    popularity = ctrl.Antecedent(popularity_list, 'popularity')
    votes =      ctrl.Antecedent(votes_list,      'votes')
    awards_nominations = ctrl.Antecedent(awards_n_list,      'awards_nominations')
    awards_wins = ctrl.Antecedent(awards_w_list,      'awards_wins')
    
    metascore.automf()
    rate.automf()
    popularity.automf()
    votes.automf()
    awards_wins.automf()
    awards_nominations.automf()
    
    metascore.view()
    # rate.view()
    popularity.view()
    # votes.view()
    awards_wins.view()
    awards_nominations.view()
    
    # rule1 = ctrl.Rule(metascore['good'] | popularity['good'], awards_nominations['good'])
    # rule2 = ctrl.Rule(awards_nominations['good'] | popularity['good'], awards_wins['good'])
    # rule3 = ctrl.Rule(votes['good'] | rate['good'], metascore['good'])
    
    # ctrl_system = ctrl.ControlSystem([rule1, rule2, rule3])
    # sim = ctrl.ControlSystemSimulation(ctrl_system)
    
    # sim.input['metascore'] = random.randrange(min(metascore_list), max(metascore_list))
    # sim.input['awards_nominations'] = random.randrange(min(awards_n_list), max(awards_n_list))
    # sim.input['popularity'] = random.randrange(min(popularity_list), max(popularity_list))
    # sim.input['votes'] = random.randrange(min(votes_list), max(votes_list))
    # sim.input['rate'] = random.uniform(min(rate_list), max(rate_list))
    # sim.input['awards_wins'] = random.randrange(min(awards_w_list), max(awards_w_list))
    
    # sim.compute()


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
    df = pd.read_csv('dataset_Pogodin.csv')
    df = df[['movie_id', 'movie', 'duration', 'gross', 
             'genre', 'rate', 'metascore', 'popularity', 
             'awards_nominations', 'awards_wins', 
             'votes', 'user_reviews', 'critic_reviews']]
    set_genre_ranking(df)
    df.dropna(inplace=True)
    
    # clustering_KMeans(df, 4, 'genre_rank', 'metascore')
    # clustering_KMeans(df, 5, 'awards_nominations', 'awards_wins')
    
    # random_forest(df)
    
    # collaborative_filtering(df)
    
    # fuzzy_logic(df, ['metascore', 'popularity'])

if __name__ == '__main__':
    main()
