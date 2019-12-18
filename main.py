# -*- coding: utf-8 -*-

import pandas as pd
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

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


def clustering(df: pd.DataFrame):
    kmeans = KMeans(n_clusters=4).fit(df)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_.astype(float)
    
    # df.plot.scatter(x=df.rate, y=df.metascore, s=df.awards_nominations, c='red')

    # plt.scatter(df.rate, df.metascore, c=labels, s=df.awards_nominations, alpha=0.5)
    # plt.scatter(centroids[:, 0], centroids[:, 1], s=df.awards_nominations, c='r')
    df['colors'] = kmeans.labels_
    df.plot.scatter('rate', 'metascore', c='colors', colormap='gist_rainbow')
    
#    n = 1000
#    d = pd.DataFrame({
#            'x': np.random.randint(0,100,n),
#            'y': np.random.randint(0,100,n),
#        })
#    
#    m = KMeans(5)
#    m.fit(d)
#    
#    d['cl'] = m.labels_
#    d.plot.scatter('x', 'y', c='cl', colormap='gist_rainbow')

def main():
    df = pd.read_csv('dataset.csv')
    df = df[['year', 'movie', 'certificate', 'duration',
             'genre', 'rate', 'metascore', 
             'awards_nominations', 'votes', 'release_date']]

    for i in range(len(df.genre)):
        df.loc[:, 'genre'][i] = df.loc[:, 'genre'][i].split('|')

    # df_copy = df[['year', 'duration', 'rate', 'metascore', 
    #               'votes', 'awards_nominations']]
    df_copy = df[['rate', 'metascore', 'awards_nominations']]
    df_copy.dropna(inplace=True)
    
    clustering(df_copy)
    

if __name__ == '__main__':
    main()
