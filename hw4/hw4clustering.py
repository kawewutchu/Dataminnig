from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
np.set_printoptions(precision=1)
def fill_missing_value(data):
    fill_index_by_near = ['Year', 'Genre', 'Sequel','Sentiment']
    fill_index_by_mean = ['Gross','Budget', 'Screens', 'Views', 'Likes','Dislikes', 'Comments','Aggregate Followers']
    
    for fill_index in fill_index_by_near:
        data[fill_index] = data[fill_index].fillna(method='bfill')
    
    for fill_index in fill_index_by_mean:
        data[fill_index] = data.groupby(['Ratings'])[fill_index]\
    .transform(lambda x: x.fillna(x.mean()))

    fill_test = data
    # print(fill_test)
    # print(data)
    MEAN_DATA = data.mean()
    STD_DATA = data.std()
    # data = data.drop(['Ratings'], axis=1)
    data = (data-data.mean())/(data.std())
    data = data*10
    return data

def dist(a, b, ax=1):
        return np.linalg.norm(a - b, axis=ax)

def clustering(k,X):
    #create cluster center  x y z
    C1 = np.random.randint(0, np.max(X)-40 ,size=k)
    C2 = np.random.randint(0, np.max(X)-40 , size=k)
    C3 = np.random.randint(0, np.max(X)-40 , size=k)
    C = np.array(list(zip(C1, C2, C3)), dtype= np.int64)
    # C = np.array(list(zip(C1, C2)), dtype= np.int64)

    #create old cluster center
    C_old = np.zeros(C.shape)
    clusters = np.zeros(len(X))

    #cluculate  error to check move new  cluster center
    error = dist(C, C_old, None)

    #if not create new cluster center, end while loop 
    while error != 0:
        for i in range(len(X)):
            #find distances and you min distances to cluster i
            distances = dist(X[i], C)
            # print(distances)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        C_old = deepcopy(C)
        for i in range(k):
            #split data group
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            #calulate new cluster center
            C[i] = np.mean(points, axis=0)

        error = dist(C, C_old, None)
        print(error)
        if(error == 1):
             error = dist(C, C, None)

    plt.rcParams['figure.figsize'] = (16, 9)
    plt.style.use('ggplot')

    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    fig, ax = plt.subplots()
    for i in range(k):
            points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
            ax.scatter(points[:, 0], points[:, 1], s=50, c=colors[i])
    ax.scatter(C[:, 0], C[:, 1], marker='X', s=200, c='#050505')
    plt.show()

    fig, ax = plt.subplots()
    for i in range(k):
            points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
            ax.scatter(points[:, 1], points[:, 2], s=50, c=colors[i])
    ax.scatter(C[:, 1], C[:, 2], marker='X', s=200, c='#050505')
    plt.show()

    fig, ax = plt.subplots()
    for i in range(k):
            points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
            ax.scatter(points[:, 0], points[:, 2], s=50, c=colors[i])
    ax.scatter(C[:, 0], C[:, 2], marker='X', s=200, c='#050505')
    plt.show()

if __name__ == "__main__":
        #read data from 
    data = pd.read_csv('xclara.csv')
    data = data.drop(['No.'], axis=1)
    x = data['V1'].values
    y = data['V2'].values
    # X = np.array(list(zip(x,y)), dtype= np.float32)
    # data.head()
    # clustering(6,X)

    data = pd.read_csv('../2014and2015CSMdataset.csv')
    data['index'] = np.arange(1, len(data)+1)
        #drop Movie attribute
    data = data.drop(['Movie'], axis=1)
        #create output form ratings
    output =  data['Ratings']
        #fill missing value and normalize
    data = fill_missing_value(data)
   
        #split data to arttribute 
    Ratings = data['Ratings'].values
    Genre = data['Genre'].values
    Gross = data['Gross'].values
    Budget = data['Budget'].values
    Screens = data['Screens'].values
    Sequel = data['Sequel'].values
    Sentiment = data['Sentiment'].values
    Views = data['Views'].values
    Likes = data['Likes'].values
    Dislikes = data['Dislikes'].values
    Aggregate = data['Aggregate Followers'].values
    Comments = data['Comments'].values

        #compress demention data to 3D
    x = Genre + Gross + Budget
    y = Screens + Sequel + Sentiment + Views
    z = Likes + Dislikes + Comments + Aggregate
        #data to list array
    X = np.array(list(zip(x,y,z)), dtype= np.float32)

    clustering(2,X)
   