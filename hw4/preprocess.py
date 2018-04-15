import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
MEAN_DATA = []
STD_DATA = []
fill_test = {}

def fill_missing_value(data):
    fill_index_by_near = ['Year', 'Genre', 'Sequel','Sentiment']
    fill_index_by_mean = ['Gross','Budget', 'Screens', 'Views', 'Likes','Dislikes', 'Comments','Aggregate Followers']
    
    for fill_index in fill_index_by_near:
        data[fill_index] = data[fill_index].fillna(method='bfill')
    
    for fill_index in fill_index_by_mean:
        data[fill_index] = data.groupby(['Ratings'])[fill_index]\
    .transform(lambda x: x.fillna(x.mean()))

    fill_test = data
    print(fill_test)
    print(data)
    MEAN_DATA = data.mean()
    STD_DATA = data.std()
    data = data.drop(['Ratings'], axis=1)
    data = (data-data.mean())/(data.std())
    return data

if __name__ == "__main__":
    data = pd.read_csv('../2014and2015CSMdataset.csv')
    data['index'] = np.arange(1, len(data)+1)
    data = data.drop(['Movie'], axis=1)
    output =  data['Ratings']
    data = fill_missing_value(data)
    output = output

    data = data.as_matrix()
    output = output.as_matrix()

    test = 
    neural_network(data, output, 13, 12, 10, 8, 1)
#Movie,Year,Ratings,Genre,Gross,Budget,Screens,Sequel,Sentiment,Views,Likes,Dislikes,Comments,Aggregate Followers
    