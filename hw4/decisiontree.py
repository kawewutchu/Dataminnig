import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from pprint import pprint


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
    data = data.drop(['Ratings'], axis=1)
    data = (data-data.mean())/(data.std())
    data = data*10
    return data

y = np.array([0, 0])

def partition(a):
    return {c: (a==c).nonzero()[0] for c in np.unique(a)}

def entropy(s):
    res = 0
    val, counts = np.unique(s, return_counts=True)
    freqs = counts.astype('float')/len(s)
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res

def mutual_information(y, x):
    res = entropy(y)
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float')/len(x)
    for p, v in zip(freqs, val):
        res -= p * entropy(y[x == v])

    return res

def is_pure(s):
    return len(set(s)) == 1

def recursive_split(x, y):
    if is_pure(y) or len(y) == 0:
        return y

    gain = np.array([mutual_information(y, x_attr) for x_attr in x.T])
    selected_attr = np.argmax(gain)

    if np.all(gain < 1e-6):
        return y

    # We split using the selected attribute
    sets = partition(x[:, selected_attr])

    res = {}
    for k, v in sets.items():
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)
        res["%d -> %d" % (selected_attr, k)] = recursive_split(x_subset, y_subset)
    return res

if __name__ == "__main__":
    data = pd.read_csv('../2014and2015CSMdataset.csv')
    data['index'] = np.arange(1, len(data)+1)
    data = data.drop(['Movie'], axis=1)
    print(data)
    # data = data.drop(['Dislikes'], axis=1)
    output =  data['Ratings']
    data = fill_missing_value(data)
    output = output
    output = np.around(output, decimals=1)
    fill_index_by_near = ["Year","Genre","Gross","Budget","Screens","Sequel","Sentiment","Views","Likes","Dislikes","Comments","Aggregate Followers"]
   
    d = []
    for fill_index in fill_index_by_near:
       a = ( data[fill_index])
       d.append(a.as_matrix())
    datatest = np.array(d)
    data = data.as_matrix()
    output = output.as_matrix()
 
    datatest = np.around(datatest, decimals=1)
    data_split =  np.split(datatest, 7, axis=1)
    output_split =  np.split(output, 7)
    print(data_split[0])
    print(output_split[0])
    # decision_tree = recursive_split(datatest.T, output)
    # pprint(decision_tree)
    # for d in decision_tree:
    #     print(d['4 -> 5'])

   
#Movie,Year,Ratings,Genre,Gross,Budget,Screens,Sequel,Sentiment,Views,Likes,Dislikes,Comments,Aggregate Followers