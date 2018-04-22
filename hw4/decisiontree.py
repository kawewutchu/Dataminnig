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
    #partition x, according to attribute values x_i
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float')/len(x)
    #calculate a weighted average of the entropy
    for p, v in zip(freqs, val):
        res -= p * entropy(y[x == v])

    return res

def is_pure(s):
    return len(set(s)) == 1

def recursive_split(x, y):
    # If there could be no split, just return the original set
    if is_pure(y) or len(y) == 0:
        return y
    #get attribute that gives the highest mutual information
    gain = np.array([mutual_information(y, x_attr) for x_attr in x.T])
    selected_attr = np.argmax(gain)

    # If there's no gain at all, nothing has to be done, just return the original set
    if np.all(gain < 1e-6):
        return y
    #split using the selected attribute
    sets = partition(x[:, selected_attr])

    res = {}
    for k, v in sets.items():
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)
        #create decision tree
        res["%d:%d" % (selected_attr, k)] = recursive_split(x_subset, y_subset)

    return res

#calculate error
def check_testcase(decision_tree, data_test, output_test):
    # pprint(decision_tree)
    error = 0.0
    per_error = 0.0
    count = 0
    data_test = data_test.astype(int)
    print(output_test)
    for i,datas in enumerate(data_test):
        error = 0.0
        #prediction answer
        answer = predic(datas)
        print(answer)
        try:
            #calculate percentage error
            
            error = abs(output_test[i] - answer[0])
            per_error += (100*error)/ output_test[i]
        except:
            pass
    #calculate mean percentage error
    # print(per_error/len(data_test))

#prediction output
def predic(datas):
    decision = 0
    #travel to decition tree
    for col, data in enumerate(datas):
            try:  
                decision = decision_tree["%d:%d" % (col, data)]
                for col, data in enumerate(datas):
                    try:  
                        decision2 = decision["%d:%d" % (col, data)]
                        return decision2
                    except:
                        pass
                if type(decision) is dict:
                    return 6.0
            except:
                pass

    return decision

if __name__ == "__main__":
    #preprocess
         #load data from csv
    data = pd.read_csv('../2014and2015CSMdataset.csv')
    data['index'] = np.arange(1, len(data)+1)
        #drop Movie attribute
    data = data.drop(['Movie'], axis=1)
        #create output form ratings
    output =  data['Ratings']
        #fill missing value 
    data = fill_missing_value(data)
    output = np.around(output, decimals=1)
    fill_index_by_near = ["Year","Genre","Gross","Budget","Screens","Sequel","Sentiment","Views","Likes","Dislikes","Comments","Aggregate Followers"]
    d = []
        #tranpost matrix
    for fill_index in fill_index_by_near:
       a = ( data[fill_index])
       d.append(a.as_matrix())
    datatest = np.array(d)
        #csc data to matrix
    data = data.as_matrix()
    output = output.as_matrix()
        #binnig out put 
    output = output.astype(int)
        #grop data 
    datatest = np.around(datatest, decimals=1)

    # split data to 7 section
    data_split =  np.split(datatest.T, 7)
    output_split =  np.split(output, 7)

    for i in range(7):
        data_test = []
        output_test = []
        data_train = []
        output_train = []
        # create test data and output 
        output_test = output_split[i]
        data_test = data_split[i]
        first_data = True
        first_output = True
        # create train data and output 
        for j in range(7):
            if(j != i):
                for data in data_split[j]:
                    if first_data:
                        data_train = data
                        first_data = False
                    else:
                        data_train = np.vstack((data_train, data))
                for output in output_split[j]:
                    if first_output:
                        output_train = output
                        first_output = False
                    else:
                        output_train = np.hstack((output_train, output))
        decision_tree = recursive_split(data_train, output_train)
        check_testcase(decision_tree, data_test, output_test)
   

   
#Movie,Year,Ratings,Genre,Gross,Budget,Screens,Sequel,Sentiment,Views,Likes,Dislikes,Comments,Aggregate Followers