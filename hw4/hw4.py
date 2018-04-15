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
    # print(fill_test)
    # print(data)
    MEAN_DATA = data.mean()
    STD_DATA = data.std()
    data = data.drop(['Ratings'], axis=1)
    data = (data-data.mean())/(data.std())
    return data

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def neural_network(datas, output, local_var_num_attb, local_var_num_hidden,local_var_num_hidden2,local_var_num_hidden3, local_var_num_output, data_test,  output_test):
    X = datas
    y = output
    np.random.seed(1)
    LEARNING_LATE = 0.5
    syn0 = 2*np.random.random((local_var_num_attb,local_var_num_hidden)) - 1
    syn1 = 2*np.random.random((local_var_num_hidden,local_var_num_hidden2)) - 1
    syn2 = 2*np.random.random((local_var_num_hidden2,local_var_num_hidden3)) - 1
    syn3 = 2*np.random.random((local_var_num_hidden3,local_var_num_output)) - 1
    # print(syn0)
    # print(syn1)
    for j in xrange(500000):
        l0 = X
        l1 = nonlin(np.dot(l0,syn0))
        l2 = nonlin(np.dot(l1,syn1))
        l3 = nonlin(np.dot(l2,syn2))
        l4 = nonlin(np.dot(l3,syn3))
        # print(y)
        # print(l3, y)
        l4_error = y - l4
        # if(j % 10000 == 0):
        #     print("----------------")
        #     print(l4_error)
        l4_delta = LEARNING_LATE*l4_error*nonlin(l4,deriv=True)
        l3_error = LEARNING_LATE*l4_delta.dot(syn3.T)
        l3_delta = l3_error * nonlin(l3,deriv=True)
        l2_error = LEARNING_LATE*l3_delta.dot(syn2.T)
        l2_delta = l2_error * nonlin(l2,deriv=True)
        l1_error = LEARNING_LATE*l2_delta.dot(syn1.T)
        l1_delta = l1_error * nonlin(l1,deriv=True)
        syn3 += l3.T.dot(l4_delta)
        syn2 += l2.T.dot(l3_delta)
        syn1 += l1.T.dot(l2_delta)
        syn0 += l0.T.dot(l1_delta)

    # l0 = X
    # l1 = nonlin(np.dot(l0,syn0))
    # l2 = nonlin(np.dot(l1,syn1))
    # l3 = nonlin(np.dot(l2,syn2))
    # l4 = nonlin(np.dot(l3,syn3))
    # # print(y)
    # # print(l3, y)
    # l4 = np.around(l4, decimals=2)
    # l4_error = y - l4
    # for index,l in enumerate(l4):
    #     print(l, output[index], l4_error[index])
    # print("/////////////////////////////////////////////")
    l0 = data_test
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))
    l3 = nonlin(np.dot(l2,syn2))
    l4 = nonlin(np.dot(l3,syn3))
    l4 = np.around(l4, decimals=2)
    l4_error = output_test - l4
    error = ((100 * l4_error) / l4)
    # print(error)
    meam_error = 0.0
    for e in error:
        meam_error += abs(e[0])
    
        # print(error)
    
    print(meam_error/len(l4))
        # print(l, output_test[index], l4_error[index])

    # print(l4_error)

if __name__ == "__main__":
    data = pd.read_csv('../2014and2015CSMdataset.csv')
    data['index'] = np.arange(1, len(data)+1)
    data = data.drop(['Movie'], axis=1)
    output =  data['Ratings']
    data = fill_missing_value(data)
    output = output
    data = data.as_matrix()
    output = output.as_matrix()

    output2 = []
    testoutput2 = []
    for d in output:
        # d2 = math.modf(d)
        output2.append([d/10])
        # ([d2[0], d2[1]/10])
    output = np.array(output2)

    data_split =  np.split(data, 7)
    output_split =  np.split(output, 7)

    # test = data_split[0]
    # outputtest = output_split[0]
    # print(data)
    for i in range(7):
        print(i)
        data_test = []
        output_test = []
        data_train = []
        output_train = []
        # print(data_split[i])
        # print(output_split[i])
        output_test = output_split[i]
        data_test = data_split[i]
        first_data = True
        for j in range(7):
            if(j != i):
                if first_data:
                    data_train = data_split[j]
                    output_train = output_split[j]
                    first_data = False
                else:
                    data_train = np.concatenate((data_train, data_split[j]), axis=0)
                    output_train = np.concatenate((output_train, output_split[j]), axis=0)
        # print(len(data_test))
        # print(len(output_test))
        neural_network(data_train, output_train, 13, 8, 6, 4, 1, data_test, output_test)
    # print(data)
    # output = output_split[0]
    # print(len(data))
   
#Movie,Year,Ratings,Genre,Gross,Budget,Screens,Sequel,Sentiment,Views,Likes,Dislikes,Comments,Aggregate Followers
    