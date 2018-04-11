import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from sklearn.utils import shuffle
from scipy import stats

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def neural_network(datas, output, local_var_num_attb, local_var_num_hidden,local_var_num_hidden2, local_var_num_output, test, testoutput):
    X = datas
    y = output
    np.random.seed(1)

    syn0 = 2*np.random.random((local_var_num_attb,local_var_num_hidden)) - 1
    syn1 = 2*np.random.random((local_var_num_hidden,local_var_num_hidden2)) - 1
    syn2 = 2*np.random.random((local_var_num_hidden2,local_var_num_output)) - 1
    # print(syn0)
    # print(syn1)
    for j in xrange(10000):
        l0 = X
        l1 = nonlin(np.dot(l0,syn0))
        l2 = nonlin(np.dot(l1,syn1))
        l3 = nonlin(np.dot(l2,syn2))

        l3_error = y - l3
        
        l3_delta = 0.1*l3_error*nonlin(l3,deriv=True)
        l2_error = 0.1*l3_delta.dot(syn2.T)
        l2_delta = l2_error * nonlin(l2,deriv=True)
        l1_error = 0.1*l2_delta.dot(syn1.T)
        l1_delta = l1_error * nonlin(l1,deriv=True)
        syn2 += l2.T.dot(l3_delta)
        syn1 += l1.T.dot(l2_delta)
        syn0 += l0.T.dot(l1_delta)
    
    l0 = test
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))
    l3 = nonlin(np.dot(l2,syn2))
    for index,l in enumerate(l3):
        print(l, testoutput[index])


if __name__ == "__main__":
    data = pd.read_csv('iris.data')
    data = data.sample(frac=1).reset_index(drop=True)
    output =  data['class'].as_matrix()
    data = data.drop(['class'], axis=1)
    data = data.as_matrix()

    test = pd.read_csv('testcase.data')
    test = test.sample(frac=1).reset_index(drop=True)
    testoutput =  test['class'].as_matrix()
    test = test.drop(['class'], axis=1)
    test = test.as_matrix()
    class_isis = { "Iris-virginica": [1, 0, 0 ] ,
                    "Iris-versicolor": [0, 1, 0],
                    "Iris-setosa": [0 , 0 ,1] }
    output2 = []
    testoutput2 = []
    for d in output:
        output2.append(class_isis[d])

    output3 = np.array(output2)
    for d in testoutput:
        testoutput2.append(class_isis[d])
    
    testoutput3 = np.array(testoutput2)
    print(data)
    print(output3)
    neural_network(data, output3, 4, 5, 4 , 3, test, testoutput2)