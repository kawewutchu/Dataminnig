import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from sklearn.utils import shuffle
from scipy import stats


def fill_missing_value(data):
    # fill_index_by_near = ['Year', 'Genre', 'Sequel','Sentiment']
    # fill_index_by_mean = ['Gross','Budget', 'Screens', 'Views', 'Likes','Dislikes', 'Comments','Aggregate Followers']
    # for fill_index in fill_index_by_near:
    #     data[fill_index] = data[fill_index].fillna(method='bfill')
    # for fill_index in fill_index_by_mean:
    #     data[fill_index] = data.groupby(['Ratings'])[fill_index]\
    # .transform(lambda x: x.fillna(x.mean()))

    # data = data.drop(['Ratings'], axis=1)

    # x = data.values #returns a numpy array
    # min_max_scaler = preprocessing.scale(x, axis=, with_mean=True, with_std=True )
    # x_scaled = min_max_scaler.fit_transform(x)
    # data = pd.DataFrame(data)
    data = (data-data.mean())/(data.std())
    df_to_nparray = data.to_records(index=False)
    print(df_to_nparray)
    # df_to_nparray = preprocessing.scale(df_to_nparray)
    
    return df_to_nparray
def neural_network(datas, output, local_var_num_attb, local_var_num_hidden, local_var_num_output, test, testoutput):
    class_isis = { "Iris-virginica": [1, 0, 0 ] ,
                    "Iris-versicolor": [0, 1, 0],
                    "Iris-setosa": [0 , 0 ,1] }
    #constant
    LEARNIN_RATE = -0.1
    weigths_input = np.random.uniform(0.1,0.9,(local_var_num_attb,local_var_num_hidden))
    delta_w_input = np.full((local_var_num_attb, local_var_num_hidden), 0.1)
    ro_input = np.full((local_var_num_attb, local_var_num_hidden), 0.0)
    #layer1 node
    v_layer1_result = np.full((local_var_num_hidden), 0.0)
    y_layer1_result = np.full((local_var_num_hidden), 0.0)
    ro_layer1 = np.full((local_var_num_hidden), 0.0)
    weigths_layer1 = np.random.uniform(0.1,0.9,(local_var_num_hidden,local_var_num_output))
    delta_w_layer1 = np.full((local_var_num_hidden, local_var_num_output), .0)

    #output node
    e_result = np.full((local_var_num_output), 0.0)
    v_output_result = np.full((local_var_num_output), 0.0)
    y_output_result = np.full((local_var_num_output), 0.0)
    ro_output = np.full((local_var_num_output), 0.0)
    weigths_output = np.full((local_var_num_output), .1)

    delta_w_output = np.full((local_var_num_output), 0.0)
    print(weigths_input)
    for kuy in range(10000):
        for index_data, data_input in enumerate(datas):
            for i in range(0, local_var_num_hidden):
                result = 0.0
                for k in range(0, local_var_num_attb):
                    result += weigths_input[k][i] * data_input[k]
                v_layer1_result[i] = result
                y_layer1_result[i] = 1/(1 +  math.exp(-1 * v_layer1_result[i])) 
            
            for i in range(0, local_var_num_output):
                result = 0.0
                for j in range(0, local_var_num_hidden):
                    result += weigths_layer1[j][i]* y_layer1_result[j]
                v_output_result[i] = result
                y_output_result[i] = 1/(1 +  math.exp(-1 * v_output_result[i])) 
                e_result[i] = y_output_result[i] - (class_isis[output[index_data]][i])
                ro_output[i] = e_result[i] * (y_output_result[i] * (1 - y_output_result[i]))  
            
            for i in range(0, local_var_num_hidden):
                ro_mul_w = 0.0
                for j in range(0, local_var_num_output):
                    ro_mul_w += weigths_layer1[i][j] * ro_output[j]
            
                ro_layer1[i] = y_layer1_result[i] * (1 - y_layer1_result[i]) * ro_mul_w

            for i in range(0, local_var_num_hidden):
                for j in range(0, local_var_num_output):
                    delta_w_layer1[i][j] = LEARNIN_RATE * ro_output[j] *  y_layer1_result[i]
                    weigths_layer1[i][j] = weigths_layer1[i][j] + delta_w_layer1[i][j]     

            for i in range(0, local_var_num_attb):
                for j in range(0, local_var_num_hidden):
                    delta_w_input[i][j] = LEARNIN_RATE * ro_layer1[j] * data_input[i]
                    weigths_input[i][j] = weigths_input[i][j] + delta_w_input[i][j]
                    

if __name__ == "__main__":
    data = pd.read_csv('iris.data')
    data = data.sample(frac=1).reset_index(drop=True)
    # data['index'] = np.arange(1, len(data)+1)
    # data = data.drop(['Movie'], axis=1)
    # print(data)
    output =  data['class']
    # print(data)
    data = data.drop(['class'], axis=1)
    data = data.to_records(index=False)

    test = pd.read_csv('testcase.data')
    test = test.sample(frac=1).reset_index(drop=True)
    # data['index'] = np.arange(1, len(data)+1)
    # data = data.drop(['Movie'], axis=1)
    # print(data)
    testoutput =  test['class']
    # print(data)
    test = test.drop(['class'], axis=1)
    test = test.to_records(index=False)
    # print(output)
    # print(data)
    # print(df_to_nparray)
    # print(output)
    # input_data = fill_missing_value(data)
    neural_network(data, output, 4, 3, 3, test, testoutput)
    
#Movie,Year,Ratings,Genre,Gross,Budget,Screens,Sequel,Sentiment,Views,Likes,Dislikes,Comments,Aggregate Followers
    