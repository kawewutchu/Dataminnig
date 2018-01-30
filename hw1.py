from pandas import DataFrame
import xlwt
import os

def ReadData():
    file = open("./crx.data.txt", "r")
    attb2 = []
    sorce = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    count = 0
    for line in file:
        if((line.split(",")[1]) != "?"):
            attb2.append(float(line.split(",")[1]))
            df = float(line.split(",")[1])
        else:
            count += 1
            attb2.append(31.56)
            df = 31.56
        min = 13.75
        for i in range(0, 10):
            if( i < 8):
                if(df >= min and df < min + 6.67):
                    sorce[i] += 1
            else:
                if(df >= min and df <= min + 6.67):
                    sorce[i] += 1
            
            min = min + 6.67

  
    label = []
    mid = []
    mean = []
    min = 13.75
    for i in range(0, 10):
        label.append(str(min) + "-" + str(min + 6.67))
        minl = (min+(min + 6.67))/2
        mid.append(minl)
        mean.append(minl * sorce[i])
        min = min + 6.67
    
    mean.append(sum(mean)/sum(sorce))
    print(sum(attb2)/len(attb2))
    print(max(attb2))
    # print(min(attb2))
    print(count)
    # df = DataFrame({'rang':label,'score':sorce, 'mid':mid, 'sum':mean})
    # df.to_excel('580610616_hw1_1_2.xls', sheet_name='sheet1', index=False, header=True, engine='xlsxwriter')
if (__name__ == "__main__"):
    ReadData()
    # l1 = ["1","2","3","4","5"]
    # df = DataFrame(l1)
    # df.to_excel('test.xlsx', sheet_name='sheet1', index=False, header=False)