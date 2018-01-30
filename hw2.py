from pandas import DataFrame
import xlwt
import os
import matplotlib.pyplot as plt
import pandas as pd

writer = pd.ExcelWriter('pandas_multiple.xlsx', engine='xlsxwriter')
def toexcel(e, startrow, startcol):
        label = []
        value = []
        size = []
        for i in range(0, 10):
            print(len(e[i]))
            key = "bin " + str(i+1) + ": "
            label.append(key)
            value.append(str(e[i]).replace("[","").replace("]",""))
            size.append(len(e[i]))

        # df = DataFrame({'bin':label, 'value':value, 'frequency': size})
        # df.to_excel(writer, sheet_name='sheet2', index=False, header=True, startrow=startrow, startcol=startcol)   


def ReadData():
    file = open("./crx.data.txt", "r")
   
    attb2 = []
    ew = {}
    ed = {}
    bm = {}
    bb = {}
    indexdep = 0
    countdep = 1
    for line in file:
        attb2.append(float(line.split(",")[2]))
  
    attb2 = sorted(attb2)
    maximum = max(attb2)
    minimum = min(attb2)
    width = int(abs(minimum - maximum) / 10) + 1
    
    for df in attb2:
        mini = minimum
        for i in range(0, 10):
                if(df >= mini and df < mini + 3):
                    tmp = ew.get(i)
                    if tmp is None:
                        ew[i] = []
                    ew[i].append(df)
                mini = mini + 3
        
        tmp = ed.get(indexdep)
        if tmp is None:
            ed[indexdep] = []
        ed[indexdep].append(df)
        if(countdep % 69 == 0):
            indexdep += 1
        countdep += 1
        
    for i in range(0, 10):
        mean = sum(ew[i])/len(ew[i]) * 1.00
        mean = float("{0:.2f}".format(mean))
        for j in range(0, len(ew[i])):
            tmp = bm.get(i)
            if tmp is None:
                bm[i] = []
            bm[i].append(mean)
    
    for i in range(0, 10):
        tmp = bb.get(i)
        if tmp is None:
            bb[i] = []
        bb[i].append(ed[i][0])
        for j in range(1, len(ed[i]) - 1):
            bb[i].append(ed[i][0])
        bb[i].append(ed[i][len(ed[i])-1])
 
    toexcel(ew,0,0)
    toexcel(bm,24,0)

    toexcel(ed,12,0)
    toexcel(bb,36,0)
    writer.save()
    
    # df = DataFrame({'name':label,'bin':value})
    # df.to_excel('580610616_hw1_1_2.xls', sheet_name='sheet2', index=False, header=True)

    
        
if (__name__ == "__main__"):
    ReadData()
   
    # l1 = ["1","2","3","4","5"]
    # df = DataFrame(l1)
    # df.to_excel('test.xlsx', sheet_name='sheet1', index=False, header=False)