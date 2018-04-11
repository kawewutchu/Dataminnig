# from pandas import DataFrame
import os
from scipy.interpolate import *
from matplotlib.pyplot import *
from numpy import *

def ReadData():
    file = open("./crx.data.txt", "r")
    attb13 = []
    attb14 = []
    missing13 = 0
    missing14 = 0
    missingindex = []
    index = 1
    for line in file:
        if(line.split(",")[13] != '?'):
            attb13.append(int(line.split(",")[13]))
            attb14.append(index)
        else:
            missingindex.append(index)                          
        index += 1
    
    p3 = polyfit(attb14, attb13, 4)
    axis([-100, max(attb14) + 100, -100, max(attb13)+ 100])
    plot(attb14, attb13, 'ro')
    plot(attb14, polyval(p3, attb14))
    show()
     
    p = poly1d(p3)
    for i in missingindex:
        print("position " + str(i) + ": " + str(p(i)))

if (__name__ == "__main__"):
    ReadData()