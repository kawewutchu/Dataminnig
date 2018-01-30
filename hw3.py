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
    index = 1
    for line in file:
        if(line.split(",")[13] != '?'):
            attb13.append(int(line.split(",")[13]))
        else:
            attb13.append(184)                              # mean of attb14 with out missing value
            print("position "+str(index) + ": 184" )
        attb14.append(index)
        index += 1

    p3 = polyfit(attb14, attb13, 3)
    axis([-100, max(attb14) + 100, -100, max(attb13)+ 100])
    plot(attb14, attb13, 'ro')
    plot(attb14, polyval(p3, attb14))
    show()

if (__name__ == "__main__"):
    ReadData()