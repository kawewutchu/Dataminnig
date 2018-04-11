import numpy as np
import pandas as pd

def shuffler(filename):
  df = pd.read_csv(filename, header=0)
  # return the pandas dataframe
  return df.reindex(np.random.permutation(df.index))


def main(outputfilename):
  shuffler('iris.data').to_csv(outputfilename, sep=',')

if __name__ == '__main__': 
  main('iris-shuffler.csv')