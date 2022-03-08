import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxopt as cv

ED = pd.read_excel("Proj2DataSet.xlsx",header=None)
index = ED.index
rows = len(index)

DataX = np.empty((0,2),float)
DataC = np.empty((0),float)
for i in range(rows):
    if(pd.isna(ED.iloc[i,0]) or pd.isna(ED.iloc[i,1]) or pd.isna(ED.iloc[i,2])):
        print("Data Missing! Skipping row " , i + 2)
    else:
        DataX = np.append(DataX,np.array([[ED.iat[i,0],ED.iat[i,1]]]),0)
        DataC = np.append(DataC, ED.iat[i,2])



plt.figure(1)

for i in range(len(DataX)):
    if DataC[i] == 1:
        plt.plot(DataX[i][0],DataX[i][1],'ro')
    else:
        plt.plot(DataX[i][0],DataX[i][1],'bo')

plt.show()