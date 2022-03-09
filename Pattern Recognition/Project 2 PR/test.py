import numpy as np
import matplotlib.pyplot as plt
import cvxopt as cv
#https://xavierbourretsicotte.github.io/SVM_implementation.html
DataX = np.array([[1,1],
                  [1,-1],
                  [2,1],
                  [3,1],
                  [3,-1],
                  [4,1]])

DataC = np.array([1,1,1,-1,-1,-1])


plt.figure(1)

for i in range(len(DataX)):
    if DataC[i] == 1:
        plt.plot(DataX[i][0],DataX[i][1],'ro')
    else:
        plt.plot(DataX[i][0],DataX[i][1],'bo')

plt.xlabel('x1')
plt.ylabel('x2')
plt.show()