import numpy as np
import pandas as pd

LR = .1
W0 = np.random.rand(2,2)
b0 = np.random.rand(2)
DataX = np.array([[-1,-1],
                  [ 1, 1],
                  [-1, 1],
                  [ 1, -1]])
N,D = DataX.shape
DataC = np.array([0,0,1,1])

W1 = np.random.rand(2)
b1 = np.random.rand(1)




B0 = np.empty((0,D),float)
for i in range(N):
    B0 = np.vstack([B0,b0])
B0 = B0.T


B1 = np.empty((0,1),float)
for i in range(N):
    B1 = np.vstack([B1,b1])
B1 = B1.T

A0 = np.dot(W0,DataX.T) + B0

#Logistic Activation Function
Z1 = 1/(1+np.exp(-1*A0))

A1 = np.dot(W1,Z1) + B1

Y = 1/(1+np.exp(-1*A1))

print(Y)