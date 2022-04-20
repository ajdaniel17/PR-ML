import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

np.set_printoptions(suppress=True)

graphX = []
graphY = []

LR = .3

DataX = np.array([[-1,-1],
                  [ 1, 1],
                  [-1, 1],
                  [ 1, -1]])

DataC = np.array([[0,1],
                  [0,1],
                  [1,0],
                  [1,0]])

N,D = DataX.shape

W0 = np.random.uniform(0,1,size=(2,2))
b0 = np.random.uniform(0,1,size = 2)

W1 = np.random.uniform(0,1,size=(2,2))
b1 = np.random.uniform(0,1,size = 2)

plt.figure(1)

# # plt.ylim(.0001, .1)

# plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
graphX = []
graphY = []
for m in range(3000):

    B0 = np.empty((0,D),float)
    for i in range(N):
        B0 = np.vstack([B0,b0])
    B0 = B0.T


    B1 = np.empty((0,2),float)
    for i in range(N):
        B1 = np.vstack([B1,b1])
    B1 = B1.T

    A0 = np.dot(W0,DataX.T) + B0
    Z0 = 1/(1+np.exp(-1*A0))

    A1 = np.dot(W1,Z0) + B1
    Y = 1/(1+np.exp(-1*A1))

    # print(Y)
    L = np.empty((0,2),float)
    for i in range(N):
        temp = np.empty(0,float)
        for j in range(2):
            temp = np.append(temp,Y[j][i]-DataC[i][j]) 
        L = np.append(L,np.array([temp]),0)
    L = L.T
    
    E = .5 * np.sum(np.linalg.norm(L,axis=0)**2)
  
    W1 = W1 - LR*np.dot(L,Z0.T)
    b1 = b1 - LR*np.mean(L,axis=1)

    delta1 = np.multiply(np.dot(W1.T,L),(np.multiply(Z0,(1-Z0))))
    W0 = W0 - LR*np.dot(delta1,DataX)
    b0 = b0 - LR*np.mean(delta1,axis=1)

    
    graphX.append(m)
    graphY.append(E)
    if m % 10 == 0:
        plt.ion()
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.xlim(0,3001)
        plt.xticks(np.arange(0,3001,500))
        plt.semilogy(graphX,graphY)
        plt.draw()
        plt.pause(.001)
        plt.clf()
plt.show()
print(Y)
print(E)
# plt.figure(2)
# ax = plt.axes(projection='3d')

# X1 = np.linspace(-2,2,1000)
# X2 = np.linspace(-2,2,1000)
# Xgraph, Ygraph = np.meshgrid(X1, X2)
# Z = np.empty((1000,1000),float)
# for i in range(1000):
#     for j in range(1000):
#         A0 = np.dot(W0,np.vstack((Xgraph[i][j],Ygraph[i][j]))) + b0
#         Z0 = 1/(1+np.exp(-1*A0))

#         A1 = np.dot(W1,Z0) + b1
#         Y = 1/(1+np.exp(-1*A1))
#         print(Y)
#         Z[i][j] = Y[0] - Y[1]

# # print(Y.shape)
# ax.plot_surface(Xgraph, Ygraph, Y, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
# plt.show()