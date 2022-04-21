import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
np.random.seed(69)
graphX = []
graphY = []

#Learning Rate
LR = .3

#Data
DataX = np.array([[-1,-1],
                  [ 1, 1],
                  [-1, 1],
                  [ 1, -1]])

#Targets
DataC = np.array([[0,1],
                  [0,1],
                  [1,0],
                  [1,0]])

N,D = DataX.shape

#Initialize random weights
W0 = np.random.rand(2,2)
b0 = np.random.rand(2)

W1 = np.random.rand(2,2)
b1 = np.random.rand(2)

plt.figure(1)
plt.title("Epochs vs Error: XOR")
graphX = []
graphY = []
for m in range(3000):
    #Forward Pass Setup
    B0 = np.empty((0,D),float)
    for i in range(N):
        B0 = np.vstack([B0,b0])
    B0 = B0.T
    
    B1 = np.empty((0,2),float)
    for i in range(N):
        B1 = np.vstack([B1,b1])
    B1 = B1.T

    #Forward Pass
    A0 = np.dot(W0,DataX.T) + B0
    Z0 = 1/(1+np.exp(-1*A0))

    A1 = np.dot(W1,Z0) + B1
    Y = 1/(1+np.exp(-1*A1))

    #Back Pass
    L = Y-DataC.T
    E = .5 * np.sum(np.linalg.norm(L,axis=0)**2)
  
    W1 = W1 - LR*np.dot(L,Z0.T)
    b1 = b1 - LR*np.mean(L,axis=1)
  
    delta1 = np.multiply(np.dot(W1.T,L),(np.multiply(Z0,(1-Z0))))
    W0 = W0 - LR*np.dot(delta1,DataX)
    b0 = b0 - LR*np.mean(delta1,axis=1)
    
    #Graph Error
    graphX.append(m)
    graphY.append(E)
    if m % 50 == 0:
        plt.ion()
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.xlim(0,3001)
        plt.xticks(np.arange(0,3001,500))
        plt.semilogy(graphX,graphY)
        plt.pause(.001)
        plt.clf()
plt.ioff()
plt.figure(1)
plt.xlabel("Epochs")
plt.title("Epochs vs Error: XOR")
plt.ylabel("Error")
plt.xlim(0,3001)
plt.semilogy(graphX,graphY)

plt.figure(2)
ax = plt.axes(projection='3d')
plt.title("XOR Decision Surface")
X1 = np.linspace(-2,2,50)
X2 = np.linspace(-2,2,50)
Xgraph, Ygraph = np.meshgrid(X1, X2)
Z = np.empty((50,50),float)

for i in range(50):
    for j in range(50):
        A0 = np.dot((W0.T),np.vstack((Xgraph[i][j],Ygraph[i][j]))) + b0.reshape((2,1))
        Z0 = 1/(1+np.exp(-1*A0))
        A1 = np.dot(W1,Z0) + b1.reshape((2,1))
        Y = 1/(1+np.exp(-1*A1))
        Z[i][j] = Y[0] - Y[1]

ax.plot_surface(Xgraph, Ygraph, Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
ax.scatter(1,1,-1,color = 'r',s=200,alpha=1)
ax.scatter(-1,-1,-1,color = 'r',s=200,alpha=1)
ax.scatter(1,-1,1,color = 'b',s=200,alpha=1)
ax.scatter(-1,1,1,color = 'b',s=200,alpha=1)

#Load Data From Excel
ED = pd.read_excel("Proj5DataSet.xlsx")
index = ED.index
rows = len(index)
DataX = np.empty((0,1),float)
DataC = np.empty((0),float)
for i in range(rows):
    if(pd.isna(ED.iloc[i,0]) or pd.isna(ED.iloc[i,1])):
        print("Data Missing! Skipping row " , i + 2)
    else:
        DataX = np.append(DataX,np.array([[ED.iat[i,0]]]),0)
        DataC = np.append(DataC, ED.iat[i,1])

N,D = DataX.shape
L = 1
U = 3
W0 = np.random.rand(U,1)
b0 = np.random.rand(U)
W1 = np.random.rand(1,U)
b1 = np.random.rand(1)
LR = .2
B = .9
prevV1 = 0
prevVb1 = 0
prevV0 = 0
prevVb0 = 0
plt.figure(3)
plt.title("Epochs vs Error: 3 Units")
graphX = []
graphY = []
prevE = 0
for m in range(3000):
    #Forward Pass Setup
    
    B0 = np.empty((0,U),float)
    for i in range(N):
        B0 = np.vstack((B0,b0))
    B0 = B0.T
    
    B1 = np.empty((0,1),float)
    for i in range(N):
        B1 = np.vstack([B1,b1])
    B1 = B1.T

    #Forward Pass
    A0 = np.dot(W0,DataX.T) + B0
    Z0 = np.tanh(A0)

    A1 = np.dot(W1,Z0) + B1
    Y = np.tanh(A1)

    #Back Pass
    L = Y-DataC.T

    E = .5 * np.sum(np.linalg.norm(L,axis=0)**2)
    V1 = (prevV1 * B) + ((1-B)* np.dot(L,Z0.T))
    Vb1 = (prevVb1 * B) + ((1-B)* np.mean(L,axis=1))
    W1 = W1 - LR*V1
    b1 = b1 - LR*Vb1
  
    delta1 = np.multiply(np.dot(W1.T,L),(1-(Z0**2)))
    V0 = (prevV0 * B) + ((1-B)* np.dot(delta1,DataX))
    Vb0 = (prevVb0 * B) + ((1-B)* np.mean(L,axis=1))
    W0 = W0 - LR*V0
    b0 = b0 - LR*Vb0
    
    prevV1 = V1
    prevVb1 = Vb1
    prevV0 = V0
    prevVb0 = Vb0
    #Graph Error
    graphX.append(m)
    graphY.append(E)
    if m % 50 == 0:
        plt.ion()
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.xlim(0,3001)
        plt.xticks(np.arange(0,3001,500))
        plt.semilogy(graphX,graphY)
        plt.pause(.001)
        plt.clf()
    if (E < (prevE + .0001) and E > (prevE - .0001)) and E < 3.7:
        break
    prevE = E
plt.ioff()
plt.figure(3)
plt.title("Epochs vs Error: 3 Units")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.xlim(0,3001)
plt.semilogy(graphX,graphY)

plt.figure(4)
plt.xlabel("X")
plt.ylabel("T")
title = '3 Units, Loss: ' + str(E)
plt.title(title)
X1 = np.linspace(-1,1,100)
Y = np.empty(100)
for i in range(100):
    A0 = np.dot((W0),X1[i]) + b0.reshape((U,1))
    Z0 = np.tanh(A0)
    A1 = np.dot(W1,Z0) + b1
    Y[i] = np.tanh(A1)
plt.plot(X1,Y)
plt.scatter(DataX,DataC)

L = 1
U = 20
W0 = np.random.rand(U,1)
b0 = np.random.rand(U)
W1 = np.random.rand(1,U)
b1 = np.random.rand(1)
LR = .05
prevV1 = 0
prevVb1 = 0
prevV0 = 0
prevVb0 = 0
B = .91
plt.figure(5)
plt.title("Epochs vs Error: 20 Units")
graphX = []
graphY = []
prevE = 0
for m in range(3000):
    #Forward Pass Setup
    
    B0 = np.empty((0,U),float)
    for i in range(N):
        B0 = np.vstack((B0,b0))
    B0 = B0.T
    
    B1 = np.empty((0,1),float)
    for i in range(N):
        B1 = np.vstack([B1,b1])
    B1 = B1.T

    #Forward Pass
    A0 = np.dot(W0,DataX.T) + B0
    Z0 = np.tanh(A0)

    A1 = np.dot(W1,Z0) + B1
    Y = np.tanh(A1)

    #Back Pass
    L = Y-DataC.T
    E = .5 * np.sum(np.linalg.norm(L,axis=0)**2)
    V1 = (prevV1 * B) + ((1-B)* np.dot(L,Z0.T))
    Vb1 = (prevVb1 * B) + ((1-B)* np.mean(L,axis=1))
    W1 = W1 - LR*V1
    b1 = b1 - LR*Vb1
  
    delta1 = np.multiply(np.dot(W1.T,L),(1-(Z0**2)))
    V0 = (prevV0 * B) + ((1-B)* np.dot(delta1,DataX))
    Vb0 = (prevVb0 * B) + ((1-B)* np.mean(L,axis=1))
    W0 = W0 - LR*V0
    b0 = b0 - LR*Vb0
    
    prevV1 = V1
    prevVb1 = Vb1
    prevV0 = V0
    prevVb0 = Vb0
    #Graph Error
    graphX.append(m)
    graphY.append(E)
    if m % 50 == 0:
        plt.ion()
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.xlim(0,3001)
        plt.xticks(np.arange(0,3001,500))
        plt.semilogy(graphX,graphY)
        plt.pause(.001)
        plt.clf()
    if (E < (prevE + .0001) and E > (prevE - .0001)) and E < 2.4:
        break
    prevE = E
plt.ioff()
plt.figure(5)
plt.xlabel("Epochs")
plt.title("Epochs vs Error: 20 Units")
plt.ylabel("Error")
plt.xlim(0,3001)
plt.semilogy(graphX,graphY)

plt.figure(6)
plt.xlabel("X")
plt.ylabel("T")
title = '20 Units, Loss: ' + str(E)
plt.title(title)
X1 = np.linspace(-1,1,100)
Y = np.empty(100)
for i in range(100):
    A0 = np.dot((W0),X1[i]) + b0.reshape((U,1))
    Z0 = np.tanh(A0)
    A1 = np.dot(W1,Z0) + b1
    Y[i] = np.tanh(A1)
plt.plot(X1,Y)
plt.scatter(DataX,DataC)
plt.show()