import numpy as np
import matplotlib.pyplot as plt
import random

Pi = np.pi
SD = .3

def LeastSquares(X,T):
    return np.matmul(np.linalg.pinv(X),T)

def LSError(X,W,T):
    Xrows, Xcols = X.shape
    y = 0
    error = 0.0
    for i in range(Xrows):
        for j in range(Xcols):
            y += X[i][j]* W[j]
        error += ((y - T[i]) ** 2)  

    RMS = np.sqrt((error / (Xrows+1)))
    return RMS

N = 10

Xtrain1 = np.random.uniform(0,1,N)

EX1 = (1/(SD*np.sqrt(2*Pi))) * np.exp(-.5*((Xtrain1-np.mean(Xtrain1)/SD)**2))
TrainTarget1 = np.sin(2*Pi*Xtrain1)  + EX1
Phi = []
for i in range(9):
    tempX = np.empty([0,i+1])
    for j in range(N):
        tempX2 = np.empty([0],float)
        for k in range(i+1):
            tempX2 = np.append(tempX2,np.array([[Xtrain1[j] ** k]]))
        tempX = np.vstack([tempX,tempX2])
    Phi.append(np.array(tempX))

W1 = []


for i in range(9):
    W1.append(LeastSquares(Phi[i], TrainTarget1))
    #print(W1[i])
print(Phi[1])
print(Phi[1][0][1])
Y = []
for i in range(N):
    Y.append(W1[1]*Phi[1][i][:])
#Y = W1[1] * Phi[1][:][1]
#print(LSError(Phi[1],W1[1],TrainTarget1))
#print(EX1)
plt.figure(1)

for i in range(len(Xtrain1)):
    plt.plot(Xtrain1[i],TrainTarget1[i],'rx')
#plt.plot(Phi[1][:][1],Y)
plt.draw()
plt.show()


