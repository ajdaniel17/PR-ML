import numpy as np
import matplotlib.pyplot as plt
import random


np.random.seed(15)
Pi = np.pi
SD = .3

def LeastSquares(X,T):
    return np.matmul(np.linalg.pinv(X),T)

def LSError(X,W,T):
    Y = np.matmul(X,W)
    error = 0
    for i in range(len(Y)):
        error += ((Y[i] - T[i]) ** 2)  
    RMS = np.sqrt((error / (len(Y))))
    return RMS

NTrain = 10
Xtrain1 = np.random.uniform(0,1,NTrain)
EX1 = np.random.normal(SD,np.mean(Xtrain1),NTrain)
TrainTarget1 = np.sin(2*Pi*Xtrain1)  + EX1
Phi = []
for i in range(10):
    tempX = np.empty([0,i+1])
    for j in range(NTrain):
        tempX2 = np.empty([0],float)
        for k in range(i+1):
            tempX2 = np.append(tempX2,np.array([[Xtrain1[j] ** k]]))
        tempX = np.vstack([tempX,tempX2])
    Phi.append(np.array(tempX))

W1 = []

for i in range(10):
    W1.append(LeastSquares(Phi[i], TrainTarget1))

RMS1 = []

for i in range(10):
    RMS1.append(LSError(Phi[i],W1[i],TrainTarget1))
    # RMS1.append(LSError(Xtrain1,W1[i],TrainTarget1))



NTest = 100
XTest1 = np.random.uniform(0,1,NTest)
EX2 = np.random.normal(SD,np.mean(Xtrain1),NTest)
TestTarget2 = np.sin(2*Pi*XTest1)  + EX2

Phi2 = []
for i in range(10):
    tempX = np.empty([0,i+1])
    for j in range(NTrain):
        tempX2 = np.empty([0],float)
        for k in range(i+1):
            tempX2 = np.append(tempX2,np.array([[XTest1[j] ** k]]))
        tempX = np.vstack([tempX,tempX2])
    Phi2.append(np.array(tempX))

RMS2 = []
for i in range(10):
    RMS2.append(LSError(Phi2[i],W1[i],TestTarget2))
    # RMS2.append(LSError(XTest1,W1[i],TestTarget2))
RMS2[9] = RMS2[9] 

plt.figure(1)
plotx = [0,1,2,3,4,5,6,7,8,9]
plt.plot(plotx,RMS2,'--ro',label = "Test")
plt.plot(plotx,RMS1,'--bo',label = "Training")
plt.xlim(0,9)
plt.ylim(0,1)
plt.xlabel('M')
plt.ylabel('ERMS')
plt.title("N Train = 10")
leg = plt.legend(loc='upper right')










NTrain = 100
Xtrain1 = np.random.uniform(0,1,NTrain)
EX1 = np.random.normal(SD,np.mean(Xtrain1),NTrain)
TrainTarget1 = np.sin(2*Pi*Xtrain1)  + EX1
Phi = []
for i in range(10):
    tempX = np.empty([0,i+1])
    for j in range(NTrain):
        tempX2 = np.empty([0],float)
        for k in range(i+1):
            tempX2 = np.append(tempX2,np.array([[Xtrain1[j] ** k]]))
        tempX = np.vstack([tempX,tempX2])
    Phi.append(np.array(tempX))

W1 = []

for i in range(10):
    W1.append(LeastSquares(Phi[i], TrainTarget1))

RMS1 = []

for i in range(10):
    RMS1.append(LSError(Phi[i],W1[i],TrainTarget1))
    # RMS1.append(LSError(Xtrain1,W1[i],TrainTarget1))



NTest = 100
XTest1 = np.random.uniform(0,1,NTest)
EX2 = np.random.normal(SD,np.mean(Xtrain1),NTest)
TestTarget2 = np.sin(2*Pi*XTest1)  + EX2

Phi2 = []
for i in range(10):
    tempX = np.empty([0,i+1])
    for j in range(NTrain):
        tempX2 = np.empty([0],float)
        for k in range(i+1):
            tempX2 = np.append(tempX2,np.array([[XTest1[j] ** k]]))
        tempX = np.vstack([tempX,tempX2])
    Phi2.append(np.array(tempX))

RMS2 = []
for i in range(10):
    RMS2.append(LSError(Phi2[i],W1[i],TestTarget2))
    # RMS2.append(LSError(XTest1,W1[i],TestTarget2))
RMS2[9] = RMS2[9] 

plt.figure(2)
plotx = [0,1,2,3,4,5,6,7,8,9]
plt.plot(plotx,RMS2,'--ro',label = "Test")
plt.plot(plotx,RMS1,'--bo',label = "Training")

plt.xlim(0,9)
plt.ylim(0,1)
plt.xlabel('M')
plt.ylabel('ERMS')
plt.title("N Train = 100")
leg = plt.legend(loc='upper right')
plt.draw()
plt.show()


