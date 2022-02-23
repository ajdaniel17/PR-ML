import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

random.seed(7)
def Misclassified(X,W,T):
    Xrows, Xcols = X.shape
    temp = 0
    B = 0
    for N in range(Xrows):
        for j in range(len(W)):
            if ((j+2) <= len(W)):
                temp += float(X[N][j])*W[(len(W)-(j+2))]
            else:
                temp += float(X[N][j])*W[j]
        if(temp > 0 and T[N] == 0):
                B += 1 
        elif(temp <= 0 and T[N] == 1 or temp == 1):
                B += 1
        temp = 0
    return B

def BatchPerceptron(X,T):
    Xrows, Xcols = X.shape
    W = np.empty(Xcols,float)
    L = 1
    maxEpochs = 10000
    MissX = np.zeros(len(W))
    temp = 0
    Num = 100
    for i in range(len(W)):
        W[i] = random.randint(1,50)
    #W = np.array([1.0,1.0])
    for N in range(maxEpochs):
        if N >= Num:
            L -= .0005
            Num += 100
        for i in range(Xrows):
            for j in range(len(W)):
                if ((j+2) <= len(W)):
                    temp += float(X[i][j])*W[(len(W)-(j+2))]
                else:
                    temp += float(X[i][j])*W[j]
            if(temp > 0 and T[i] == 0):
                MissX += X[i,:]*-1.0
            elif(temp <= 0 and T[i] == 1 or temp == 1):
                MissX += X[i,:]
            temp = 0

        if np.sum(MissX) == 0:
            print(N , "# of Epochs")
            return W,N

        for j in range(len(W)):
            W[j] += MissX[j]*L
        
        MissX = np.zeros(len(W))

    print(maxEpochs , "# of Epohs")
    return W,maxEpochs

    
def LeastSquares(X,T):
    return (np.matmul((np.linalg.pinv(X)),T))



X = np.array([[1,1,1],
              [2,1,1],
              [4,5,1],
              [8,9,1]])

Xtru = np.array([0,2,3,5])
Xreal = np.array([1,2,4,5])

Y = np.array([[1],
              [1],
              [0],
              [0]])


X2 = np.array([[1,1,1],
              [2,1,1],
              [4,5,1],
              [8,9,1]])

Y3 = np.array([[1,0],
              [1,0],
              [0,1],
              [0,1]])

W1, N1= BatchPerceptron(X,Y)
#print("Batch Perceptron Weight Vectors: ",W1)

W2 = LeastSquares(X,Y)

W3 = LeastSquares(X2,Y3)

#print("Least Squares Weight Vectors: ",W2)
X1Range = np.linspace(0, 9)
X2Range = np.linspace(0, 9)

#testvalues = W3[0][0]*X2Range + W3[1][0]*X1Range + W3[2][0]
d1 = W3[1][0]*X2Range + W3[0][0]*X1Range + W3[2][0] 
d2 = W3[1][1]*X2Range + W3[0][1]*X1Range + W3[2][1] 
#d3 = W3[0][2]*X2Range + W3[1][2]*X1Range + W3[2][2] 

Y4 = (X1Range*(W3[0][1]-W3[0][0])+(W3[2][1]-W3[2][0]))/(W3[1][0]-W3[1][1])
#Y5 = d1 - d3
#Y6 = d3 - d2

Y2 = W2[0]*X2Range + W2[1]*X1Range + W2[2] + .5

Predict = np.matmul(X,W2) 

Y1 = (-1.0*(W1[2]+W1[1]*X[:,0]))/W1[0]

#print(Predict)
print(W3)
print(W2)
#print(Y2)
Num1 = Misclassified(X,W1,Y)
Num2 = Misclassified(X,W2,Y)
#print("Batch Errors:" , Num1)
#print("LS Errors:" , Num2)

plt.figure(1)

for i in range(len(Y)):
    if Y[i] == 0:
        plt.plot(X[i][0],X[i][1],'rx')
    else:
        plt.plot(X[i][0],X[i][1],'bx')
        
#plt.plot(Xreal,Y2,'y')
plt.plot(X[:,0],Y1,'g:',label = "Batch Perceptron")
plt.plot(X1Range,Y2,'b:',label = "Least Sqaures")
plt.plot(X1Range,Y4,'r',label = "Least Sqaures MUlti1")
#plt.plot(X1Range,Y5,'b',label = "Least Sqaures MUlti2")
#plt.plot(X1Range,Y6,'g',label = "Least Sqaures MUlti3")
#plt.plot(line_x, Y2)
leg = plt.legend(loc='upper right')
plt.draw()
plt.show()