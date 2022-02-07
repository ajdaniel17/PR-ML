import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def Misclassified(X,W,T):
    Xrows, Xcols = X.shape
    temp = 0
    B = 0
    for N in range(Xrows):
        for j in range(len(W)):
                temp += X[N][j]*W[j]
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
    maxEpochs = 1000
    MissX = np.zeros(len(W))
    temp = 0
    for i in range(len(W)):
        W[i] = random.randint(1,10)
    
    for N in range(maxEpochs):
        for i in range(Xrows):
            for j in range(len(W)):
                temp += X[i][j]*W[j]
            if(temp > 0 and T[i] == 0):
                MissX += (X[i,:]*-1)
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
    return np.matmul(np.linalg.pinv(X),T)



X = np.array([[1,1],[2,1],[4,1],[5,1]])
Xtru = np.array([0,2,3,5])
Xreal = np.array([1,2,4,5])
Y = np.array([[1],[1],[0],[0]])

#Xthing = np.transpose([1,np.transpose(X)])

W1, N1= BatchPerceptron(X,Y)

W2 = LeastSquares(X,Y)


#print(W2)
#print(W1)

#print(W2)

#Y2 = Xtru*W2[0] + W2[1]
Y3 = Xtru*W1[0] + W1[1]
#Y2 = -W2[0] / W2[1] * Xtru
#print(Y3)
#print(W1)
Num1 = Misclassified(X,W1,Y)
Num2 = Misclassified(X,W2,Y)
print("Batch Errors:" , Num1)
print("LS Errors:" , Num2)

line_x = np.linspace(0, 9)
Y2 = -W2[0] / W2[1]  * line_x

plt.figure(1)
plt.plot(Xreal,Y,'rx')
#plt.plot(Xreal,Y2,'y')
plt.plot(Xtru,Y3,'g:')
plt.plot(line_x, Y2)
plt.draw()
plt.show()