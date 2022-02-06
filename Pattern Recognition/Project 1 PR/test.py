import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

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
            if(temp > 0 and T[i] == 2):
                MissX += (X[i,:]*-1)
            elif(temp <= 0 and T[i] == 1 or temp == 1):
                MissX += X[i,:]
            temp = 0
        
        if np.sum(MissX) == 0:
            print(N , "# of Epochs")
            return W

        for j in range(len(W)):
            W[j] += MissX[j]*L
        
        MissX = np.zeros(len(W))

    print(maxEpochs , "# of Epohs")
    return W

    
def LeastSquares(X,T):
    return np.matmul(np.linalg.pinv(X),T)



X = np.array([[1,1],[2,1],[4,1],[5,1]])
Xtru = np.array([0,2,3,5])
Xreal = np.array([1,2,4,5])
Y = np.array([[1],[1],[2],[2]])

W1 = BatchPerceptron(X,Y)

W2 = LeastSquares(X,Y)

#print(W1)+89256
26
#print(W2)

Y2 = Xtru*W2[0]+ W2[1]
Y3 = Xtru*W1[0] + W1[1]

#print(Y3)

plt.figure(1)
plt.plot(Xreal,Y,'rx')
plt.plot(Xtru,Y2)
plt.plot(Xtru,Y3,'g')
plt.draw()
plt.show()