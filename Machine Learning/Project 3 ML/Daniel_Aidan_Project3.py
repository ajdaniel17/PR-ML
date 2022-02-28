import numpy as np

np.random.seed(69)
Pi = np.pi
SD = .3
L = 100 
N = 25
np.set_printoptions(threshold=np.inf)

#Setup X values
X = np.empty((0,N),float)  
for i in range(L):
    temp = np.random.uniform(0,1,N)
    X = np.append(X,np.array([temp]),0)
#print(X.shape)

#Setup Target Values
T = np.empty((0,N),float)
for i in range(L):
    EX1 = np.random.normal(SD,np.mean(X[i][:]),N)
    temp = np.sin(2*Pi*X[i][:])  + EX1
    T = np.append(T,np.array([temp]),0)

M = 20

Phi = np.empty((N,M+1,0),float)

Mu = np.random.uniform(0,1,M)
Mu = np.sort(Mu)
s = .1

#Setup all Phi
for i in range(L):
    temp2 = np.empty((0,(M+1)),float)
    for j in range(N):
        temp = np.empty((0),float)
        for k in range(M+1):
            if k == 0:
                temp = np.append(temp,np.array([1]))
            else:
                temp3 = np.exp(-1*((X[i][j]-Mu[k-1])**2)/(2*(s**2)))
                temp = np.append(temp,np.array([temp3]))
        temp2 = np.append(temp2,np.array([temp]),0)
    #print(temp2.shape)
    Phi = np.dstack((Phi,temp2))
print(Phi.shape)

l = np.random.uniform(.01,5,1000)
l = np.sort(l)

