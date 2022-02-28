import numpy as np
import matplotlib.pyplot as plt


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
#print(T.shape)
M = 20

Phi = np.empty((0,N,M+1),float)

Mu = np.random.uniform(0,1,M)
#Mu = np.sort(Mu)
#Mu = np.linspace(0, 1,M)
s = .1
#print(Mu)
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
    Phi = np.append(Phi,np.array([temp2]),0)
#print(Phi.shape)
#print(Phi[0])

lam = np.random.uniform(.01,5,300)
lam = np.sort(lam)
#print(lam)
I = np.identity(M+1)
W = np.empty((0,L,(M+1)),float)
# Setup all Weight Vectors

for i in range(len(lam)):
    temp1 = np.empty((0,(M+1)),float)
    for j in range(L):
        temp =  np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(Phi[j][:][:]),Phi[j][:][:])+lam[i]*I),np.transpose(Phi[j][:][:])),T[j][:])
        #print(temp.shape)
        temp1 = np.append(temp1,np.array([temp]),0)
    W = np.append(W,np.array([temp1]),0)
#print(W.shape)

Xrange = np.linspace(0, 1,500)


temp2 = np.empty((0,(M+1)),float)
for j in range(len(Xrange)):
        temp = np.empty((0),float)
        for k in range(M+1):
            if k == 0:
                temp = np.append(temp,np.array([1]))
            else:
                temp3 = np.exp(-1*((Xrange[j]-Mu[k-1])**2)/(2*(s**2)))
                temp = np.append(temp,np.array([temp3]))
        temp2 = np.append(temp2,np.array([temp]),0)


plt.figure(1)
numlam = 299
for i in range(N):
    for j in range(L):
        plt.plot(X[j][i],T[j][i],'bo')
# for i in range(1):
#     plt.plot(X[i][:],np.matmul(Phi[i][:][:],W[0][i][:]))
for i in range(L):
    plt.plot(Xrange,np.matmul(temp2,W[numlam][i][:]),'r')

plt.figure(2)
TruY = np.sin(2*Pi*Xrange)
plt.plot(Xrange,TruY,'b')

#print(temp2.shape)
#print(W[0][i][:].shape)
sum = np.zeros(500)
for i in range(L):
    sum += np.matmul(temp2,W[numlam][i][:])
sum = sum/300
plt.plot(Xrange,sum,'r')
plt.show()