import numpy as np
import matplotlib.pyplot as plt

#Initializing Values
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


#Setup Target Values
T = np.empty((0,N),float)
for i in range(L):
    EX1 = np.random.normal(0,SD,N)
    temp = np.sin(2*Pi*X[i][:])  + EX1
    T = np.append(T,np.array([temp]),0)


#Setup all Phi
M = 20
Phi = np.empty((0,N,M+1),float)
Mu = np.random.uniform(0,1,M)
s = .1
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
    Phi = np.append(Phi,np.array([temp2]),0)


# Setup all Weight Vectors
lamamount = 300
lam = np.random.uniform(.01,5,lamamount)
lam = np.sort(lam)
I = np.identity(M+1)
W = np.empty((0,L,(M+1)),float)
for i in range(len(lam)):
    temp1 = np.empty((0,(M+1)),float)
    for j in range(L):
        temp =  np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(Phi[j][:][:]),Phi[j][:][:])+lam[i]*I),np.transpose(Phi[j][:][:])),T[j][:])
        temp1 = np.append(temp1,np.array([temp]),0)
    W = np.append(W,np.array([temp1]),0)

#Calculate Average Weight Vectors for Data Sets
Fhat = np.empty((0,M+1),float)
for i in range(lamamount):
    temp = np.zeros(M+1)
    for j in range(L):
        temp += W[i][j][:]
    temp = temp * float(1.0/L)
    Fhat = np.append(Fhat,np.array([temp]),0)


bias = np.empty((0),float)
#Calculate Bias for all data Points
for i in range(lamamount):
    temp = 0
    temp2 = 0
    temp3 = 0
    for j in range(L):
        temp = np.matmul(Fhat[i][:],np.transpose(Phi[j]))
        for k in range(N):
            temp2 += (temp[k] - (np.sin(2*Pi*X[j][k])))**2
        temp2 = temp2 / float(N)
        temp3 += temp2
    temp3 = temp3 / float(L)
    bias = np.append(bias,temp3)


variance = np.empty((0),float)
temp = np.empty((0,300,25),float)
#Setup an array to help with indexing
for i in range(L):
    temp2 = np.matmul(Fhat,np.transpose(Phi[i]))
    temp = np.append(temp,np.array([temp2]),0)
#Calculate Variance for all data points
for i in range(lamamount):
    temp2 = 0
    temp3 = 0
    temp4 = 0
    for j in range(L):
        temp2 = np.matmul(W[i],np.transpose(Phi[j]))
        for k in range(N):
            temp3 += (temp2[j][k] - temp[j][i][k])**2
        temp3 = temp3 / float(N) 
        temp4 += temp3
    temp4 = temp4 / float(L)
    variance = np.append(variance,temp4)

#Setup A Test Set of 1000 Values
Xtest = np.linspace(0, 1, 1000)
testPhi = np.empty((0,(M+1)),float)
for j in range(len(Xtest)):
        temp = np.empty((0),float)
        for k in range(M+1):
            if k == 0:
                temp = np.append(temp,np.array([1]))
            else:
                temp3 = np.exp(-1*((Xtest[j]-Mu[k-1])**2)/(2*(s**2)))
                temp = np.append(temp,np.array([temp3]))
        testPhi = np.append(testPhi,np.array([temp]),0)

#Find the error of the test set
FX = np.matmul(Fhat,np.transpose(testPhi))
TruY = np.sin(2*Pi*Xtest)
TestError = np.empty((0),float)
for i in range(lamamount):
    temp = 0
    for j in range(1000):
        temp += (TruY[j] - FX[i][j])**2
    temp = np.sqrt(temp / float(1000))
    TestError = np.append(TestError,temp)


plt.figure(1)
plt.plot(np.log(lam),bias,label = '(bias)²')
plt.plot(np.log(lam),variance,label = 'variance')
plt.plot(np.log(lam),variance+bias,label = '(bias)² + variance')
plt.plot(np.log(lam),TestError,label = 'test error')
plt.xlim(-3,2)
plt.ylim(0,.18)
plt.xlabel('ln λ')
leg = plt.legend(loc='upper right')

# Xrange = np.linspace(0, 1,500)
# temp2 = np.empty((0,(M+1)),float)
# for j in range(len(Xrange)):
#         temp = np.empty((0),float)
#         for k in range(M+1):
#             if k == 0:
#                 temp = np.append(temp,np.array([1]))
#             else:
#                 temp3 = np.exp(-1*((Xrange[j]-Mu[k-1])**2)/(2*(s**2)))
#                 temp = np.append(temp,np.array([temp3]))
#         temp2 = np.append(temp2,np.array([temp]),0)

# plt.figure(2)
# numlam = 299
# for i in range(N):
#     for j in range(L):
#         plt.plot(X[j][i],T[j][i],'bo')
# for i in range(20):
#     plt.plot(Xrange,np.matmul(temp2,W[numlam][i][:]),'r')
# plt.figure(3)
# TruY = np.sin(2*Pi*Xrange)
# plt.plot(Xrange,TruY,'b')
# plt.plot(Xrange,np.matmul(temp2,Fhat[numlam][:]),'r')

plt.show()