import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxopt as cv
import time
import math
import libsvm.svmutil as svm


def SoftSVM(X,Y,C,Lrange):
    Xrows, Xcols = X.shape

    Xprim = np.empty((0,Xcols),float)
    q = np.empty((0),float)
    b = np.zeros(1)
    G = np.identity(Xrows) * -1
    h = np.zeros(Xrows)
    A = Y
    
    for i in range(Xrows):
        temp = Y[i] * X[i]
        Xprim = np.append(Xprim,np.array([temp]),0)
        q = np.append(q,-1)

    P = np.matmul(Xprim,np.transpose(Xprim)) * 1.0
    P = cv.matrix(P)
    q = cv.matrix(q)
    b = cv.matrix(b,(1,1),'d')
    G = cv.matrix(G)
    h = cv.matrix(h)
    A = cv.matrix(A,(1,Xrows),'d')

    x = cv.solvers.qp(P,q,G,h,A,b)
    lambdas = np.array(x['x'])

    LinS = x['status']

    #Check to see if data is linearly Seperable
    if LinS != 'optimal':
        print("Data is not Lineraly Serperable!")
        print("Rerunning Calculations!")
        G = np.vstack((np.identity(Xrows) * -1,np.identity(Xrows)))
        h = np.hstack((np.zeros(Xrows), np.ones(Xrows) * C))
        G = cv.matrix(G)
        h = cv.matrix(h)
        x = cv.solvers.qp(P,q,G,h,A,b)
        lambdas = np.array(x['x'])

    #Round Low lambdas to zero
    for i in range(len(lambdas)):
        if lambdas[i] < Lrange:
            lambdas[i] = 0

    #Calculate W
    w = np.zeros(Xcols)
    for i in range(Xrows):
        w += X[i]*Y[i]*lambdas[i]
  
    #Calculate Offset
    b = 0 
    j = 0
    for i in range(Xrows):
        if lambdas[i] != 0:
            b += (Y[i]-np.matmul(X[i],w))
            j += 1
    b = b/j*1.0

    return w,b,lambdas

#Load Data From Excel
ED = pd.read_excel("Proj2DataSet.xlsx",header=None)
index = ED.index
rows = len(index)
DataX = np.empty((0,2),float)
DataC = np.empty((0),float)
for i in range(rows):
    if(pd.isna(ED.iloc[i,0]) or pd.isna(ED.iloc[i,1]) or pd.isna(ED.iloc[i,2])):
        print("Data Missing! Skipping row " , i + 2)
    else:
        DataX = np.append(DataX,np.array([[ED.iat[i,0],ED.iat[i,1]]]),0)
        DataC = np.append(DataC, ED.iat[i,2])

#Solve for C = 0.1
W1,B1,Lam1 = SoftSVM(DataX,DataC,0.1,1e-6)
#Find the Margins
G1 = (np.abs(((B1-1)/W1[1])-((B1+1)/W1[1])))/np.sqrt(1+(-1*W1[0]/W1[1])**2)
G1 = np.round(G1,5)
#Find the Error
E1 = 0
for i in range(len(DataX)):
    temp = DataX[i][1] * W1[1] + DataX[i][0] * W1[0] + B1

    if DataC[i] == 1 and temp <= 1:
        E1 += 1
    elif DataC[i] == -1 and temp >= -1:
        E1 += 1

#Solve for C = 100
W2,B2,Lam2 = SoftSVM(DataX,DataC,100,.01)
#Find the Margins
G2 = (np.abs(((B2-1)/W2[1])-((B2+1)/W2[1])))/np.sqrt(1+(-1*W2[0]/W2[1])**2)
G2 = np.round(G2,5)
#Find the Error
E2 = 0
for i in range(len(DataX)):
    temp = DataX[i][1] * W2[1] + DataX[i][0] * W2[0] + B2

    if DataC[i] == 1 and temp <= 1:
        E2 += 1
    elif DataC[i] == -1 and temp >= -1:
        E2 += 1
#Calculate the Decision Hyper-Plane line and its margins
x1 = np.linspace(-2, 8)
x2 = -1*(((W1[0]*x1)+B1)/W1[1])
mar1 =  (((W1[0]*x1*-1)-B1+1)/W1[1])
mar2 =  (((W1[0]*x1*-1)-B1-1)/W1[1])
#Plot the Data points and plot the support vectors
plt.figure(1)
for i in range(len(DataX)):
    if DataC[i] == 1:
        plt.plot(DataX[i][0],DataX[i][1],'ro')
    else:
        plt.plot(DataX[i][0],DataX[i][1],'gs')
    if Lam1[i] != 0:
        plt.plot(DataX[i][0],DataX[i][1],'bx')
#Plot the Decision Hyper-Plane and its margins
plt.plot(DataX[0][0],DataX[0][1],'ro',label = 'Class 1')
plt.plot(DataX[len(DataX)-1][0],DataX[len(DataX)-1][1],'gs',label = 'Class 2')
plt.plot(x1,x2,'b',label= 'Decision Boundary')
plt.plot(x1,mar1,'b--',label = 'Margin Boundary')
plt.plot(x1,mar2,'b--',label = 'Margin Boundary')
plt.xlim([-3, 8])
plt.ylim([-2, 6])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title("C = %.1f;" %0.1 + "Pts in margin or misclass=%i;" %E1 +  "Margin=%.5f" %G1)
leg = plt.legend(loc='upper right')
#Calculate the Decision Hyper-Plane line and its margins
x2 = -1*(((W2[0]*x1)+B2)/W2[1])
mar1 =  (((W2[0]*x1*-1)-B2+1)/W2[1])
mar2 =  (((W2[0]*x1*-1)-B2-1)/W2[1])
#Plot the Data points and plot the support vectors
plt.figure(2)
for i in range(len(DataX)):
    if DataC[i] == 1:
        plt.plot(DataX[i][0],DataX[i][1],'ro')
    else:
        plt.plot(DataX[i][0],DataX[i][1],'gs')
    if Lam2[i] != 0:
        plt.plot(DataX[i][0],DataX[i][1],'bx')
#Plot the Decision Hyper-Plane and its margins
plt.plot(DataX[0][0],DataX[0][1],'ro',label = 'Class 1')
plt.plot(DataX[len(DataX)-1][0],DataX[len(DataX)-1][1],'gs',label = 'Class 2')
plt.plot(x1,x2,label= 'Decision Boundary')
plt.plot(x1,mar1,'b--',label = 'Margin Boundary')
plt.plot(x1,mar2,'b--',label = 'Margin Boundary')
plt.xlim([-3, 8])
plt.ylim([-2, 6])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title("C = %.1f;" %100 + "Pts in margin or misclass=%i;" %E2 +  "Margin=%.5f" %G2)
leg = plt.legend(loc='upper right')


#Part 2
np.random.seed(1)
time1 = []
time2 = []
N = []
options = '-t 0'
for i in range(10,100):
    N.append(i)
    class1=np.random.multivariate_normal([1,3],[[1,0],[0,1]],math.ceil(i/2.0))
    class2=np.random.multivariate_normal([4,1],[[2,0],[0,2]],math.floor(i/2.0))
    X = np.vstack((class1,class2))
    Y = np.hstack((np.ones(math.ceil(i/2.0)),np.ones(math.floor(i/2.0))*-1))

    start_time = time.time()
    temp = SoftSVM(X,Y,1,.01)
    end_time = time.time()
    time1.append(end_time-start_time)

    start_time = time.time()
    model = svm.svm_train(Y.astype(int).tolist(), X.tolist(), options)
    end_time = time.time()
    time2.append(end_time-start_time)
    

plt.figure(3)
plt.plot(N,time1,'--ro',label = 'My Function')
plt.plot(N,time2,'--bo',label = 'LibSVM Function')
plt.xlabel('Number of Training Samples')
plt.ylabel('Time')
plt.title('Training Samples Vs Time')
leg = plt.legend(loc='upper left')
plt.show()