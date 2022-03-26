import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxopt as cv

cv.solvers.options['show_progress'] = False
cv.solvers.options['abstol'] = 1e-10
cv.solvers.options['reltol'] = 1e-10
cv.solvers.options['feastol'] = 1e-10

PI = np.pi
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

X1avg = np.average(DataX[:][0]) 
X2avg = np.average(DataX[:][1]) 

sig = 1.75
const = 1 / sig*np.sqrt(PI*2)
DataY = np.empty((0,2),float)
for i in range(rows):
    DataY = np.append(DataY,np.array([[const*np.exp(-.5*((DataX[i][0]-X1avg)**2)/sig**2),const*np.exp(-.5*((DataX[i][1]-X2avg)**2)/sig**2)]]),0)
 

#W1,B1,Lam2 = SoftSVM(DataX,DataC,10,.01)

W1,B1,Lam2 = SoftSVM(DataY,DataC,10,.01)


x1 = np.linspace(-2, 8)
x2 = -1*(((W1[0]*x1)+B1)/W1[1])
plt.figure(1)

plt.plot(x1,x2,'b',label= 'Decision Boundary')
for i in range(len(DataX)):
    if DataC[i] == 1:
        plt.plot(DataX[i][0],DataX[i][1],'ro')
    else:
        plt.plot(DataX[i][0],DataX[i][1],'gs')

plt.show()