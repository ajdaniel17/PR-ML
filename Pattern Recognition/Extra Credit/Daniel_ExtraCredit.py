import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxopt as cv

cv.solvers.options['show_progress'] = False
cv.solvers.options['abstol'] = 1e-10
cv.solvers.options['reltol'] = 1e-10
cv.solvers.options['feastol'] = 1e-10

PI = np.pi
def SoftSVM(X,Y,C,kernal):
    Xrows, Xcols = X.shape
    Lrange = 1E-5

    if kernal == 'gaussian':
        K = np.zeros((Xrows, Xrows))
        for i in range(Xrows):
            for j in range(Xrows):
                K[i,j] = np.exp((-1.0*np.linalg.norm(DataX[i]-DataX[j])**2)/(2*(sig)**2))
    
    q = np.empty((0),float)
    b = np.zeros(1)
    G = np.identity(Xrows) * -1
    h = np.zeros(Xrows)
    A = Y
    
    for i in range(Xrows):
        q = np.append(q,-1)

    P = cv.matrix(np.outer(Y,Y) * K)
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

print(DataC)
X1avg = np.average(DataX[:][0]) 
X2avg = np.average(DataX[:][1]) 

sig = 1.75
const = 1 / sig*np.sqrt(PI*2)
DataY = np.empty((0,2),float)
for i in range(rows):
    DataY = np.append(DataY,np.array([[np.exp((-1.0*np.linalg.norm(DataX[i][0]-1)**2)/(2*(sig)**2)),np.exp((-1.0*np.linalg.norm(DataX[i][1]+1)**2)/(2*(sig)**2))]]),0)
 

#W1,B1,Lam2 = SoftSVM(DataX,DataC,10,.01)

W1,B1,Lam2 = SoftSVM(DataY,DataC,10,'gaussian')
print(W1)
print(B1)

delta = 0.025
xrange = np.arange(-2.0, 8.0, delta)
yrange = np.arange(-2.0, 6.0, delta)
x, y = np.meshgrid(xrange, yrange)
equation = W1[0]*x + W1[1]*y+B1
plt.figure(1)
plt.contour(x, y, equation, [0])
for i in range(len(DataX)):
    if DataC[i] == 1:
        plt.plot(DataX[i][0],DataX[i][1],'ro')
    else:
        plt.plot(DataX[i][0],DataX[i][1],'gs')

plt.show()