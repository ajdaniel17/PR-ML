import numpy as np
import matplotlib.pyplot as plt
import cvxopt as cv
#https://xavierbourretsicotte.github.io/SVM_implementation.html

cv.solvers.options['show_progress'] = False
cv.solvers.options['abstol'] = 1e-10
cv.solvers.options['reltol'] = 1e-10
cv.solvers.options['feastol'] = 1e-10

def SoftSVM(X,Y,C):
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
    print(x)

    #Round Low lambdas to zero
    for i in range(len(lambdas)):
        if lambdas[i] < 1e-10:
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

    


DataX = np.array([[1,1],
                  [1,-1],
                  [2,0],
                  [5,0],
                  [6,1],
                  [6,-1]])

DataC = np.array([1,1,-1,1,-1,-1])


# DataX = np.array([[1,1],
#                   [1,-1],
#                   [2,1],
#                   [3,1],
#                   [3,-1],
#                   [4,1]])

# DataC = np.array([1,1,1,-1,-1,-1])


C = .1
w ,b,lam = SoftSVM(DataX,DataC,C)


x1 = np.linspace(1, 7)

x2 = -1*(((w[0]*x1)+b)/w[1])

mar1 =  (((w[0]*x1*-1)-b+1)/w[1])
mar2 =  (((w[0]*x1*-1)-b-1)/w[1])
gap = (np.abs(((b-1)/w[1])-((b+1)/w[1])))/np.sqrt(1+(-1*w[0]/w[1])**2)
gap = np.round(gap,5)
print("gap is ", gap)
plt.figure(1)

for i in range(len(DataX)):
    if DataC[i] == 1:
        plt.plot(DataX[i][0],DataX[i][1],'ro')
    else:
        plt.plot(DataX[i][0],DataX[i][1],'gs')

    if lam[i] != 0:
        plt.plot(DataX[i][0],DataX[i][1],'bx')

plt.plot(x1,x2)
plt.plot(x1,mar1)
plt.plot(x1,mar2)
plt.xlim([-3, 7])
plt.ylim([-3, 3])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()