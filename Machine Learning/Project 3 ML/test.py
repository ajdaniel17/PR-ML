import numpy as np
import matplotlib.pyplot as plt

N = 5
X = np.array([1,2,3,4,5])

lamda = 2


t = np.array([[1],[2],[3],[4],[5]])


M = 3
Mu = np.linspace(0, 1,M)
s = 1
#print(Mu)
#Setup all Phi
Phi = np.empty((0,(M+1)),float)
temp3 = 0
for j in range(N):
    temp = np.empty((0),float)
    for k in range(M+1):
        if k == 0:
            temp = np.append(temp,np.array([1]))
        else:
            temp3 = np.exp(-1*((X[j]-Mu[k-1])**2)/(2*(s**2)))
            #print(temp3)
            temp = np.append(temp,np.array([temp3]))
        #print(temp)
    Phi = np.append(Phi,np.array([temp]),0)

I = np.identity(M+1)
print(Phi.shape)
#print(Phi[0])


# print(I.shape)
# print(t.shape)

W = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(Phi),Phi)+lamda*I),np.transpose(Phi)),t)

print(W)
plt.plot(1)

for i in range(3):
    plt.plot(X,t,'bo')
plt.plot(X,np.matmul(Phi,W),'r')
plt.show()