import numpy as np

np.random.seed(69)
L = 100 
N = 25
np.set_printoptions(threshold=np.inf)
X = np.empty((0,L),float)  

for i in range(N):
    temp = np.random.uniform(0,1,L)
    X = np.append(X,np.array([temp]),0)
print(X.shape)

