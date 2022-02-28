import numpy as np

Phi = np.array([[2,1],
                [4,1],
                [5,1]
                ])

lamda = 2

I = np.array([[1,0],
              [0,1]])

t = np.array([[3],[5],[6]])

print(Phi.shape)
print(I.shape)
print(t.shape)

W = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(Phi),Phi)+lamda*I),np.transpose(Phi)),t)

print(W)