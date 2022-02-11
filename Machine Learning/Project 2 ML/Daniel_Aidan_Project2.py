import numpy as np
import matplotlib.pyplot as plt
import random

Pi = np.pi
SD = .3

N = 1000

Xtrain1 = np.random.uniform(0,1,N)

E = (1/(SD*np.sqrt(2*Pi))) * np.exp(-.5*((Xtrain1-np.mean(Xtrain1)/SD)**2))
Train1 = np.sin(2*Pi*Xtrain1) + E
print(E)
plt.figure(1)

for i in range(len(Xtrain1)):
    plt.plot(Xtrain1[i],Train1[i],'rx')

plt.draw()
plt.show()


