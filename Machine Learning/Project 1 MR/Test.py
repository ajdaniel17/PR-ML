import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# This class is for easy storage of our data, it contains ways to set the data and retrive it
class car():
    Weight = 0
    HorsePower = 0
    
    def __init__(self,W,H):
        self.Weight = W
        self.HorsePower = H

    def getWeight(self):
        return self.Weight
    
    def getHorsePower(self):
        return self.HorsePower

    def setWeight(self,W):
        self.Weight = W

    def setHorsePower(self,H):
        self.HorsePower = H

car1 = car(3,6)
car2 = car(5,4)
car3 = car(4,2)

Data = [car1,car2,car3]

#Code to do ML HERE
x = np.empty((0,1),float)
for i in range(len(Data)):
    x = np.append(x,np.array([[Data[i].getWeight()]]))


bigX = np.empty((0,2),float)
for i in range(len(Data)):
    bigX = np.append(bigX,np.array([[Data[i].getWeight(),1]]),0)
#print(bigX)

t = np.empty((0,1),float)
for i in range(len(Data)):
    t = np.append(t,np.array([[Data[i].getHorsePower()]]))
#print(t)

w = np.linalg.pinv(bigX)
W = np.matmul(w,t)
#print(W)


W2 = np.empty((0,1),float)
for i in range(len(bigX[0])):
    W2 = np.append(W2,np.array(random.randint(1,5)))
# W2 = [[4],[5]]

basicx = np.linspace(3,5,100)
L = .1


for i in range(10000):

    JW = np.matmul(bigX,  W2)
    for i in range(len(JW)):
        JW[i] = JW[i] - t[i]
        
    JW = (L*(1/len(t))) * JW
  
    temp1 = np.matmul(np.transpose(JW),bigX[:,0])
    temp2 = np.matmul(np.transpose(JW),bigX[:,1])

    W2[0] = W2[0] - temp1
    W2[1] = W2[1] - temp2
    print(W2)


Y2 = W2[0]*x + W2[1]
Y = np.matmul(bigX,W)

#Plot First Figure : Closed Form
plt.figure(1)
plt.plot(x,Y)

#Plot Second Figure : Gradient Descent
plt.plot(x,Y2,color = 'r', ls = ':')

for i in range(len(Data)):
    plt.plot(Data[i].getWeight(),Data[i].getHorsePower(),'rx')


plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.title("""Matlab's "carbig" dataset""")

plt.draw()
plt.show()