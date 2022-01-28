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
x = np.empty((0,1),int)
for i in range(len(Data)):
    x = np.append(x,np.array([[Data[i].getWeight()]]))


bigX = np.empty((0,2),int)
for i in range(len(Data)):
    bigX = np.append(bigX,np.array([[Data[i].getWeight(),1]]),0)
#print(bigX)

t = np.empty((0,1),int)
for i in range(len(Data)):
    t = np.append(t,np.array([[Data[i].getHorsePower()]]))
#print(t)

w = np.linalg.pinv(bigX)
W = np.matmul(w,t)
#print(W)

Y = np.matmul(bigX,W)
print(Y)
#Plot First Figure : Closed Form
plt.figure(1)
plt.plot(x,Y)

for i in range(len(Data)):
    plt.plot(Data[i].getWeight(),Data[i].getHorsePower(),'rx')


plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.title("""Matlab's "carbig" dataset""")

plt.draw()
plt.show()