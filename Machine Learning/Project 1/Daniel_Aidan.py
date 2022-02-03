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

#Open the Excel sheet
ED = pd.read_excel("proj1Dataset.xlsx")
index = ED.index
rows = len(index)

#Initialize Empty List
Data = []

# Fill the array with the cars class, skip points of data with missing parts
for i in range(rows):
    if(pd.isna(ED.iloc[i,0]) or pd.isna(ED.iloc[i,1])):
        print("Data Missing! Skipping row " , i + 2)
    else:
        TempCar = car(ED.iat[i,0], ED.iat[i,1])
        Data.append(TempCar)
        #print(Data[i].getHorsePower())
        #print(Data[i].getWeight())


#Code to do ML HERE

#Closed Form
x = np.empty((0,1),float)
for i in range(len(Data)):
    x = np.append(x,np.array([[Data[i].getWeight()]]))
#print(x)

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
print(W)
Y = np.matmul(bigX,W)
#print(Y)

#Gradient Descent

# W2 = np.empty((0,1),float)
# for i in range(len(bigX[0])):
#     W2 = np.append(W2,np.array(random.randint(1,100)))
# print(W2.shape)
W2 = [[4],[5]]

L = .0000001

for i in range(15000):

    JW = np.matmul(bigX,  W2)
    for i in range(len(JW)):
        JW[i] = JW[i] - t[i]
        
    JW = (L*(float(1/len(t)))) * np.transpose(JW)
  
    temp1 = np.matmul(JW,bigX[:,0])
    temp2 = np.matmul(JW,bigX[:,1])

    W2[0] = W2[0] - temp1
    W2[1] = W2[1] - temp2
print(W2)


Y2 = W[0]*x + W[1]


#Plot First Figure : Closed Form
plt.figure(1)
plt.plot(x,Y,label = "Closed Form")

for i in range(len(Data)):
    plt.plot(Data[i].getWeight(),Data[i].getHorsePower(),'rx')


plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.title("""Matlab's "carbig" dataset""")
leg = plt.legend(loc='upper right')


#Plot Second Figure : Gradient Descent
plt.figure(2)

plt.plot(x,Y2,label = "Gradient Descent Form")

for i in range(len(Data)):
    plt.plot(Data[i].getWeight(),Data[i].getHorsePower(),'rx')


plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.title("""Matlab's "carbig" dataset""")
leg = plt.legend(loc='upper right')

plt.draw()
plt.show()