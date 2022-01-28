import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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






#Plot First Figure : Closed Form
plt.figure(1)
for i in range(len(Data)):
    plt.plot(Data[i].getWeight(),Data[i].getHorsePower(),'rx')


plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.title("""Matlab's "carbig" dataset""")

#Plot Second Figure : Gradient Descent
plt.figure(2)
for i in range(len(Data)):
    plt.plot(Data[i].getWeight(),Data[i].getHorsePower(),'rx')


plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.title("""Matlab's "carbig" dataset""")

plt.show()