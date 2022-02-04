import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

class plant():
    def __init__(self,m1,m2,m3,m4,n):
        self.M1 = m1
        self.M2 = m2
        self.M3 = m3
        self.M4 = m4
        self.name = n

    def getM1(self):
        return self.M1

    def getM2(self):
        return self.M2
    
    def getM3(self):
        return self.M3
    
    def getM4(self):
        return self.M4

    def getName(self):
        return self.name

def findInfo(data,M,name,ty):
    temp = np.empty(0,float)
    for i in range(len(data)):
       
        if (data[i].getName() == name or name == "all"):
            if M == 1:
                temp = np.append(temp,(data[i].getM1()))
            elif M == 2:
                temp = np.append(temp,(data[i].getM2()))
            elif M == 3:
                temp = np.append(temp,(data[i].getM3()))
            elif M == 4:
                temp = np.append(temp,(data[i].getM4()))
            else:
                print("ERROR WRONG M")
                return
    if ty == 1:
        return np.amin(temp)
    elif ty == 2:
        return np.amax(temp)
    elif ty == 3:
        return np.average(temp)
    elif ty == 4:
        return np.var(temp)
    else:
        print("ERROR WRONG TY")
        return
    return

#Open the Excel sheet
ED = pd.read_excel("Proj1DataSet.xlsx")
index = ED.index
rows = len(index)

#Initialize Empty List
Data = []

for i in range(rows):
    if(pd.isna(ED.iloc[i,0]) or pd.isna(ED.iloc[i,1]) or pd.isna(ED.iloc[i,2]) or pd.isna(ED.iloc[i,3]) or pd.isna(ED.iloc[i,4])):
        print("Data Missing! Skipping row " , i + 2)
    else:
        TempPlant = plant(ED.iat[i,0], ED.iat[i,1], ED.iat[i,2], ED.iat[i,3], ED.iat[i,4])
        Data.append(TempPlant)
        #print(Data[i].getName())

try:
    print("Min is ",findInfo(Data, 1, "setosa", 1))
    print("Max is ",findInfo(Data, 1, "setosa", 2))
    print("Average is ",findInfo(Data, 1, "setosa", 3))
    print("Variance is ",findInfo(Data, 1, "setosa", 4))
except ValueError:
    pass