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
        print(Data[i].getM1())
        #print(Data[i].getWeight())