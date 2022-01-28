import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

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

# Fill the array with the cars class
for i in range(rows):
    TempCar = car(ED.iat[i,0], ED.iat[i,1])
    Data.append(TempCar)
    #print(Data[i].getHorsePower())

