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

    def getM(self,i):
        if i == 1:
            return self.M1
        elif i == 2:
            return self.M2
        elif i == 3:
            return self.M3
        elif i == 4:
            return self.M4
        else:
            return

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


def Smaller(X,Y):
    if len(X) > len(Y):
        return Y
    else:
        return X


def correlationCoefficient(X,Y):
    Xsum = np.sum(X)
    Ysum = np.sum(Y)
    XYsum = 0
    XXsum = 0
    YYsum = 0
    Len = len(Smaller(X,Y))

    for i in range(Len):
        XYsum += X[i]*Y[i]

    for i in range(len(X)):
        XXsum += X[i]*X[i]

    for i in range(len(Y)):
        YYsum += Y[i]*Y[i]
    
    R = (float)(Len * XYsum - Xsum * Ysum)/(float)(np.sqrt((Len * XXsum - Xsum * Xsum)* (Len * YYsum - Ysum * Ysum)))

    return R


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

plt.figure(1)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
for i in range(len(Data)):
    if Data[i].getName() == "setosa":
        plt.plot(Data[i].getM1(),Data[i].getM2(),'r+')
    elif Data[i].getName() == "versicolor":
        plt.plot(Data[i].getM1(),Data[i].getM2(),'bx')
    elif Data[i].getName() == "virginica":
        plt.plot(Data[i].getM1(),Data[i].getM2(),'go')
plt.title("Sepal Length vs Sepal Width")

plt.figure(2)
plt.xlabel("Pedal Length")
plt.ylabel("Pedal Width")
for i in range(len(Data)):
    if Data[i].getName() == "setosa":
        plt.plot(Data[i].getM3(),Data[i].getM4(),'r+')
    elif Data[i].getName() == "versicolor":
        plt.plot(Data[i].getM3(),Data[i].getM4(),'bx')
    elif Data[i].getName() == "virginica":
        plt.plot(Data[i].getM3(),Data[i].getM4(),'go')
plt.title("Pedal Length vs Pedal Width")


print("Min Sepal Length is",findInfo(Data, 1, "all", 1))
print("Max Sepal Length  is",findInfo(Data, 1, "all", 2))
print("Average Sepal Length  is",findInfo(Data, 1, "all", 3))
print("Variance Sepal Length  is",findInfo(Data, 1, "all", 4))

print("\nMin Sepal Width is",findInfo(Data, 2, "all", 1))
print("Max Sepal Width  is",findInfo(Data, 2, "all", 2))
print("Average Sepal Width  is",findInfo(Data, 2, "all", 3))
print("Variance Sepal Width  is",findInfo(Data, 2, "all", 4))

print("\nMin Pedal Length is",findInfo(Data, 3, "all", 1))
print("Max Pedal Length  is",findInfo(Data, 3, "all", 2))
print("Average Pedal Length  is",findInfo(Data, 3, "all", 3))
print("Variance Pedal Length  is",findInfo(Data, 3, "all", 4))

print("\nMin Pedal Width is",findInfo(Data, 4, "all", 1))
print("Max Pedal Width  is",findInfo(Data, 4, "all", 2))
print("Average Pedal Width  is",findInfo(Data, 4, "all", 3))
print("Variance Pedal Width  is",findInfo(Data, 4, "all", 4))

WCV1 = float(1/3)*(findInfo(Data,1,"setosa",4) + findInfo(Data,1,"versicolor",4) + findInfo(Data,1,"virginica",4))
WCV2 = float(1/3)*(findInfo(Data,2,"setosa",4) + findInfo(Data,2,"versicolor",4) + findInfo(Data,2,"virginica",4))
WCV3 = float(1/3)*(findInfo(Data,3,"setosa",4) + findInfo(Data,3,"versicolor",4) + findInfo(Data,3,"virginica",4))
WCV4 = float(1/3)*(findInfo(Data,4,"setosa",4) + findInfo(Data,4,"versicolor",4) + findInfo(Data,4,"virginica",4))

print("\nWithin-Class Variance for Sepal Length is" , WCV1)
print("Within-Class Variance for Sepal Width is" , WCV2)
print("Within-Class Variance for Pedal Length is" , WCV3)
print("Within-Class Variance for Pedal Width is" , WCV4)

BCV1 = float(1/3)*(pow(findInfo(Data, 1, "setosa", 3) - (findInfo(Data, 1, "all", 3)),2)+pow(findInfo(Data, 1, "versicolor", 3) - (findInfo(Data, 1, "all", 3)),2)+pow(findInfo(Data, 1, "virginica", 3) - (findInfo(Data, 1, "all", 3)),2))
BCV2 = float(1/3)*(pow(findInfo(Data, 2, "setosa", 3) - (findInfo(Data, 2, "all", 3)),2)+pow(findInfo(Data, 2, "versicolor", 3) - (findInfo(Data, 2, "all", 3)),2)+pow(findInfo(Data, 2, "virginica", 3) - (findInfo(Data, 2, "all", 3)),2))
BCV3 = float(1/3)*(pow(findInfo(Data, 3, "setosa", 3) - (findInfo(Data, 3, "all", 3)),2)+pow(findInfo(Data, 3, "versicolor", 3) - (findInfo(Data, 3, "all", 3)),2)+pow(findInfo(Data, 3, "virginica", 3) - (findInfo(Data, 3, "all", 3)),2))
BCV4 = float(1/3)*(pow(findInfo(Data, 4, "setosa", 3) - (findInfo(Data, 4, "all", 3)),2)+pow(findInfo(Data, 4, "versicolor", 3) - (findInfo(Data, 4, "all", 3)),2)+pow(findInfo(Data, 4, "virginica", 3) - (findInfo(Data, 4, "all", 3)),2))

print("\nBetween-Class Variance for Sepal Length is", BCV1)
print("Between-Class Variance for Sepal Width is", BCV2)
print("Between-Class Variance for Pedal Length is", BCV3)
print("Between-Class Variance for Pedal Width is", BCV4, "\n")

M1 = np.empty((0,1),float)
M2 = np.empty((0,1),float)
M3 = np.empty((0,1),float)
M4 = np.empty((0,1),float)
M5 = np.empty((0,1),float)

for i in range(len(Data)):
    M1 = np.append(M1,Data[i].getM1())

for i in range(len(Data)):
    M2 = np.append(M2,Data[i].getM2())

for i in range(len(Data)):
    M3 = np.append(M3,Data[i].getM3())

for i in range(len(Data)):
    M4 = np.append(M4,Data[i].getM4())

for i in range(len(Data)):
    if Data[i].getName() == "setosa":
        M5 = np.append(M5,1)
    elif Data[i].getName() == "versicolor":
        M5 = np.append(M5,2)
    elif Data[i].getName() == "virginica":
        M5 = np.append(M5,3)

Measurements = [M1,M2,M3,M4,M5]

hml = ["SepL","SepW","PetL","PetW","Class"]
 
Corrarry = np.corrcoef(Measurements)

fig, ax = plt.subplots()
im = ax.imshow(Corrarry,cmap=plt.cm.coolwarm)

ax.set_xticks(np.arange(len(hml)))
ax.set_yticks(np.arange(len(hml)))
ax.set_xticklabels(hml)
ax.set_yticklabels(hml)
plt.title("Correlation Coefficient Heat Map")
plt.colorbar(im)

plt.show()