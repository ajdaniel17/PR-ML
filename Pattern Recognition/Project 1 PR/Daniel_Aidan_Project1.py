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

def BatchPerceptron(X,T):
    Xrows, Xcols = X.shape
    W = np.empty(Xcols,float)
    L = .001
    maxEpochs = 1000
    MissX = np.zeros(len(W))
    temp = 0
    for i in range(len(W)):
        W[i] = random.randint(1,10)
    
    for N in range(maxEpochs):
        for i in range(Xrows):
            for j in range(len(W)):
                temp += X[i][j]*W[j]
            if(temp > 0 and T[i] == 0):
                MissX += (X[i,:]*-1)
            elif(temp <= 0 and T[i] == 1 or temp == 1):
                MissX += X[i,:]
            temp = 0
        
        if np.sum(MissX) == 0:
            return W,N

        for j in range(len(W)):
            W[j] += MissX[j]*L
        
        MissX = np.zeros(len(W))

    return W,maxEpochs

def LeastSquares(X,T):
    return np.matmul(np.linalg.pinv(X),T)

def Misclassified(X,W,T):
    Xrows, Xcols = X.shape
    temp = 0
    B = 0
    for N in range(Xrows):
        for j in range(len(W)):
                temp += X[N][j]*W[j]
        if(temp > 0 and T[N] == 0):
                B += 1 
        elif(temp <= 0 and T[N] == 1 or temp == 1):
                B += 1
        temp = 0
    return B


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


#Plotting Data in 2D form
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

#General Data Observations
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

#Correlation Coefficient Heat Map
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

#Feature VS Class Label Graphs
plt.figure(4)

plt.subplot(2,2,1)
plt.plot(M1,M5,'rx')
plt.title("SepL Vs Class")

plt.subplot(2,2,2)
plt.plot(M2,M5,'rx')
plt.title("SepW Vs Class")

plt.subplot(2,2,3)
plt.plot(M3,M5,'rx')
plt.title("PetL Vs Class")

plt.subplot(2,2,4)
plt.plot(M4,M5,'rx')
plt.title("PetW Vs Class")
plt.tight_layout()
#Classification Tasks

#Setosa Vs Versi+Virigi
X1 = np.empty((0,5),float)
for i in range(len(Data)):
    X1 = np.append(X1,np.array([[Data[i].getM1(),Data[i].getM2(),Data[i].getM3(),Data[i].getM4(),1]]),0)
#print(X1)

T1 = np.empty((0,1),float)
for i in range(len(Data)):
    if Data[i].getName() == "setosa":
        T1 = np.append(T1,np.array([[1]]),0)
    elif Data[i].getName() == "versicolor":
        T1 = np.append(T1,np.array([[0]]),0)
    elif Data[i].getName() == "virginica":
        T1 = np.append(T1,np.array([[0]]),0)   
#print(Y1)

print("Setosa VS Versi+Virigi : All Features")
BW1,N1 = BatchPerceptron(X1,T1)
LSW1 = LeastSquares(X1,T1)
MBW1 = Misclassified(X1,BW1,T1)
if(N1 != 1000):
    print("Batch Perceptron Converged!")
else:
    print("Batch Perceptron did Not Converge!")

print("Batch Perceptron # of Epochs:",N1)
print("Batch Perceptron Weight Vectors:",BW1)
print("Batch Perceptron Misclassifications:", MBW1)
print("Least Squares Weight Vectors:",np.transpose(LSW1))
print("Least Sqaurs Misclassifications:")

print("\n\n")

#Setosa Vs Versi+Virigi
X2 = np.empty((0,3),float)
for i in range(len(Data)):
    X2 = np.append(X2,np.array([[Data[i].getM3(),Data[i].getM4(),1]]),0)

T2 = T1

print("Setosa VS Versi+Virigi: Features 3 and 4")
BW2,N2 = BatchPerceptron(X2,T2)
LSW2 = LeastSquares(X2,T2)
MBW2 = Misclassified(X2,BW2,T2)
if(N2 != 1000):
    print("Batch Perceptron Converged!")
else:
    print("Batch Perceptron did Not Converge!")

print("Batch Perceptron # of Epochs:",N2)
print("Batch Perceptron Weight Vectors:",BW2)
print("Batch Perceptron Misclassifications:", MBW2)
print("Least Squares Weight Vectors:",np.transpose(LSW2))
print("Least Sqaurs Misclassifications:")

plt.figure(5)
X2P = np.array([0,1,2,3,4,5,6,7])
for i in range(len(Data)):
    if T2[i] == 1:
        plt.plot(Data[i].getM3(),Data[i].getM4(),'ro')
    else:
        plt.plot(Data[i].getM3(),Data[i].getM4(),'bo')
BY2 = BW2[2] * X2P + BW2[1]* X2P + BW2[0]

plt.plot(X2P,BY2,label = "Batch Perceptron",color = "green")

plt.xlabel('Pedal Length')
plt.ylabel('Pedal Width')
plt.title("Setosa VS Versi+Virgi: Features 3 and 4")
leg = plt.legend(loc='upper right')


plt.tight_layout()
plt.show()