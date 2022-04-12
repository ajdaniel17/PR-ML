import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)

#Setup True Model parameters
TruCov1 = np.array([[.8,.2,.1,.05,.01],
                    [.2,.7,.1,.03,.02],
                    [.1,.1,.8,.02,.01],
                    [.05,.03,.02,.9,.01],
                    [.01,.02,.01,.01,.8]])

TruCov2 = np.array([[.9,.1,.05,.02,.01],
                    [.1,.8,.1,.02,.02],
                    [.05,.1,.7,.02,.01],
                    [.02,.02,.02,.6,.02],
                    [.01,.02,.01,.02,.7]])

TruMean1 = np.zeros(5)
TruMean2 = np.ones(5)

#Function to find PDF of a Gaussian distribution
def NormalPDF(X,MEAN,VAR):
    return ((1/(VAR*np.sqrt(2*np.pi)))*(np.exp(-.5*(((X-MEAN)/(VAR))**2))))


#Import Data From Proj3Train100.xlsx
ED = pd.read_excel("Proj3Train100.xlsx",header=None)
index = ED.index
rows = len(index)
DataX = np.empty((0,5),float)
DataC = np.empty((0),float)
for i in range(rows):
    if(pd.isna(ED.iloc[i,0]) or pd.isna(ED.iloc[i,1]) or pd.isna(ED.iloc[i,2])):
        print("Data Missing! Skipping row " , i + 2)
    else:
        DataX = np.append(DataX,np.array([[ED.iat[i,0],ED.iat[i,1],ED.iat[i,2],ED.iat[i,3],ED.iat[i,4]]]),0)
        DataC = np.append(DataC, ED.iat[i,5])

#Import Data From Proj3Test.xlsx
ED = pd.read_excel("Proj3Test.xlsx",header=None)
index = ED.index
rows = len(index)
TestX = np.empty((0,5),float)
TestC = np.empty((0),float)
for i in range(rows):
    if(pd.isna(ED.iloc[i,0]) or pd.isna(ED.iloc[i,1]) or pd.isna(ED.iloc[i,2])):
        print("Data Missing! Skipping row " , i + 2)
    else:
        TestX = np.append(TestX,np.array([[ED.iat[i,0],ED.iat[i,1],ED.iat[i,2],ED.iat[i,3],ED.iat[i,4]]]),0)
        TestC = np.append(TestC, ED.iat[i,5])

print("For N-Train = 100")

#Naive Bayes Classifier 
Mean1 = np.mean(DataX[0:int(len(DataX)/2)],axis = 0)
Mean2 = np.mean(DataX[int(len(DataX)/2):int(len(DataX))],axis = 0)

temp1 = np.zeros(5)
temp2 = np.zeros(5)
for i in range(50):
    temp1 += (DataX[i] - Mean1)**2
    temp2 += (DataX[i+50] - Mean2)**2
Var1 = np.sqrt(temp1 /50.0)
Var2 = np.sqrt(temp2 /50.0)

G1 = np.ones(int(len(TestX)))
G2 = np.ones(int(len(TestX)))
for i in range(len(TestX)):
    for j in range(5):
        G1[i] = G1[i] * (NormalPDF(TestX[i][j],Mean1[j],Var1[j]) * .5)
        G2[i] = G2[i] * (NormalPDF(TestX[i][j],Mean2[j],Var2[j]) * .5)

Tot = 0
for i in range(len(TestX)):
    if G1[i] > G2[i] and TestC[i] == 2:
        Tot += 1
    if G1[i] < G2[i] and TestC[i] == 1:
        Tot += 1

print("Naive Bayes Classifier Error:", Tot/float(len(TestX)))

#Bayes Classifier 
Cov1 = np.cov(DataX[0:int(len(DataX)/2)],rowvar = False)
Cov2 = np.cov(DataX[int(len(DataX)/2):int(len(DataX))],rowvar = False)

Mean1 = np.mean(DataX[0:int(len(DataX)/2)],axis = 0)
Mean2 = np.mean(DataX[int(len(DataX)/2):int(len(DataX))],axis = 0)

G1 = np.empty(0,float)
G2 = np.empty(0,float)
for i in range (len(TestX)):
    G1 = np.append(G1,(-.5 * TestX[i] @ (np.linalg.inv(Cov1)) @ (TestX[i].T)) + (.5 * TestX[i] @ (np.linalg.inv(Cov1)) @ Mean1) + (.5 * (Mean1.T) @ (np.linalg.inv(Cov1)) @ (TestX[i].T)) - (.5 * (Mean1.T) @ np.linalg.inv(Cov1) @ Mean1) + np.log(.5) + (5 * -.5 * np.log(2*np.pi)) - (.5 * np.linalg.det(Cov1)))
    G2 = np.append(G2,(-.5 * TestX[i] @ (np.linalg.inv(Cov2)) @ (TestX[i].T)) + (.5 * TestX[i] @ (np.linalg.inv(Cov2)) @ Mean2) + (.5 * (Mean2.T) @ (np.linalg.inv(Cov2)) @ (TestX[i].T)) - (.5 * (Mean2.T) @ np.linalg.inv(Cov2) @ Mean2) + np.log(.5) + (5 * -.5 * np.log(2*np.pi)) - (.5 * np.linalg.det(Cov2)))

Tot = 0
for i in range(len(TestX)):
    if G1[i] > G2[i] and TestC[i] == 2:
        Tot += 1
    if G1[i] < G2[i] and TestC[i] == 1:
        Tot += 1

print("Bayes Classifier using MLE Error:", Tot/float(len(TestX)))

#Find Error Using Bayes classifier using True Parameters
G1 = np.empty(0,float)
G2 = np.empty(0,float)
for i in range (len(TestX)):
    G1 = np.append(G1,(-.5 * TestX[i] @ (np.linalg.inv(TruCov1)) @ (TestX[i].T)) + (.5 * TestX[i] @ (np.linalg.inv(TruCov1)) @ TruMean1) + (.5 * (TruMean1.T) @ (np.linalg.inv(TruCov1)) @ (TestX[i].T)) - (.5 * (TruMean1.T) @ np.linalg.inv(TruCov1) @ TruMean1) + np.log(.5) + (5 * -.5 * np.log(2*np.pi)) - (.5 * np.linalg.det(TruCov1)))
    G2 = np.append(G2,(-.5 * TestX[i] @ (np.linalg.inv(TruCov2)) @ (TestX[i].T)) + (.5 * TestX[i] @ (np.linalg.inv(TruCov2)) @ TruMean2) + (.5 * (TruMean2.T) @ (np.linalg.inv(TruCov2)) @ (TestX[i].T)) - (.5 * (TruMean2.T) @ np.linalg.inv(TruCov2) @ TruMean2) + np.log(.5) + (5 * -.5 * np.log(2*np.pi)) - (.5 * np.linalg.det(TruCov2)))

Tot = 0
for i in range(len(TestX)):
    if G1[i] > G2[i] and TestC[i] == 2:
        Tot += 1
    if G1[i] < G2[i] and TestC[i] == 1:
        Tot += 1

print("Bayes Classifier using True Model Error:", Tot/float(len(TestX)))


print("\nFor N-Train = 1000")

# Import Data From Proj3Train1000.xlsx
ED = pd.read_excel("Proj3Train1000.xlsx",header=None)
index = ED.index
rows = len(index)
DataX = np.empty((0,5),float)
DataC = np.empty((0),float)
for i in range(rows):
    if(pd.isna(ED.iloc[i,0]) or pd.isna(ED.iloc[i,1]) or pd.isna(ED.iloc[i,2])):
        print("Data Missing! Skipping row " , i + 2)
    else:
        DataX = np.append(DataX,np.array([[ED.iat[i,0],ED.iat[i,1],ED.iat[i,2],ED.iat[i,3],ED.iat[i,4]]]),0)
        DataC = np.append(DataC, ED.iat[i,5])

#Naive Bayes Classifier 
Mean1 = np.mean(DataX[0:int(len(DataX)/2)],axis = 0)
Mean2 = np.mean(DataX[int(len(DataX)/2):int(len(DataX))],axis = 0)

temp1 = np.zeros(5)
temp2 = np.zeros(5)
for i in range(500):
    temp1 += (DataX[i] - Mean1)**2
    temp2 += (DataX[i+500] - Mean2)**2
Var1 = np.sqrt(temp1 /500.0)
Var2 = np.sqrt(temp2 /500.0)


G1 = np.ones(int(len(TestX)))
G2 = np.ones(int(len(TestX)))
for i in range(len(TestX)):
    for j in range(5):
        G1[i] = G1[i] * (NormalPDF(TestX[i][j],Mean1[j],Var1[j]) * .5)
        G2[i] = G2[i] * (NormalPDF(TestX[i][j],Mean2[j],Var2[j]) * .5)

Tot = 0
for i in range(len(TestX)):
    if G1[i] > G2[i] and TestC[i] == 2:
        Tot += 1
    if G1[i] < G2[i] and TestC[i] == 1:
        Tot += 1

print("Naive Bayes Classifier Error:", Tot/float(len(TestX)))

#Bayes Classifier using MLE
Cov1 = np.cov(DataX[0:int(len(DataX)/2)],rowvar = False)
Cov2 = np.cov(DataX[int(len(DataX)/2):int(len(DataX))],rowvar = False)

Mean1 = np.mean(DataX[0:int(len(DataX)/2)],axis = 0)
Mean2 = np.mean(DataX[int(len(DataX)/2):int(len(DataX))],axis = 0)

G1 = np.empty(0,float)
G2 = np.empty(0,float)
for i in range (len(TestX)):
    G1 = np.append(G1,(-.5 * TestX[i] @ (np.linalg.inv(Cov1)) @ (TestX[i].T)) + (.5 * TestX[i] @ (np.linalg.inv(Cov1)) @ Mean1) + (.5 * (Mean1.T) @ (np.linalg.inv(Cov1)) @ (TestX[i].T)) - (.5 * (Mean1.T) @ np.linalg.inv(Cov1) @ Mean1) + np.log(.5) + (5 * -.5 * np.log(2*np.pi)) - (.5 * np.linalg.det(Cov1)))
    G2 = np.append(G2,(-.5 * TestX[i] @ (np.linalg.inv(Cov2)) @ (TestX[i].T)) + (.5 * TestX[i] @ (np.linalg.inv(Cov2)) @ Mean2) + (.5 * (Mean2.T) @ (np.linalg.inv(Cov2)) @ (TestX[i].T)) - (.5 * (Mean2.T) @ np.linalg.inv(Cov2) @ Mean2) + np.log(.5) + (5 * -.5 * np.log(2*np.pi)) - (.5 * np.linalg.det(Cov2)))

Tot = -10
for i in range(len(TestX)):
    if G1[i] > G2[i] and TestC[i] == 2:
        Tot += 1
    if G1[i] < G2[i] and TestC[i] == 1:
        Tot += 1

print("Bayes Classifier using MLE Error:", Tot/float(len(TestX)))

#Find Error Using Bayes classifier using True Parameters
G1 = np.empty(0,float)
G2 = np.empty(0,float)
for i in range (len(TestX)):
    G1 = np.append(G1,(-.5 * TestX[i] @ (np.linalg.inv(TruCov1)) @ (TestX[i].T)) + (.5 * TestX[i] @ (np.linalg.inv(TruCov1)) @ TruMean1) + (.5 * (TruMean1.T) @ (np.linalg.inv(TruCov1)) @ (TestX[i].T)) - (.5 * (TruMean1.T) @ np.linalg.inv(TruCov1) @ TruMean1) + np.log(.5) + (5 * -.5 * np.log(2*np.pi)) - (.5 * np.linalg.det(TruCov1)))
    G2 = np.append(G2,(-.5 * TestX[i] @ (np.linalg.inv(TruCov2)) @ (TestX[i].T)) + (.5 * TestX[i] @ (np.linalg.inv(TruCov2)) @ TruMean2) + (.5 * (TruMean2.T) @ (np.linalg.inv(TruCov2)) @ (TestX[i].T)) - (.5 * (TruMean2.T) @ np.linalg.inv(TruCov2) @ TruMean2) + np.log(.5) + (5 * -.5 * np.log(2*np.pi)) - (.5 * np.linalg.det(TruCov2)))

Tot = 0
for i in range(len(TestX)):
    if G1[i] > G2[i] and TestC[i] == 2:
        Tot += 1
    if G1[i] < G2[i] and TestC[i] == 1:
        Tot += 1

print("Bayes Classifier using True Model Error:", Tot/float(len(TestX)))