import numpy as np
import pandas as pd


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


Cov1 = np.cov(DataX[0:int(len(DataX)/2)],rowvar = False)
Cov2 = np.cov(DataX[int(len(DataX)/2):int(len(DataX))],rowvar = False)

Mean1 = np.mean(DataX[0:int(len(DataX)/2)],axis = 0)
Mean2 = np.mean(DataX[0:int(len(DataX)/2):int(len(DataX))],axis = 0)

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


print("Total Error:", Tot/float(len(TestX)))




#Import Data From Proj3Train1000.xlsx
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

Cov1 = np.cov(DataX[0:int(len(DataX)/2)],rowvar = False)
Cov2 = np.cov(DataX[int(len(DataX)/2):int(len(DataX))],rowvar = False)

Mean1 = np.mean(DataX[0:int(len(DataX)/2)],axis = 0)
Mean2 = np.mean(DataX[0:int(len(DataX)/2):int(len(DataX))],axis = 0)

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


print("Total Error:", Tot/float(len(TestX)))