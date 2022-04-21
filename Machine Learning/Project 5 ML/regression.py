import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def readDataset():
    dataFrame = pd.read_excel('Proj5Dataset.xlsx')
    X = dataFrame[['X']].to_numpy()
    T = dataFrame[['T']].to_numpy()
    return X.T, T.T

def tanhPrime(Z):
    return 1 - (Z ** 2)

def forwardPropagation(X, W0, B0, W1, B1):
    AHiddenLayer = np.dot(W0, X) + B0
    ZHiddenLayer = np.tanh(AHiddenLayer)
    AOutputLayer = np.dot(W1, ZHiddenLayer) + B1
    ZOutputLayer = np.tanh(AOutputLayer)
    return AHiddenLayer, ZHiddenLayer, AOutputLayer, ZOutputLayer
    
def backPropagation(N, X, AHiddenLayer, ZHiddenLayer, AOutputLayer, ZOutputLayer, W0, B0, W1, B1, T, NLO, prevGradient0, prevGradient1):
    rho= 0.085
    beta = 0.91
    delta = ZOutputLayer - T
    gradient = np.dot(delta, ZHiddenLayer.T)
    gradient = (beta * prevGradient1) + ((1 - beta) * gradient)
    prevGradient1 = gradient
    W1 = W1 - rho * gradient
    B1 = B1 - rho * np.tile(np.mean(delta, axis=1), (NLO, N))
    lossHidden = np.dot(W1.T, delta)
    delta = lossHidden * tanhPrime(ZHiddenLayer)
    gradient = np.dot(delta, X.T)
    gradient = (beta * prevGradient0) + ((1 - beta) * gradient)
    prevGradient0 = gradient
    W0 = W0 - rho * gradient
    B0 = B0 - rho * np.tile(np.mean(delta, axis=1)[np.newaxis].T, (1, N))
    return W0, B0, W1, B1

def smoothCurvePlot(W0, B0, W1, B1):
    X = np.linspace(-1, 1, 50)[np.newaxis]
    AHiddenLayer = np.dot(W0, X) + B0
    ZHiddenLayer = np.tanh(AHiddenLayer)
    AOutputLayer = np.dot(W1, ZHiddenLayer) + B1
    ZOutputLayer = np.tanh(AOutputLayer)
    return X, ZOutputLayer

def plotInputVsModel(X, T, W0, B0, W1, B1, loss, NHO):
    title = 'Input Data vs Model (' + str(NHO) + ' Hidden Units)\nLOSS = ' + str(loss)
    plt.figure(title)
    if NHO == 20:
        plt.figure(title)
    plt.title(title)    
    plt.xlim(-1, 1)
    plt.xticks(np.arange(-1, 1.1, 0.5))
    plt.xlabel('X')
    plt.ylim(-1.5, 1.5)
    plt.yticks(np.arange(-1.5, 1.6, 0.5))
    plt.ylabel('T')
    plt.scatter(X, T, color='b')
    plotX, model = smoothCurvePlot(W0, B0, W1, B1)
    plt.plot(plotX.T, model.T, color='r')
    if NHO == 20:
        plt.show()

def plotLossFunction(epochs, loss, NHO):
    title = 'Loss Function (Regression with ' + str(NHO) + ' Hidden Units)'
    plt.figure(title)
    plt.ion
    plt.xlim(0, 3001)
    plt.xticks(np.arange(0, 3001, 500))
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.semilogy(epochs, loss, color='r')
    plt.draw()
    plt.pause(0.001)
    plt.clf()

def fitNNRegression(L, N, X, T, InputUnits, HiddenUnits, OutputUnits):
    np.random.seed(4)
    W0 = np.random.uniform(size=(HiddenUnits, InputUnits))
    W1 = np.random.uniform(size=(OutputUnits, HiddenUnits))
    b0 = np.random.uniform(size=(HiddenUnits, 1))
    B0 = np.tile(b0, (1, N))
    b1 = np.random.uniform(size=(OutputUnits, 1))
    B1 = np.tile(b1, (1, N))

    epochs = 3000
    lossArray = []
    epochsArray = []
    prevGradient0 = np.zeros((HiddenUnits, OutputUnits))
    prevGradient1 = np.zeros((OutputUnits, HiddenUnits))
    for i in range(epochs):
        AHiddenLayer, ZHiddenLayer, AOutputLayer, ZOutputLayer = forwardPropagation(X, W0, B0, W1, B1)
        loss = 0.5 * np.sum((ZOutputLayer - T)**2)
        lossArray.append(loss)
        epochsArray.append(i)
        if (i % 50 == 0):
            plotLossFunction(epochsArray, lossArray, HiddenUnits)
        W0, B0, W1, B1 = backPropagation(N, X, AHiddenLayer, ZHiddenLayer, AOutputLayer, ZOutputLayer, W0, B0, W1, B1, T, OutputUnits, prevGradient0, prevGradient1)
    print(lossArray[len(lossArray) - 1])
    return W0, B0, W1, B1, epochsArray, lossArray

X, T = readDataset()

L = 2
InputUnits = 1
HiddenUnits = 3
OutputUnits = 1
N = X.shape[1]

W0, B0, W1, B1, epochs, loss = fitNNRegression(L, N, X, T, InputUnits, HiddenUnits, OutputUnits)
plotInputVsModel(X, T, W0, B0, W1, B1, loss[len(loss) - 1], HiddenUnits)

HiddenUnits = 20

W0, B0, W1, B1, epochs, loss = fitNNRegression(L, N, X, T, InputUnits, HiddenUnits, OutputUnits)
plotInputVsModel(X, T, W0, B0, W1, B1, loss[len(loss) - 1], HiddenUnits)