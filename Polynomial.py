import numpy as np
import matplotlib.pyplot as plt


def getValues(): # Reads csv file and eturns training set and test set in two lists
    inputvalues = np.loadtxt(open("polynomial200.csv", "r"), delimiter=",")
    rowCount, trainingSet, testSet = 0, [], []
    for row in inputvalues:
        rowCount += 1
        if (rowCount <= 100):
            trainingSet.append([row[0], row[1]])
        else:
            testSet.append([row[0], row[1]])
    return trainingSet, testSet


def getNpArrays(array): # Converts a regular list into two numpy arrays, for x- and y- values (sorted)
    x1, y1 = [], []
    npArray = np.array(array)
    arraySorted = npArray[npArray[:, 0].argsort()]
    for row in arraySorted:
        x1.append(row[0])
        y1.append(row[1])
    x = np.array(x1)
    y = np.array(y1)
    return x, y


def oneXtwo(): # Prints training set and test set in a plot, side by side
    fig, (ax1, ax2) = plt.subplots(1, 2)
    trainingSet, testSet = getValues()
    trX, trY = getNpArrays(trainingSet)
    ax1.plot(trX, trY) # Use this for plot
    ax1.scatter(trX, trY, s=15) # Use this for scatter
    ax1.set_title('Training set')
    ax1.legend(['train'], loc=2)
    testX, testY = getNpArrays(testSet)
    ax2.plot(testX, testY) # Use this for plot
    ax2.scatter(testX, testY, s=15) # Use this for scatter
    ax2.set_title('Test set')
    ax2.legend(['test'], loc=2)
    plt.figure(1)


def getRegression(tSet, x, k): # Get the y-value for the regression line
    dist, test = [], []
    for point in tSet:
        if (point[0] < x):
            test.append(x - point[0])
        else:
            test.append(point[0] - x)

    dist = np.insert(tSet, 2, test, axis=1)
    distSorted = dist[dist[:, 2].argsort()]
    mean = 0
    for i in range(k):
        mean += distSorted[i][1]
    mean = mean / k
    return mean


def getMSE(trX, trY, listSorted, k): # Calculates MSE and returns it
    MSEtot = 0
    for i in range(trX.size):
        reg = getRegression(listSorted, trX[i], k)
        if (reg >  trY[i]):
            MSEtot += pow((reg - trY[i]), 2)
        else:
            MSEtot += pow((trY[i] - reg), 2)
    return (MSEtot/trX.size)


def printRegressionLine(ax, a, b, k): # Print regression line and training set dots in a subplot
    trainingSet, testSet = getValues()
    trX, trY = getNpArrays(trainingSet)
    testX, testY = getNpArrays(testSet)
    ax[a, b].scatter(trX, trY, s=10, c='blue')
    
    npTrainingSet = np.array(trainingSet)
    trainSorted = npTrainingSet[npTrainingSet[:, 0].argsort()]
    regResult, regX = [], []
    tStep = (trX.max() - trX.min())/200
    for x in np.arange(trX.min(), trX.max(), tStep):
        regX.append(x)
        regResult.append(getRegression(trainSorted, x, k))

    MSEval = round(getMSE(trX, trY, trainSorted, k), 2)

    MSEtest = round(getMSE(testX, testY, trainSorted, k), 2)

    ax[a, b].plot(regX, regResult)
    ax[a, b].legend(['training set', 'regression line'], loc=2)
    ax[a, b].set_title('polynomial_train, k = ' + str(k) + ', MSE: train=' + str(MSEval) + ', test=' + str(MSEtest))


def twoXthree(): # Call on printer to print 6 subplots
    fig, ax = plt.subplots(2, 3)
    printRegressionLine(ax, 0, 0, 1)
    printRegressionLine(ax, 0, 1, 3)
    printRegressionLine(ax, 0, 2, 5)
    printRegressionLine(ax, 1, 0, 7)
    printRegressionLine(ax, 1, 1, 9)
    printRegressionLine(ax, 1, 2, 11)
    plt.figure(2)
    plt.show()


def run():
    print('I would say that k=9 gives the best regression, see comments in code for motivation.')
    oneXtwo()
    twoXthree()


run()


# Best regression:
# When looking at the mean deviation of the training set, k=1 gives MSE = 0 because it is clearly overfitting. It has 
# the highest MSE for the test set (49!!). Basically all other k's (3, 5, 7, 9, 11) have reasonable and quite similar MSE's. 
# They only differ between 27.7-31.58, where k=9 has the lowest MSE. Comparing the MSE results with the regression results, 
# we get that the order of the best regression is as follows: 9, 5, 7, 11, 3, 1. The lower mean deviation, the better. Therefore->

# Conclusion: k = 9 has the lowest mean deviation and is therefore the best k value for regression.