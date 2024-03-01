from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

def getValues():
    inputvalues = np.loadtxt(open("microchips.csv", "r"), delimiter=",")
    return inputvalues

def partOne(): # Reads csv file
    mapcolor = ListedColormap(['red', 'green'])
    plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap=mapcolor)
    plt.figure(1)
    plt.title('Original data') 


def getNN(a, b): # Calculate distance
    dist = np.sqrt(pow(b[0] - a[0], 2) + pow(b[1] - a[1], 2))
    return dist


def nearestNeighbors(x, y, k): # Checks k nearest neighbors and returns which it belongs to
    chip = [x, y]
    chip1List = []
    for row in data:
        dist = getNN(chip, [row[0], row[1]])
        chip1List.append([dist, row[2]])
    chip1List.sort()
    count = 0
    for i in range(k):
        count += chip1List[i][1]
    if (count > (k/2)):
        return 1
    else:
        return 0


def checkChips(k): # Checks NN for three cases in chiplist
    chiplist = [[-0.3, 1.0], [-0.5, -0.1], [0.6, 0.0]]
    i = 0
    print("k =", k)
    for chip in chiplist:
        i += 1
        if (nearestNeighbors(chip[0], chip[1], k) == 1):
            print(f"\tchip{i}: {chip} ==> OK")
        else:
            print(f"\tchip{i}: {chip} ==> FAIL")


def getTrainingErrors(k):
    errorSum = 0
    for row in data:
        if nearestNeighbors(row[0], row[1], k):
            if not row[2]:
                errorSum += 1
        else:
            if row[2]:
                errorSum += 1
    return errorSum


def printPartThree(xx, yy, ax, a, b, k): # Printer for decision boundary
    xymesh = np.c_[xx.ravel(), yy.ravel()]
    mapcolor = ListedColormap(['red', 'green'])
    Z = np.array([nearestNeighbors(x, y, k) for x, y in xymesh])
    Z = Z.reshape(xx.shape)
    ax[a, b].set_title("k = " + str(k) + ", training errors = " + str(getTrainingErrors(k)))
    ax[a, b].contourf(xx, yy, Z, cmap=mapcolor, alpha=0.4)
    ax[a, b].scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap=mapcolor)


def partThree(): # Create subplots, figure, meshgrid and calls on printer
    fig, ax = plt.subplots(2, 2)
    plt.figure(2)
    pixels = 100 # Meshsize per axis, ex 100 -> 100x100 (pixels is an incorrect name, but it describes its purpose)
    x_min, x_max = (data[:, 0].min() - 0.1), (data[:, 0].max() + 0.1)
    y_min, y_max = (data[:, 1].min() - 0.1), (data[:, 1].max() + 0.1)
    meshSize = (x_max - x_min) / pixels
    xx, yy = np.meshgrid(np.arange(x_min, x_max, meshSize), np.arange(y_min, y_max, meshSize))

    print('\nProgress:')
    printPartThree(xx, yy, ax, 0, 0, 1)
    print('|xx------| 25% complete')
    printPartThree(xx, yy, ax, 0, 1, 3)
    print('|xxxx----| 50% complete')
    printPartThree(xx, yy, ax, 1, 0, 5)
    print('|xxxxxx--| 75% complete')
    printPartThree(xx, yy, ax, 1, 1, 7)
    print('|xxxxxxxx| 100% complete')


def runner():
    partOne()

    checkChips(1)
    checkChips(3)
    checkChips(5)
    checkChips(7)

    partThree()
    plt.show()

data = getValues()
runner()