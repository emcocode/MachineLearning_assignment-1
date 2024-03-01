from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt


def runner():
    #partOne
    data = np.loadtxt(open("microchips.csv", "r"), delimiter=",")
    mapcolor = ListedColormap(['red', 'green'])
    plt.figure(1)
    plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap=mapcolor)
    plt.title('Original data') 

    #partTwo
    dots, res = [], []
    for row in data:
        dots.append([row[0], row[1]])
        res.append(int(row[2]))
    for k in [1, 3, 5, 7]:
        knn = KNeighborsClassifier(k)
        knn.fit(dots, res)
        print(f'k={k}\t')
        chipCounter = 0
        for chip in [[-0.3, 1.0], [-0.5, -0.1], [0.6, 0.0]]:
            chipCounter += 1
            predChip = knn.predict([[chip[0], chip[1]]])
            if predChip:
                print(f"\tchip{chipCounter}: [{chip[0]}, {chip[1]}] ==> OK")
            else:
                print(f"\tchip{chipCounter}: [{chip[0]}, {chip[1]}] ==> FAIL")

    #partThree
    fig, ax = plt.subplots(2, 2)
    plt.figure(2)
    for k in [1, 3, 5, 7]:
        a, b = getCoords(k)
        pixels = 100 # Meshsize per axis, ex 100 -> 100x100 (pixels is an incorrect name, but it describes its purpose)
        x_min, x_max = (data[:, 0].min() - 0.1), (data[:, 0].max() + 0.1)
        y_min, y_max = (data[:, 1].min() - 0.1), (data[:, 1].max() + 0.1)
        meshSize = (x_max - x_min) / pixels
        xx, yy = np.meshgrid(np.arange(x_min, x_max, meshSize), np.arange(y_min, y_max, meshSize))
    
        knn = KNeighborsClassifier(k)
        knn.fit(data[:, :2], data[:, 2])

        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax[a, b].contourf(xx, yy, Z, alpha=0.4, cmap=mapcolor)
        ax[a, b].scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap=mapcolor)
        
        res = knn.predict(data[:, :2])
        errors = np.sum(res != data[:, 2])
        ax[a, b].set_title("k = " + str(k) + ", training errors = " + str(errors))

    plt.show()    


def getCoords(k):
    if (k == 1):
        return 0, 0
    elif (k == 3):
        return 0, 1
    elif (k == 5):
        return 1, 0
    elif (k == 7):
        return 1, 1    
    

runner()