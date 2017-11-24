import numpy as np
from sklearn import datasets


def lr(x, w):
    return 1 / (1 + np.exp(np.dot(x, w)))


def gradient(x, w, y):
    return (y - lr(x, w)) * x


if __name__ == "__main__":

    iris = datasets.load_iris()
    x = iris.data[:-50]
    y = iris.target[:-50]
    w = np.random.random([4, 1])
    eta = 0.001

    for j in range(1000):
        for i in range(x.shape[0]):
            x_ = x[i]
            y_ = y[i]
            w -= eta * gradient(x_, w, y_).reshape(4,1)



