import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def sigmoid_backward(da, z):
    sig = sigmoid(z)
    return da * sig * (1 - sig)


def relu_backward(da, z):
    dz = np.array(da, copy=True)
    dz[z <= 0] = 0
    return dz
