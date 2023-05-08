import numpy as np


def sigmoid(predicted, real):
    error = np.sum(np.abs(predicted - real))

    return 1 / (1 + error)


def sse(predicted, real):
    return np.sum((predicted - real) ** 2)


def mse(predicted, real):
    return 1 / len(predicted) * np.sum((predicted - real) ** 2)


def rmse(predicted, real):
    return np.sqrt(1 / len(predicted) * np.sum((predicted - real) ** 2))


