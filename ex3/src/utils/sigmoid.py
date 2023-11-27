import numpy as np


def sigmoid(x):
    """
    Sigmoid function.

    :param x: input vector
    :return: 1 / (1 + e^(-x))
    """
    return 1.0 / (1.0 + np.power(np.e, np.dot(-1, x)))


# def sigmoid(x):
#     """
#     Sigmoid function.
#
#     :param x: input vector
#     :return: 1 / (1 + e^(-x))
#     """
#     return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))