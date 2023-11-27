import numpy as np
from .sigmoid import sigmoid


def lr_cost_function(theta, X, y, lam, alpha=3):
    """
    Logistic Regression Cost Function.

    Compute cost and gradient for logistic regression with regularization

    m = size(y)
    cost = 1/m * (sum(-y * log(h_x) - (1-y) * log(1-h_x))) + lambda * sum(theta^2))

    k = size(theta)
    regularized_gradient = [grad1, grad2, ... grad_k]

    :param theta: theta parameters of the model
    :param x: training set
    :param y: training set labels
    :param lam: lambda for regularization
    :param alpha: alpha parameter for gradient

    :return: (cost, gradient) for the given parameters of the model
    """

    m = np.size(y)

    """
    ================================================ YOUR CODE HERE ====================================================
    Instructions: Compute the cost of a particular choice of theta. Compute the partial derivatives and set grad to the
                  partial derivatives of the cost w.r.t. each parameter in theta.

    Hint: The computation of the cost function and gradients can be efficiently vectorized.
          For example, consider the following computation:
    
          ```
          h_x = sigmoid(np.matmul(x, theta))
          ```
    
          Each row of the resulting matrix will contain the value of the prediction for that example.
          You can make use of this to vectorize the cost function and gradient computations.

    Hint: Computing the regularized gradient can be done the following way:
          ```
          grad = <NOT REGULARIZED GRADIENT>
          tmp = theta
          tmp[0] = 0
          grad_reg = <YOUR CODE>

          grad = grad + grad_reg
          ```
    """

    one = y * np.log(sigmoid(np.matmul(X, theta)))
    two = (1 - y) * (np.log(1 - sigmoid(np.matmul(X, theta))))
    reg = (float(lam) / (2 * m)) * sum(theta[1:] ** 2)
    J = -(1. / m) * (one + two).sum() + reg

    theta_c = theta.copy()
    theta_c[0] = 0

    h = np.matmul(X, theta)
    g = sigmoid(h)
    pre = sigmoid(np.matmul(X, theta)) - np.reshape(y, np.shape(g))
    pre2 = np.matmul(np.transpose(X), pre)
    dif = (1. / m) * pre2

    grad = alpha*(dif + (float(lam) / m) * theta_c)


    return J, grad
