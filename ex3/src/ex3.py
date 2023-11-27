import numpy as np
from utils import load_data, lr_cost_function, one_vs_all, predict_one_vs_all, feature_normalize

def ex3():
    """
    Machine Learning Class - Exercise 3 | Part 1: One-vs-all

    Instructions
    ------------
    This file contains code that helps you get started on the
    linear exercise. You will need to complete the following functions:

       - lr_cost_function.py (logistic regression cost function)
       - one_vs_all.py
       - predict_one_vs_all.py

    For this exercise, you will not need to change any code in this file,
    or any other files other than those mentioned above.
    """

    print()
    print('--------------------------------')
    print('Logistic regression via OneVsAll')
    print('--------------------------------')
    print()

    # Setup the parameters you will use for this part of the exercise
    # 16 features for 13,611 instances of dry beans
    # 7 labels
    num_labels = 7

    """"
    =========== Part 1: Loading and Visualizing Data =============
    We start the exercise by first loading the dataset.
    You will be working with a dataset that contains features of dry beans.
    
    """
    (X, y) = load_data()

    X, mu, sigma = feature_normalize(X)

    """"
    ============ Part 2a: Vectorize Logistic Regression ============
    
    In this part of the exercise, you will reuse your logistic regression
    code from the last exercise. You task here is to make sure that your
    regularized logistic regression implementation is vectorized. After
    that, you will implement one-vs-all classification for the handwritten
    digit dataset.
    
    """
    # Test case for lr_cost_function
    print('Testing lr_cost_function() with regularization')
    print('Example data:')

    theta_t = np.array([-2, -1, 1, 2])
    print('theta=', theta_t)

    x_t = np.c_[np.ones((5, 1)), np.reshape(range(1, 16), (3, 5)).T / 10]
    print('x=', x_t)

    y_t = np.array([1, 0, 1, 0, 1]) >= 0.5
    print('y=', y_t)

    lambda_t = 3
    print('lambda=', lambda_t)

    (J, grad) = lr_cost_function(theta_t, x_t, y_t, lambda_t)

    print()
    print('Cost: ', J)
    print('Expected cost: 2.534819')
    print()
    print('Gradients:')
    print(grad)
    print('Expected gradients:')
    print('[ 0.146561 -0.548558 0.724722 1.398003 ]')
    print()

    """
    ============ Part 2b: One-vs-All Training ============
    """
    input('\nPress ENTER to start One-vs-All')
    print('Training One-vs-All Logistic Regression...')

    lam = 0.1
    all_theta = one_vs_all(X, y, num_labels, lam)
    np.savetxt('all_theta.out', all_theta, delimiter=',')
    print('Saved all_theta to all_theta.out')

    """
    ================ Part 3: Predict for One-Vs-All ================
    """

    pred = predict_one_vs_all(all_theta, X)
    print('Predicted classes for X')
    print(pred)
    print('Training Set Accuracy: ', np.mean(pred == y) * 100, '%')
    #
    # print('Visualising thetas for each class')
    #
    # thetas = []
    # for i in range(0, np.size(all_theta, 0)):
    #     thetas.append(all_theta[i][1:].T)
    #
    # display_data(thetas, 'Trained thetas')

if __name__ == "__main__":
    ex3()