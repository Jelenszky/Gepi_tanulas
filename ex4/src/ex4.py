# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize
from .utils.predict import predict
from .utils.nnCostFunction import nnCostFunction
from .utils.randInitializeWeights import randInitializeWeights
from .utils.load_data import load_data
from .utils.feature_normalize import feature_normalize
from matplotlib import pyplot as plot

def ex4():
    """
    Machine Learning Online Class - Exercise 4 Neural Network Learning
    
    Instructions
    ------------
     
    This file contains code that helps you get started on the linear exercise. 
    You will need to complete the following functions in this exericse:
  
       sigmoidGradient.py
       randInitializeWeights.py
       nnCostFunction.py
  
    For this exercise, you will not need to change any code in this file,
    or any other files other than those mentioned above.   
    """

    hidden_layer_sizes = [8, 10, 12, 14]
    training_accuracies = []
    validation_accuracies = []
    ## Setup the parameters you will use for this exercise
    input_layer_size = 16  # 16 features
    # hidden_layer_size = 10   # 10 hidden units
    num_labels = 7  # 7 labels, from 0 to 6

    """
    =========== Part 1: Loading  Data =============
    We start the exercise by first loading the dataset. 
    """

    (X, y) = load_data()

    X, mu, sigma = feature_normalize(X)

    # Calculate the number of records for each set
    total_records = len(X)
    train_size = int(0.7 * total_records)
    val_size = test_size = (total_records - train_size) // 2

    # Create training set
    X_train, y_train = X[:train_size], y[:train_size]

    # Create validation set
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]

    # Create testing set
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    # input("Program paused. Press Enter to continue...")
    for hidden_layer_size in hidden_layer_sizes:

        """
        ================ Part 2: Initializing Pameters ================
        In this part of the exercise, you will be starting to implment a two
        layer neural network. You will start by implementing a function to initialize
        the weights of the neural network
        (randInitializeWeights.py)
        """

        print('Initializing Neural Network Parameters ...')

        initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
        initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

        # Unroll parameters
        initial_nn_params = np.hstack((initial_Theta1.T.ravel(), initial_Theta2.T.ravel()))

        """
        =================== Part 3: Training NN using the training set ===================
        You have now implemented all the code necessary to train a neural network. 
        To train your neural network, we will now use minimize function with ‘CG’ method.
        Recall that these advanced optimizers are able to train our cost functions 
        efficiently as long as we provide them with the gradient computations.
        """

        print('Training Neural Network... ')

        #  I tried different lambdas : 0.1 , 1, 3, 5 and reached the best precision with 0.1
        Lambda = 0.1

        # Create "short-hand" for the cost function to be minimized
        costFunc = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X_train, y_train, Lambda)[0]
        gradFunc = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X_train, y_train, Lambda)[1]

        result = minimize(costFunc, initial_nn_params, method='CG', jac=gradFunc, options={'disp': True, 'maxiter': 100})
        nn_params = result.x

        # Obtain Theta1 and Theta2 back from nn_params
        Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                           (hidden_layer_size, input_layer_size + 1), order='F').copy()
        Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                           (num_labels, (hidden_layer_size + 1)), order='F').copy()

        # input("Program paused. Press Enter to continue...")


        """
        ================= Part 4: Prediction on the validation set to determine the best hidden layer size =================
        After training the neural network, we would like to use it to predict the labels.
        """

        train_pred = predict(Theta1, Theta2, X_train)
        train_predList = np.array(train_pred).flatten().tolist()
        # print(predList)
        train_yList = np.array(y_train).flatten().tolist()
        # print(yList)
        train_accuracy = np.mean(np.double(np.equal(train_predList, train_yList)*1)) * 100
        training_accuracies.append(train_accuracy)
        print('Training Set Accuracy: %f\n'% train_accuracy, 'reached with hidden layer size: ', hidden_layer_size)

        valid_pred = predict(Theta1, Theta2, X_val)
        valid_predList = np.array(valid_pred).flatten().tolist()
        # print(predList)
        valid_yList = np.array(y_val).flatten().tolist()
        # print(yList)
        valid_accuracy = np.mean(np.double(np.equal(valid_predList, valid_yList) * 1)) * 100
        validation_accuracies.append(valid_accuracy)
        print('Validation Set Accuracy: %f\n' % valid_accuracy, 'reached with hidden layer size: ', hidden_layer_size)

    plot.plot(hidden_layer_sizes, training_accuracies)
    plot.plot(hidden_layer_sizes, validation_accuracies)
    plot.xlabel('hidden layer size')
    plot.ylabel('Accuracy')
    plot.show()

        # input("Program paused. Press Enter to exit...")
    best_hidden_layer_size = hidden_layer_sizes[np.argmax(validation_accuracies)]
    print('The best hidden layer size is: ', best_hidden_layer_size,
          'and the reached accuracy on the validation set is: ', np.max(validation_accuracies))

    """
    ================ Part 5: Training the NN again with the determined best hidden layer size================
    """

    print('Initializing Neural Network Parameters ...')

    initial_Theta1 = randInitializeWeights(input_layer_size, best_hidden_layer_size)
    initial_Theta2 = randInitializeWeights(best_hidden_layer_size, num_labels)

    # Unroll parameters
    initial_nn_params = np.hstack((initial_Theta1.T.ravel(), initial_Theta2.T.ravel()))

    print('Training Neural Network... ')

    #  I tried different lambdas : 0.1 , 1, 3, 5 and reached the best precision with 0.1
    Lambda = 0.1

    # Create "short-hand" for the cost function to be minimized
    costFunc = lambda p: nnCostFunction(p, input_layer_size, best_hidden_layer_size, num_labels, X_train, y_train, Lambda)[0]
    gradFunc = lambda p: nnCostFunction(p, input_layer_size, best_hidden_layer_size, num_labels, X_train, y_train, Lambda)[1]

    result = minimize(costFunc, initial_nn_params, method='CG', jac=gradFunc, options={'disp': True, 'maxiter': 100})
    nn_params = result.x

    # Obtain Theta1 and Theta2 back from nn_params
    Theta1 = np.reshape(nn_params[:best_hidden_layer_size * (input_layer_size + 1)],
                        (best_hidden_layer_size, input_layer_size + 1), order='F').copy()
    Theta2 = np.reshape(nn_params[best_hidden_layer_size * (input_layer_size + 1):],
                        (num_labels, (best_hidden_layer_size + 1)), order='F').copy()

    # input("Program paused. Press Enter to continue...")

    """
    ================ Part 5: Calculating the accuracy of prediction on the test set 
    using the determined best hidden layer size================
    """

    test_pred = predict(Theta1, Theta2, X_test)
    test_predList = np.array(test_pred).flatten().tolist()
    print(test_predList)
    test_yList = np.array(y_test).flatten().tolist()
    print(y_test)
    test_accuracy = np.mean(np.double(np.equal(test_predList, test_yList) * 1)) * 100
    print('Test Set Accuracy: ', test_accuracy, '% reached with hidden layer size: ', best_hidden_layer_size)

if __name__ == '__main__':
        ex4()
