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

    first_hidden_layer_sizes = [8, 10, 12, 14]
    second_hidden_layer_sizes = [8, 10, 12, 14]
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

    for first_hidden_layer_size in first_hidden_layer_sizes:
        for second_hidden_layer_size in second_hidden_layer_sizes:

            """
            ================ Part 2: Initializing Pameters ================
            In this part of the exercise, you will be starting to implment a two
            layer neural network. You will start by implementing a function to initialize
            the weights of the neural network
            (randInitializeWeights.py)
            """

            print('Initializing Neural Network Parameters ...')

            initial_Theta1 = randInitializeWeights(input_layer_size, first_hidden_layer_size)
            initial_Theta2 = randInitializeWeights(first_hidden_layer_size, second_hidden_layer_size)
            initial_Theta3 = randInitializeWeights(second_hidden_layer_size, num_labels)

            # Unroll parameters
            initial_nn_params = np.hstack((initial_Theta1.T.ravel(), initial_Theta2.T.ravel()
                                           , initial_Theta3.T.ravel()))

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
            costFunc = lambda p: nnCostFunction(p, input_layer_size, first_hidden_layer_size, second_hidden_layer_size ,num_labels, X_train, y_train, Lambda)[0]
            gradFunc = lambda p: nnCostFunction(p, input_layer_size, first_hidden_layer_size, second_hidden_layer_size ,num_labels, X_train, y_train, Lambda)[1]

            result = minimize(costFunc, initial_nn_params, method='CG', jac=gradFunc, options={'disp': True, 'maxiter': 100})
            nn_params = result.x

            # Obtain Theta1, Theta2, and Theta3 back from nn_params
            Theta1 = np.reshape(nn_params[:first_hidden_layer_size * (input_layer_size + 1)],
                                (first_hidden_layer_size, input_layer_size + 1), order='F').copy()

            start_index = first_hidden_layer_size * (input_layer_size + 1)
            end_index = start_index + second_hidden_layer_size * (first_hidden_layer_size + 1)
            Theta2 = np.reshape(nn_params[start_index:end_index],
                                (second_hidden_layer_size, first_hidden_layer_size + 1), order='F').copy()

            Theta3 = np.reshape(nn_params[end_index:],
                                (num_labels, (second_hidden_layer_size + 1)), order='F').copy()

            # input("Program paused. Press Enter to continue...")


            """
            ================= Part 4: Prediction on the validation set to determine the best hidden layer size =================
            After training the neural network, we would like to use it to predict the labels.
            """

            train_pred = predict(Theta1, Theta2, Theta3, X_train)
            train_predList = np.array(train_pred).flatten().tolist()
            # print(predList)
            train_yList = np.array(y_train).flatten().tolist()
            # print(yList)
            train_accuracy = np.mean(np.double(np.equal(train_predList, train_yList)*1)) * 100
            training_accuracies.append(train_accuracy)
            print('Training Set Accuracy: %f\n'% train_accuracy,
                  'reached with hidden layer sizes: ', first_hidden_layer_size, ' and ', second_hidden_layer_size)

            valid_pred = predict(Theta1, Theta2, Theta3 ,X_val)
            valid_predList = np.array(valid_pred).flatten().tolist()
            # print(predList)
            valid_yList = np.array(y_val).flatten().tolist()
            # print(yList)
            valid_accuracy = np.mean(np.double(np.equal(valid_predList, valid_yList) * 1)) * 100
            validation_accuracies.append(valid_accuracy)
            print('Validation Set Accuracy: %f\n' % valid_accuracy,
                  'reached with hidden layer sizes: ', first_hidden_layer_size, ' and ', second_hidden_layer_size)

    hidden_layer_sizes_to_plot = [f"{x}x{y}" for x in first_hidden_layer_sizes for y in second_hidden_layer_sizes]
    plot.plot(hidden_layer_sizes_to_plot, training_accuracies)
    plot.plot(hidden_layer_sizes_to_plot, validation_accuracies)
    plot.xlabel('hidden layer size')
    plot.ylabel('Accuracy')
    plot.show()

        # input("Program paused. Press Enter to exit...")
    best_hidden_layer_sizes = [
        first_hidden_layer_sizes[int(np.argmax(validation_accuracies) / 4)],
        second_hidden_layer_sizes[int((np.argmax(validation_accuracies) % 4))]
    ]
    print('The best hidden layer sizes are: ', best_hidden_layer_sizes[0], ' and ', best_hidden_layer_sizes[1],
          'and the reached accuracy on the validation set is: ', np.max(validation_accuracies))

    """
    ================ Part 5: Training the NN again with the determined best hidden layer size================
    """

    print('Initializing Neural Network Parameters ...')

    initial_Theta1 = randInitializeWeights(input_layer_size, best_hidden_layer_sizes[0])
    initial_Theta2 = randInitializeWeights(best_hidden_layer_sizes[0], best_hidden_layer_sizes[1])
    initial_Theta3 = randInitializeWeights(best_hidden_layer_sizes[1], num_labels)

    # Unroll parameters
    initial_nn_params = np.hstack((initial_Theta1.T.ravel(), initial_Theta2.T.ravel(), initial_Theta3.T.ravel()))

    print('Training Neural Network... ')

    #  I tried different lambdas : 0.1 , 1, 3, 5 and reached the best precision with 0.1
    Lambda = 0.1

    # Create "short-hand" for the cost function to be minimized
    costFunc = lambda p: nnCostFunction(p, input_layer_size, best_hidden_layer_sizes[0], best_hidden_layer_sizes[1], num_labels, X_train, y_train, Lambda)[0]
    gradFunc = lambda p: nnCostFunction(p, input_layer_size, best_hidden_layer_sizes[0], best_hidden_layer_sizes[1], num_labels, X_train, y_train, Lambda)[1]

    result = minimize(costFunc, initial_nn_params, method='CG', jac=gradFunc, options={'disp': True, 'maxiter': 100})
    nn_params = result.x

    # Obtain Theta1, Theta2, and Theta3 back from nn_params
    Theta1 = np.reshape(nn_params[:best_hidden_layer_sizes[0] * (input_layer_size + 1)],
                        (best_hidden_layer_sizes[0], input_layer_size + 1), order='F').copy()

    start_index = best_hidden_layer_sizes[0] * (input_layer_size + 1)
    end_index = start_index + best_hidden_layer_sizes[1] * (best_hidden_layer_sizes[0] + 1)
    Theta2 = np.reshape(nn_params[start_index:end_index],
                        (best_hidden_layer_sizes[1], best_hidden_layer_sizes[0] + 1), order='F').copy()

    Theta3 = np.reshape(nn_params[end_index:],
                        (num_labels, (best_hidden_layer_sizes[1] + 1)), order='F').copy()

    # input("Program paused. Press Enter to continue...")

    """
    ================ Part 5: Calculating the accuracy of prediction on the test set 
    using the determined best hidden layer size================
    """

    test_pred = predict(Theta1, Theta2, Theta3, X_test)
    test_predList = np.array(test_pred).flatten().tolist()
    print(test_predList)
    test_yList = np.array(y_test).flatten().tolist()
    print(y_test)
    test_accuracy = np.mean(np.double(np.equal(test_predList, test_yList) * 1)) * 100
    print('Test Set Accuracy: ', test_accuracy, '% reached with hidden layer sizes: ',
          best_hidden_layer_sizes[0], ' and ', best_hidden_layer_sizes[1])

if __name__ == '__main__':
        ex4()
