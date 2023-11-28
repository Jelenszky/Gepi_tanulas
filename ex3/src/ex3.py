import numpy as np
from utils import load_data, one_vs_all, predict_one_vs_all, feature_normalize
from matplotlib import pyplot as plot

def ex3():

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

    """
    ============ One-vs-All Training with training set ============
    """
    input('\nPress ENTER to start One-vs-All on training set')
    print('Training One-vs-All Logistic Regression...')

    """
        ================ Predict for One-Vs-All on validation set to determine the best lambda value ================
    """

    lambdas = [0.03, 0.1, 0.3, 1, 3, 5, 7, 10, 15]
    training_accuracies = []
    validation_accuracies = []
    for lambda_value in lambdas:
        all_theta = one_vs_all(X_train, y_train, num_labels, lambda_value)
        pred = predict_one_vs_all(all_theta, X_train)
        print('Predicted classes for X on the training set using lambda = ', lambda_value,' for regularization')
        print(pred)
        training_accuracy = np.mean(pred == y_train) * 100
        print('Accuracy of predictions on the training set: ', training_accuracy, '%')
        training_accuracies.append(training_accuracy)
        pred = predict_one_vs_all(all_theta, X_val)
        print('Predicted classes for X on the validation set using lambda = ', lambda_value, ' for regularization')
        print(pred)
        validation_accuracy = np.mean(pred == y_val) * 100
        print('Accuracy of predictions on the validation set: ', validation_accuracy, '%')
        validation_accuracies.append(validation_accuracy)

    plot.plot(lambdas, training_accuracies)
    plot.plot(lambdas, validation_accuracies)
    plot.xlabel('lambda')
    plot.ylabel('Accuracy')
    plot.show()

    best_lambda = lambdas[np.argmax(validation_accuracies)]
    print('The best lambda is: ', best_lambda,
          'and the reached accuracy on the validation set is: ', np.max(validation_accuracies))

    """
    ================ Predict for One-Vs-All on test set using the determined best lambda value ================
    """

    input('\nPress ENTER to start One-vs-All on test set')
    all_theta = one_vs_all(X_train, y_train, num_labels, best_lambda)
    np.savetxt('all_theta.out', all_theta, delimiter=',')
    print('Saved all_theta to all_theta.out')
    pred = predict_one_vs_all(all_theta, X_test)
    print('Predicted classes for X on the test set using lambda = ', best_lambda,' for regularization')
    print(pred)
    print('Accuracy of predictions on the test set: ', np.mean(pred == y_test) * 100, '%')


if __name__ == "__main__":
    ex3()