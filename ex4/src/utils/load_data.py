import pandas as pd
from os import getcwd, path
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


def load_data():
    """
    Loads the training data to variables X, y.

    The training data can be found under /src/data/ex3data1.mat

    :return: X, y the training set and associated labels
    """

    print('Loading Data ...')

    file_name = path.join(getcwd(),'ex4', 'src' ,'data', 'Dry_Bean_Dataset.xlsx')

    df = pd.read_excel(file_name)

    # Separate features (X) and target variable (y)
    X = df.drop(columns=['Class'])  # Drop the 'Class' column to get the features
    y = df['Class']  # Extract the 'Class' column as the target variable


    # Shuffle the data while keeping X and y aligned
    X, y = shuffle(X, y, random_state=42)

    # Create a dictionary similar to the one loaded with scipy.io.loadmat
    data_dict = {'X': X.values, 'y': y.values}

    # print(data_dict['X'])  # Features
    # print(data_dict['y'])  # Target variable
    # print("Size of X:", len(data_dict['X']))
    # print("Size of y:", len(data_dict['y']))

    label_encoder = LabelEncoder()
    data_dict['y'] = label_encoder.fit_transform(data_dict['y'])

    # print(data_dict['y'])  # Target variable
    # print("Size of y:", len(data_dict['y']))

    # scaler = StandardScaler()
    # data_dict['X'] = scaler.fit_transform(data_dict['X'])

    return data_dict['X'], data_dict['y']

