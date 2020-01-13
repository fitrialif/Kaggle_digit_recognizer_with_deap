# import statements needed for the functions below
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import deap_functions
import augment_data


# this function is to change data received from pandas reading files from kaggle that are csv
# from xxx integers it creates matrices of floats
# not sure why this function needs labels as input!!!!
def reformat(dataset, image_size, num_channels=1):
    # .astype - Copy of the array, cast to a specified type, -1 means that numpy should figure it out
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    return dataset


# this function takes a set and adds padding with zeros on each side of every picture by "padding_no"
# this function assumes that padding on every side is the same
# it is not tested what happens if padding is negative
def padding(set_to_pad, padding_no):
    set_padded = np.pad(set_to_pad, ((0, 0), (padding_no, padding_no), (padding_no, padding_no), (0, 0)), 'constant')
    return set_padded


# this function will read the data from kaggle set and prepare if for
# processing with tensorflow
def data_load(input_file, image_size):
    # with pandas read input files
    train_set = pd.read_csv(input_file)

    # train_set = pd.read_csv('/home/sebastian/kaggle/digit_recognizer/dataset/train.csv')

    # i could read also submit data that is used for submit but i will do it in some other function
    # df_submit = pd.read_csv('/home/sebastian/kaggle/digit_recognizer/dataset/test.csv')

    # from train data get labels
    # ----------------------------label_0 label_1 label_2 label_3 label_4 label_5 label_6 ..... label_9
    # it is changing digit 4 into       0       0       0       0       1       0       0 .....       0
    train_set = pd.get_dummies(train_set, columns=['label'])

    # this assigns(or maybe copies)all columns apart from last 10 to a new variable
    # new variable is of type ndarray
    train_set_features = train_set.iloc[:, :-10].values

    # and this just extract last 10 columns because they will be used as labels
    train_set_labels = train_set.iloc[:, -10:].values

    # this is about splitting set to train and validation
    # training_features is list of features (inputs) and training_labels are labels used for training
    # testing_all_features  is list of features (inputs) and testing_all_labels are labels used for testing
    # and validation at the end of training
    # i am not sure about random_state: should I keep seed equal 1212 or maybe leave it random?
    training_features, testing_all_features, training_labels, testing_all_labels = \
        train_test_split(train_set_features, train_set_labels, test_size=0.2, random_state=1212)

    # here test set is split again for testing and validation (of the accuracy at the end of learning)
    testing_features, validating_features, testing_labels, validating_labels = \
        train_test_split(testing_all_features, testing_all_labels, test_size=0.5, random_state=0)

    # reformat data
    training_dataset = reformat(training_features, image_size)
    testing_dataset = reformat(testing_features, image_size)

    validating_dataset = reformat(validating_features, image_size)

    return training_dataset, testing_dataset, validating_dataset, training_labels, testing_labels, validating_labels



# call data augmentation for both training and testing data
def data_load2(input_file, image_size, multiplier):
    # with pandas read input files
    train_set = pd.read_csv(input_file)

    # train_set = pd.read_csv('/home/sebastian/kaggle/digit_recognizer/dataset/train.csv')

    # i could read also submit data that is used for submit but i will do it in some other function
    # df_submit = pd.read_csv('/home/sebastian/kaggle/digit_recognizer/dataset/test.csv')

    # from train data get labels
    # ----------------------------label_0 label_1 label_2 label_3 label_4 label_5 label_6 ..... label_9
    # it is changing digit 4 into       0       0       0       0       1       0       0 .....       0
    train_set = pd.get_dummies(train_set, columns=['label'])

    # this assigns(or maybe copies)all columns apart from last 10 to a new variable
    # new variable is of type ndarray
    train_set_features = train_set.iloc[:, :-10].values

    # and this just extract last 10 columns because they will be used as labels
    train_set_labels = train_set.iloc[:, -10:].values

    # this is about splitting set to train and validation
    # training_features is list of features (inputs) and training_labels are labels used for training
    # testing_all_features  is list of features (inputs) and testing_all_labels are labels used for testing
    # and validation at the end of training
    # i am not sure about random_state: should I keep seed equal 1212 or maybe leave it random?
    training_features, testing_all_features, training_labels, testing_all_labels = \
        train_test_split(train_set_features, train_set_labels, test_size=0.2, random_state=1212)

    # here test set is split again for testing and validation (of the accuracy at the end of learning)
    testing_features, validating_features, testing_labels, validating_labels = \
        train_test_split(testing_all_features, testing_all_labels, test_size=0.5, random_state=0)

    # reformat data
    training_dataset = reformat(training_features, image_size)
    testing_dataset = reformat(testing_features, image_size)
    validating_dataset = reformat(validating_features, image_size)

    # call augmentation
    training_dataset, training_labels = augment_data.augment_data(training_dataset, training_labels, multiplier)
    testing_dataset, testing_labels = augment_data.augment_data(testing_dataset, testing_labels, multiplier)




    return training_dataset, testing_dataset, validating_dataset, training_labels, testing_labels, validating_labels





# this function can be used to load initial population from csv file
# this initial population can be created by running fast_evaluation on randomly
# created individuals and saving those that passed fast_evaluation with fitness more than 0.1-0.4
# this in theory should speed up optimisation of the individuals
def load_initial_population(input_file_):
    input_data = pd.read_csv(input_file_, header=None)

    # create toolbox using deap_functions
    toolbox = deap_functions.create_toolbox()
    population_ = toolbox.population(len(input_data))

    for i in range(len(input_data)):
        for j in (1, 4, 5, 6, 7, 8):
            population_[i][j-1] = int(input_data.loc[i, j])
        population_[i][1] = round(population_[i][1], 3)
        population_[i][2] = round(population_[i][2], 3)

    return population_
