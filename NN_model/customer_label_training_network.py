# !/usr/bin/env python2.7
__author__ = "Arkenstone"

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from check_input import CheckInput
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from logging.config import fileConfig
import logging
import numpy as np
import pandas as pd
import os
import re
import random

class CustomerLabelingNN():
    def __init__(self, training_set_dir, model_save_dir, training_set_col_pattern='X\d+', prediction_save_dir=None, prefix=None, train_test_ratio=4, **kwargs):
        """
        :param training_set_dir (str): training set path
        :param model_save_dir (str): directory to hold the trained model and model weight
        :param training_set_col_pattern (str): training set column pattern in the file, like 'X1' => 'X\d+'
        :param prediction_save_dir (str): directory for prediction output if prediction function is performed
        :param prefix (str): file prefix for output file
        :param train_test_ratio (int): ratio to separate input data to training data set and test data set
        :param kwargs: parameter for model training: hidden_units (int): number of hidden unit in dense network;
                                                    activation (str): activation function of NN. See Keras activation
                                                    layer (int): number of layers of the network
                                                    drop_out_rate (float): ratio of unit to be drop before importing to next layer
                                                    nb_epoch (int): number of epoches during training
                                                    batch_size (int): size of batch during training
                                                    loss (str): loss function of nn. See Keras loss to detailed information
                                                    optimizer (str): optimizer method during training. See Keras optimizer for detailed information
        """
        self.training_set_dir = training_set_dir
        self.model_save_dir = model_save_dir
        # self.training_set_col_pattern = training_set_col_pattern
        # self.prediction_save_dir = prediction_save_dir
        self.prefix = prefix
        # self.train_test_ratio = train_test_ratio
        self.hidden_units = kwargs.get('hidden_units', 11)
        self.activation = kwargs.get('activation', 'relu')
        self.layer = kwargs.get('layer', 3)  # at least 2 layers
        self.drop_out = kwargs.get('drop_out', 0.2)
        self.nb_epoch = kwargs.get('nb_epoch', 40)
        self.batch_size = kwargs.get('batch_size', 1000)
        self.loss = kwargs.get('loss', 'categorical_crossentropy')
        self.optimizer = kwargs.get('optimizer', 'adam')

    def NN_model_train(self, trainX, trainY, testX, testY):
        """
        set up the NN architecture and store the trained model to output directory
        :param trainX (df): training set data input
        :param trainY (df): training set data output
        :param testX (df): test data set input
        :param textY (df): test data set output
        :return:  trained model
        """
        def _baseline_model(input_dim, output_dim):
            logging.info(
                "This network is a fully-connected network, with %d input dimension, %d output dimension and %d layers",
                input_dim, output_dim, self.layer)
            model = Sequential()
            # input layer
            self.hidden_units = input_dim * 20
            model.add(Dense(output_dim=self.hidden_units, input_dim=input_dim, activation=self.activation))
            model.add(Dropout(self.drop_out))
            # hidden layer
            for i in range(1, self.layer - 2):
                model.add(Dense(output_dim=self.hidden_units, input_dim=self.hidden_units, activation=self.activation))
                model.add(Dropout(self.drop_out))
            # output layer
            model.add(Dense(output_dim=output_dim, input_dim=self.hidden_units, activation='softmax'))
            # compile
            logging.info("Compiling model...")
            model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
            logging.info("Compiling done!")
            return model
        # convert df input to np.array
        trainX = np.asarray(trainX, dtype='int').reshape((trainX.shape[0], trainX.shape[1]))
        testX = np.asarray(testX,dtype='int').reshape((testX.shape[0], testX.shape[1]))
        trainY = np.asarray(trainY, dtype='int').reshape((trainY.shape[0], trainY.shape[1]))
        testY = np.asarray(testY, dtype='int').reshape((testY.shape[0], testY.shape[1]))
        # get input and output dimension of the customer
        input_dim = trainX.shape[1]
        output_dim = trainY.shape[1]
        model = _baseline_model(input_dim, output_dim)
        # estimator for sklearn cross-validated score
        # estimator = KerasClassifier(build_fn=model, nb_epoch=self.nb_epoch, batch_size=self.batch_size)
        # estimator.fit(trainX, trainY)
        logging.info("Staring training!")
        model.fit(trainX, trainY, nb_epoch=self.nb_epoch, batch_size=self.batch_size)
        logging.info("Training done!")
        score = model.evaluate(testX, testY, batch_size=self.batch_size, verbose=1)
        # logging.info("The score of the trained model is %f", score)
        # save model to json files
        if self.prefix:
            json_file = self.model_save_dir + "/" + self.prefix + "-model.json"
            model_weight_file = self.model_save_dir + "/" + self.prefix + "-model.weight.h5"
        else:
            json_file = self.model_save_dir + "/model.json"
            model_weight_file = self.model_save_dir + "/model.weight.h5"
        logging.info("Model is saved in %s and model weight is saved in %s", json_file, model_weight_file)
        model_json = model.to_json()
        with open(json_file, "w") as json_string:
            json_string.write(model_json)
        # save weights
        if os.path.exists(model_weight_file):
            os.remove(model_weight_file)
        model.save_weights(model_weight_file)
        return None #estimator

    def NN_prediction(self, data, model_path=None):
        """
        Predicting output with given data
        :param data (df): input data
        :param model_path (str): trained model directory. If none, run NN_model train and save the trained modle to the model path
        :return: predicted result of the given data
        """
        # check na data
        data = CheckInput().check_na(data)
        data = np.array(data).reshape((data.shape[0], data.shape[1]))
        # load model
        model_file = model_path + "/"+ self.prefix + "-model.json"
        model_weight_file = model_path + "/" + self.prefix + "-model.weight.h5"
        logging.info("Loading model...")
        model = None
        try:
            with open(model_file, 'r') as json_string:
                logging.info("Loading model...")
                model_json = json_string.read()
                model = model_from_json(model_json)
                json_string.close()
        except Exception, e:
            logging.error("Model file is not found! This model should be provided by %s", model_path, exc_info=True)
        logging.info("Loading model weight...")
        model.load_weights(model_weight_file)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        logging.info("Model loading done!")
        logging.info("Predicting data!")
        predict = model.predict(data, batch_size=self.batch_size, verbose=1)
        # predict_prob = model.predict_proba(data, batch_size=self.batch_size, verbose=1)
        logging.info("Prediction done!")
        return predict #, predict_prob

def _dummy_label(labels):
    """
    convert labels to dummy variables using one hot encoding
    :param labels (np.array): labels as strings
    :return: dummy variables
    """
    encoder = LabelEncoder()
    encoder_labels = encoder.fit_transform(labels)
    dummy_labels = np_utils.to_categorical(encoder_labels, len(set(encoder_labels)))
    return dummy_labels

def _inverse_dummy_label(labels):
    """
    convert dummy labels to initial categorical labels
    :param labels (np.array): dummy label
    :return: categorical label (np.array)
    """
    re = []
    for i in labels:
        max_ele = max(i)
        re.append(list(i).index(max_ele))
    return np.asarray(re)

def _statistic_discrepancy(tar, ref, file_save_path):
    """
    Statistic discrepancies between predicted value and real value and output a bar plot
    :param tar (pd.Series): prediction data
    :param ref (pd.Series): real data
    :param file_save_path: path to save the output bar plot
    :return: None
    """
    discrepancy = np.abs(tar - ref)
    bins = [0.0, 2.0, 5.0, 10.0, 15.0, 30.0, np.inf]
    statistic, bin_edges, binnumber = binned_statistic(x=discrepancy, values=discrepancy, statistic='count', bins=bins)
    x_index = np.arange(len(statistic))
    logging.info("Start plotting...")
    plt.subplot(1,1,1)
    plt.bar(x_index, statistic, color='b')
    plt.xlabel('|x_real - x_expected|')
    plt.ylabel('Counts')
    plt.title("Statistic of predicted and real time intervals")
    # plot counts values
    for x, y in zip(x_index, statistic):
        plt.text(x, y+0.1, str(y), ha='center', va='center',fontweight=2)
    plt.xticks(x_index, bins[:-1])
    assert os.path.exists(file_save_path), "Statistic bar plot directory should exist! Check your input: %s" % file_save_path
    plt.savefig(file_save_path+'/statistic_discrepancy.png')
    plt.close()
    logging.info("Plotting done!")
    return None

def main():
    intdir = "/home/fanzong/Desktop/RNN-prediction/enterprises-train.5-5/clustering-mbkm"
    outdir = "/home/fanzong/Desktop/RNN-prediction/enterprises-train.5-5/clustering-mbkm/prediction"
    # data_file = intdir + "/all_data_with_label.csv"
    input_col_pattern = 'X\d+'
    label_col = ['labels']
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    training_ratio = 4
    CL = CustomerLabelingNN(intdir, outdir, prefix='mbkm')
    logging.info("Loading data...")
    # df = pd.read_csv(data_file)
    # get line count of input file
    # line_count_cmd = "cat %s | wc -l" %data_file
    # line_count = int(os.popen(line_count_cmd).read().rstrip())
    df = pd.DataFrame()
    for file in os.listdir(intdir):
        if re.search(r"cluster-\d+.csv", file):
            cur_df = pd.read_csv(intdir+"/"+file)
            df = pd.concat([df, cur_df])
    line_count = len(df.index)
    df.index = range(line_count)
    """
    select_ratio = 1.0
    logging.info("There are %d lines in the input file. Randomly select %f for clustering!", line_count, select_ratio)
    # head = df.columns
    select_index = random.sample(range(line_count), int(line_count * select_ratio))
    df = df.ix[select_index, ]
    df.index = range(len(df.index))
    """
    input_col = re.findall(input_col_pattern, str(df.columns))
    """
    train_X_row = random.sample(df.index, len(df.index) * training_ratio / (training_ratio + 1))
    train_Y_row = [i for i in df.index if i not in train_X_row]
    """
    # convert categorical labels to one hot encoding
    dummy_labels = _dummy_label(np.asarray(df.ix[:, label_col], dtype='float'))
    """
    df_train_X = df.ix[train_X_row, input_col]
    df_test_X = df.ix[train_Y_row, input_col]
    df_label_train = dummy_labels[train_X_row]
    df_label_test = dummy_labels[train_Y_row]
    """
    df_train_X, df_test_X, df_label_train, df_label_test = train_test_split(df.ix[:,input_col], dummy_labels, test_size=training_ratio/(training_ratio+1.0), random_state=0)
    total_X = df.ix[:, input_col]
    logging.info("Splitting data done! Starting training model!")
    CL.NN_model_train(df_train_X, df_label_train, df_test_X, df_label_test)
    predict = CL.NN_prediction(total_X, outdir)
    # transform one hot encoding to original categorical label
    predict = _inverse_dummy_label(predict)
    # predict = np.asarray(predict, dtype='int')
    df['predict'] = predict
    # df['prob'] = prob.ravel()
    df.to_csv(outdir + "/prediction.csv")
    # statistic_discrepancy(np.asarray(df['predict'],dtype='float32'), np.asarray(df['label'],dtype='float32'), outdir)
    x = np.asarray(df[label_col]).ravel()
    y = np.asarray(df['predict']).ravel()
    match_count = sum(x == y)
    logging.info("Labeling Precision: %f", float(match_count)/len(df.predict))
    """
    # estimate model with k-Fold cross validation
    seed = 7
    np.random.seed(seed)
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(estimator, np.asarray(total_X), dummy_labels, cv=kfold)
    logging.info("CrossValidation Baseline: %f(%f)", results.mean(), results.std())
    """

if __name__ == '__main__':
    fileConfig('logging_conf.ini')
    logging = logging.getLogger(__name__)
    main()










