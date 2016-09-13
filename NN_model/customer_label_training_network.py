# !/usr/bin/env python2.7
__author__ = "Arkenstone"

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from check_input import CheckInput
import logging
import numpy as np
import pandas as pd
import os

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
        self.training_set_col_pattern = training_set_col_pattern
        self.prediction_save_dir = prediction_save_dir
        self.prefix = prefix
        self.train_test_ratio = train_test_ratio
        self.hidden_units = kwargs.get('hidden_units', 8)
        self.activation = kwargs.get('activation', 'softmax')
        self.layer = kwargs.get('layer', 4)  # at least 2 layers
        self.drop_out = kwargs.get('drop_out', 0.2)
        self.nb_epoch = kwargs.get('nb_epoch', 10)
        self.batch_size = kwargs.get('batch_size', 100)
        self.loss = kwargs.get('loss', 'mean_squared_error')
        self.optimizer = kwargs.get('optimizer', 'rmsprop')

    def NN_model_train(self, trainX, trainY, testX, textY):
        """
        set up the NN architecture and store the trained model to output directory
        :param trainX (df): training set data input
        :param trainY (df): training set data output
        :param testX (df): test data set input
        :param textY (df): test data set output
        :return:  trained model
        """
        # check na
        trainX = CheckInput().check_na(trainX)
        trainY = CheckInput().check_na(trainY)
        testX = CheckInput().check_na(testX)
        testY = CheckInput().check_na(textY)
        # check dimension
        CheckInput().check_dimension(trainX, testX)
        CheckInput().check_dimension(trainY, testY)
        # convert df input to np.array
        trainX = np.array(trainX).reshape((trainX.shape[0], trainX.shape[1]))
        testX = np.array(testX).reshape((testX.shape[0], testX.shape[1]))
        col_re = -1
        if len(trainY.shape) == 1:
            col_re = 1
        else:
            col_re = trainY.shape[1]
        trainY = np.array(trainY).reshape((trainY.shape[0], col_re))
        testY = np.array(testY).reshape((testY.shape[0], col_re))
        # get input and output dimension of the customer
        input_dim = trainX.shape[1]
        output_dim = trainY.shape[1]
        logging.info("This network is a fully-connected network, with %d input dimension, %d output dimension and %d layers", (input_dim, output_dim, self.layer))
        model = Sequential()
        # input layer
        model.add(Dense(output_dim=self.hidden_units, input_dim=input_dim, activation=self.activation))
        model.add(Dropout(self.drop_out))
        # hidden layer
        for i in range(1,self.layer-2):
            model.add(Dense(output_dim=self.hidden_units, input_dim=self.hidden_units, activation=self.activation))
            model.add(Dropout(self.drop_out))
        # output layer
        model.add(Dense(output_dim=output_dim, input_dim=self.hidden_units, activation=self.activation))
        # compile
        logging.info("Compiling model...")
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        logging.info("Compiling donw!")
        logging.info("Staring training!")
        model.fit(trainX, trainY, nb_epoch=self.nb_epoch, batch_size=self.batch_size)
        logging.info("Training done!")
        score = model.evaluate(testX, textY, batch_size=self.batch_size, verbose=1)
        logging.info("The score of the trained model is %f", score)
        # save model to json files
        if self.prefix:
            json_file = self.model_save_dir + "/" + self.prefix + "-model.json"
            model_weight_file = self.model_save_dir + "/" + self.prefix + "-model.weight.h5"
        else:
            json_file = self.model_save_dir + "/model.json"
            model_weight_file = self.model_save_dir + "/model.weight.h5"
        logging.info("Model is saved in %s and model weight is saved in %s", (json_file, model_weight_file))
        model_json = model.to_json()
        with open(json_file, "w") as json_string:
            json_string.write(model_json)
        # save weights
        if os.path.exists(model_weight_file):
            os.remove(model_weight_file)
        model.save_weights(model_weight_file)
        return model

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
        model_file = model_path + "/model.json"
        model_weight_file = model_path + "/model.weight.h5"
        try:
            with open(model_file, 'r') as json_string:
                logging.info("Loading model...")
                model_json = json_string.read()
                model = model_from_json(model_json)
                json_string.close()
        except Exception, e:
            logging.error("Model file is not found! This model should be provided by %s", model_path, exc_info=True)
        model.load_weights(model_weight_file)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        logging.info("Model loading done!")
        logging.info("Predicting data!")
        predict = model.predict(data, batch_size=self.batch_size, verbose=1)
        predict_prob = model.predict_proba(data, batch_size=self.batch_size, verbose=1)
        return predict, predict_prob

def customer_labeling():
    intdir = "/home/fanzong/Desktop/RNN-prediction/enterprises-train.5-5"
    outdir = "/home/fanzong/Desktop/RNN-prediction/enterprises-train.5-5"



