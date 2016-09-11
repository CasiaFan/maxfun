"""
import log_format

from process_functions import *
from trainingset_selection import trainingSetSelection
from check_input import CheckInput
from Xmeans import XMeans
from clustering_methods import ClusteringMethod
from customer_clustering import CustomerClustering
from customer_label_training_network import CustomerLabelingNN
from neural_network_run import NeuralNetwork


import os
cwd = os.getcwd()
os.system("export PYTHONPATH=$PYTHONPATH:%s" %cwd)
"""
