#!/usr/env/python2.7
# -*- coding:utf-8 -*-
__name__ = "Arkenstone"

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from sklearn.cluster import MiniBatchKMeans
from gensim.models.word2vec import Word2Vec
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class TokenClustering():
    def __init__(self, n_clusters=10, batch_size=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.batch_size = batch_size # batch size for mini batch kmeans clustering
        self.tol = tol # tolerance for centroid change during kmeans clustering


    def clustering(self, word2vec_model_path, plot_result=True):
        # load trained word2vec model
        word2vec_model = Word2Vec.load(word2vec_model_path)
        # get feature vector
        word_vec = word2vec_model.syn0
        n_words = word_vec.shape[0]
        vec_dim = word_vec.shape[1]
        # clustering using mini batch kmeans in case of memory overflow
        mbkm = MiniBatchKMeans(n_clusters=self.n_clusters, init='kmeans++', batch_size=self.batch_size, tol=self.tol)
        mbkm.fit(word_vec)
        # get word vector cluster centroids and labels
        centroids = mbkm.cluster_centers_
        labels = mbkm.labels_
        # return cluster centroids and label of each sample
        if plot_result:
            self.plot_clustering_results(word_vec=word_vec, centroids=centroids, labels=labels)
        return centroids, labels


    def plot_clustering_results(self, word_vec, centroids, labels):
        # create a new figure
        fig = plt.figure(figsize=(8, 3))
        # adjust layout
        fig.subplots_adjust(left=0.3, right=0.3, bottom=0.2, top=0.2)
