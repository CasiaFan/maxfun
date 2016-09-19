#!/usr/bin/env python2.7
__author__ = "Arkenstone"

from Xmeans import XMeans
from check_input import CheckInput
from sklearn.cluster import DBSCAN, MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import random

class ClusteringMethod():
    def __init__(self):
        pass

    def DBSCAN_clustering(self, df, eps=0.5, min_samps=8):
        """
        :param df: input df for clustering rows
        :param eps: maximum distance between 2 samples to be considered as in same cluster
        :param min_samps: minimum number of neighbouring samples for a point to be considered as core point
        :return: df with labels of each row
        """
        # check input df
        df = CheckInput().check_na(df)
        scaler = StandardScaler().fit(df)
        # scale along columns
        df_scale = scaler.fit_transform(df)
        # compute DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samps, algorithm='ball_tree').fit(df_scale)
        # logging.info("Silhouette Coefficient: %s", str(silhouette_samples(df, db.labels_)))
        df['label'] = db.labels_
        return df

    def hierarchical_clustering(self, df, kmax=20, method='ward', dist='euclidean', treeplot_dir='.', show_plot=True):
        """
        :param df (pd.dataframe): input df
        :param kmax (int): max number of clusters to generate
        :param method (str): algorithm to perform hierarchical clustering;
                        'ward', 'complete', 'average'
        :param dist (str): distance method to compute the linkage;
                    'euclidean', 'l1', 'l2', 'manhatton', 'cosine', 'precomputed'
                    when method is 'ward', only 'euclidean' is accepted.
        :param treeplot_dir (str): directory to get the hierarchical clustering tree plot.
                            PS: The name indicate the sample length
        :param show_plot (bool): show cophenetic correlation coefficient and dendrogram plot or not
        :return: df with labels
        """
        if not os.path.exists(treeplot_dir):
            os.makedirs(treeplot_dir)
        df = CheckInput().check_na(df)
        scaler = StandardScaler().fit(df)
        df_scale = scaler.fit_transform(df)
        hc_z = linkage(df_scale, method=method, metric=dist)
        # compute the cophenetic correlation coefficient to check if the clustering preserve original distances
        coph_coef, coph_dist = cophenet(hc_z, pdist(df_scale))
        logging.info("Cophenetic correlation coefficient of this hierarchical clustering is %0.3f", coph_coef)
        # use elbow method to automatically determine the number of clusters
        last_30 = hc_z[-kmax:, 2]
        idx = np.arange(1, len(last_30)+1)
        # 2nd derivatives of the distances
        acce = np.diff(last_30, 2)
        # get the knee point: if the last iteration is the max value in acce, we want to 2 clusters
        n_clusters = acce[::-1].argmax() + 2
        logging.info("Clusters: %d", n_clusters)
        # visualize the elbow method plot
        file_symbol = len(df.columns)
        out_elbow = treeplot_dir + "/" + "hc-elbow.%s.S%s.png" %(method, file_symbol)
        # plot the distance of last 30 clusters iteratively (inverse the order of distance)
        if show_plot:
            plt.title("Elbow method for cluster number selection")
            plt.xlabel("Iteration")
            plt.ylabel("Distance")
            plt.plot(idx, last_30[::-1])
            # the first and the last distance cannot compute 2nd derivatives
            plt.plot(idx[1:-1], acce[::-1])
            plt.tight_layout()
            plt.savefig(out_elbow, dpi=200)
            plt.close()
        # get labels of each point
        labels = fcluster(hc_z, n_clusters, criterion='maxclust')
        # compute the silhoutte coefficient
        # logging.info("Sihoutte coeffient of hierarchical clustering is %s", str(silhouette_samples(df, labels)))
        df['label'] = labels
        # visualize the dendrogram clustering tree for further check
        out_dendro = treeplot_dir + "/" + "hc-dendrogram.%s.N%d.S%s.png" %(method, n_clusters, file_symbol)
        if show_plot:
            plt.title("Hierarchical clustering dendrogram (lastp)")
            plt.xlabel("Cluster size")
            plt.ylabel("Distance")
            dendrogram(
                hc_z,
                p=n_clusters,
                truncate_mode='lastp',
                show_leaf_counts=True,
                leaf_rotation=90,
                leaf_font_size=12,
                show_contracted=True,
            )
            plt.tight_layout()
            plt.savefig(out_dendro, dpi=200)
            plt.close()
        return df

    def X_means(self, df, **kwargs):
        """
        X-means algorithm for clustering input dataframe
        :param df: input dataframe
        :param kwargs: parameters for kmeans algorithm, including:
                        kmin (int): the minimum clusters classified, default: 2
                        kmax (int): maximum clusters classified, default: None
                        init (str or np.ndarray): ways to initialize first set of centers -
                                                'kmeans++', 'random', or user-provided array of center coordinates
                        bic_cutoff (float): bic criterion for terminate center splitting, default: 1.0
                        max_iter (int): maximum iteration for kmeans cluster in specified region, default: 1000
                        kmeans_tole (float): center distance change criterion during kmeans clustering, default: 0.01
        :return: df with labels
        """
        df = CheckInput().check_na(df)
        scaler = StandardScaler().fit(df)
        df_scale = scaler.fit_transform(df)
        # convert df to np.array
        logging.info("Initializing X-means clustering!")
        model = XMeans(**kwargs).fit(np.asarray(df_scale))
        # logging.info("Silhouette Coefficient: %0.3s", str(silhouette_samples(df, model.labels)))
        df['labels'] = model.labels
        return df

    def MiniBatchKmeans(self, df, **kwargs):
        """
        Mini-Batch Kmeans clustering, especially suitable for large data sets. See detailed arguments in sklearn.cluster.MiniBatchKmeans method
        :param df (pd.Dataframe): input df
        :param kwargs: parameters for mini-Batch-Kmeans clustering
                         kmin (int):  minimum cluster numbers generated
                         kmax (int): maximum cluster numbres generated
                         max_iter (int): maximum iterations before stopping
                         batch_size (int): size of mini batches
                         verbose (bool): verbosity mode.
        :return: df with labels
        """
        kmin = kwargs.pop('kmin', 8)
        kmax = kwargs.pop('kmax', 20)
        df = CheckInput().check_na(df)
        scaler = StandardScaler().fit(df)
        df_scale = scaler.fit_transform(df)
        # init dictionaries to hold inetia scores, labels, BIC scores
        labels, bics, inertias = {}, {}, {}
        for n_cluster in range(kmin, kmax):
            mbkm = MiniBatchKMeans(n_clusters=n_cluster, **kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                mbkm.fit(np.array(df_scale))
            labels[n_cluster] = mbkm.labels_
            cur_centers = mbkm.cluster_centers_
            inertias[n_cluster] = mbkm.inertia_
            # get index of data in each cluster
            cluster_index = [[] for i in range(n_cluster)]
            for i in range(n_cluster):  # or np.sort(np.unique(labels[n_cluster])) if not start with 0
                cluster_index[i] = np.arange(len(labels[n_cluster]))[labels[n_cluster] == i]
            bics[n_cluster] = XMeans().BIC(np.array(df_scale), cluster_index, cur_centers)
        # get the maximum bic value and corresponding labels
        opt_n_cluster = max(bics, key=bics.get)
        logging.info("Optimal cluster number has BIC value %f", bics[opt_n_cluster])
        opt_labels = labels[opt_n_cluster]
        logging.info("The optimum clusters found is %d. And the inertia scores for each cluster count are %s", int(opt_n_cluster), str(inertias))
        df['labels'] = opt_labels
        return df

    def Kmeans(self, df, **kwargs):
        # similar parrameter settings with previous minibatch kmeans
        kmin = kwargs.pop('kmin', 8)
        kmax = kwargs.pop('kmax', 20)
        df = CheckInput().check_na(df)
        scaler = StandardScaler().fit(df)
        df_scale = scaler.fit_transform(df)
        # init dictionaries to hold inetia scores, labels, BIC scores
        labels, bics, inertias = {}, {}, {}
        for n_cluster in range(kmin, kmax):
            km = KMeans(n_clusters=n_cluster, **kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                km.fit(np.array(df_scale))
            labels[n_cluster] = km.labels_
            cur_centers = km.cluster_centers_
            inertias[n_cluster] = km.inertia_
            # get index of data in each cluster
            cluster_index = [[] for i in range(n_cluster)]
            for i in range(n_cluster):  # or np.sort(np.unique(labels[n_cluster])) if not start with 0
                cluster_index[i] = np.arange(len(labels[n_cluster]))[labels[n_cluster] == i]
            bics[n_cluster] = XMeans().BIC(np.array(df_scale), cluster_index, cur_centers)
        # get the maximum bic value and corresponding labels
        opt_n_cluster = max(bics, key=bics.get)
        logging.info("Optimal cluster number has BIC value %f", bics[opt_n_cluster])
        # clustering on whole df
        opt_labels = labels[opt_n_cluster]
        logging.info("The optimum clusters found is %d. And the inertia scores for each cluster count are %s",
                     int(opt_n_cluster), str(inertias))
        df['labels'] = opt_labels
        return df

    ## def SOM_clustering(self, df):



