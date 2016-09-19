#!/usr/bin/env python2.7
__author__ = "Arkenstone"

from clustering_methods import ClusteringMethod
from check_input import CheckInput
from logging.config import fileConfig
import logging
import pandas as pd
import numpy as np
import os
import sys
import re
import random
import matplotlib.pyplot as plt

class CustomerClustering():
    def __init__(self):
        pass

    def clustering_run(self, df, method='xmeans', **kwargs):
        """
        Performing clustering with specified method: Xmeans, DBSCAN, hierarchical clustering
        :param df (pd.Dataframe): input df. DONT scale df now! Scalling will be took within functions
        :param method (str): specified method. Only support 3 methods mentioned above. 'xmeans', 'dbscan', 'hc', 'mbkm', 'km'
        :param kwargs: arguments required for according method. See clustering_methods.py for detailed infomation
        :return: df with clustering labels
        """
        logger.warning("DONT scale df now! It will be scaled within clustering method function.")
        cm = ClusteringMethod()
        # select one method for clustering
        df_labels = None
        if method is "xmeans":
            logger.info("Current clustering method is Xmeans! Clustering start!")
            df_labels = cm.X_means(df, **kwargs)
        elif method is "dbscan":
            logger.info("Current clustering method is DBSCAN! Clustering start!")
            df_labels = cm.DBSCAN_clustering(df, **kwargs)
        elif method is "hc":
            logger.info("Current clustering method is hierarchical clustering! Clustering start!")
            df_labels = cm.hierarchical_clustering(df, **kwargs)
        elif method is 'mbkm':
            logger.info("Current clustering method is Mini-Batch Kmeans! Clustering start!")
            df_labels = cm.MiniBatchKmeans(df, **kwargs)
        elif method is 'km':
            logger.info("Current clustering method is Kmeans! Clustering start!")
            df_labels = cm.Kmeans(df, **kwargs)
        else:
            logger.error("Clustering method must be xmeans, dbscan or hc! Chcek your input!")
        return df_labels

    def plot_clustering(self, df, save_fig_file, restrict=True):
        """
        Plot scatter-line figure to shoe cluster pattern
        :param df: input df
        :param save_fig_file (str): save figure to this file
        :param restrict: restrict large data sets for plot in case memory overflow
        :return: None
        """
        logger.info("Start to plot cluster pattern...")
        """
        # initialize a figure map with 4 columns and output each clusters pattern in a subplot
        n_cluster = len(np.unique(df['labels']))
        subplot_len = int((n_cluster - 1) / 4) + 1
        fig, axes = plt.subplots(subplot_len, 4, figsize=(8, 12))
        fig.canvas.set_window_title("Time interval patterns")
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.legend()
        cols = re.findall(r'X\d+', str(df.columns))
        for i in range(n_cluster):
            df_label = df.ix[df['labels'] == i, cols]
            df_label.columns = range(len(cols))
            label_mean = df_label.mean(axis=0)
            # plot_index = 100 * subplot_len + 40 + i + 1
            # plt.subplot(plot_index)
            axes[int(i / 4)][i % 4].plot(df_label.T, 'b', alpha=0.3)
            axes[int(i / 4)][i % 4].plot(np.array(label_mean), '-or', lw=2)
            axes[int(i / 4)][i % 4].set_ylabel("time intervals")
            axes[int(i / 4)][i % 4].set_title("cluster %d" %i)
        plt.savefig(save_fig_file, dpi=320)
        """
        row_count = len(df.index)
        selected_row = df.index
        if restrict:
            if row_count > 1e5:
                selected_row = random.sample(df.index, int(1e5))
        plt.subplot(1, 1, 1)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        cols = re.findall(r'X\d+', str(df.columns))
        df = df.ix[selected_row, cols]
        df.columns = [int(i) for i in range(len(cols))]
        label_mean = df.mean(axis=0)
        plt.plot(df.T, 'b', alpha=0.2, lw=0.5)
        plt.plot(np.array(label_mean), '-or', lw=4)
        plt.ylabel("time interval - days")
        plt.title("Time interval patterns")
        # plt.legend()
        plt.savefig(save_fig_file)
        plt.close()
        logger.info("Plot done!")
        return None


def main():
    # get input directory
    intdir = "/home/fanzong/Desktop/RNN-prediction/enterprises-train.5-5"
    intfile = intdir + "/all_data.csv"
    method = 'xmeans'
    outdir = intdir + "/clustering-" + method
    training_set_col_pattern = 'X\d+'
    # make directories if not exist
    if not os.path.exists(intdir):
        os.makedirs(intdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # initialize clustering method parameters

    pars = {
        'xmeans':{
            'kmin': 8,
            'kmax': None,
            'init': 'kmeans++',
            'bic_cutoff': 100,
            'kmeans_tole': 0.01,
        },
        # time and spatial cost of these algorithms are O(n^2). May cause Memory errors.
        'dbscan':{
            'eps': 0.5,
            'min_samps': 8,
        },
        'hc':{
            'kmax': 20,
            'treeplot_dir': outdir,
        },
        'mbkm':{
            'kmin': 8,
            'kmax': 20,
            'max_iter': 100,
            'batch_size': 100,
            'init': 'random',
            'verbose': 1,
        },
        'km':{
            'kmin': 8,
            'kmax': 20,
            'max_iter': 400,
            'init': 'random',
            'verbose': 1,
        }
    }
    cc = CustomerClustering()
    stdout_backup = sys.stdout
    logfile = outdir + "/clustering_log.txt"
    loghandler = open(logfile, "w")
    sys.stdout = loghandler
    logger.info("Read in data from %s", intfile)
    """
    # get line count of input file
    line_count_cmd = "cat %s | wc -l" %intfile
    line_count = int(os.popen(line_count_cmd).read().rstrip())
    select_ratio = 0.001
    logger.info("There are %d lines in the input file. Randomly select %f for clustering!", line_count, select_ratio)
    """
    df = pd.read_csv(intfile)
    """
    select_index = random.sample(range(line_count-1), int(line_count * select_ratio))
    df =df.ix[select_index,]
    """
    # get training set length
    training_length = len(re.findall(training_set_col_pattern, str(df.columns)))
    X_cols = ['X' + str(i) for i in range(1, 1+training_length)]
    df_train = df[X_cols]
    ############################################
    # sort training set
    df_train = df_train.apply(lambda x: np.sort(x), axis=1)
    ############################################
    df_labels = cc.clustering_run(df_train, method=method, **pars[method])
    df_labels[['customer_id', 'enterprise_id']] = df[['customer_id', 'enterprise_id']]
    # cc.plot_clustering(df_labels, save_fig_file=outdir + "/clustering_pattern.png")
    # split df based on there labels
    logger.info("Splitting data based on labels!")
    for label in pd.unique(df_labels['labels']):
        logger.info("Current label is %d", label)
        df_cur_label = df_labels[df_labels['labels'] == label]
        outfile = outdir + "/cluster-" + str(label) + ".csv"
        df_cur_label.to_csv(outfile)
        # plot cluster pattern
        cc.plot_clustering(df_cur_label, save_fig_file=outdir + "/cluster-" + str(label) + ".png")
    logger.info("Splitting done!")
    sys.stdout = stdout_backup
    loghandler.close()

if __name__ == "__main__":
    log_conf_file = "logging_conf.ini"
    fileConfig(log_conf_file, disable_existing_loggers=False)
    logger = logging.getLogger(__name__)
    main()