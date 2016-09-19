#!/usr/bin/env python2.7
__author__ = "Arkentstone"

from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import euclidean
import logging
import numpy as np

class XMeans():
    def __init__(self, kmin=2, kmax=None, init='kmeans++', **kwargs):
        """
        :param kmin: minimum cluster numbers
        :param kmax: maximum cluster numbers
        :param init: method of cluster initialization the cluster centers.
                    'kmeans++': k cluster centers with best variance will be selected.
                    'random': randomly select k cluster centers
                     or ndarray of pre-definded centers
        :param **kwaggs: bic_cutoff: maximum change in BIC score that terminate the batch run
                        kmeans_tole: stop condition for kmeans clustering in specified region

        """
        self.clusters = None
        self.centers = None
        self.labels = None
        self.kmin = kmin
        self.kmax = kmax
        self.init = init
        self.bic_cutoff = kwargs.get('bic_cutoff', 10.0)
        self.max_iter = kwargs.get('max_iteration', 400)
        self.kmeans_tole = kwargs.get('kmeans_tole', 0.025)

    def fit(self, data):
        """
        Fit model with current input data
        :param data: input data; np arrrays
        :return: fitted models with: clusters (list): index of each point in each cluster (according to initial data)
                                    centers (list): center coordinates
                                    labels (np.array): cluster label of each point according to each cluster
        """
        # ensure the data is np array type
        data = np.asarray(data)
        k = self.kmin
        # make a list to hold cluster centers
        logging.info("Fitting with k=%d", k)
        n_samples = len(data)
        logging.info("Initializing centers!")
        centers = None
        if self.init == 'kmeans++':
            centers = self.__k_int(data, k)
        elif self.init == 'random':
            centers = np.random.random((k, data.shape[1]))
        elif isinstance(self.init, np.ndarray):
            centers = np.array(self.init, dtype=data.dtype, copy=True)
        logging.info("Initializing centers done! Initializing clusters!")
        clusters, centers = self.__improve_params(data, centers)
        logging.info(("Initilizing clusters done! Start clustering."))
        iter = 1
        while (len(centers) < self.kmax) or (self.kmax is None):
            logging.debug("Current iteration is %d", iter)
            current_center_count = len(centers)
            # repeat until cluster number not increment or surpass the maximum threshold
            allocated_centers = self.__improve_structure(data=data, clusters=clusters, centers=centers)
            clusters, centers = self.__improve_params(data=data, centers=allocated_centers)
            logging.debug("There are %d clusters classified during the %d iteration", len(centers), iter)
            iter += 1
            # if no longer splitting child clusters, stop
            if len(centers) == current_center_count:
                break
            # cluster numbers should not surpass all data point count
            if len(centers) >= len(data):
                break
        # add label to each data point
        logging.info("Clustering done! Generating lables!")
        labels = np.array([-1 for i in range(len(data))])  # -1 means the point belongs none within the cluster
        for cur_label in range(len(centers)):
            labels[clusters[cur_label]] = cur_label
        logging.info("Labeling done!")
        logging.info("All works done!")
        self.clusters = clusters
        self.centers = centers
        self.labels = labels
        return self

    @classmethod
    def __k_int(cls, data, n_clusters, n_local_trials=None):
        """
        Initializing n_cluster seeds according to k-means++ algorithm (modified from sklearn _k_init function)
        :param data: input data; shape (n_samples, n_features)
        :param n_clusters: number cluster seeds
        :return: seeds after initialization
        """
        # set the number of local seeding trial: 2+log(k) (Arthur/Vassilvitskii tried)
        if n_local_trials is None:
            n_local_trials = 2 + int(np.log(n_clusters))
        # run k-means++ algorithm
        # select first centroid randomly from the data set
        n_samples, n_fetures = data.shape
        center_id = np.random.randint(n_samples)
        centers = np.empty((n_clusters, n_fetures))
        centers[0] = data[center_id]
        # initialize list of closest distances and calculate current potential.
        x_squared_norms = np.einsum('ij,ij->i', data, data)  # Y_norm_squared is sum of squared row elements
        closest_dist_sq = euclidean_distances(centers[0].reshape(1, -1),
                                              data,
                                              Y_norm_squared=x_squared_norms,
                                              squared=True)
        current_pot = closest_dist_sq.sum()
        # pick the remaining n_cluster-1 centroid
        for k in range(1, n_clusters):
            # choose centroid candidates by sampling with probability proportional
            # to the squared distances to closest existing centroid
            rand_vals = np.random.random_sample(n_local_trials) * current_pot
            candidate_ids = np.searchsorted(closest_dist_sq.cumsum(), rand_vals) # searchsorted fidn the later value's closest index in former array with solid order
            # compute distance to center candidates
            dist_to_candidates = euclidean_distances(data[candidate_ids],
                                                     data,
                                                     Y_norm_squared=x_squared_norms,
                                                     squared=True)
            # decide which candidates is best
            best_candi = None
            best_pot = None
            best_dist_sq = None
            for trial in range(n_local_trials):
                # compute the potential when including center candidate.
                # Reassign point to new center if it is closer than to initial center
                new_dist_sq = np.minimum(closest_dist_sq, dist_to_candidates[trial])
                new_pot = new_dist_sq.sum()
                if best_candi is None or new_pot < best_pot:  # first half ensure there are at least 2 centers,
                    # the later make sure the sum of distances is decreasing
                    best_candi = candidate_ids[trial]
                    best_pot = new_pot
                    best_dist_sq = new_dist_sq
            # permanantly add best center candidate found in local tries
            centers[k] = data[best_candi]
            closest_dist_sq = best_dist_sq
            current_pot = best_pot
        return centers

    def BIC(self, data, clusters, centers):
        """
        Compute splitting criterion for input clusters using bayesian information criterion.
        (modified from pyClustering/xmeans.py) ===> seem to be wrong when state the maximum likelihood estimate for the variance
        use BIC formula from github/gomeans/BIC_notes
        :param data: input data
        :param clusters (list): index of points in each cluster for which splitting criterion should be computed
        :param centers (list): centers of the clusters
        :return: BIC value of current model. In this formula, the BIC values are negative.
                Lower value of splitting criterion means current structure is much better (http://stanfordphd.com/BIC.html)
        """
        scores = [.0] * len(clusters)  # splitting criterion
        M = data.shape[1]
        # log likelihood of data: see formula in paper:
        # "X-means: extending K-means with efficient estimation of numbers of clusters"
        sigma = .0
        K = len(clusters)
        N = data.shape[0]
        for idx_cluster in range(len(clusters)):
            for j in clusters[idx_cluster]:  # this is the index of point in total data set
                sigma += euclidean(data[j], centers[idx_cluster]) ** 2
        # N should always be larger than K
        if N > K:
            sigma /= ((N - K) * M)
            for idx_cluster in range(len(clusters)):
                n = len(clusters[idx_cluster])
                scores[idx_cluster] = - n * M * np.log(2 * np.pi * sigma) / 2 - (n - 1) * M / 2 + n * np.log(
                    n) - n * np.log(N)
        else:
            logging.error("Cluster numbers should not overpass the number of total data points!")
        return sum(scores) - (M + 1) * K * np.log(N) / 2

    @classmethod
    def __update_clusters(cls, data, centers):
        """
        Compute euclidean distance to each point from each cluster. Nearest points are captured by according clusters and clusters are updated
        :param data (np.ndarray): input data
        :param centers (list): coordinates of centers of each cluster
        :return: updated clusters (index of initial data set)
        """
        clusters = [[] for i in range(len(centers))]
        for i in range(len(data)):
            # compute the distance to each center and append the id the optimum cluster
            optim_dist = .0
            optim_clust = -1
            for j in range(len(centers)):
                dist = euclidean(data[i], centers[j])
                if (dist < optim_dist) or (j is 0):
                    # Must be <, not <= . So when coordinates of 2 centers are identical,
                    # only first centers will be returned with nearest points. The second one is null.
                    optim_dist = dist
                    optim_clust = j
            clusters[optim_clust].append(i)
        return clusters

    @classmethod
    def __update_centers(cls, data, clusters):
        """
        Update cluster center coordinates
        :param data (np.ndarray): input data
        :param clusters (list): index of each cluster points (according to initial data)
        :return: centers
        """
        centers = [[] for i in range(len(clusters))]
        for j in range(len(clusters)):
            centers[j] = data[clusters[j]].mean(axis=0)
        return centers

    def __improve_params(self, data, centers):
        """
        Perform k-means clustering in specified region
        :param data (np.ndarray): input data point of specified region
        :param centers (list): centers of clusters
        :return: list of indexes of points in each cluster (according to initial data set);
                and centers coordinates of each cluster
        """
        change = np.inf
        stop_condition = self.kmeans_tole
        clusters = []
        iter = 0
        while (change > stop_condition) and (iter < self.max_iter):
            clusters = self.__update_clusters(data, centers)
            # remove clusters without any data
            clusters = [cluster for cluster in clusters if len(cluster) > 0]
            updated_centers = self.__update_centers(data, clusters)  # get the maximum changes during this iteration
            change = np.max([euclidean(centers[i], updated_centers[i]) for i in range(len(updated_centers))]) # use updated centers just incase all label in same cluster
            # logging.debug("Current change is %f", change)
            centers = updated_centers
            iter += 1
        return clusters, centers

    def __improve_structure(self, data, centers, clusters):
        """
        Check for best structure: decide to divide specified cluster into two or not based on BIC criterion
        :param data (np.ndarray): input data set
        :param centers (list): centers of each clusters
        :param clusters (list): indexes of points in each cluster according to initial data set
        :param bic_criterion: BIC criterion to determine if current cluster should be divided
        :return: allocated centers
        """
        difference = 0.001  # move distance from initial coordinate to generate 2 child centroids
        allocated_centers = []
        # split each centroid
        for idx_cluster in range(len(centers)):
            child_centers = []
            child_centers.append((np.asarray(centers[idx_cluster]) - difference).tolist())
            child_centers.append((np.asarray(centers[idx_cluster]) + difference).tolist())
            # get point coordinates of this cluster
            cluster_data = data[clusters[idx_cluster]]
            # k-means to cluster this cluster after centroid splitting
            child_clusters, child_centers = self.__improve_params(cluster_data, child_centers)
            # determine if splitting this centroid is necessary or not. (There does exist 2 child points)
            if len(child_clusters) > 1:
                # compute the BIC criterion
                parent_bic = self.BIC(data, [clusters[idx_cluster]], [centers[idx_cluster]])
                child_bic = self.BIC(data, child_clusters, child_centers)
                if parent_bic - child_bic > self.bic_cutoff:  # split the centers. In this formula, BIC value the lower the better
                    allocated_centers.append(child_centers[0])
                    allocated_centers.append(child_centers[1])
                else:
                    allocated_centers.append(centers[idx_cluster])
            else:
                # not split the center
                allocated_centers.append(centers[idx_cluster])
        return allocated_centers