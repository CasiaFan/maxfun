#!/usr/env/python2.7
# -*- coding:utf-8 -*-
__author__ = "Arkenstone"

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
from gensim.models.word2vec import Word2Vec
from matplotlib.font_manager import FontProperties
from pylab import mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import preprocessing as pp
import ConfigParser
import codecs
import os
import re
import logging
from logging.config import fileConfig

class DensityPeakCluster():
    def __init__(self):
        pass


    def get_vocab_distance_file(self, wv_model_path, distance_outfile, text_thred=5):
        # get pairwise distance between word in vocab
        # use word2vec model 1-word.similarity as distance
        # The similarity is just cosine distance: see here http://reference.wolfram.com/language/ref/CosineDistance.html
        # words whose frequencies are larger than text_thred will be analyzed
        model = Word2Vec.load(wv_model_path)
        model_vocab = model.vocab.keys()
        filtered_vocab = [word for word in model_vocab if model.vocab[word].count >= text_thred]
        # get similarity
        with codecs.open(distance_outfile, mode='wb', encoding='utf-8') as of:
            for i in xrange(len(filtered_vocab)):
                for j in xrange(i, len(filtered_vocab)):
                    distance = 1 - model.similarity(filtered_vocab[i], filtered_vocab[j])
                    of.write("{} {} {} {} {}\n".format(i, j, filtered_vocab[i], filtered_vocab[j], distance))
        of.close()


    def load_word_distance(self, distance_outfile, tokens=None):
        # word similarity file should have 5 columns corresponding to word1 index, word2 index, word1, word2, similarity
        # return a dictionary whose key is word pair and value is distance and vocabulary list
        # tokens: only load data related to given words
        if not os.path.exists(distance_outfile):
            logging.error("Word distance file not exits! Perform get_vocab_distance_file function first!")
        with codecs.open(distance_outfile, encoding='utf-8') as fi:
            distance = {}
            if isinstance(tokens, type(None)):
                vocab = []
            else:
                try:
                    assert isinstance(tokens, list) or isinstance(tokens, np.ndarray)
                except:
                    logging.error("Tokens specified in word distance module should be list or np.array")
                    exit(-1)
                vocab = tokens
            line_index = 1
            for line in fi:
                elements = line.strip().split(" ")
                try:
                    assert len(elements) == 5
                except:
                    logging.error("Word distance file only has 5 columns! {} line has {} columns".format(line_index, len(elements)))
                    exit(-1)
                if isinstance(tokens, type(None)):
                    if elements[2] not in vocab:
                        vocab.append(elements[2])
                    if elements[3] not in vocab:
                        vocab.append(elements[3])
                # sort word1 and word2
                sw1, sw2 = sorted([elements[2], elements[3]])
                distance[(sw1, sw2)] = float(elements[4])
                line_index += 1
        fi.close()
        self.distance = distance
        self.distance_vocab = np.asarray(vocab)


    def select_distance_cutoff(self):
        """
        One can choose dc so that the average number of neighbors is around 1 to 2% of the total number of points in the data set
        :param distance: dictionary of pairwise words distance
        :param vocab: unique vocabulary list. vocab has n * (n + 1) / 2 words
        :return: distance cutoff
        """
        try:
            assert self.distance and len(self.distance_vocab)
        except:
            logging.error("distance dictionary and distance_vocab array not exist! Run load word distance first!")
        percent = 0.05
        word_count = len(self.distance_vocab)
        dc_index_in_dict = int(word_count * (word_count + 1) / 2 * percent)
        dc = sorted(self.distance.values())[dc_index_in_dict]
        return dc


    def local_density(self, dc):
        rho = {}
        distance_vocab_count = len(self.distance_vocab)
        for i in xrange(distance_vocab_count):
            rho_i = 0
            for j in xrange(i, distance_vocab_count):
                sw1, sw2 = sorted([self.distance_vocab[i], self.distance_vocab[j]])
                rho_i += 1 if self.distance[(sw1, sw2)] <= dc else 0
            rho[self.distance_vocab[i]] = rho_i
        return rho


    def min_distance_higher_density(self, rho):
        delta = {}
        nearest_neighbour = {}
        sorted_rho_list = sorted(rho.items(), key=lambda x: x[1], reverse=True)
        for i, (word_i, rho_i) in enumerate(sorted_rho_list[1:]):
            min_dij = np.inf
            for (word_j, rho_j) in sorted_rho_list[0:i+1]:
                sw1, sw2 = sorted([word_i, word_j])
                if self.distance[(sw1, sw2)] < min_dij:
                    min_dij = self.distance[(sw1, sw2)]
                    nearest_neighbour[word_i] = word_j
            delta[word_i] = min_dij
        delta[sorted_rho_list[0][0]] = max(delta.values())
        return delta, nearest_neighbour


    def find_centers_by_decision_graph(self, rho, delta, decision_graph_file=None):
        try:
            assert len(self.distance_vocab) == len(rho) == len(delta)
        except:
            logging.error("Given rho, delta, vocab should have same length, while number of vocab is {}, rho is {}, delta is {}".format(len(self.distance_vocab), len(rho), len(delta)))
            exit(-1)
        logging.info("Plot decision graph: rho-delta and n-gamma...")
        delta_values = []
        rho_values = []
        for word in self.distance_vocab:
            delta_values.append(delta[word])
            rho_values.append(rho[word])
        delta_values = np.asarray(delta_values)
        rho_values = np.asarray(rho_values)
        gamma_index_tuple = zip(range(len(rho_values)), delta_values * rho_values)
        sorted_gamma = sorted(gamma_index_tuple, key=lambda x: x[1], reverse=True)
        # plot rho-delta plot
        fig = plt.figure(figsize=(5, 10))
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.scatter(rho_values, delta_values)
        ax1.set_title("decision graph - rho vs delta")
        ax1.set_xlabel("rho")
        ax1.set_ylabel("delta")
        # plot n-gamma plot
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.scatter(range(1, 1+len(sorted_gamma)), [gamma_tuple[1] for gamma_tuple in sorted_gamma])
        ax2.set_title("decision graph - n vs gamma")
        ax2.set_xlabel("n")
        ax2.set_ylabel("gamma")
        if not decision_graph_file:
            plt.show()
        else:
            try:
                assert os.path.exists(os.path.dirname(decision_graph_file))
            except:
                os.makedirs(os.path.dirname(decision_graph_file))
            fig.savefig(decision_graph_file, dpi=300)
        # define number of centers from raw input
        logging.info("Input number of centers by decision graph:")
        center_count = int(raw_input().strip())
        # center_count = 10
        center_word = [self.distance_vocab[gamma_index[0]] for gamma_index in sorted_gamma[:center_count]]
        return np.asarray(center_word)


    def cluster(self, tokens, word2vec_model_path, distance_outfile, distance_word_thred=5, distance_file_override=False, decision_graph_file=None):
        # distance_file_override: if true, regenerate word distance file
        # decision_save_file: plot save file for decision graph
        # distance_outfile: word distance output file
        if not os.path.exists(distance_outfile) or distance_file_override:
            self.get_vocab_distance_file(word2vec_model_path, distance_outfile, text_thred=distance_word_thred)
        # load word distance
        self.load_word_distance(tokens=tokens, distance_outfile=distance_outfile)
        # get dc
        dc = self.select_distance_cutoff()
        # get local density
        rho = self.local_density(dc)
        # get min distance to higher density
        delta, nearest_neighbour = self.min_distance_higher_density(rho)
        # get center_word
        center_words = self.find_centers_by_decision_graph(rho=rho, delta=delta, decision_graph_file=decision_graph_file)
        # assign rest points to clusters
        center_labels = {}
        cluster = {}
        for index, center in enumerate(center_words):
            cluster[center] = index
            center_labels[center] = index
        # get vocab list in reverse rho order
        sorted_rho_items = sorted(rho.items(), key=lambda x: x[1], reverse=True)
        sorted_vocab = [item[0] for item in sorted_rho_items]
        for word in sorted_vocab:
            if word in center_words:
                continue
            neighbour = nearest_neighbour[word]
            # if label of nearest_neighbour is defined, assign neighbour's label to current one
            # ======== it seems need outlier screen rule here to assign outliers to a single group labeld with -1========
            if neighbour in cluster.keys():
                cluster[word] = cluster[neighbour]
            else:
                cluster[word] = -1
        return cluster, center_labels



class TokenClustering():
    def __init__(self, method='mbkm', n_clusters=10, batch_size=100, tol=1e-4, distance_word_thred=5, distance_outfile=None, distance_file_override=False, decision_graph_file=None, show_text_thred=15, plot_result=True, cluster_fig_save_path=None):
        self.method=method
        if method == 'mbkm':
            self.n_clusters = n_clusters
            self.batch_size = batch_size # batch size for mini batch kmeans clustering
            self.tol = tol # tolerance for centroid change during kmeans clustering
        elif method == 'cfsdp':
            self.distance_word_thred = distance_word_thred
            self.distance_outfile = distance_outfile
            self.distance_file_override = distance_file_override
            self.decision_graph_file = decision_graph_file
        # text_thred: text count with minimum count will be displayed. Only work when word_count is specified
        self.show_text_thred = show_text_thred
        self.plot_result = plot_result
        self.cluster_fig_save_path = cluster_fig_save_path


    def clustering(self, word2vec_model_path, tags=None):
        # tags are array or list of tags (in unicode). If given, doing clustering on these tags rather than on whole vocabulary
        # method: kmeans or cfsdp (clustering by fast search and find of density peaks)
        # if method is cfsdp, distance_outfile should be specified
        try:
            assert os.path.exists(word2vec_model_path)
        except:
            raise AssertionError("Word2Vec model not found! Check given path: {}".format(word2vec_model_path))
        # load trained word2vec model
        word2vec_model = Word2Vec.load(word2vec_model_path)
        # get vocab in model
        vocab = word2vec_model.vocab.keys()
        # get feature vector of each word
        tokens = []
        word_vecs = []
        word_count = []
        if self.distance_word_thred:
            logging.info("Word for distance calculation should have minimum count of {}".format(self.distance_word_thred))
            min_text_count = self.distance_word_thred
        else:
            min_text_count = 1
        if isinstance(tags, list) or isinstance(tags, np.ndarray):
            # statistic unique tag count and larger counter means larger size
            unique_tags, count = np.unique(tags, return_counts=True)
            for index, word in enumerate(unique_tags):
                # only use word in model vocab
                if word in vocab and word2vec_model.wv.vocab[word].count >= min_text_count:
                    tokens.append(unique_tags[index])
                    word_vecs.append(word2vec_model[word])
                    word_count.append(count[index])
            word_count = np.asarray(word_count)
            tokens = np.asarray(tokens)
        else:
            for word in vocab:
                word_vecs.append(word2vec_model[word])
            tokens = np.asarray(vocab)
            word_count = np.asarray([1] * len(vocab))
        word_vecs = np.asarray(word_vecs)
        if self.method == 'mbkm':
            # clustering using mini batch kmeans in case of memory overflow
            mbkm = MiniBatchKMeans(n_clusters=self.n_clusters, init='k-means++', batch_size=self.batch_size, tol=self.tol)
            mbkm.fit(word_vecs)
            # get word vector cluster centroids and labels
            centroids = mbkm.cluster_centers_
            labels = mbkm.labels_
        elif self.method == 'cfsdp':
            try:
                assert self.distance_outfile != None
            except:
                logging.error("Clustering method is chosen cfsdp, distance file path should be specified!")
                distance_outfile = raw_input().strip()
            cfsdp = DensityPeakCluster()
            # get cluster, center dict with word as key and label as value
            cluster, center = cfsdp.cluster(tokens=tokens, word2vec_model_path=word2vec_model_path, distance_outfile=self.distance_outfile,
                                            distance_file_override=self.distance_file_override, decision_graph_file=self.decision_graph_file)
            centroids = center.items()
            labels = np.asarray([cluster[token] for token in tokens])
        # return cluster centroids and label of each sample
        if self.plot_result:
            self.plot_clustering_results(tokens=tokens, word_vec=word_vecs, centroids=centroids, labels=labels, word_count=word_count)
        return tokens, word_count, centroids, labels


    def plot_clustering_results(self, tokens, word_vec, centroids, labels, word_count=None):
        chinese_font = FontProperties(fname='/usr/share/fonts/MyFonts/YaHei.Consolas.1.11b.ttf')
        # mpl.rcParams['font.sans-serif'] = ['SimHei']
        # create a new figure
        fig = plt.figure(figsize=(10, 10))
        # adjust layout
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
        n_clusters = len(centroids)
        cmap = plt.get_cmap(name='gnuplot')
        # +1 here is for outliers whose label is -1
        colors = [cmap(i) for i in np.linspace(0, 1, num=n_clusters+1)]
        # print out colors
        print "Color for each cluster is:\n"
        for index, color in enumerate(colors):
            print "{}/{}".format(index, plc.to_hex(color))
        # use t-SNE algorithm to embed high dimension word vectors into 2D
        tsne_model = TSNE(n_components=2, random_state=0)
        wv_embed_coord = tsne_model.fit_transform(np.asarray(word_vec))
        # get color of each point
        wv_color = np.asarray([colors[i] for i in labels])
        ax = fig.add_subplot(1, 1, 1)
        size = np.asarray([1] * len(labels))
        if not isinstance(word_count, type(None)):
            size = word_count
            if self.show_text_thred:
                filter_tokens = tokens[word_count >= self.show_text_thred]
                filter_tokens_coor = wv_embed_coord[word_count >= self.show_text_thred]
        ax.scatter(wv_embed_coord[:, 0], wv_embed_coord[:, 1], s=size, c=wv_color)
        # print filtered tokens in fig
        if self.show_text_thred:
            for index, token in enumerate(filter_tokens):
                plt.text(filter_tokens_coor[index][0], filter_tokens_coor[index][1], token, fontsize=12, fontproperties=chinese_font)
        # plt.show()
        if self.cluster_fig_save_path:
            plt.savefig(self.cluster_fig_save_path, dpi=300)


def main():
    config = ConfigParser.ConfigParser()
    config.read("sentiment_config.ini")
    word2vec_model = config.get("model_save", "word2vec_comment_model_file")
    model_save_path = config.get("model_save", "model_save_path")
    cluster_fig_save_path = model_save_path + "/token_clustering.tags.cfsdp.png"
    n_clusters = 20
    clustering_result = model_save_path + "/clustering_result.tags.cfsdp.csv"
    # retrieve tags from database
    localhost=config.get('database', 'localhost')
    username=config.get('database', 'username')
    password=config.get('database', 'password')
    dbname=config.get('database', 'dbname')
    tbname="comments_phrase_nlp_results"
    field = 'tags'
    method = 'cfsdp'
    distance_outfile = model_save_path + "/word_distance.txt"
    distance_file_override = False
    decision_graph_file = model_save_path + "/decision_graph.png"
    distance_word_thred = 5
    show_text_thred = 15
    tags_sentences = np.asarray(next(pp.get_df_from_db(localhost=localhost, username=username, password=password, dbname=dbname, tbname=tbname, fields=field))).ravel()
    tags_array = np.asarray([re.split(u" ", sentence) for sentence in tags_sentences if sentence])
    tags = np.asarray([x for array in tags_array for x in array])
    token_cluster_obj = TokenClustering(method=method, n_clusters=n_clusters, cluster_fig_save_path=cluster_fig_save_path, show_text_thred=show_text_thred,
                                        distance_word_thred=distance_word_thred, distance_outfile=distance_outfile,
                                        distance_file_override=distance_file_override, decision_graph_file=decision_graph_file)
    tokens, word_count, _, labels = token_cluster_obj.clustering(word2vec_model_path=word2vec_model, tags=tags)
    with codecs.open(clustering_result, mode="wb", encoding='utf-8') as of:
        of.write("token,count,label\n")
        for index, label in enumerate(labels):
            of.write("{},{},{}\n".format(tokens[index], word_count[index], label))
    of.close()


if __name__ == '__main__':
    # load log format
    fileConfig("logging_conf.ini")
    main()
