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


class TokenClustering():
    def __init__(self, n_clusters=10, batch_size=100, tol=1e-4, text_thred=15, plot_result=True, fig_save_path=None):
        self.n_clusters = n_clusters
        self.batch_size = batch_size # batch size for mini batch kmeans clustering
        self.tol = tol # tolerance for centroid change during kmeans clustering
        self.text_thred = text_thred  # text_thred: text count with minimum count will be displayed. Only work when word_count is specified
        self.plot_result = plot_result
        self.fig_save_path = fig_save_path


    def clustering(self, word2vec_model_path, tags=None):
        # tags are array or list of tags (in unicode). If given, doing clustering on these tags rather than on whole vocabulary
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
        if not isinstance(tags, type(None)):
            # statistic unique tag count and larger counter means larger size
            unique_tags, count = np.unique(tags, return_counts=True)
            for index, word in enumerate(unique_tags):
                # only use word in model vocab
                if word in vocab:
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
        # clustering using mini batch kmeans in case of memory overflow
        mbkm = MiniBatchKMeans(n_clusters=self.n_clusters, init='k-means++', batch_size=self.batch_size, tol=self.tol)
        mbkm.fit(word_vecs)
        # get word vector cluster centroids and labels
        centroids = mbkm.cluster_centers_
        labels = mbkm.labels_
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
        colors = [cmap(i) for i in np.linspace(0, 1, num=n_clusters)]
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
            if self.text_thred:
                filter_tokens = tokens[word_count >= self.text_thred]
                filter_tokens_coor = wv_embed_coord[word_count >= self.text_thred]
        ax.scatter(wv_embed_coord[:, 0], wv_embed_coord[:, 1], s=size, c=wv_color)
        # print filtered tokens in fig
        if self.text_thred:
            for index, token in enumerate(filter_tokens):
                plt.text(filter_tokens_coor[index][0], filter_tokens_coor[index][1], token, fontsize=12, fontproperties=chinese_font)
        # plt.show()
        if self.fig_save_path:
            plt.savefig(self.fig_save_path, dpi=300)


def main():
    config = ConfigParser.ConfigParser()
    config.read("sentiment_config.ini")
    word2vec_model = config.get("model_save", "word2vec_comment_model_file")
    model_save_path = config.get("model_save", "model_save_path")
    fig_save_path = model_save_path + "/token_clustering.tags.png"
    n_clusters = 20
    clustering_result = model_save_path + "/clustering_result.tags.csv"
    # retrieve tags from database
    localhost=config.get('database', 'localhost')
    username=config.get('database', 'username')
    password=config.get('database', 'password')
    dbname=config.get('database', 'dbname')
    tbname="comments_phrase_nlp_results"
    field = 'tags'
    tags_sentences = np.asarray(next(pp.get_df_from_db(localhost=localhost, username=username, password=password, dbname=dbname, tbname=tbname, fields=field))).ravel()
    tags_array = np.asarray([re.split(u" ", sentence) for sentence in tags_sentences if sentence])
    tags = np.asarray([x for array in tags_array for x in array])
    token_cluster_obj = TokenClustering(n_clusters=n_clusters, fig_save_path=fig_save_path)
    tokens, word_count, _, labels = token_cluster_obj.clustering(word2vec_model_path=word2vec_model, tags=tags)
    with codecs.open(clustering_result, mode="wb", encoding='utf-8') as of:
        of.write("token,count,label\n")
        for index, label in enumerate(labels):
            of.write("{},{},{}\n".format(tokens[index], word_count[index], label))
    of.close()


if __name__ == '__main__':
    main()
