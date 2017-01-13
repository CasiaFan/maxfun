#!/usr/env/python2.7
# -*- coding:utf-8 -*-
__author__ = "Arkenstone"

import sys
reload(sys) # for sys.setdefaultencoding() is removed from sys when python starts
sys.setdefaultencoding('utf8')
import numpy as np
import jieba
import jieba.analyse
import jieba.posseg
import pandas as pd
import logging
import os
import codecs
import re
import multiprocessing # for multicore machine
from multiprocessing.pool import ThreadPool
from threading import Thread
from zhon.hanzi import punctuation
from logging.config import fileConfig
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential, model_from_json,save_model, load_model
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from Preprocessing import *

class Sentiment():
    def __init__(self, **kwargs):
        # sentences input preprocessing
        self.entity_mark_file = kwargs.get("entity_mark_file", "./time_and_price_entity_mark.txt")
        # params for sentence tokenizing
        self.html_tag_file = kwargs.get("html_tag_file", "./html_tag_replacement.txt")      # path to file defining html tags
        self.delimiter_file = kwargs.get("delimiter_file", "./full_stop_punctuations.txt")  # path to file defining delimiters
        self.user_add_dict = kwargs.get("user_add_dict", './comment_add_dict.txt')          # path to user defined tokenizing dictionary
        self.user_del_dict = kwargs.get("user_del_dict", "./comment_del_dict.txt")          # path to user defined dictionary to be deleted
        # params for word embedding
        self.vocab_dim = kwargs.get("vocab_dim", 100)        # deifne the dimensionality of feature
        self.min_count = kwargs.get("min_count", 5)          # min word frequency used for word embedding
        self.window = kwargs.get("window", 5)                # maximum distance between current and predicted word within the sentence
        self.workers = multiprocessing.cpu_count()           # multi-processes
        self.iter = kwargs.get("iter", 4)                    # number of interations for word embedding model training
        # params for lstm network
        self.label_degree = kwargs.get("label_degree", 5)    # label levels
        self.maxlen = kwargs.get("maxlen", 100)              # max number of tokens in a sentence used
        self.droupout = kwargs.get("dropout", 0.2)           # dropout rate
        self.nb_epoch = kwargs.get("nb_epoch", 5)            # # of epoches for lstm training
        self.batch_size= kwargs.get("batch_size", 100)       # batch size during lstm training
        # params for model save path
        self.model_save_path = kwargs.get("model_save_path", "./sentiment/model")

    def _load_dictionary(self):
        # load domain specific vocabulary
        if self.user_add_dict:
            try:
                jieba.load_userdict(self.user_add_dict)
            except ValueError(), e:
                raise("No such user deined dictionaty! Check your input.")

    def _delete_vocabulary(self):
        # delete unwanted tokens
        if self.user_del_dict:
            with codecs.open(self.user_del_dict, "rb", encoding="utf-8") as f:
                for line in f:
                    jieba.del_word(line.strip())
    def tokenizer(self, sentences, sub_split=True, remove_redundant_punc=True, replace_uncommon=True):
        """
        :param sentences: comment list (1D)
        :param user_dict: path to user defined tokenizing dictionary
        :param sub_split: Boolean. If true, split sentences into subsentences using full stop delimiters
        :return: list of filtered sentencese, list of tokens of each sentence and list of tokens of each sub-sentences separated by full stop delimiters (2D)
        """
        self._load_dictionary()
        self._delete_vocabulary()
        sentences = parse_html_tag(sentences, self.html_tag_file)  # convert html before processing punctuations
        # remove \n
        sentences = [re.sub(r"\n", "", sentence) for sentence in sentences]
        ################## statistic ##################################################################
        # statistic average sentence length
        # average_sentence_len = sum([len(sentence) for sentence in sentences]) / len(sentences)
        # logging.info("Average sentence lenghth of input sentences is %s" %str(average_sentence_len))
        # statistic average sentence characters count
        # average_character_count = sum([len(set(sentence)) for sentence in sentences]) / len(sentences)
        # logging.info("Average sentence character amount is %s" %str(average_character_count))
        ###############################################################################################
        # replace uncommon punctuations with common "。"
        if replace_uncommon:
            sentences = replace_uncommon_punc(sentences)
        # remove redundant punctuations
        if remove_redundant_punc:
            sentences = rm_redundant_punc(sentences)
        # filter out sentences with little information (short text length)
        sentences = filter_sentences(sentences)
        # replace marked entities
        sentences = mark_entity(sentences, self.entity_mark_file)
        sentences2tokens = np.asarray([jieba.lcut(sentence, cut_all=False, HMM=True) for sentence in sentences])
        sub_sentences2tokens = []
        if sub_split:
            sub_sentences = sentence_splitter(sentences, self.delimiter_file)
            ################# statistic ###############################################################
            # statistic average sentence length, run one time only
            # average_sub_sentence_len = sum([len(sub_sentence) for sub_sentence in sub_sentences]) / len(sub_sentences)
            # logging.info("Average clause length of input sentences is %s" %str(average_sub_sentence_len))
            ###########################################################################################
            # merge too short sentences
            sub_sentences = merge_sub_sentences(sub_sentences)
            for sub_sentence in sub_sentences:
                sub_sentences2tokens.append(jieba.lcut(sub_sentence, cut_all=False, HMM=True))
        return sentences, np.asarray(sentences2tokens), np.asarray(sub_sentences2tokens)

    def idf_statistic(self, sentences2tokens):
        # statistic of inversed document frequency of all comments
        N = len(sentences2tokens)
        all_words = set([x for sentence2tokens in sentences2tokens for x in sentence2tokens])
        ####################### use multiple thread to accelerate idf statistic ###############
        dict = {}
        for word in all_words:
            # remove empty element
            dict[word] = 0
        def _statistic(dict, all_words, sentences2tokens):
            for word in all_words:
                for sentence in sentences2tokens:
                    if word in sentence:
                        dict[word] += 1
                        continue
            return dict
        pool = ThreadPool(processes=10)
        results = pool.apply_async(_statistic, (dict, all_words, sentences2tokens))
        dict = results.get()
        for key in dict.keys():
            dict[key] = np.log(N/dict[key])
        ########################################################################################
        # sort dict in descending order
        dict = sorted(dict.items(), key=lambda x:x[1], reverse=True)    # dict is tuple list type after sorting
        try:
            with codecs.open(self.model_save_path + "/idf.txt", mode="wb", encoding="utf-8") as of:
                for key, value in dict:
                    of.write(key + " " + str(value) + "\n")
            of.close()
        except ValueError:
            logging.error("File not found: %s" %(self.model_save_path + "/idf.txt"))

    def keyword_extraction(self, sentences, k, pos=None):
        """ Extract top k keywords of comment
        :param k: number of keywords selected
        :param pos: allowed POS list for keywords selection
        """
        # check if idf file exists, if not train one
        idf_file = self.model_save_path + "/idf.txt"
        jieba.analyse.set_idf_path(idf_file)
        keywords = []
        self._load_dictionary()
        self._delete_vocabulary()
        for sentence in sentences:
            keyword = jieba.analyse.extract_tags(sentence, topK=k, allowPOS=pos, withWeight=True)
            keywords.append(keyword)
        return np.asarray(keywords)

    def _save_sorted_vocab(self, model):
        token_list = [(token, model.vocab[token].count) for token in model.vocab]
        sorted_tokens = sorted(token_list, key=lambda x: x[1], reverse=True)
        with codecs.open(self.model_save_path + "/word2vec.vocab", "wb", encoding="utf8") as vf:
            for (token, count) in sorted_tokens:
                vf.write(token + "\t" + str(count) + "\n")
        vf.close()

    def _word2Vec_model(self, sentences2tokens):
        """
        :param sentences: sentences iterable could be list or iterable streams directly from file or corpus, eg: LineSentence
        :model_save_path: model save path
        """
        # training word embedding model by import sentences
        # check if import is not empty
        if len(sentences2tokens) == 0:
            raise logging.error("No data imported!")
        else:
            """
            #NOTE: gensim word2vec not support add new vocabulary to pre-trained model. It ONLY support to continue to train previous trained model
            # check if word2vec model exist. If exists, load and update it; else initialize one
            if not os.path.exists(self.model_save_path + "/word2vec.model"):
                # initialize model
                model = Word2Vec(size=self.vocab_dim,
                                 window=self.window,
                                 min_count=self.min_count,
                                 workers=self.workers,
                                 iter=self.iter,
                                 sorted_vocab=1)
                logging.debug("Tokenizing must be performed before this step!")
                logging.info("Build up trainng vocabulary ...")
                model.build_vocab(sentences2tokens)  # pass in collected words. Sentences should be list of unicode
                model.save(self.model_save_path + "/word2vec.model")
                # save vocabulary
                self._save_sorted_vocab(model)
            else:
                model = Word2Vec(size=self.vocab_dim,
                                 window=self.window,
                                 min_count=self.min_count,
                                 workers=self.workers,
                                 iter=self.iter,
                                 sorted_vocab=1)
                model.load(self.model_save_path + "/word2vec.model")
                model.build_vocab(sentences2tokens)
                model.train(sentences2tokens)        # train the model
                # save models
                model.save(self.model_save_path + "/word2vec.model")  # save_word2vec_format ??
                # save vocabulary
                self._save_sorted_vocab(model)
                logging.info("Number of vocabulary in the model: %d", len(model.vocab))
            """
            model = Word2Vec(size=self.vocab_dim,
                             window=self.window,
                             min_count=self.min_count,
                             workers=self.workers,
                             iter=self.iter,
                             sorted_vocab=1)
            model.build_vocab(sentences2tokens)
            model.train(sentences2tokens)  # train the model
            # save models
            model.save(self.model_save_path + "/word2vec.model")  # save_word2vec_format ??
            # save model vocabulary
            self._save_sorted_vocab(model)
            logging.info("Number of vocabulary in the model: %d", len(model.vocab))

    def word2Vec_model_train(self, tbs, word2vec_fields, override=True):
        """
        Training word2vec model (training data is from mysql database)
        :param tbs: list of tables of dataset
        """
        logging.info("Loading data from DB ...")
        sentences = np.array([])
        for cur_sentences_df in getDataFromDB(tbs, word2vec_fields):
            if len(sentences) == 0:
                sentences = np.asarray(cur_sentences_df.ix[:, 0])
            else:
                sentences = np.concatenate((sentences, np.asarray(cur_sentences_df.ix[:, 0])))
        logging.info("Tokenizing ...")
        filter_sentences, sentences2tokens, sub_sentences2tokens = self.tokenizer(sentences)
        df_sentences = pd.DataFrame(sentences, columns=['initial_comment'])
        df_filter_sentences = pd.DataFrame(filter_sentences, columns=['filtered_comment'])
        df_tokens = pd.DataFrame(sentences2tokens, columns=['tokens'])
        logging.info("Extracting keywords...")
        if not os.path.exists(self.model_save_path + "/idf.txt") or override:
            logging.info("Constructing IDF file...")
            self.idf_statistic(sentences2tokens)
        keywords = self.keyword_extraction(filter_sentences, k=10, pos=None)
        df_keywords = pd.DataFrame(keywords, columns=['keywords'])
        df = pd.concat([df_sentences, df_filter_sentences, df_tokens, df_keywords], axis=1)
        df.to_csv(self.model_save_path + "/comments2tokens.csv", encoding='utf-8')
        logging.info("Updating word2vec model ...")
        self._word2Vec_model(sub_sentences2tokens)
        logging.info("Training done!")

    def _document2index(self, sentences2tokens, model):
        """
        convert tokens of document to word index
        :param sentences: comment sentences in unicode format
        :param model: trained word embedding model
        :param polarity_dgr: comment polarity. It is provided during model training
        :return: training set and test set for lstm training
        """
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.vocab.keys(), allow_update=True) # model.vocab contains tokens, token counts pairs during model training
        # transfer tokens in a sentence to word index in gensim vocabulary dictionary, for keras embedding use word index
        word2index = {word: count for word, count in gensim_dict.items()}
        word2vec = {word: model[word] for word in gensim_dict.keys()}   # word vector matrix as keras initial embedding weight
        sentences2index = []
        for sentence in sentences2tokens:
            tokens2index = []
            for word in sentence:
                if word in gensim_dict.keys:
                    tokens2index.append(word2index[word])
                else:
                    tokens2index.append(0) # words whose frequency <= min_count don;t have word vector, their index are assigned to 0
            sentences2index.append(tokens2index)
        # format index list to matrix
        sentences2index = np.asarray(sentences2index)   # convert to np.array type
        formatedSentences2index = sequence.pad_sequences(sentences2index, maxlen=self.vocab_dim)
        return word2vec, formatedSentences2index

    @classmethod
    def _format_lstm_model_training_data(cls, sentences2index, word2vec, label):
        """
        :param sentences2index: list of sentences with token indexes
        :param word2vec: word2vec dictionary
        :param polarity_dgr: labels
        :return:
        """
        # wordsCount = len(word2vec.keys()) + 1   # 1 is for index 0
        embedding_weights = np.asarray(pd.DataFrame.from_dict(word2vec, orient='index'))
        # dummy encoding of labels
        dummy_labels = np.asarray(pd.get_dummies(label))
        train_x, test_x, train_y, test_y = train_test_split(sentences2index, dummy_labels, test_size=0.2)
        return train_x, test_x, train_y, test_y, embedding_weights

    def _lstm_model_training(self, train_x, train_y, test_x, test_y, embedding_weights):
        input_dim = embedding_weights.shape[0] + 1    # word embedding. 1 for index 0
        model = Sequential()
        model.add(Embedding(input_dim=input_dim,
                            output_dim=self.vocab_dim,
                            input_length=self.maxlen,
                            weights=embedding_weights,
                            mask_zero=True)) # mask_zero is useful in recurrent network
        model.add(LSTM(output_dim=int(self.vocab_dim/2), activation='softmax'))
        model.add(Dropout(0.2))
        model.add(Dense(output_dim=test_x.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['categorical_accuracy'])
        model.fit(train_x, train_y, batch_size=self.batch_size, nb_epoch=self.nb_epoch,validation_data=(test_x, test_y))
        # evaluate the model
        score = model.evaluate(train_x, train_y, self.batch_size)
        # save model and model weight
        model_path = self.model_save_path + "/lstm_model.h5"
        save_model(model,model_path)
        """
        json_file = model_save_path + "/lstm_model.json"
        model_weight_file = model_save_path + "/lstm_model_weight.h5"
        json_string = model.to_json()
        with open(json_file, 'wb') as jf:
            jf.write(json_string)
        model.save_weights(model_weight_file)
        """
        return score

    def lstm_model_train(self,lstm_training_file_path, lstm_fields):
        """
        Training lstm model (training dataset is from file)
        :param fields: list of fields in the training file
        :return:
        """
        logging.info("Loading word2vec model ...")
        model = Word2Vec.load(self.model_save_path + "/word2vec.model")
        logging.info("Loading LSTM training Data ...")
        sentences, label = getDataFromFile(file_path=lstm_training_file_path, fields=lstm_fields)
        logging.info("Tokenizing ...")
        _, sentences2tokens, sub_sentences2tokens = self.tokenizer(sentences)
        logging.info("Convert document tokens to word index ...")
        word2vec, formatedSentences2index = self._document2index(sub_sentences2tokens, model)
        logging.info("Training LSTM model using word2vec as initial weights and token index matrix ...")
        train_x, test_x, train_y, test_y, embedding_weights = self._format_lstm_model_training_data(formatedSentences2index, word2vec, label)
        score = self._lstm_model_training(train_x, train_y, test_x, test_y, embedding_weights)
        logging.info("Evaluating model ... ")
        logging.info(score)

    def predict(self, sentences):
        """
        :param sentences: raw input comment sentences
        :param model_save_path: word2vec and lstm model file path
        :return: sentiment label
        """
        logging.info("Predicting ... ")
        sentences2tokens = self.tokenizer(sentences)
        logging.info("Loading trained word embedding model...")
        word2vec_model = Word2Vec.load(self.model_save_path + "/word2vec.model")
        _, formatedSentences2index = self._document2index(sentences2tokens, word2vec_model)
        logging.info("Loading trained lstm model ...")
        lstm_model = load_model(self.model_save_path + "/lstm_model.h5")
        predict_dummy_class = lstm_model.predict_classes(formatedSentences2index)
        # restore dummy labels to initial degree labels
        predict_class = np.asarray([list(x).index(1) +1 for x in predict_dummy_class])
        class_prob = lstm_model.predict_proba(formatedSentences2index)
        return predict_class, class_prob

def main():
    ## --- REPLACMENT ---- ##
    lstm_training_file_path = "."
    lstm_fields = []
    ## --- --------------- ##
    tbs = ['baidu_waimai_comments', 'dianping_comments', 'eleme_comments', 'meituan_comments',
           'meituan_waimai_comments', 'nuomi_comments', 'sina_comments']
    # test
    # tbs = ['baidu_waimai_comments']
    word2vec_fields = ['comment']
    model_save_path = "sentiment/model"
    predict_file_save_path = "sentiment/predict"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if not os.path.exists(predict_file_save_path):
        os.makedirs(predict_file_save_path)
    sentiObj = Sentiment(model_save_path=model_save_path)
    sentiObj.word2Vec_model_train(tbs, word2vec_fields)
    """
    sentiObj.lstm_model_train(lstm_training_file_path, lstm_fields)

    # output to csv file
    outfile = predict_file_save_path + "/predict_class.csv"
    for predict_sentences in sentiObj.getDataFromDB(tbs, word2vec_fields):
        predict_sentences = sentiObj.getDataFromDB(tbs, word2vec_fields)
        predict_class, class_prob = sentiObj.predict(predict_sentences)
        unicode_predict_class = sentiObj.polarity_conversion(predict_class)
        df_sentences = pd.DataFrame([",".join(sentence) for sentence in predict_sentences], columns=['sentence'])
        df_predict_class = pd.DataFrame(predict_class, columns=['class'])
        df_unicode_class = pd.DataFrame(unicode_predict_class, columns=['unicode_classes'])
        df_prob = pd.DataFrame(class_prob, columns=['prob'])
        df_out = pd.concat([df_sentences, df_predict_class, df_unicode_class, df_prob], axis=1)
        df_out.to_csv(outfile, header=False, mode='a')
    """
    
if __name__ == "__main__":
    # logging format
    fileConfig("../NN_model/logging_conf.ini")
    main()