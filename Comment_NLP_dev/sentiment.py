#!/usr/env/python2.7
# -*- coding:utf-8 -*-
__author__ = "Arkenstone"


import sys
reload(sys) # for sys.setdefaultencoding() is removed from sys when python starts
sys.setdefaultencoding('utf8')
import numpy as np
import jieba
import jieba.analyse
import jieba.posseg as pseg
import pandas as pd
import preprocessing as pp
import logging
import os
import codecs
import re
import sqlalchemy
import ConfigParser
import argparse
import pymysql
import datetime
import multiprocessing # for multicore machine
import time
from sqlalchemy import create_engine
from multiprocessing.pool import ThreadPool
from logging.config import fileConfig
from gensim import utils
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
from keras.models import Sequential, save_model, load_model
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from ThemeSummarization import NameNormalization, ThemeSummarization
from sklearn.model_selection import train_test_split


class SentimentScore():
    def __init__(self, sentiment_score_file=None, sentiment_score_tb=None, negation_word_file=None, negation_word_tb=None, localhost=None, username=None, password=None, dbname=None):
        self.localhost = localhost
        self.username = username
        self.password = password
        self.dbname = dbname
        self.sentiment_score_tb = sentiment_score_tb
        self.sentiment_score_file = None if self.sentiment_score_tb else sentiment_score_file
        self.negation_word_tb = negation_word_tb
        self.negation_word_file = None if self.negation_word_tb else negation_word_file
        # at least one of them should be specified
        assert negation_word_file or negation_word_tb
        self.word_sentiment_polarity_dict = {}
        self.word_sentiment_strength_dict = {}
        self.PUNCTUATIONS = [u',', u'.', u'!', u'?', u'~', u';', u'，', u'。', u'！', u'？', u'；']


    def _get_dict_from_sentiment_score_ref(self, key_header, value_header):
        if self.sentiment_score_tb:
            logging.info("Load sentiment polarity strength table ... Note header of this table should be token, pos, polarity, strength!")
            try:
                df = next(pp.get_df_from_db(self.localhost, self.username, self.password, self.dbname, self.sentiment_score_tb, fields='*'))
            except:
                logging.error("Sentiment polarity strength table not found! Check your table: %s" %self.negation_word_tb)
                exit(-1)
        elif self.sentiment_score_file:
            logging.info("Load sentiment polarity strength file ... Note header of this file should be token, pos, polarity, strength!")
            try:
                assert os.path.exists(self.sentiment_score_file)
            except:
                logging.error("Sentiment polarity strength file is not found! Check your path: %s!" %self.sentiment_score_file)
                exit(-1)
            df = pd.read_csv(self.sentiment_score_file, encoding='utf-8')
        else:
            logging.error("Sentiment polarity strength file or table must be specified!")
            exit(-1)
        key_col = np.asarray(df[key_header])
        value_col = np.asarray(df[value_header])
        score_dict = {}
        for index, word in enumerate(key_col):
            score_dict[word] = score_dict.get(word, value_col[index])
        return score_dict


    def _sentiment_score_dict(self):
        token_header, strength_header, polarity_header = ['token', 'strength', 'polarity']
        if not self.word_sentiment_polarity_dict:
            self.word_sentiment_strength_dict = self._get_dict_from_sentiment_score_ref(token_header, strength_header)
        if not self.word_sentiment_polarity_dict:
            self.word_sentiment_polarity_dict = self._get_dict_from_sentiment_score_ref(token_header, polarity_header)


    def score(self, sentences2tokens):
        """
        Strategy to deal with negation word: When meeting with negation words, inverse polarity of following words until punctuations or end of comment
        If whole_comment parameter is true, punctuations are included in the comment text; otherwise, only phrases are present.
        """
        logging.info("Loading chinese negation word list ...")
        if self.negation_word_file:
            try:
                assert os.path.exists(self.negation_word_file)
                negation_words = pp.file2list(self.negation_word_file)
            except:
                logging.error("Chinese negation file is not found. Check your file path: %s" %self.negation_word_file)
                exit(-1)
        elif self.negation_word_tb:
            try:
                negation_words = np.asarray(next(pp.get_df_from_db(self.localhost, self.username, self.password, self.dbname, self.negation_word_tb, fields='*'))).ravel()
            except:
                logging.error("Chinese negation table is not found. Check your table: %s" %self.negation_word_tb)
        else:
            logging.warning("Negation word file not found! Check your input!")
            negation_words = []
        logging.info("Loading word polarity and strength dictionaries ...")
        self._sentiment_score_dict()
        total_score = []
        for sentence2tokens in sentences2tokens:
            cur_score = 0
            negation_flag = False
            negation_count = 0
            for token in sentence2tokens:
                # if with pos or string
                if isinstance(token, jieba.posseg.pair):
                    token = token.word
                if not negation_flag:
                    cur_score += self.word_sentiment_polarity_dict.get(token, 0.0) * self.word_sentiment_strength_dict.get(token, 0.0)
                else:
                    cur_score -= self.word_sentiment_polarity_dict.get(token, 0.0) * self.word_sentiment_strength_dict.get(token, 0.0)
                if token in negation_words:
                    negation_count += 1
                    negation_flag = True if negation_count % 2 else False
                elif token in self.PUNCTUATIONS:
                    negation_flag = False
                    negation_count = 0
                else:
                    pass
            total_score.append(cur_score)
        return np.asarray(total_score)


class Sentiment():
    def __init__(self, config=None, **kwargs):
        ## If reference table is specified, reference file will be ignored
        ## If config and kwargs are specified, use arguments in kwargs as standard reference
        ## sentences input preprocessing
        self.localhost = kwargs.get("localhost") or config.get("database", "localhost")
        self.username = kwargs.get("username") or config.get("database", "username")
        self.password = kwargs.get("password") or config.get("database", "password")
        self.dbname = kwargs.get("dbname") or config.get("database", "dbname")
        self.enter_tb = kwargs.get("enter_tb") or config.get("database", "enter_tb")
        self.enter_fields = kwargs.get("enter_fields") or eval(config.get("database", "enter_fields"))
        ## params for sentence tokenizing
        # replace detailed time & price entity and cantonese
        self.entity_mark_tb = kwargs.get("entity_mark_tb") or config.get("database", "entity_mark_tb")
        if not self.entity_mark_tb:
            self.entity_mark_file = kwargs.get("entity_mark_file") or config.get("sentiment", "entity_mark_file")
        else:
            self.entity_mark_file = None
        # path to file defining html tags
        self.html_tag_file = kwargs.get("html_tag_file") or config.get("sentiment", "html_tag_file")
        # path to user defined tokenizing dictionary
        self.vocab_add_dict_tb = kwargs.get("vocab_add_dict_tb") or config.get("database", "vocab_add_dict_tb")
        if not self.vocab_add_dict_tb:
            self.vocab_add_dict_file = kwargs.get("vocab_add_dict_file") or config.get("sentiment", "vocab_add_dict_file")
        else:
            self.vocab_add_dict_file = None
        # path to user defined dictionary to be deleted
        self.vocab_del_dict_tb = kwargs.get("vocab_del_dict_tb") or config.get("database", "vocab_del_dict_tb")
        if not self.vocab_del_dict_tb:
            self.vocab_del_dict_file = kwargs.get("vocab_del_dict_file") or config.get("sentiment", "vocab_del_dict_file")
        else:
            self.vocab_del_dict_file = None
        # path to stop words file
        self.stop_words_tb = kwargs.get("stop_words_tb") or config.get("database", "stop_words_tb")
        if not self.stop_words_tb:
            self.stop_words_file = kwargs.get("stop_words_file") or config.get("sentiment", "stop_words_file")
        else:
            self.stop_words_file = None
        # screen tags base on pos. If None, all pos will be extracted, ohterwise list of pos should be provided
        self.pos_of_tags = kwargs.get("tag_use_pos") or eval(config.get("tokenizing", "pos_of_tag"))
        ## params for word embedding
        # define the dimensionality of feature
        self.vocab_dim = kwargs.get("vocab_dim") or eval(config.get("word2vec", "vocab_dim"))
        # min word frequency used for word embedding
        self.min_count = kwargs.get("min_count") or eval(config.get("word2vec", "min_count"))
        # maximum distance between current and predicted word within the sentence
        self.window = kwargs.get("window") or eval(config.get("word2vec", "window"))
        # multi-processes
        self.workers = multiprocessing.cpu_count()
        # number of interations for word embedding model training
        self.iter = kwargs.get("iter") or eval(config.get("word2vec", "iter"))
        ## params for lstm network
        # max number of tokens in a sentence used
        self.maxlen = kwargs.get("maxlen") or eval(config.get("lstm", "maxlen"))
        # dropout rate
        self.droupout = kwargs.get("dropout") or eval(config.get("lstm", "dropout"))
        # num of epoches for lstm training
        self.nb_epoch = kwargs.get("nb_epoch") or eval(config.get("lstm", "nb_epoch"))
        # batch size during lstm training
        self.batch_size= kwargs.get("batch_size") or eval(config.get("lstm", "batch_size"))
        # activation function for network layers
        self.activation = kwargs.get("activation") or config.get("lstm", "activation")
        # use generator or not to input data during model training (in case of memory overflow)
        self.use_generator = kwargs.get("use_generator") or eval(config.get("lstm", "use_generator"))
        # number of samples returned each time by generator
        self.generator_chunksize = kwargs.get("generator_chunksize") or eval(config.get("lstm", "generator_chunksize"))
        # dummy labeling header during lstm model training and prediction
        self.lstm_label_header = kwargs.get("lstm_label_header") or eval(config.get("lstm", "lstm_label_header"))
        ## params for model save path
        self.model_save_path = kwargs.get("model_save_path") or config.get("model_save", "model_save_path")
        self.word2vec_model_file = kwargs.get("word2vec_model_file") or config.get("model_save", "word2vec_phrase_model_file")
        self.word2vec_vocab_file = kwargs.get("word2vec_vocab_file") or config.get("model_save", "word2vec_vocab_file")
        self.lstm_model_file = kwargs.get("lstm_model_file") or config.get("model_save", "phrase_lstm_model_file")
        self.tf_file = kwargs.get("tf_file") or config.get("model_save", "tf_file")
        self.idf_file = kwargs.get("idf_file") or config.get("model_save", "phrase_idf_file")
        ## params for theme summarization and meal recognition
        self.branch_store_tb = kwargs.get("branch_store_tb") or config.get("database", "branch_store_tb")
        if not self.branch_store_tb:
            self.branch_store_file = kwargs.get("branch_store_file") or config.get("sentiment", "branch_store_file")
        else:
            self.branch_store_file = None
        self.sentiment_score_tb = kwargs.get("sentiment_score_tb") or config.get("database", "sentiment_score_tb")
        if not self.sentiment_score_tb:
            self.sentiment_score_file = kwargs.get("sentiment_score_file") or config.get("sentiment", "sentiment_score_file")
        else:
            self.sentiment_score_file = None
        self.negation_word_tb = kwargs.get("negation_word_tb") or config.get("database", "negation_word_tb")
        if not self.negation_word_tb:
            self.negation_word_file = kwargs.get("negation_word_file") or config.get("sentiment", "negation_word_file")
        else:
            self.negation_word_file = None


    def _load_dictionary(self):
        # load domain specific vocabulary
        if self.vocab_add_dict_file:
            try:
                jieba.load_userdict(self.vocab_add_dict_file)
            except ValueError(), e:
                logging.error("No such user defined dictionary file! Check your dictionary path: %s" %self.vocab_add_dict_file)
        elif self.vocab_add_dict_tb:
            try:
                words = np.asarray(next(pp.get_df_from_db(self.localhost, self.username, self.password, self.dbname, self.vocab_add_dict_tb, fields='*'))).ravel()
                for word in words:
                    jieba.add_word(word)
            except:
                logging.error("No such user defined add dictionary table! Check your dictionary table name: %s" %self.vocab_add_dict_tb)
        else:
            logging.warning("Cannot find user-defined add dictionary! Check your input!")

    def _delete_vocabulary(self):
        # delete unwanted tokens
        if self.vocab_del_dict_file:
            if os.path.exists(self.vocab_del_dict_file):
                with codecs.open(self.vocab_del_dict_file, "rb", encoding="utf-8") as f:
                    for line in f:
                        jieba.del_word(line.strip())
        elif self.vocab_del_dict_tb:
            try:
                words = np.asarray(next(pp.get_df_from_db(self.localhost, self.username, self.password, self.dbname, self.vocab_del_dict_tb, fields='*'))).ravel()
                for word in words:
                    jieba.del_word(word)
            except:
                logging.error("No such user-defined del dictionary table! Check your dictionary table name: %s" %self.vocab_del_dict_tb)
        else:
            logging.warning("Cannot find user-defined del dictionary! Check your input!")

    def tokenizer(self, sentences, remove_redundant_punc=True, replace_uncommon_punc=True, pos=()):
        """
        :return: list of formatted sentence and list of tokens of each sentence
        """
        logging.info("Loading user-defined dictionary ...")
        self._load_dictionary()
        # delete unwanted vocabulary
        logging.info("Delete undesired vocabulary ...")
        self._delete_vocabulary()
        logging.info("Parse html tags left during extracting data with web crawler ...")
        sentences = pp.parse_html_tag(sentences, self.html_tag_file)
        sentences = [re.sub(r"\n", "", sentence) for sentence in sentences]
        # convert traditional chinese to simplified chinese
        logging.info("Convert traditional chinese to simplified chinese...")
        sentences = pp.trad2simp(sentences)
        # replace uncommon punctuations with "。"
        if replace_uncommon_punc:
            logging.info("Replace uncommon punctuations ...")
            sentences = pp.replace_uncommon_punctuation(sentences)
        # remove redundant punctuations
        if remove_redundant_punc:
            logging.info("Remove redundant punctuations ...")
            sentences = pp.remove_redundant_punctuation(sentences)
        # replace detailed time or price phrases to symbol time or price; replace cantonese characters to mandarin ones
        logging.info("Mark entities about time, price and cantonese characters...")
        sentences = pp.mark_entity(sentences, self.entity_mark_file, self.localhost, self.username, self.password, self.dbname, self.entity_mark_tb)
        logging.info("Generating an generator for sentences tokenizing ...")
        # return tokens with pos or not
        if pos:
            logging.warning("When pos is true, return a list of pair of token and its pos")
            sentences2tokens = np.asarray([jieba.posseg.lcut(sentence) for sentence in sentences])
        else:
            sentences2tokens = np.asarray([jieba.lcut(sentence, cut_all=False) for sentence in sentences])
        return sentences, np.asarray(sentences2tokens)


    def tf_statistic(self, sentences2tokens, update=False):
        # statistic of term frequencies. if update is True, update existing tf dictionary with current input
        if update:
            tf_origin = pp.file2tuple_list(self.tf_file)
            tf_dict = {pp.strdecode(word): float(count) for word, count in tf_origin}
        else:
            tf_dict = {}
        for sentence2tokens in sentences2tokens:
            for word in sentence2tokens:
                tf_dict[word] = tf_dict.get(word, 0.0) + 1
        tf_dict_tuple = sorted(tf_dict.items(), key=lambda x: x[1], reverse=True)
        try:
            with codecs.open(self.tf_file, mode='wb', encoding='utf-8') as of:
                for key, value in tf_dict_tuple:
                    of.write(key + " " + str(value) + '\n')
            of.close()
        except:
            logging.error("TF file not found: %s" %(self.tf_file))
            exit(-1)


    def idf_statistic(self, sentences2tokens, override=True, update=False):
        # Statistic of reversed document frequency of all comments. If update is True, update idf file wtih current dataset.
        # NOW we need a specific dictionary key called CURRENT_IDF_DOCUMENT_COUNTS to store how may documents used for current idf statistic file
        if override:
            logging.warn("When override is set true, update parameter will not be ignored!")
        def _idf_statistic(sentences2tokens, idf_dict, doc_count):
            all_words = set([x for sentence2tokens in sentences2tokens for x in sentence2tokens])
            for word in all_words:
                for sentence in sentences2tokens:
                    if word in sentence:
                        idf_dict[word] = idf_dict.get(word, 0.0) + 1.0
            for key in idf_dict.keys():
                idf_dict[key] = np.log(doc_count/idf_dict[key])
            # sort dict in descending order
            idf_dict_tuple = sorted(idf_dict.items(), key=lambda x: x[1], reverse=True)    # dict is tuple list type after sorting
            try:
                with codecs.open(self.idf_file, mode="wb", encoding="utf-8") as of:
                    of.write("CURRENT_IDF_DOCUMENT_COUNTS" + " " + str(doc_count) + "\n")
                    for key, value in idf_dict_tuple:
                        if key == "CURRENT_IDF_DOCUMENT_COUNTS":
                            continue
                        of.write(key + " " + str(value) + "\n")
                of.close()
            except:
                logging.error("IDF file not found: %s" %(self.idf_file))
                exit(-1)

        if not os.path.exists(self.idf_file) or override:
            idf_dict = {}
            N = len(sentences2tokens)
            _idf_statistic(sentences2tokens, idf_dict, N)
        elif update:
            idf_origin = pp.file2tuple_list(self.idf_file)
            idf_dict = {pp.strdecode(word): float(count) for word, count in idf_origin}
            N = len(sentences2tokens) + int(idf_dict["CURRENT_IDF_DOCUMENT_COUNTS"])
            _idf_statistic(sentences2tokens, idf_dict, N)
        else:
            logging.info("IDF file already exists!")


    def keywords_extraction(self, sentences2tokens, topk=None, freq_thred=5, pos=(), with_flag=False, with_weight=True):
        """
        Extract top k keywords of comment
        :param k: number of keywords selected
        :param freq_thred: only token whose word frequency > freq_thred will be treated as keyword
        :param with_flag: return tags with pos flag. This parameter only works when pos parameter is not none.
        """
        logging.info("Loading tf, idf statistic results from file ...")
        try:
            assert os.path.exists(self.idf_file)
        except:
            logging.error("IDF file is not found! Check file path: %s" %self.idf_file)
            exit(-1)
        try:
            assert os.path.exists(self.tf_file)
        except:
            logging.error("TF file is not found! Check file path: %s" %self.tf_file)
            exit(-1)
        idf_ref = pp.file2tuple_list(self.idf_file)
        idf_dict = {word: float(value) for word, value in idf_ref}
        tf_ref = pp.file2tuple_list(self.tf_file)
        tf_dict = {word: float(value) for word, value in tf_ref}
        logging.info("Remove stop words after keywords extraction ...")
        # pos flag set initialization
        if pos:
            if 'a' in pos:
                pos += ['ad', 'ab', 'al']
            if 'n' in pos:
                pos += ['nr', 'ns', 'nt', 'nz', 'nl']
            if 'v' in pos:
                pos += ['vd', 'vn', 'vi', 'vl']
            pos = frozenset(pos)
        pool = ThreadPool(processes=self.workers)
        results = pool.apply_async(pp.remove_stop_words, (sentences2tokens, self.stop_words_file, self.stop_words_tb, self.localhost, self.username, self.password, self.dbname))
        sentences2tokens = results.get()
        keywords = []
        median_idf = sorted(idf_dict.values())[len(idf_dict) // 2]
        for sentence2tokens in sentences2tokens:
            # check if sentence is none after removing stop words
            if len(sentence2tokens) == 0:
                keywords.append([])
                continue
            freq = {}
            # compute token frequencies of this sentence
            for word in sentence2tokens:
                freq[word] = float(freq.get(word, 0.0)) + 1
            total = len(sentence2tokens)
            # compute weight
            for word in freq:
                k = word.word if isinstance(word, jieba.posseg.pair) else word
                freq[word] *= idf_dict.get(k, median_idf) / total
            tags = sorted(freq.items(), key=lambda x: x[1], reverse=True)
            # filter tags overpass threshold
            if freq_thred:
                if isinstance(sentence2tokens[0], jieba.posseg.pair):
                    sig_tags = [(word, value) for word, value in tags if tf_dict.get(word.word, 0) >= freq_thred]
                else:
                    sig_tags = [(word, value) for word, value in tags if tf_dict.get(word, 0) >= freq_thred]
            else:
                sig_tags = tags
            if pos and isinstance(sentence2tokens[0], jieba.posseg.pair):
                sig_tags = [(word, value) for word, value in sig_tags if word.flag in pos]
                if not with_flag:
                    sig_tags = [(word.word, value) for word, value in sig_tags]
            if not with_weight:
                sig_tags = [word for word, value in sig_tags]
            if topk:
                keywords.append(sig_tags[:topk])
            else:
                keywords.append(sig_tags)
        return np.asarray(keywords)


    def get_tokens_and_tags_df(self, sentences, override=False, update=True, topk=None, freq_thred=3):
        # override and update are parameters for tf-idf file generation; the rest are for keyword extraction
        logging.info("Start tokenizing ...")
        filter_sentences, sentences2tokens = self.tokenizer(sentences, pos=self.pos_of_tags)
        if self.pos_of_tags:
            logging.info("Words with specified pos will be used ...")
            sentences2tokens = np.asarray([[pair.word for pair in sentence] for sentence in sentences2tokens])
        df_formatted_sentences = pd.DataFrame(filter_sentences, columns=['formatted_comment'])
        df_tokens = pd.DataFrame({'tokens': list(sentences2tokens)}, index=df_formatted_sentences.index)
        logging.info("Update IDF file ...")
        pool1 = ThreadPool(processes=self.workers)
        pool1.apply_async(self.idf_statistic, (sentences2tokens, override, update))
        logging.info("Update TF file ...")
        pool2 = ThreadPool(processes=self.workers)
        pool2.apply_async(self.tf_statistic, (sentences2tokens, update))
        time.sleep(10)
        logging.info("Extracting keywords ...")
        total_keywords = self.keywords_extraction(sentences2tokens, topk=topk, freq_thred=freq_thred, pos=self.pos_of_tags)
        df_total_keywords = pd.DataFrame({'tags': list(total_keywords)}, index=df_formatted_sentences.index)
        logging.info("Scoring comment sentiment ...")
        sentiScoreObj = SentimentScore(sentiment_score_file=self.sentiment_score_file,
                                       sentiment_score_tb=self.sentiment_score_tb,
                                       negation_word_file=self.negation_word_file,
                                       negation_word_tb=self.negation_word_tb,
                                       localhost=self.localhost,
                                       username=self.username,
                                       password=self.password,
                                       dbname=self.dbname)
        sentiment_scores = sentiScoreObj.score(sentences2tokens)
        df_senti_score = pd.DataFrame({'sentiment_score':sentiment_scores}, index=df_formatted_sentences.index)
        df = pd.concat([df_formatted_sentences, df_tokens, df_total_keywords, df_senti_score], axis=1)
        return df


    def normalize_meals(self, sentences_df):
        # Normalize meal names with given rules. If rule file not specified, return with null; else append meals column to initial_df
        # check enterprise_id and comment within input dataframe column headers
        # return df append with meals column
        try:
            assert 'enterprise_id' in sentences_df.columns and 'comment' in sentences_df.columns
        except:
            logging.error("Input dataframe doesn't have enterprise_id and comment fields! Check your input df.")
            exit(-1)
        enterprises = list(set(sentences_df['enterprise_id']))
        meal_df = pd.DataFrame({'meals': [None] * len(sentences_df.index)}, index=sentences_df.index)
        mealObj = NameNormalization(branch_store_file=self.branch_store_file, branch_store_tb=self.branch_store_tb,
                                    localhost=self.localhost,username=self.username, password=self.password,
                                    dbname=self.dbname, enter_tb=self.enter_tb, enter_fields=self.enter_fields)
        for enterprise in enterprises:
            cur_enter_df = sentences_df['comment'][sentences_df['enterprise_id'] == enterprise]
            enterprise_comments = np.asarray(cur_enter_df)
            cur_meals = mealObj.normalize(enterprise_comments, enterprise)
            meal_df.ix[cur_enter_df.index, 0] = cur_meals
        sentences_df = pd.concat([sentences_df, meal_df], axis=1)
        return sentences_df


    def word2vec_model_train(self, sentences, update=False):
        """
        :param sentences: sentences iterable could be list or iterable streams directly from file or corpus, eg: LineSentence
        :model_save_path: model save path
        """
        # training word embedding model by import sentences
        # check if import is not empty
        if len(sentences) == 0:
            logging.error("No data imported! Cannot train word2vec model ...")
            exit(-1)
        else:
            _, sentences2tokens = self.tokenizer(sentences)
            if update:
                logging.info("Updating word2vec model ...")
                try:
                    assert os.path.exists(self.word2vec_model_file)
                except:
                    logging.error("Word2Vec model file doesn't exist. Train it first ... ")
                    exit(-1)
                model = Word2Vec.load(self.word2vec_model_file)
                vocab_dict = dict([(token, model.vocab[token].count) for token in model.vocab])
            else:
                model = Word2Vec(size=self.vocab_dim,
                             window=self.window,
                             min_count=self.min_count,
                             workers=self.workers,
                             iter=self.iter,
                             sorted_vocab=1)
                vocab_dict = {}
                model.build_vocab(sentences2tokens)
            sentences2tokens = [list(sentence2tokens) for sentence2tokens in sentences2tokens]
            model.train(sentences2tokens)  # train the model
            # save models
            model.save(self.word2vec_model_file)
            # save model vocabulary
            for token in model.vocab:
                vocab_dict[token] = vocab_dict.get(token, 0) + model.vocab[token].count
            sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
            with codecs.open(self.word2vec_vocab_file, "wb", encoding="utf8") as vf:
                for (token, count) in sorted_vocab:
                    vf.write(token + "\t" + str(count) + "\n")
            vf.close()
            logging.info("Number of vocabulary in the model: %d", len(vocab_dict))
            logging.info("Word2vec model training done!")


    def _document2index(self, sentences2tokens, model):
        """
        convert tokens of document to word index
        :param sentences: comment sentences in unicode format
        :param model: trained word embedding model
        :param polarity_dgr: comment polarity. It is provided during model training
        :return: training set and test set for lstm training
        """
        model_dict = Dictionary()
        # model.vocab contains tokens, token counts pairs during model training
        model_dict.doc2bow(model.vocab.keys(), allow_update=True)
        # transfer tokens in a sentence to word index in gensim vocabulary dictionary, for keras embedding use word index
        word2index = {value: key+1 for key, value in model_dict.items()}   # gensim_dict.items(): key is index from 0; value is word
        word2vec = {word: model[word] for word in word2index.keys()}   # word vector matrix as keras initial embedding weight
        sentences2index = []
        for sentence in sentences2tokens:
            tokens2index = []
            for word in sentence:
                if word in word2index.keys():
                    tokens2index.append(word2index[word])
                else:
                    # words whose frequency <= min_count don't have word vector, their index are assigned to 0
                    tokens2index.append(0)
            sentences2index.append(tokens2index)
        # format index list to matrix
        sentences2index = np.asarray(sentences2index)
        # truncate vector length with max length
        formatedSentences2index = sequence.pad_sequences(sentences2index, maxlen=self.vocab_dim)
        return word2index, word2vec, formatedSentences2index


    def _format_lstm_model_training_data(self, sentences2index, label, word2index, word2vec):
        """
        :param sentences2index: list of sentences with token indexes
        :param label: labels corresponding to each sentence
        :param word2vec: word2vec dictionary used for initial weight
        :param word2index: word index in model
        """
        # All words count whose frequencies are larger than 5. The extra +1 represents those words whose frequencies are less than 5
        word_count = len(word2vec) + 1
        embedding_weights = np.zeros((word_count, self.vocab_dim))
        for word, index in word2index.items():
            embedding_weights[index] = word2vec[word]
        # dummy encoding of labels
        dummy_labels = np.asarray(pd.get_dummies(label))
        train_x, test_x, train_y, test_y = train_test_split(sentences2index, dummy_labels, test_size=0.2)
        return train_x, test_x, train_y, test_y, embedding_weights


    def _train_test_data_generator(self, train_data, test_data):
        # return a generator with yielding batch of sample data each time where batch is defined in self.batch_size
        train_data = np.asarray(train_data)
        test_data = np.asarray(test_data)
        try:
            assert len(train_data) == len(test_data)
        except:
            logging.error("Train data should have same length with test data when yield generator!")
            exit(-1)
        while 1:
            for i in range(0, len(train_data), self.batch_size):
                yield (train_data[i:i+self.batch_size], test_data[i:i+self.batch_size])


    def _lstm_model_training(self, train_x, train_y, test_x, test_y, embedding_weights):
        input_dim = embedding_weights.shape[0]  # input dim is the number of vocabulary
        model = Sequential()
        model.add(Embedding(input_dim=input_dim,
                            output_dim=self.vocab_dim,
                            input_length=self.maxlen,
                            weights=[embedding_weights],
                            mask_zero=True)) # mask_zero is useful in recurrent network
        model.add(LSTM(output_dim=int(self.vocab_dim/2), activation=self.activation))
        model.add(Dropout(0.2))
        # model.add(Dense(output_dim=int(self.vocab_dim/2), activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(output_dim=train_y.shape[1], activation=self.activation))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['categorical_accuracy'])
        if not self.use_generator:
            model.fit(train_x, train_y, batch_size=self.batch_size, nb_epoch=self.nb_epoch,validation_data=(test_x, test_y))
            # evaluate the model
            score = model.evaluate(train_x, train_y, self.batch_size)
        else:
            train_generator = self._train_test_data_generator(train_x, train_y)
            test_generator = self._train_test_data_generator(test_x, test_y)
            model.fit_generator(generator=train_generator, samples_per_epoch=self.generator_chunksize, nb_epoch=self.nb_epoch, validation_data=test_generator, nb_val_samples=self.generator_chunksize)
            score = model.evaluate_generator(generator=train_generator, val_samples=self.generator_chunksize)
        # save model and model weight
        save_model(model, self.lstm_model_file)
        logging.info("Evaluating lstm model score... ")
        logging.info(score)


    def lstm_model_train(self, sentences, labels):
        # labels: list of training data set labels for supervised learning
        logging.info("Loading word2vec model ...")
        try:
            assert os.path.exists(self.word2vec_model_file)
        except:
            logging.error("Fail to load word2vec model during lstm model training! Check path %s" %self.word2vec_model_file)
        self.word2vec_model_train(sentences, update=True)
        model = Word2Vec.load(self.word2vec_model_file)
        logging.info("Tokenizing sentences during lstm model training...")
        _, sentences2tokens = self.tokenizer(sentences)
        logging.info("Convert document tokens to word index ...")
        word2index, word2vec, formatedSentences2index = self._document2index(sentences2tokens, model)
        logging.info("Training LSTM model for sentiment analysis using word2vec as initial weights and token index matrix ...")
        train_x, test_x, train_y, test_y, embedding_weights = self._format_lstm_model_training_data(formatedSentences2index, labels, word2index, word2vec)
        self._lstm_model_training(train_x, train_y, test_x, test_y, embedding_weights)
        logging.info("LSTM classifier for sentiment analysis done!")


    def _predict_data_generator(self, sentences, chunksize=2000):
        sentences = np.asarray(sentences)
        for i in range(0, len(sentences), chunksize):
            yield sentences[i:i+chunksize]


    def lstm_predict(self, sentences, chunksize=2000):
        logging.info("Predicting sentiment using lstm classifier... ")
        sentences = np.asarray(sentences)
        _, sentences2tokens = self.tokenizer(sentences)
        logging.info("Loading trained word2vec model...")
        word2vec_model = Word2Vec.load(self.word2vec_model_file)
        _, _, formatedSentences2index = self._document2index(sentences2tokens, word2vec_model)
        logging.info("Loading trained lstm model ...")
        lstm_model = load_model(self.lstm_model_file)
        logging.info("Predicting input sentence labels ...")
        predict_class = []
        class_prob = []
        if self.use_generator:
            for cur_formatedSentences2index in self._predict_data_generator(formatedSentences2index, chunksize=chunksize):
                cur_predict_dummy_class = lstm_model.predict_classes(cur_formatedSentences2index)
                # restore dummy labels to initial degree labels
                cur_predict_class = np.asarray([self.lstm_label_header[label] for label in cur_predict_dummy_class])
                cur_class_prob = lstm_model.predict_proba(cur_formatedSentences2index)
                predict_class += list(cur_predict_class)
                class_prob += list(cur_class_prob)
            predict_class = np.asarray(predict_class)
            class_prob = np.asarray(class_prob)
        else:
            predict_dummy_class = lstm_model.predict_classes((formatedSentences2index))
            predict_class = np.asarray([self.lstm_label_header[label] for label in predict_dummy_class])
            class_prob = lstm_model.predict_proba(formatedSentences2index)
        logging.info("Predicted label possibility assessment ...")
        return predict_class, class_prob


    @staticmethod
    def adjust_sentiment(sentences_df, rating_field, sentiment_field, pos_field, neg_field, prob_no_diff=0.2):
        # prob_no_diff: labels whose difference of positive prediction and negative prediction within given value will be reconsidered
        logging.info("Adjusting predicted sentiment using whole comment rating ... ")
        sentiment_array = np.asarray(sentences_df[sentiment_field]).copy()
        rating_array = np.asarray(sentences_df[rating_field]).copy().astype(float)
        # descriptions with rating == 4 is ambiguous to separate positive sentiment and negative sentiment
        # rating == 0 means rating data is blank for this sentence
        rating_label_array = np.asarray([None] * len(rating_array))
        for index, value in enumerate(rating_array):
            if value == 5:
                rating_label_array[index] = 'pos'
            elif value <= 3:
                rating_label_array[index] = 'neg'
            else:
                pass
        pos_prob_array = np.asarray(sentences_df[pos_field])
        neg_prob_array = np.asarray(sentences_df[neg_field])
        # filter sentences whose pos-neg prob discrepancy is less than 0.2 (0.4 ~ 0.6)
        diff_bool_array = np.abs(pos_prob_array - neg_prob_array) < prob_no_diff
        for index, value in enumerate(diff_bool_array):
            if value and rating_label_array[index]:
                sentiment_array[index] = rating_label_array[index]
        sentences_df[sentiment_field] = sentiment_array
        return sentences_df


    @staticmethod
    def adjust_rating(phrases_df, rating_field, sentiment_field):
        logging.info("Adjusting rating using whole comment sentiment and predicted sentiment ...")
        # use whole comment rating to rate separated phrases
        sentiment_array = np.asarray(phrases_df[sentiment_field])
        # 5 for pos and 1 for neg
        sentiment_score_array = np.asarray([5] * len(sentiment_array))
        for index, label in enumerate(sentiment_array):
            if unicode(sentiment_array[index]) == u'neg':
                sentiment_score_array[index] = 1
        rating_score_array = np.asarray(phrases_df[rating_field]).copy().astype(float)
        # replace rating == 0 with sentiment_score
        for index, value in enumerate(rating_score_array):
            if not value:
                rating_score_array[index] = sentiment_score_array[index]
        # statistic count of conflict between rating and predict sentiment: score difference >= 3
        diff_array = np.abs(rating_score_array - sentiment_score_array)
        conflict_count = (diff_array >= 3).sum()
        logging.warning("Number of conflicts between comment rating and predicted sentiment labels: {}/{}".format(conflict_count, len(diff_array)))
        # add comment rating and calculate their average score
        adjust_score_array = (sentiment_score_array / 2.0 + rating_score_array / 2.0).astype(int)
        phrases_df['adjusted_rating'] = adjust_score_array
        return phrases_df


    def _df2sql_dtype_conversion_dict(self, df):
        cols = df.columns
        dtype_dict = {}
        text_length = 1024
        for col in cols:
            if df[col].dtype == np.dtype('int'):
                dtype_dict[col] = sqlalchemy.types.INTEGER
            elif df[col].dtype == np.dtype('datetime64[ns]'):
                dtype_dict[col] = sqlalchemy.types.TIMESTAMP
            elif df[col].dtype == np.dtype('float'):
                dtype_dict[col] = sqlalchemy.types.Float(precision=3)
            elif df[col].dtype == np.dtype('object'):
                dtype_dict[col] = sqlalchemy.types.NVARCHAR(length=text_length)
            else:
                pass
        return dtype_dict


    def store_nlp_analysis_results_to_db(self, df, out_tb, mode='replace'):
        """
        :paam out_tb: table name for
        :param mode:
            fail: If table exists, do nothing.
            replace: If table exists, drop it, recreate it, and insert data.
            append: If table exists, insert data. Create if does not exist.
        """
        connect_string = "mysql+pymysql://{}:{}@{}/{}?charset=utf8".format(self.username, self.password, self.localhost, self.dbname)
        engine = create_engine(connect_string, encoding='utf-8')
        # engine = pymysql.connect(host=self.localhost, user=self.username, password=self.password, database=self.dbname, charset='utf8', use_unicode=True)
        dtype_dict = self._df2sql_dtype_conversion_dict(df)
        df.to_sql(name=out_tb, con=engine, if_exists=mode, flavor='mysql', index=False, dtype=dtype_dict)


def main_total_run(config, model_override=False, database_override=False, start_date=None, end_date=None):
    def initial_w2v_model_train(config, CommentSentiObj, PhraseSentiObj):
        fields = config.get('database', 'w2v_tb_fields')
        comment_initial_run = True
        phrase_initial_run = True
        for sentences_df in pp.get_df_from_db(localhost=config.get('database', 'localhost'),
                                        username=config.get('database', 'username'),
                                        password=config.get('database', 'password'),
                                        dbname=config.get('database', 'dbname'),
                                        tbname=eval(config.get('database', 'w2v_train_tbnames')),
                                        fields=fields,
                                        chunksize=eval(config.get('database', 'chunksize'))):
            # remove rows with null comments
            w2v_sentences = np.asarray(sentences_df[fields])
            w2v_sentences = [sentence for sentence in w2v_sentences if sentence]
            _, sentences2tokens = CommentSentiObj.tokenizer(w2v_sentences)
            if comment_initial_run:
                logging.info("Statistic tf information ...")
                CommentSentiObj.tf_statistic(sentences2tokens, update=False)
                logging.info("Statistic comment idf information ...")
                CommentSentiObj.idf_statistic(sentences2tokens, override=True, update=False)
                logging.info("Training whole comment w2c model ...")
                CommentSentiObj.word2vec_model_train(w2v_sentences, update=False)
                comment_initial_run = False
            else:
                logging.info("Update comment tf and idf information ...")
                CommentSentiObj.tf_statistic(sentences2tokens, update=True)
                CommentSentiObj.idf_statistic(sentences2tokens, override=False, update=True)
                logging.info("Training whole comment w2c model ...")
                CommentSentiObj.word2vec_model_train(w2v_sentences, update=True)
            logging.info("Split whole comments into subsentences ...")
            sub_sentences = pp.sentence_splitter(w2v_sentences)
            # ravel enclosed list
            sub_sentences = [x for sub_sentence in sub_sentences for x in sub_sentence]
            _, sub_sentences2tokens = PhraseSentiObj.tokenizer(sub_sentences)
            if phrase_initial_run:
                logging.info("Statistic phrase idf information ...")
                PhraseSentiObj.idf_statistic(sub_sentences2tokens, override=True, update=False)
                logging.info("Training phrase comment w2c model ...")
                PhraseSentiObj.word2vec_model_train(np.asarray(sub_sentences), update=False)
                phrase_initial_run = False
            else:
                logging.info("Update phrase idf information ...")
                PhraseSentiObj.idf_statistic(sub_sentences2tokens, override=False, update=True)
                logging.info("Training phrase comment w2c model ...")
                PhraseSentiObj.word2vec_model_train(np.asarray(sub_sentences), update=True)


    def run_sub_sentences_extraction(config, sentences_df):
        fields = eval(config.get("database", "fields"))
        comment_field = config.get("database", "comment_field")
        w2v_sentences = np.asarray(sentences_df[comment_field])
        # split sentences
        sub_sentences_df = np.array([])
        comment_field_index = fields.index(comment_field)
        for index, sentence in enumerate(w2v_sentences):
            sub_sentences = pp.sentence_splitter(sentence)
            current_row = np.asarray(sentences_df.iloc[index, :])
            for i in range(len(sub_sentences)):
                # replace initial sentence part into phrase
                current_row[comment_field_index] = sub_sentences[i]
                sub_sentences_df = np.concatenate((sub_sentences_df, current_row))
        sub_sentences_df = pd.DataFrame(
            sub_sentences_df.reshape((len(sub_sentences_df) / len(fields), len(fields))), columns=fields)
        # format data type of each columns
        for field in fields:
            try:
                sub_sentences_df[field] = sub_sentences_df[field].astype(sentences_df[field].dtype)
            except:
                sub_sentences_df[field] = sub_sentences_df[field].astype('object')
        return sub_sentences_df


    def run_predict_word2vec_update(config, sentiObj, sentences_df, topk):
        comment_field = config.get("database", "comment_field")
        if not os.path.exists(sentiObj.model_save_path):
            os.makedirs(sentiObj.model_save_path)
        # generate comment dataset for word2vec model training: only comments column required
        w2v_sentences = np.asarray(sentences_df[comment_field])
        # TF_IDF files are generated in the following function
        tokens_tags_df = sentiObj.get_tokens_and_tags_df(sentences=w2v_sentences, override=False, update=True, topk=topk,freq_thred=eval(config.get("tokenizing", "tag_min_tf")))
        tokens_tags_df.index = sentences_df.index
        # get normalized meal names
        total_df = sentiObj.normalize_meals(sentences_df)
        total_df.index = sentences_df.index
        # merge dfs
        total_df = pd.concat([total_df, tokens_tags_df], axis=1)
        sentiObj.word2vec_model_train(w2v_sentences, update=True)
        return total_df


    def run_lstm_train(sentiObj, lstm_training_file):
        lstm_fields = ['comment', 'label']
        sentences, labels = pp.readInTrainingSetFromFile(lstm_training_file, fields=lstm_fields)
        # too few neu samples. Remove them to improve model performance
        sentences = [sentences[index] for index in range(len(labels)) if labels[index] != 'neu']
        labels = [label for label in labels if label != 'neu']
        # normalize each group sample count
        # sentences, labels = pp.normalize_data_groups(sentences, labels)
        # remove polarity strength
        labels = pp.remove_label_polarity_strength(labels, prefix=sentiObj.lstm_label_header)
        sentiObj.lstm_model_train(sentences, labels)


    def _format_df_dtype(df):
        cols = df.columns
        for col in cols:
            if df[col].dtype == np.dtype('int'):
                df[col] = map(int, df[col])
            elif df[col].dtype == np.dtype('float'):
                df[col] = map(float, df[col])
            elif df[col].dtype == np.dtype('object'):
                # check if element is list or not
                ele = None
                for x in df[col]:
                    if len(x) > 0:
                        ele = x
                        break
                if isinstance(ele, list) or isinstance(ele, np.ndarray):
                    df[col] = map(lambda x: " ".join(x), df[col])
                elif not ele:
                    df[col] = map(lambda x: "", df[col])
                else:
                    pass
            else:
                pass
        return df


    def run_lstm_predict(config, sentiObj, sentences_df, tbname, mode='phrase',sql_mode='replace'):
        # mode: phrase or comment. If first, use phrases as input data; else use whole comments as input data
        try:
            assert os.path.exists(sentiObj.lstm_model_file)
        except:
            logging.info("LSTM classifier model not exit! Check file path: %s" %sentiObj.lstm_model_file)
        sentences = sentences_df['formatted_comment']
        # get prob of labeling neg and pos
        predict_chunksize = eval(config.get("lstm_predict", "predict_chunksize"))
        labels, labels_prob = sentiObj.lstm_predict(sentences, chunksize=predict_chunksize)
        sentences_df['label'] = labels
        label_prob_dict = {}
        n_label = len(labels_prob[0])
        for i in range(n_label):
            label_header = "prob_" + sentiObj.lstm_label_header[i]
            label_prob_dict[label_header] = np.asarray([probs[i] for probs in labels_prob])
            sentences_df[label_header] = label_prob_dict[label_header]
        rating_field = config.get("rating_score", "rating_field")
        sentiment_field = config.get("rating_score", "sentiment_field")
        pos_field = 'prob_pos'
        neg_field = 'prob_neg'
        prob_no_diff = eval(config.get("rating_score", "prob_no_diff"))
        # adjust sentiment label with comment rating
        sentences_df = sentiObj.adjust_sentiment(sentences_df, rating_field=rating_field, sentiment_field=sentiment_field, pos_field=pos_field, neg_field=neg_field, prob_no_diff=prob_no_diff)
        # adjust rating with whole comment rating and predicted sentiment score
        sentences_df = sentiObj.adjust_rating(phrases_df=sentences_df, rating_field=rating_field, sentiment_field=sentiment_field)
        # get themes
        themeObj = ThemeSummarization(process_keyword_rule_file=config.get("sentiment", "process_keyword_rule_file"),
                                      experience_keyword_rule_file=config.get("sentiment", "experience_keyword_rule_file"),
                                      regexp_rule_file=config.get("sentiment", "regexp_rule_file"),
                                      sentiment_rule_file=config.get("sentiment", "sentiment_rule_file"),
                                      process_keyword_rule_tb=config.get("database", "process_keyword_rule_tb"),
                                      experience_keyword_rule_tb=config.get("database", "experience_keyword_rule_tb"),
                                      regexp_rule_tb=config.get("database", "regexp_rule_tb"),
                                      sentiment_rule_tb=config.get("database", "sentiment_rule_tb"),
                                      theme_header=eval(config.get("sentiment", "theme_header")),
                                      keyword_header=config.get("sentiment", "keyword_header"),
                                      localhost=sentiObj.localhost,
                                      username=sentiObj.username,
                                      password=sentiObj.password,
                                      dbname=sentiObj.dbname)
        main_themes, sub_themes, thd_themes = themeObj.summarize(sentences, labels)
        if len(main_themes.ravel()) > 0:
            sentences_df['main_theme'] = main_themes
        else:
            sentences_df['main_theme'] = [""] * len(sentences)
        if len(sub_themes.ravel()) > 0:
            sentences_df['sub_theme'] = sub_themes
        else:
            sentences_df['sub_theme'] = [""] * len(sentences)
        if len(thd_themes.ravel()) > 0:
            sentences_df['thd_theme'] = thd_themes
        else:
            sentences_df['thd_theme'] = [""] * len(sentences)
        # output the df to db
        # format data frame for mysql_output. Complex data structure is not allowed.
        # NOTE: Convert str to datetime type should be ahead of formatting
        logging.info("Format result dataframe dtype to make it suitable for writing to mysql database ...")
        sentences_df['tag_weight'] = map(lambda x: " ".join([str(tag[1]) for tag in x]), sentences_df['tags'])
        sentences_df['tags'] = map(lambda x: " ".join([tag[0] for tag in x]), sentences_df['tags'])
        # convert datetime instance to TimeStamp
        sentences_df['comment_time'] = pd.to_datetime(sentences_df['comment_time'])
        # formatting
        sentences_df = _format_df_dtype(sentences_df)
        logging.info("Save nlp results to database table: %s ..." %(tbname))
        if mode == 'comment':
            sentences_df = sentences_df.drop(['main_theme', 'sub_theme', 'thd_theme'], axis=1)
        elif mode == 'phrase':
            pass
        else:
            logging.error("lstm predict model only accept 2 modes: phrase for phrases, comment for whole comment")
            exit(-1)
        sentiObj.store_nlp_analysis_results_to_db(sentences_df, tbname, mode=sql_mode)


    # make output directory if not exists
    outdir = config.get("model_save", "model_save_path")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    CommentSentiObj = Sentiment(config, word2vec_model_file=config.get("model_save", "word2vec_comment_model_file"),
                         lstm_model_file=config.get("model_save", "comment_lstm_model_file"),
                         idf_file=config.get("model_save", "comment_idf_file"))
    PhraseSentiObj = Sentiment(config)
    comment_lstm_tb_override = True
    phrase_lstm_tb_override = True
    if model_override or not os.path.exists(config.get("model_save", "word2vec_comment_model_file")):
        # train initial w2c model
        initial_w2v_model_train(config=config, CommentSentiObj=CommentSentiObj, PhraseSentiObj=PhraseSentiObj)
    # train lstm classifier for whole comment and phrase
    comment_lstm_training_file = config.get("model_train", "comment_label_file")
    phrase_lstm_training_file = config.get("model_train", "phrase_label_file")
    if model_override or not os.path.exists(CommentSentiObj.lstm_model_file):
        run_lstm_train(CommentSentiObj, comment_lstm_training_file)
    if model_override or not os.path.exists(PhraseSentiObj.lstm_model_file):
        run_lstm_train(PhraseSentiObj, phrase_lstm_training_file)
    # analysis comment imported every day
    if model_override:
        start_time_str = None
        end_time_str = None
        database_override = True
    if not database_override:
        if not start_date:
            start_time_str = datetime.datetime.today().strftime("%Y-%m-%d")
        else:
            start_time_str = start_date
        if not end_date:
            end_time_str = datetime.datetime.today().strftime("%Y-%m-%d")
        else:
            end_time_str = end_date
    else:
        start_time_str = None
        end_time_str = None
    for sentences_df in pp.get_df_from_db(localhost=config.get('database', 'localhost'),
                                  username=config.get('database', 'username'),
                                  password=config.get('database', 'password'),
                                  dbname=config.get('database', 'dbname'),
                                  tbname=config.get("database", "readin_tbname"),
                                  fields=eval(config.get("database", "fields")),
                                  chunksize=eval(config.get("database", "chunksize")),
                                  time_field=config.get("database", "comment_import_time_field"),
                                  start_time=start_time_str,
                                  end_time=end_time_str):
        # remove data without raw comments
        comment_field = config.get("database", "comment_field")
        sentences_df = sentences_df[(sentences_df[comment_field].notnull()) & (sentences_df[comment_field] != "")]
        comment_model_file = CommentSentiObj.lstm_model_file
        comment_topk = eval(config.get("tokenizing", "comment_topk"))
        # whole sentences training and prediction
        total_df = run_predict_word2vec_update(config, CommentSentiObj, sentences_df, comment_topk)
        if comment_lstm_tb_override and database_override:
            run_lstm_predict(config=config, sentiObj=CommentSentiObj, sentences_df=total_df, tbname=config.get("database", "comment_output_tbname"), mode='comment')
            comment_lstm_tb_override = False
        else:
            run_lstm_predict(config=config, sentiObj=CommentSentiObj, sentences_df=total_df, tbname=config.get("database", "comment_output_tbname"), mode='comment', sql_mode='append')
        # phrase training and prediction
        phrase_model_file = PhraseSentiObj.lstm_model_file
        phrase_topk = eval(config.get("tokenizing", "phrase_topk"))
        sub_sentences_df = run_sub_sentences_extraction(config, sentences_df)
        sub_total_df = run_predict_word2vec_update(config, PhraseSentiObj, sub_sentences_df, phrase_topk)
        if phrase_lstm_tb_override and database_override:
            run_lstm_predict(config, PhraseSentiObj, sub_total_df, tbname=config.get("database", "phrase_output_tbname"))
            phrase_lstm_tb_override = False
        else:
            run_lstm_predict(config, PhraseSentiObj, sub_total_df, tbname=config.get("database", "phrase_output_tbname"), sql_mode="append")



if __name__ == "__main__":
    # load arguments in configuration file
    config = ConfigParser.ConfigParser()
    config.read('sentiment_config.ini')
    # load log format
    fileConfig("logging_conf.ini")
    # get model_override variable from command line using argparse module
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_override", action='store_true')
    parser.add_argument("--database_override", action='store_true')
    parser.add_argument("--start_date", type=str, default=None)
    parser.add_argument("--end_date", type=str, default=None)
    args = parser.parse_args()
    main_total_run(config=config, model_override=args.model_override, database_override=args.database_override, start_date=args.start_date, end_date=args.end_date)
