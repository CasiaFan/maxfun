# -*- coding:utf-8 -*-

import os
import re
import logging
import codecs
import numpy as np
import pandas as pd
from preprocessing import file2tuple_list, get_df_from_db

class NameNormalization():
    # normalize meal/service/experience... name in comment to a standard one
    # If branch_store_tb is true, branch_store file will be ignored
    def __init__(self, localhost, username, password, dbname, enter_tb, enter_fields, branch_store_tb=None, branch_store_file=None):
        self.branch_store_tb = branch_store_tb
        if self.branch_store_tb:
            self.branch_store_file = None
        else:
            self.branch_store_file = branch_store_file
        self.localhost = localhost
        self.username = username
        self.password = password
        self.dbname = dbname
        self.tbname = enter_tb
        self.enter_fields = enter_fields
        self.enter2rule_dict = {}
        self.enter_dict = {}


    def _get_enterprise_id_dict(self):
        enter_df = next(get_df_from_db(self.localhost, self.username, self.password, self.dbname, self.tbname, self.enter_fields))
        self.enter_dict = {enter_df[self.enter_fields[0]][index]: enter_df[self.enter_fields[1]][index] for index in range(len(enter_df.index))}


    def _corefernce_branch_store_from_file(self, enter_dict):
        try:
            assert os.path.exists(self.branch_store_file)
        except:
            logging.error("Branch store file: %s file does not exist" % self.branch_store_file)
            exit(-1)
        rule_file_dir = os.path.dirname(self.branch_store_file)
        name2rule_tuple = file2tuple_list(self.branch_store_file)
        for id, name in enter_dict.items():
            for name_rule_tuple in name2rule_tuple:
                if re.search(name_rule_tuple[0], name):
                    self.enter2rule_dict[id] = rule_file_dir + "/" + name_rule_tuple[1]


    def _coreference_brach_store_from_db(self, enter_dict):
        enter_meal_df = next(get_df_from_db(self.localhost, self.username, self.password, self.dbname, self.branch_store_tb, fields='*'))
        for id, name in enter_dict.items():
            for index in range(len(enter_meal_df.index)):
                if re.search(enter_meal_df.iloc[index, 0], name):
                    self.enter2rule_dict[id] = enter_meal_df.iloc[index, 1]

    def _read_in_rule_from_file(self, rule_file):
        first_order_rule, second_order_rule, rule = [], [], []
        try:
            with codecs.open(rule_file, 'rb', encoding='utf-8') as f:
                for row in f:
                    if row:
                        if re.search(u'first', row.strip()):
                            continue
                        elif re.search(u'second', row.strip()):
                            first_order_rule = rule
                            rule = []
                        else:
                            rule.append(tuple(re.split(r"\s+", row.strip())))
                second_order_rule = rule
            f.close()
            return first_order_rule, second_order_rule
        except:
            logging.warning("Meal summarization rule file: %s not exist!" %rule_file)


    def _read_in_rule_from_db(self, rule_tb):
        first_order_rule, second_order_rule, rule = [], [], []
        try:
            df = np.asarray(next(get_df_from_db(self.localhost, self.username, self.password, self.dbname, rule_tb, fields='*')))
            rule_array = np.asarray(df).ravel()
            for row in rule_array:
                if row:
                    if re.search(u'first', row):
                        continue
                    elif re.search(u'second', row):
                        first_order_rule = rule
                        rule = []
                    else:
                        rule.append(tuple(re.split(ur"\s+", row)))
            second_order_rule = rule
            return first_order_rule, second_order_rule
        except:
            logging.warning("Meal summarization rule tb: %s not exist!" %rule_tb)

    def normalize(self, sentences, enterprise_id):
        # When normalizing, check first order rule first then check the second rule
        # return list of mentioned meals with specified enterprise
        logging.info("Normalize meal names for enterprise: %s ... " %str(enterprise_id))
        if not self.enter_dict:
            self._get_enterprise_id_dict()
        if not self.enter2rule_dict:
            if self.branch_store_file:
                self._corefernce_branch_store_from_file(self.enter_dict)
            elif self.branch_store_tb:
                self._coreference_brach_store_from_db(self.enter_dict)
            else:
                logging.error("No branch store reference specified!")
                exit(-1)
        if enterprise_id not in self.enter2rule_dict.keys():
            logging.info("Name normalization rule for enterprise_id %d doesn't exist!" %enterprise_id)
            return [[] for i in range(len(sentences))]
        else:
            rule_file = self.enter2rule_dict[enterprise_id]
            if self.branch_store_file:
                first_order_rule, second_order_rule = self._read_in_rule_from_file(rule_file)
            else:
                first_order_rule, second_order_rule = self._read_in_rule_from_db(rule_file)
            sentences2meals = [[] for i in range(len(sentences))]
            for index, sentence in enumerate(sentences):
                for phrase, meal in first_order_rule:
                    if re.search(phrase, sentence):
                        sentences2meals[index].append(meal)
            for index, sentence in enumerate(sentences):
                for phrase_match, phrase_exclude, meal in second_order_rule:
                    if not re.search(phrase_exclude, sentence) and re.search(phrase_match, sentence):
                        sentences2meals[index].append(meal)
            # remove redundant meals in a single sentence
            sentences2meals = [list(set(meals)) for meals in sentences2meals]
            return sentences2meals


class ThemeSummarization():
    # map keywords to hierarchical classes. If rule tbs are provided, rule file will be ignored
    def __init__(self, theme_header, keyword_header, experience_keyword_rule_tb=None, process_keyword_rule_tb=None, regexp_rule_tb=None, sentiment_rule_tb=None,
                  experience_keyword_rule_file=None, process_keyword_rule_file=None, regexp_rule_file=None, sentiment_rule_file=None,
                 localhost=None, username=None, password=None, dbname=None):
        self.experience_keyword_rule_tb = experience_keyword_rule_tb
        self.experience_keyword_rule_file = None if self.experience_keyword_rule_tb else experience_keyword_rule_file
        self.process_keyword_rule_tb = process_keyword_rule_tb
        self.process_keyword_rule_file = None if self.process_keyword_rule_tb else process_keyword_rule_file
        self.regexp_rule_tb = regexp_rule_tb
        self.regexp_rule_file = None if self.regexp_rule_tb else regexp_rule_file
        self.sentiment_rule_tb = sentiment_rule_tb
        self.sentiment_rule_file = None if self.sentiment_rule_tb else sentiment_rule_file
        self.theme_header = theme_header # from highest class to lowest class
        self.keyword_header = keyword_header
        if self.experience_keyword_rule_tb or self.process_keyword_rule_tb or self.regexp_rule_tb or sentiment_rule_tb:
            assert localhost != None
            self.localhost = localhost
            self.username = username
            self.password = password
            self.dbname = dbname
        self.process_main_keyword_dict, self.process_sub_keyword_dict, self.process_thd_keyword_dict = {}, {}, {}
        self.experience_main_keyword_dict, self.experience_sub_keyword_dict, self.experience_thd_keyword_dict = {}, {}, {}
        self.main_regexp_dict, self.sub_regexp_dict, self.thd_regexp_dict = {}, {}, {}
        self.main_sentiment_dict, self.sub_sentiment_dict, self.thd_sentiment_dict = {}, {}, {}
        self.initial_process_keywords_list = []
        self.initial_experience_keywords_list = []

    def _get_theme_reference_dict(self, mode='process_keyword'):
        # mode: keyword: use keywords to map;
        #       regexp: use regular expression to map;
        #       sentiment: use keywords + phrase sentiment to map
        try:
            assert mode in ['process_keyword', 'experience_keyword', 'regexp', 'sentiment']
        except:
            logging.error("Theme summarization mode should be within 'keyword', 'regexp' and 'sentiment'")
            exit(-1)
        theme_rule_ref = self.process_keyword_rule_tb if self.process_keyword_rule_tb else self.process_keyword_rule_file
        if mode == 'experience_keyword':
            theme_rule_ref = self.experience_keyword_rule_tb if self.experience_keyword_rule_tb else self.experience_keyword_rule_file
        elif mode == 'regexp':
            theme_rule_ref = self.regexp_rule_tb if self.regexp_rule_tb else self.regexp_rule_file
        elif mode == 'sentiment':
            theme_rule_ref = self.sentiment_rule_tb if self.sentiment_rule_tb else self.sentiment_rule_file
        else:
            pass
        try:
            assert len(self.theme_header) == 3
        except:
            logging.error("Theme header list must have 3 items")
            exit(-1)
        try:
            df = pd.read_csv(theme_rule_ref, encoding='utf-8')
        except:
            df = next(get_df_from_db(self.localhost, self.username, self.password, self.dbname, theme_rule_ref, fields="*"))
        df = df.fillna(method='ffill')
        key_col = np.asarray(df[self.keyword_header])
        main_theme_col = np.asarray(df[self.theme_header[0]])
        sub_theme_col = np.asarray(df[self.theme_header[1]])
        thd_theme_col = np.asarray(df[self.theme_header[2]])
        # sort keywords in order
        for index, keywords in enumerate(key_col):
            # replace chinese comma to english one
            keywords = re.sub(u"，", u",", keywords)
            # remove blanks
            keywords = re.sub(ur"\s+", u"", keywords)
            # remove last ,
            keywords = re.sub(ur",", u"", keywords)
            keyword_list = re.split(u',', keywords)
            for key in keyword_list:
                if mode == 'process_keyword':
                    self.initial_process_keywords_list.append(key)
                    self.process_main_keyword_dict[key] = main_theme_col[index]
                    self.process_sub_keyword_dict[key] = sub_theme_col[index]
                    self.process_thd_keyword_dict[key] = thd_theme_col[index]
                elif mode == 'experience_keyword':
                    self.initial_experience_keywords_list.append(key)
                    self.experience_main_keyword_dict[key] = main_theme_col[index]
                    self.experience_sub_keyword_dict[key] = sub_theme_col[index]
                    self.experience_thd_keyword_dict[key] = thd_theme_col[index]
                elif mode == 'regexp':
                    self.main_regexp_dict[key] = main_theme_col[index]
                    self.sub_regexp_dict[key] = sub_theme_col[index]
                    self.thd_regexp_dict[key] = thd_theme_col[index]
                else:
                    self.main_sentiment_dict[key] = main_theme_col[index]
                    self.sub_sentiment_dict[key] = sub_theme_col[index]
                    self.thd_sentiment_dict[key] = thd_theme_col[index]


    def summarize(self, phrases, phrases_sentiment):
        # mode: if all, keywords, regular expression and sentiment mode are executed.
        logging.info("Summarizing main, sub and third themes ...")
        phrases = np.asarray(phrases)
        phrases_sentiment = np.asarray(phrases_sentiment)
        if not self.process_main_keyword_dict:
            self._get_theme_reference_dict(mode='process_keyword')
        if not self.experience_main_keyword_dict:
            self._get_theme_reference_dict(mode='experience_keyword')
        if not self.main_regexp_dict:
            self._get_theme_reference_dict(mode='regexp')
        if not self.main_sentiment_dict:
            self._get_theme_reference_dict(mode='sentiment')
        main_theme = [[] for i in range(len(phrases))]
        sub_theme = [[] for i in range(len(phrases))]
        thd_theme = [[] for i in range(len(phrases))]
        for index, phrase in enumerate(phrases):
            # use keywords to map first
            for key in self.initial_process_keywords_list:
                if re.search(key, phrase):
                    main_theme[index].append(self.process_main_keyword_dict[key])
                    sub_theme[index].append(self.process_sub_keyword_dict[key])
                    thd_theme[index].append(self.process_thd_keyword_dict[key])
                    break
            for key in self.initial_experience_keywords_list:
                if re.search(key, phrase):
                    main_theme[index].append(self.experience_main_keyword_dict[key])
                    sub_theme[index].append(self.experience_sub_keyword_dict[key])
                    thd_theme[index].append(self.experience_thd_keyword_dict[key])
                    break
            # then use regular expression to map
            for key in self.thd_regexp_dict:
                if re.search(key, phrase):
                    main_theme[index].append(self.main_regexp_dict[key])
                    sub_theme[index].append(self.sub_regexp_dict[key])
                    thd_theme[index].append(self.thd_regexp_dict[key])
            # finally use sentiment rules to map
            sentiment_keywords = [re.sub(u"\w+", u"", x) for x in self.thd_sentiment_dict.keys()]
            for key in sentiment_keywords:
                if re.search(key, phrase):
                    senti_key = key + phrases_sentiment[index]
                    main_theme[index].append(self.main_sentiment_dict.get(senti_key, ""))
                    sub_theme[index].append(self.sub_sentiment_dict.get(senti_key, ""))
                    thd_theme[index].append(self.thd_sentiment_dict.get(senti_key, ""))
            # remove redundant themes in same phrase
            main_theme[index] = list(set(main_theme[index]))
            sub_theme[index] = list(set(sub_theme[index]))
            thd_theme[index] = list(set(thd_theme[index]))
        return np.asarray(main_theme), np.asarray(sub_theme), np.asarray(thd_theme)

