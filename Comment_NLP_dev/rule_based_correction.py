#! -*- encoding: utf-8 -*-
__author__ = "Arkenstone"

import preprocessing as pp
import pandas as pd
import numpy as np
import logging
import re

class SentimentCorrection():
    """
    Correct predicted sentiment label with end2end feedback and rules
    """
    def __init__(self, localhost, username, password, dbname, feedback_tb, rule_correction_tb=None):
        self.localhost = localhost
        self.username = username
        self.password = password
        self.dbname = dbname
        self.feedback_tb = feedback_tb
        self.rule_correction_tb = rule_correction_tb


    def _load_correction_rule_dict(self, tb, fields):
        df = next(pp.get_df_from_db(self.localhost, self.username, self.password, self.dbname, tb, fields))
        rule_dict = {df[fields[0]][i]:df[fields[1]][i] for i in range(len(df.index))}
        return rule_dict


    def _correct(self, sentences, labels, rule_dict):
        for i, sentence in enumerate(sentences):
            for rule in rule_dict:
                if re.search(rule, sentence) or re.search(sentence, rule):
                    labels[i] = rule_dict[rule]
                    break
        return labels


    def correct(self, sentences, labels):
        # both sentences and labels are np.array
        logging.info("Modify predicted sentiment based on end2end feedback ...")
        feedback_dict = self._load_correction_rule_dict(self.feedback_tb, fields=['comment', 'label'])
        labels = self._correct(sentences=sentences, labels=labels, rule_dict=feedback_dict)
        if self.rule_correction_tb:
            logging.info("Modify predicted sentiment based on manually specified rule ...")
            rule_dict = self._load_correction_rule_dict(self.rule_correction_tb, fields=['keyword', 'label'])
            labels = self._correct(sentences=sentences, labels=labels, rule_dict=rule_dict)
        return labels
