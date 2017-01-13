# -*- coding:utf-8 -*-
import logging
import os
import codecs
import re
import numpy as np
import pandas as pd
import jieba.posseg
from sqlalchemy import create_engine
import pymysql
from zhon.cedict import simp, trad


# chinese character unicode range
BASIC_LATIN_PUNC_UNICODE_RANGE = (u'\u0020', u'\u007f')
GENERAL_PUNC_UNICODE_RANGE = (u'\u2000', u'\u206f')
CJK_PUNC_UNICODE_RANGE = (u'\u3000', u'\u303f')
HALFWIDTH_AND_FULL_WIDTH_PUNC_UNICODE_RANGE = (u'\uff00', u'\uffef')
# average sentence length
AVERAGE_SENTENCE_LENGTH = 7
# punctuations
SENTENCE_DELIMITER = [u',', u'.', u'!', u'?', u'~', u';', u'，', u'。', u'！', u'？', u'；']

def strdecode(sentence):
    # decode str to utf-8 or gbk
    if not isinstance(sentence, unicode):
        try:
            sentence = sentence.decode('utf-8')
        except UnicodeDecodeError:
            sentence = sentence.decode('gbk', 'ignore')
    return sentence


def file2tuple_list(file, sep="\s+"):
    # parse file input to list of tuple: divide the line into 2 parts at sep position and the 1st part is key and 2nd part is value
    with codecs.open(file, mode="rb", encoding="utf8") as f:
        tuple_list = []
        for line in f:
            if line:
                key, value = re.split(sep, line.strip(), maxsplit=1)
                tuple_list.append((key, value))
    f.close()
    return tuple_list


def df2tuple_list(df, key_col=0, value_col=1, sep='\s+'):
    # get list of tuples from df with specified columns index or name  as key and value
    tuple_list = []
    try:
        keys = np.asarray(df.iloc[:, key_col])
    except:
        keys = np.asarray(df[key_col])
    try:
        values = np.asarray(df.iloc[:, value_col])
    except:
        values = np.asarray(df[value_col])
    for index in range(len(df.index)):
        tuple_list.append((keys[index], values[index]))
    return tuple_list


def file2list(file):
    delimiter = []
    with codecs.open(file, 'rb', encoding='utf8') as f:
        for line in f:
            delimiter.append(line.strip().replace("\n", ""))
    f.close()
    # convert to unicode
    return [x.decode("utf-8") for x in delimiter]


def readInTrainingSetFromFile(file_path, fields):
    """
    :param file_path: only accepted excel and csv format
    :param fields: field list of input file for analysis. <= 2 columns: 1st is the sentence; 2nd is the label (optional)
    :return: np.array of test strings and sentiment polarity degrees (2D)
    """
    assert os.path.exists(file_path)
    assert len(fields) == 2
    if file_path.endswith("xlsx") or file_path.endswith("xls"):
        df = pd.read_excel(file_path, encoding='utf-8')
        df = df[fields]
    elif file_path.endswith("csv"):
        df = pd.read_csv(file_path, usecols=fields, encoding='utf-8')
    else:
        ext = os.path.splitext(file_path)
        raise ValueError("only excel or csv files are accepted., But %s is found" % ext)
    sentences = np.asarray(df[fields[0]])
    labels = np.asarray(df[fields[1]])
    return sentences, labels


def get_df_from_db(localhost, username, password, dbname, tbname, fields, chunksize=None, time_field=None, start_time=None, end_time=None):
    """
    Read in comments data as dataframe from mysql database
    :param chunksize: If specified, return an iterator where chunksize is the number of rows to include in each chunk.
    :param time_field: Unless specified, start_time and end_time arguments will be ignored
    :param start time and end_time: define the time period of data to be used. Its format must be str like yyyy-mm-dd if given
    :return: df or df iterator
    """
    # con = pymysql.connect(host=localhost, user=username, password=password, database=dbname, charset='utf8', use_unicode=True)
    connect_string = "mysql+pymysql://{}:{}@{}/{}?charset=utf8".format(username, password, localhost, dbname)
    con = create_engine(connect_string, encoding='utf-8')
    time_cond = ""
    if time_field:
        if not end_time:
            time_cond = " WHERE " + time_field + " <= NOW()"
        else:
            time_cond = " WHERE " + time_field + " <= " + end_time
        if start_time:
            time_cond += " AND " + time_field + " >=" + start_time
    if isinstance(tbname, unicode):
        tbname = str(tbname)
    if isinstance(tbname, str):
        if isinstance(fields, unicode):
            fields = str(fields)
        if isinstance(fields, str):
            fields = [fields]
        sql_cmd = "SELECT " + ",".join(fields) + " FROM " + tbname + time_cond
        if chunksize:
            for cur_df in pd.read_sql(sql_cmd, con, chunksize=chunksize):
                yield cur_df
        else:
            cur_df = pd.read_sql(sql_cmd, con)
            yield cur_df
    elif isinstance(tbname, list):
        if isinstance(fields, unicode):
            fields = str(fields)
        if isinstance(fields, str):
            fields = [fields]
        for cur_tb in tbname:
            sql_cmd = "SELECT " + ",".join(fields) + " FROM " + cur_tb + time_cond
            if chunksize:
                for cur_df in pd.read_sql(sql_cmd, con, chunksize=chunksize):
                    yield cur_df
            else:
                cur_df = pd.read_sql(sql_cmd, con)
                yield cur_df
    else:
        logging.error("Argument tbname only accept a string or a list of string! But input type is %s" %(type(tbname)))
        exit(-1)


def trad2simp(sentences):
    # replace traditional chinese characters with simplified ones
    dict_zh = {trad[i]:simp[i] for i in range(len(trad))}
    simp_sentences = [u""] * len(sentences)
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            if sentences[i][j] in trad:
                simp_sentences[i] += dict_zh[sentences[i][j]]
            else:
                simp_sentences[i] += sentences[i][j]
    return np.asarray(simp_sentences)


def parse_html_tag(sentences, html_tag_file):
    """Parse html tages in sentences
    :param ref_path: html tag replacement reference file path. If exists, each items in line should be separated by commas.
    """
    tag_list = file2tuple_list(html_tag_file, ",")
    for key, value in tag_list:
        for index in range(len(sentences)):
            sentences[index] = re.sub(key, value, sentences[index])
    return sentences


def mark_entity(sentences, entity_mark_file=None, localhost=None, username=None, password=None, dbname=None, entity_mark_tb=None):
    # replace detailed prices/time with entity mark: price / time
    # if table is specified, ref file will be ignored
    logging.info("Entity mark step should be performed before tokenizing!")
    entity_list = None
    if entity_mark_file and not entity_mark_tb:
        logging.info("Use entity mark rule from file: %s" %entity_mark_file)
        entity_list = file2tuple_list(entity_mark_file)
    elif entity_mark_tb and localhost and username and password and dbname:
        logging.info("Use entity mark rule from table: %s" %entity_mark_tb)
        mark_rule_df = next(get_df_from_db(localhost, username, password, dbname, entity_mark_tb, fields='*'))
        entity_list = df2tuple_list(mark_rule_df)
    else:
        logging.warning("No entity mark rule specified ...")
    marked_sentences = []
    if entity_list:
        for sentence in sentences:
            for entity, replacement in entity_list:
                sentence = re.sub(ur"%s" % (entity.decode('utf-8')), replacement.decode("utf-8"),
                                  sentence)  # value is string type
            marked_sentences.append(sentence)
    else:
        marked_sentences = sentences
    return marked_sentences


def remove_redundant_punctuation(sentences):
    # format chinese input: remove redundant delimiters and other unknown punctuations
    # remove redundant punctuation
    # check if input type is str or unicode
    if isinstance(sentences, str) or isinstance(sentences, unicode):
        sentences = [sentences]
    rm_redun_sentences = [re.sub(ur"([%s-%s%s-%s%s-%s%s-%s])+"
                                 % (GENERAL_PUNC_UNICODE_RANGE[0], GENERAL_PUNC_UNICODE_RANGE[1],
                                    CJK_PUNC_UNICODE_RANGE[0], CJK_PUNC_UNICODE_RANGE[1],
                                    BASIC_LATIN_PUNC_UNICODE_RANGE[0], BASIC_LATIN_PUNC_UNICODE_RANGE[1],
                                    HALFWIDTH_AND_FULL_WIDTH_PUNC_UNICODE_RANGE[0],
                                    HALFWIDTH_AND_FULL_WIDTH_PUNC_UNICODE_RANGE[1]),
                                 r"\1", strdecode(sentence))
                          for sentence in sentences]
    # rm_redun_sentences = [re.sub(ur"([%s])+" %punctuation, r"\1", self.strdecode(sentence)) for sentence in sentences]
    return np.asarray(rm_redun_sentences)


def replace_uncommon_punctuation(sentences):
    # replace uncommon punctaions with '。'
    if isinstance(sentences, str) or isinstance(sentences, unicode):
        sentences = [sentences]
    sentences = [re.sub(ur"[^，！？：、\"',.!?:\d\w\u4e00-\u9fff]", u"。", sentence) for sentence in sentences]
    sentences = [re.sub(ur" ", u"。", sentence) for sentence in sentences]
    return sentences


def _remove_null_ele(list):
    # remove null element in list (after sub-sentence splitting)
    return [x for x in list if x]


def remove_stop_words(sentences2tokens, stopwords_file=None, stopwords_tb=None, localhost=None, username=None, password=None, dbname=None):
    stopwords = []
    if stopwords_file and not stopwords_tb:
        try:
            assert os.path.exists(stopwords_file)
        except:
            logging.error("Cannot open file: %s" % (stopwords_file))
            exit(-1)
        with codecs.open(stopwords_file, mode="rb", encoding="utf-8") as f:
            for line in f:
                stopwords.append(line.strip())
        f.close()
    elif stopwords_tb and localhost and username and password and dbname:
        stopwords = np.asarray(next(get_df_from_db(localhost, username, password, dbname, stopwords_tb, fields="*"))).ravel()
    filtered_sentences = []
    if isinstance(sentences2tokens[0][0], unicode):
        for sentence in sentences2tokens:
            filtered_sentences.append([word for word in sentence if word not in stopwords])
    elif isinstance(sentences2tokens[0][0], tuple):
        for sentence in sentences2tokens:
            filtered_sentences.append([(word, weight) for word, weight in sentence if word not in stopwords])
    elif isinstance(sentences2tokens[0][0], jieba.posseg.pair):
        for sentence in sentences2tokens:
            filtered_sentences.append([pair for pair in sentence if pair.word not in stopwords])
    else:
        logging.warning("Unwanted data structure! eg: %s" %str(sentences2tokens[0][0]))
    return np.asarray(filtered_sentences)


def sentence_splitter(sentences):
    """
    Format the chinese sentences - remove unwanted punctuations. Split sentences by each delimiter
    :return: If input a list or ndarray, return 2d ndarray; If input is a string or unicode, return 1d ndarray
    """
    # only accept list/ndarray or string/unicode type
    try:
        assert isinstance(sentences, list) or isinstance(sentences, np.ndarray) or isinstance(sentences, str) or isinstance(sentences, unicode)
    except:
        logging.error("Split sentences failed! Only list/ndarray or string/unicode type are allowed. Check your input: %s" %type(sentences))
        exit(-1)
    str_mark = False
    if isinstance(sentences, str) or isinstance(sentences, unicode):
        sentences = [sentences]
        str_mark = True
    delimiter = SENTENCE_DELIMITER
    sentences = replace_uncommon_punctuation(sentences)
    sentences = remove_redundant_punctuation(sentences)
    # split sentence
    sub_sentences = []
    for sentence in sentences:
        sub_sentence = re.split(r"[" + "".join(delimiter) + "]+", sentence)
        sub_sentence = _remove_null_ele(sub_sentence)
        sub_sentences.append(sub_sentence)
    if str_mark:
        sub_sentences = sub_sentences[0]
    return np.asarray(sub_sentences)


def average_sentence_length(sentences):
    # statistic average sentence length
    average_sentence_len = sum([len(sentence) for sentence in sentences]) / len(sentences)
    logging.info("Average sentence lenghth of input sentences is %s" %str(average_sentence_len))
    # statistic average sentence unique tokens count
    average_character_count = sum([len(set(sentence)) for sentence in sentences]) / len(sentences)
    logging.info("Average sentence character amount is %s" %str(average_character_count))


def remove_label_polarity_strength(labels, prefix):
    # converge multiple labels with common prefix to 3 kinds of labels
    # label format should be pos/neu/neg+\d
    # prefix is a list of neg, neu, pos labels
    triple_labels = [re.sub(r'(%s)\d+' %("|".join(prefix)), '\\1', label) for label in labels]
    return np.asarray(triple_labels)


def remove_tokenizing_dict_redundancy(dict_file):
    with codecs.open(dict_file, 'rb', encoding='utf-8') as f:
        word_dict = {}
        for word in f:
            word_dict[word.strip()] = word_dict.get(word.strip(), 0) + 1
    f.close()
    with codecs.open(dict_file, 'wb', encoding='utf-8') as of:
        for uniq_word in word_dict.keys():
            of.write(uniq_word)
            of.write('\n')
    of.close()

def normalize_data_groups(sentences, labels):
    # count the number of each labels
    sentences = np.asarray(sentences)
    labels = np.asarray(labels)
    unique_labels = list(set(labels))
    label_count_dict, label_sentence_dict = {}, {}
    for label in unique_labels:
        label_count_dict[label] = sum(labels == label)
    for index, label in enumerate(labels):
        label_sentence_dict[label] = label_sentence_dict.get(label, [])
        label_sentence_dict[label].append(sentences[index])
    # use the count of largest labeling group as reference to expand other groups
    max_count = max([x[1] for x in label_count_dict.items()])
    addition_count = {label: max_count+1 - count for label, count in label_count_dict.items()}
    normalized_sentences_dict = {label: label_sentence_dict[label]+list(np.asarray(label_sentence_dict[label])[[np.random.randint(label_count_dict[label]) for i in range(addition_count[label])]])
                           for label in unique_labels}
    normalized_labels = np.asarray([label for label in unique_labels for i in range(max_count+1)])
    normalized_sentences = np.asarray([sentence for label in unique_labels for sentence in normalized_sentences_dict[label]])
    return normalized_sentences, normalized_labels