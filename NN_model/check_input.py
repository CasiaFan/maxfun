#!/usr/bin/env python2.7
__author__ = "Arkenstone"

from log_format import get_logger
import pandas as pd
import numpy as np
import datetime as dt

logger = get_logger(__name__)

class CheckInput():
    def __init__(self):
        pass

    def check_na(self, df, deal_na='rm', na_rp=0):
        """
        :param df: input df
        :param deal_na: the way to deal with na. 'rm' means remove; 'rp' means replace;
                        if option is 'rp', replacement value could be provided. Default: 0
        :return: df after filtering
        """
        # check if na exists
        if df.isnull().any().sum():
            if deal_na is 'rm':
                logger.warn("There are NAs in the data frame. Remove the line with NAs!")
                df = df.ix[df.isnull().any(axis=1), :]
            elif deal_na is 'rp':
                logger.warn("There are NAs in the data frame. Replace NAs with %s" %str(na_rp))
                df = df.fillna(na_rp)
        return df

    def check_columns(self, df, column_names):
        """
        :param df: input df
        :param column_names: column names (list) checked if exists
        :return: none
        """
        if len([1 for i in column_names if i in df.columns.ravel()]) == len(column_names):
            logger.debug("All checked columns %s exist." %str(column_names))
        else:
            not_exist_list = [i for i in column_names if i not in df.columns.ravel()]
            logger.error("Checked columns %s not exist." %str(not_exist_list))
        return

    def check_dimension(self, df1, df2, axis='row'):
        """
        Check if 2 data sets have same dimension along specified axis
        :param df1 (pd.DataFrame): first df
        :param df2 (pd.DataFrame): second df
        :param axis (str): axis to check. 'row or 'col' only
        :return: raise error if the dimension check not accord to each other
        """
        dim1 = -1
        dim2 = -1
        if axis is 'row':
            dim1 = df1.shape[0]
            dim2 = df2.shape[0]
        elif axis is 'col':
            dim1 = df1.shape[1]
            dim2 = df2.shape[1]
        else:
            logger.error("Axis argument could only be 'row' or 'col'! Check your setting!")
        if dim1 == dim2:
            logger.info("Check dimension done! Pass!")
        else:
            logger.error("Dimensions not accord! Check your data dimensions or if you choose the wrong dimension to check! Dim of df1 is %d, dim of df2 is %d", (dim1, dim2))
        return
