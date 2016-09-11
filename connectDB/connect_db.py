#!/usr/bin/env python2.7
__author__ = "Arkenstone"

from NN_model.log_format import get_logger
import MySQLdb as msdb
import datetime as dt
import pandas as pd
import numpy as np
import random

logger = get_logger(__name__)

# class for connect to database
class extractDataFromDB:
    def __init__(self, localhost="120.24.87.197", username="root", password="78iU5478oT0hg", dbname="maxfun_qf", tbname="customer_behavior", enterprise_id="256"):
        self.localhost = localhost
        self.username = username
        self.password = password
        self.dbname = dbname
        self.tbname = tbname
        self.enterprise_id = enterprise_id

    def connect_db(self):
        # connect to the database
        db = msdb.connect(host=self.localhost, user=self.username, passwd=self.password, db=self.dbname)
        db_cursor = db.cursor()
        # return a db cursor
        return db_cursor

    def disconnect_db(self, cursor):
        cursor.close()

    def get_RFM_from_customer_behavior(self, db_cursor, selected, timemarker='last_purchase_time', lastday=dt.datetime.today().strftime('%Y-%m-%d')):
        # input: columns need to be retrieved, data before the lastday will be retrieved.
        # output: df with frequency, age, recency and other so on
        # retrieve data in selected column before the last date set
        # choose table
        tbname = self.tbname
        enterprise_id = self.enterprise_id
        # items need to be selected
        selected = selected
        # data before last day chose (item marker in the database should be known in previous)
        timemarker =timemarker
        lastday = lastday
        # set sql filtering command
        sql = "SELECT " + ", ".join(selected) + " FROM " + tbname + " WHERE enterprise_id = " + enterprise_id + " and " + timemarker + " < \'" + lastday + "\'"
        # set dictionary to hold data
        data ={}
        for item in selected:
            data[item] = []
        try:
            # retrieve data
            db_cursor.execute(sql)
            results = db_cursor.fetchall()
            for row in results:
                for j in range(len(selected)):
                    data[selected[j]].append(row[j])
        except:
            logger.error("Error: unable to fetch data from %s" %(tbname))
        # return a data frame containing the retrieved data
        df = pd.DataFrame()
        for i in selected:
            df[i] = data[i]
        return df

    def get_RFM_from_transaction(self, db_cursor, selected, lastday, timemarker='create_time'):
        # input: columns need to be retrieved, transactions until the lastday, column header of time series
        # output: df with recency, frequency, and age
        # retrieve data in selected column before the last date set
        # choose table
        tbname = self.tbname
        enterprise_id = self.enterprise_id
        # items need to be selected
        selected = selected
        # data before last day chose
        timemarker = timemarker
        lastday = lastday
        # set sql filtering command
        sql = "SELECT " + ", ".join(selected) + " FROM " + tbname + " WHERE enterprise_id = " + enterprise_id + " and " + timemarker + " < \'" + lastday + "\'"
        # set dictionaries to hold information
        dict_cus_frequency, dict_cus_age, dict_cus_recency, dict_cus_first_date, dict_cus_last_date = [{} for i in range(5)]
        # retrieve data from the database
        try:
            db_cursor.execute(sql)
            results = db_cursor.fetchall()
            for row in results:
                cus = row[0]
                date = row[1]
                # check if the customer id exists
                if cus in dict_cus_age:
                    # check the date is first or last
                    if (dict_cus_first_date[cus] - date).days > 0:
                        dict_cus_first_date[cus] = date
                    if (dict_cus_last_date[cus] - date).days < 0:
                        dict_cus_last_date[cus] = date
                    # recency = current date - last purchase date
                    recency = (dt.datetime.strptime(lastday, "%Y-%m-%d") - dict_cus_last_date[cus]).days
                    dict_cus_recency[cus] = recency
                    # age = current date - first purchase date
                    age = (dt.datetime.strptime(lastday, "%Y-%m-%d") - dict_cus_first_date[cus]).days
                    dict_cus_age[cus] = age
                    dict_cus_frequency[cus] += 1
                else:
                    dict_cus_first_date[cus] = date
                    dict_cus_last_date[cus] = date
                    age = (dt.datetime.strptime(lastday, "%Y-%m-%d") - date).days
                    dict_cus_age[cus] = age
                    dict_cus_recency[cus] = age
                    dict_cus_frequency[cus] = 1
        except:
            logger.error("Error: unable to fetch data from database %s" %(tbname))
        # convert the dic to data frame
        df_freq = pd.DataFrame.from_dict(dict_cus_frequency, orient='index')
        df_freq.columns = ['total_purchase_count_before_' + lastday]
        df_age = pd.DataFrame.from_dict(dict_cus_age, orient='index')
        df_age.columns = ['transaction_duration_until_' + lastday]
        df_recency = pd.DataFrame.from_dict(dict_cus_recency, orient='index')
        df_recency.columns = ['last_purchase_date_to_' + lastday]
        # merge to a single data frame along rows
        df = pd.concat([df_freq, df_recency, df_age], axis=1)
        return df

    def get_data_from_db(self, db_cursor, selected, filter=None):
        # input: columns title need to be retrieved;
        # output: columns retrieved
        ### NOTE: filter should be list format: ["create_time < '2016-06-02'", "enterprise_id = 256"] ####
        # and selected should be list like ['customer_id', 'create_time']
        # choose table
        tbname = self.tbname
        # choose items
        outID = selected
        selected = ', '.join(selected)
        # filter conditions if exist
        if filter:
            cond = ' and '.join(filter)
            # sql filtering command
            sql = "SELECT " + selected + " FROM " + tbname + " WHERE " + cond
        else:
            sql = "SELECT " + selected + " FROM " + tbname
        # initial a dictionary for holding the data
        my_data = {}
        try:
            # fetch all data selected
            db_cursor.execute(sql)
            results = db_cursor.fetchall()
            # exit the function if the return results tuple is empty
            if not results:
                logger.warn("No data retrieved! Please check if your sql command is correct!")
                return pd.DataFrame()
            count = 0
            for row in results:
                my_data[count] = row
                count += 1
        except:
            logger.error("Error: cannot fetch data from %s" %(tbname))
        # convert the data in dictionary ro data frame
        df = pd.DataFrame.from_dict(my_data, orient='index')
        df.columns = outID
        return df

    def get_data_by_sql_cmd(self, db_cursor, sql_cmd, selected_cols=None):
        """
        Retrieving data from database DIRECTLY by SQL command
        :param db_cursor (db,cursor): cursor in database returned by msdb.connect function
        :param sql_cmd (str): SQL command
        :param selected (list): columns names of returned df
        :return: df with selected columns
        """
        data = {}
        try:
            db_cursor.execute(sql_cmd)
            results = db_cursor.fetchall()
            if not results:
                logger.error("No data retrieved! Please check your SQL command: %s", sql_cmd)
            else:
                count = 0
                for row in results:
                    data[count] = row
                    count += 1
        except Exception, e:
            logger.error("Cannot fetch data from %s", self.tbname, exc_info=True)
        df = pd.DataFrame.from_dict(data, orient='index')
        if selected_cols:
            df.columns = selected_cols
        return df

