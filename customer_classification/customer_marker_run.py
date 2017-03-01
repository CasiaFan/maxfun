#!/usr/bin/env python2.7
__author__ = 'Arkenstone'

from customer_behavior_functions import *
import pandas as pd
import numpy as np
from connectDB.connect_db import *
import os
import re

def customer_behavior_run(df, remove_not_work_days, high_line, low_line):
    # input df from transaction db/file, output result df
    # get time intervals, total frequency, total price
    df_intervals, df_tras_count, df_tras_amount = calculate_time_interval(df,
                                                                          transaction_count=True,
                                                                          transaction_amount=True,
                                                                          remove_not_work_days=remove_not_work_days)
    # merge personal transaction
    df_persoanl_transaction_list = merge_time_intervals(df_intervals)
    # get high and low line of total transactions of this enterprise
    # low_outline, high_outline = get_up_and_down_line_of_potential_churn(df_intervals, up=75, down=25, auto_adjust=True)
    low_outline, high_outline = low_line, high_line
    # get churn and potential churn cutoff
    df_churn = get_churn_cutoff(df_persoanl_transaction_list,
                                high_outline=high_outline,
                                low_outline=low_outline,
                                filter=True,
                                largest_k=5,
                                auto_adjust_churn_factor=True,
                                churn_cutoff=60)
    # get recency and age
    df_recency, df_age = get_recency_age(df_intervals, remove_not_work_days=remove_not_work_days)
    # merge customer information output
    df_total = pd.concat([df_persoanl_transaction_list, df_tras_count, df_tras_amount, df_recency, df_age, df_churn], axis=1)
    # mark customers
    df_total = set_customer_churn_mark(df_total)
    df_total['customer_id'] = df_total.index
    return df_total

def customer_marker(db,
                    enterprise_list,
                    int_dir,
                    out_dir,
                    prefix_filein,
                    prefix_result,
                    remove_not_work_days=True,
                    enterprise_low_line_list=None,
                    enterprise_high_line_list=None,):
    # If bd argument is true, df is from database and enterprise in enterprise_list will be used,
    # then argument in_dir will be overlook, otherwise from files in in_dir.
    # output result files to out_dir with file prefix. Time format is format to parse input time string from files
    # NOTE: db should be qa, tp or false.

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # get transaction data from db transaction
    selected = ['customer_id', 'enterprise_id', 'create_time', 'price']

    if db == "tp":
        # init a database connector class
        currentDB = extractDataFromDB()
        # init the parameters required for database connection
        ###--------------TP-------------------------------###
        currentDB.localhost = "your_localhost"
        currentDB.username = "username"
        currentDB.password = "password"
        currentDB.dbname = "maxfun_tp"
        currentDB.tbname = "transaction"
        db_cursor=currentDB.connect_db()
        for index, enterprise in enumerate(enterprise_list):
            # data from db
            df = currentDB.get_data_from_db(db_cursor, selected, filter='enterprise_id = ' + str(enterprise))
            # check if enterprise high line and low line is provided
            if enterprise_low_line_list:
                try:
                    if len(enterprise_low_line_list) == len(enterprise_list):
                        high_line = enterprise_high_line_list[index]
                        low_line = enterprise_low_line_list[index]
                except:
                    print "enterprise id outline list should be provided as same length as enterprise id list! " \
                          "please check the input of enterprise id and enterprise outline list"
            df_total = customer_behavior_run(df, remove_not_work_days, high_line=high_line, low_line=low_line)
            outfile = out_dir + str(prefix_result) + '-' +str(enterprise) + ".csv"
            df_total.to_csv(outfile)
    elif db == "qa":
        # init a database connector class
        currentDB = extractDataFromDB()
        ###------------------QA----------------------------###
        currentDB.localhost = "your_localhost"
        currentDB.username = "username"
        currentDB.password = "password"
        currentDB.dbname = "maxfun_qf"
        currentDB.tbname = "transaction"
        db_cursor = currentDB.connect_db()
        for index, enterprise in enumerate(enterprise_list):
            # data from db
            df = currentDB.get_data_from_db(db_cursor, selected, filter='enterprise_id = ' + str(enterprise))
            # check if enterprise high line and low line is provided
            if enterprise_low_line_list:
                try:
                    if len(enterprise_low_line_list) == len(enterprise_list):
                        high_line = enterprise_high_line_list[index]
                        low_line = enterprise_low_line_list[index]
                except:
                    print "enterprise id outline list should be provided as same length as enterprise id list! " \
                          "please check the input of enterprise id and enterprise outline list"
            df_total = customer_behavior_run(df, remove_not_work_days, high_line=high_line, low_line=low_line)
            outfile = out_dir + str(prefix_result) + '-' + str(enterprise) + ".csv"
            df_total.to_csv(outfile)
    elif db == False:
        # get input files start with transaction
        files = [file for file in os.listdir(int_dir) if re.match(prefix_filein + '[0-9]+', file)]
        for file in files:
            print file
            filename = os.path.join(int_dir, file)
            enterprise = re.findall(r'\d+', file)   # it returns a list
            # read in date type
            dateparser1 = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
            dateparser2 = lambda x: pd.datetime.strptime(x, '%m/%d/%Y %H:%M')
            try:
                df = pd.read_csv(filename, parse_dates=['create_time'], date_parser=dateparser1)
            except:
                df = df = pd.read_csv(filename, parse_dates=['create_time'], date_parser=dateparser2)
            df_total = customer_behavior_run(df, remove_not_work_days)
            outfile = out_dir + str(prefix_result) + '-' + ''.join(enterprise) + ".csv"
            df_total.to_csv(outfile)
    else:
        raise ValueError("Argument db must be qa, tp or False, please check your input!")

db = 'tp'
enterprise_list = [1212, 1917, 1918, 84, 86, 330, 866]
enterprise_low_line_list = [15, 30, 30, 15, 15, 30, 15]
enterprise_high_line_list = [30, 60, 60, 30, 30, 45, 30]
remove_not_work_days = True
int_dir = 'C:/Users/fanzo/Desktop/enterprise_predefined_outline/'
# output file for results
out_dir = int_dir
prefix_filein = 'transaction_'
prefix_result = 'test'
customer_marker(db=db,
                enterprise_list=enterprise_list,
                enterprise_low_line_list=enterprise_low_line_list,
                enterprise_high_line_list=enterprise_high_line_list,
                int_dir=int_dir,
                out_dir=out_dir,
                prefix_filein=prefix_filein,
                prefix_result=prefix_result,
                remove_not_work_days=remove_not_work_days)



