#!/usr/bin/env python2.7
__auther__ = "Arkenstone"

from connectDB.connect_db import extractDataFromDB
import pandas as pd
import numpy as np
import os

def get_conversion_rate(df):
    """
    Compute the conversion rate
    :param df: transaction count distribution with index is purchase count and the value is customer counts corresponding to the purchase count
    :return: conversion rate df
    """
    pur_count_list = np.array(df.index)
    cus_count_list = np.array(df)
    conversion_list = []
    for index in range(len(cus_count_list)-1):
        rate = np.sum(cus_count_list[index+1:], dtype=float) / np.sum(cus_count_list[index:], dtype=float)
        conversion_list.append(rate)
    df_conv = pd.DataFrame(conversion_list, index=pur_count_list[1:], columns=['conversion_rate'])
    return df_conv

def get_customer_counts_distribution(localhost="112.74.30.59",
                                        username="fanzong",
                                        password="maxfun",
                                        dbname="maxfun_tp",
                                        tbname="transaction",
                                        min_cus=5000,
                                        outdir = ".",
                                        k=10
                                        ):
    """
    Compute the knee point in customer conversion curse in first k transaction times
    :param min_cus (int): enterprise selected with minimum customers
    :param outdir: output dir for knee point file
    :param k: first k transaction times used for analyze the conversion rate change
    :return: files with conversion rate change and knee point
    """
    # make output dir if not exists
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    connectDB = extractDataFromDB()
    connectDB.localhost = localhost
    connectDB.username = username
    connectDB.password = password
    connectDB.dbname = dbname
    connectDB.tbname = tbname
    db_cursor = connectDB.connect_db()
    # get enterprise ids df
    sql_enter = "SELECT enterprise_id FROM %s GROUP BY enterprise_id HAVING COUNT(enterprise_id) >= %d" %(connectDB.tbname, min_cus)
    df_enter = connectDB.get_data_by_sql_cmd(db_cursor, sql_enter)
    # get distribution of all data
    for enter in df_enter.ix[:, 0]:
        sql_cus = "SELECT customer_id, COUNT(customer_id) AS counts FROM %s WHERE enterprise_id = %s GROUP BY customer_id" %(connectDB.tbname, str(enter))
        df_cus = connectDB.get_data_by_sql_cmd(db_cursor, sql_cus)
        # skip current enterprise if returned df is empty
        if not df_cus.empty:
            continue
        df_cus.columns = ['customer_id', 'counts']
        # get customer transaction count distributions
        df_group = df_cus.groupby(['counts'], sort=True).size()
        df_conv = get_conversion_rate(df_group)
        # calculate the 2nd derivatives using the first k element.
        acceleration = np.diff(np.array(df_conv.conversion_rate), 2)
        # get the minimum acceleration value as the knee point. Due to the 2nd derivative,
        # if the first one is the minimum, we will use the 2nd one as knee point
        knee_point_index = acceleration[:k].argmin() + 2
        # get the purchase count corresponding to the knee point
        knee_pur_count = df_conv.index[knee_point_index]
        df_out = pd.concat([df_conv, pd.DataFrame(knee_pur_count, index=['knee_point'], columns=df_conv.columns)], axis=0)
        # output to file
        out_file = outdir + "/" + str(enter) + ".conversion.knee.point.csv"
        df_out.to_csv(out_file)
    # disconnect to db
    connectDB.disconnect_db(db_cursor)

if __name__ == "__main__":
    get_customer_counts_distribution(min_cus=5000, outdir="C:/Users/fanzo/Desktop/conversion_rate", k=6)
