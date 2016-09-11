__author__ = 'Arkenstone'

import numpy as np
import pandas as pd
import MySQLdb as msdb
from CLV_BG_NBD_model import *
import datetime as dt
from connectDB.connect_db import *

def run_pareto_nbd_model(df, header, k, out_file):
    # variables required
    frequency = np.asarray(df[header[1]])
    age = np.asarray(df[header[3]])
    recency_r = age - np.asarray(df[header[2]])
    # initial the maxfun model object
    current_pareto_model = BGNBD()
    # training the model parameters using this frequency, age, recency data
    current_pareto_model.fit_BG_NBD_pars(frequency, recency_r, age)
    # calculate the possibility the customer is still alive now
    p_alive = current_pareto_model.p_alive_present(frequency, recency_r, age)
    # estimate the number of transaction win next k days
    tras_k = current_pareto_model.freq_future_k_days(frequency, recency_r, age, k)
    # format a data frame for output
    new_df = pd.DataFrame({'p_alive': p_alive, 'frequency_in_next_k_days': tras_k, 'k': pd.Series([k] * len(p_alive))})
    out_df_frame = [df, new_df]
    # concatenate origin and result data frame
    df_out = pd.concat(out_df_frame, axis=1)
    df_out.to_csv(out_file)

def real_time_run():
    # initial db connect class
    currentDB = extractDataFromDB()
    # database IP, user name, password, database selected
    currentDB.localhost = "120.24.87.197"
    currentDB.username = "root"
    currentDB.password = "78iU5478oT0hg"
    currentDB.dbname = "maxfun_qf"
    currentDB.tbname = "customer_behavior"
    currentDB.enterprise_id = "256"
    db_cursor = currentDB.connect_db()
    # factors should be retrieved: it should follow the order: customer, frequency, recency, age
    selected = ["customer_id", "real_purchase_times", "not_purchase_days",  "memership_days"]
    ######################## NOTE: date format must be %Y-%m-%d like 2016-06-16 ##################
    lastday = dt.datetime.today().strftime('%Y-%m-%d')
    timemarker = 'last_purchase_time'
    # predict transactions made during future k days
    k = 60
    # output file name
    out_file = "C:/Users/fanzo/Desktop/test-" + currentDB.enterprise_id + ".csv"

    # retrieve data from db table: customer behavior
    df = currentDB.get_RFM_from_customer_behavior(db_cursor, selected, timemarker, lastday)
    currentDB.disconnect_db()
    # run the model
    run_pareto_nbd_model(df, selected, k, out_file)

def test_run_with_solidation():
    # initial db connect class
    currentDB = extractDataFromDB()
    # database IP, user name, password, database selected
    currentDB.localhost = "120.24.87.197"
    currentDB.username = "root"
    currentDB.password = "78iU5478oT0hg"
    currentDB.dbname = "maxfun_qf"
    currentDB.tbname = "transaction"
    currentDB.enterprise_id = "256"
    db_cursor=currentDB.connect_db()
    # factors should be retrieved
    selected = ["customer_id", "create_time"]
    # predict transactions made during future k days
    k = 60
    # NOTE: date format must be %Y-%m-%d like 2016-06-16
    test_currentday = '2016-07-01'
    lastday = (dt.datetime.strptime(test_currentday, "%Y-%m-%d") - dt.timedelta(k)).strftime("%Y-%m-%d")
    timemarker = 'create_time'
    # output file name
    out_file = "C:/Users/fanzo/Desktop/test-bg-nbd-" + currentDB.enterprise_id + ".csv"

    # retrieve data from transaction db
    # get transcation count to test last day (k days previous to current day)
    df_lastday = currentDB.get_RFM_from_transaction(db_cursor, selected, lastday, timemarker)
    # add customer to df
    df_lastday['customer'] = df_lastday.index
    # remove customer whose frequency is 0
    # reorder into order: customer, frequency, recency, age
    order = ['customer', 'total_purchase_count_before_' + lastday, 'last_purchase_date_to_' + lastday, 'transaction_duration_until_' + lastday]
    df_lastday_reoder = df_lastday[order]
    # get transaction count to current day
    df_test_currentday = currentDB.get_RFM_from_transaction(selected, test_currentday, timemarker)
    currentDB.disconnect_db()
    # obtain the real transactions happen during the k days
    df_lastday_reoder['real_transactons_made_within_' + str(k)] = df_test_currentday['total_purchase_count_before_' + test_currentday] - df_lastday_reoder['total_purchase_count_before_' + lastday]
    # NOTE: remove customers whose both duration is less than 7, for they are difficult to predict future behaviour
    df_lastday_reoder_filter = df_lastday_reoder[df_lastday_reoder['transaction_duration_until_' + lastday] >= 7 ]
    # filter those with only 1 purchase, for they are too uncertain to make a prediction
    df_lastday_reoder_filter = df_lastday_reoder_filter[df_lastday_reoder_filter['total_purchase_count_before_' + lastday] > 1]
    # reindex row of data frame from 0 on
    df_lastday_reoder_filter.index = range(len(df_lastday_reoder_filter))
    # run pareto NBD model using df_lastday
    run_pareto_nbd_model(df_lastday_reoder_filter, order, k, out_file)

test_run_with_solidation()


