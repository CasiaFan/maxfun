__author__ = "Arkenstone"

import os

import pandas as pd

from customer_classification.customer_behavior_functions import *
from connectDB.connect_db import *

# enterprise ids
enterprise_id_list = [1314, 807, 1472, 1398, 913, 1212, 866, 84, 86, 1917, 1918]
# customer counts selected
cus_number = 500
selection_type = 1
# output directory
out_dir = "C:/Users/fanzo/Desktop/selected_enterprise"
def retrieve_data_from_db(enterprise_list, cus_number=500, selection_type=1, output_dir=out_dir):
    # input: enterprise ids: eg:[1314, 807]; selection customer count: 500, selection_type: based on frequency: 1; based on recency: 2
    # output: customer df and transaction df

    # initial connect db
    currentDB = extractDataFromDB()
    # initial the db connector parameters
    currentDB.localhost = "112.74.30.59"
    currentDB.username = "fanzong"
    currentDB.password = "maxfun"
    currentDB.dbname = "maxfun_tp"
    # make the output direactory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for id in enterprise_id_list:
        ############### get data from customer behavioud table #####################
        currentDB.tbname = "customer_behavior"
        # selected items from customer behavior table
        selected_cb = ['customer_id', 'enterprise_id', 'total_purchase_amount', 'last_purchase_time', 'purchase_times', 'r_score', 'not_purchase_days']
        df_cb = currentDB.get_data_from_db(selected_cb, filter='enterprise_id = ' + str(id))
        # replace the r_score: 1-3: active, 4: potential_churn, 5: churn
        df_cb['r_score'] = df_cb['r_score'].replace([1, 2, 3, 4, 5], ['active', 'active', 'active', 'potential_churn', 'churn'])
        ############################################################################

        ################ get data from transaction table ############################
        currentDB.tbname = "transaction"
        selected_tr = ['customer_id', 'create_time', 'price']
        df_tr = currentDB.get_data_from_db(selected_tr, filter='enterprise_id = ' + str(id))
        # remove those transaction price is less than 0
        df_tr = df_tr[df_tr['price'] > 0]
        ##############################################################################

        # get the time interval between continuous 2 transactions made by a customer
        df_tr_interval, df_cus_ave_interval, df_tras_count = calculate_time_interval(df_tr, customer_average=True, transaction_count=True)
        # get transaction week date
        df_tr_interval['weekday'] = df_tr_interval[selected_tr[1]].apply(lambda x: x.weekday())
        # convert week integer to str: 0: Mon, 1: Tue, 2: Wed, 3: Thu, 4: Fri, 5: Sat, 6: Sun
        df_tr_interval['weekday'] = df_tr_interval['weekday'].replace(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        # reorder transaction df
        order_tr = ['customer_id', 'create_time', 'weekday', 'price', 'time_interval']
        df_tr_reorder = df_tr_interval[order_tr]

        # get all customers'average transaction interval
        all_intervals = [x for x in df_tr_reorder['time_interval'] if x != 0]
        all_cus_ave_interval = pd.Series(all_intervals).mean()

        # merge all-customer average transaction interval and personal average transaction interval
        # set index of df_cb to customer_id, then concatenation will be done by customer_id automatically,
        # for index of df_cus_ave_interval has been set to customer_id
        df_cb.index = df_cb['customer_id']
        ####### replace purchase count in customer_behavior with those from transaction ########
        df_cb['purchase_times'] = df_tras_count
        #######################################################################
        df_cb_result = pd.concat([df_cb, df_cus_ave_interval], axis=1)
        df_cb_result['enterprise_average_time_interval'] = all_cus_ave_interval
        # reorder the df_cb_results
        order_cb = ['customer_id', 'enterprise_id', 'total_purchase_amount', 'last_purchase_time', 'purchase_times', 'personal_average_time_interval', 'enterprise_average_time_interval', 'r_score', 'not_purchase_days']
        df_cb_result_reorder = df_cb_result[order_cb]

        # select cus_number (500) customers from all customers based on frequency/recency
        df_cb_ran = customer_selection(df_cb_result_reorder, cus_number, selection_type)
        # customer_id of selected customers
        cb_ran_id = df_cb_ran.index
        df_tr_reorder_ran = df_tr_reorder.ix[[i for i in df_tr_reorder.index if df_tr_reorder.at[i, 'customer_id'] in cb_ran_id],:]
        # sort output df by customer_id and create_time in ascendancy
        df_cb_ran = df_cb_ran.sort_values(['customer_id'], ascending=[True])
        df_tr_reorder_ran = df_tr_reorder_ran.sort_values(['customer_id', 'create_time'], ascending=[True, True])
        # write the 2 dfs to an excel file
        out_file = output_dir + "/test_sample_from_" + str(id) + ".xlsx"
        writer = pd.ExcelWriter(out_file)
        df_cb_ran.to_excel(writer, 'customer_info')
        df_tr_reorder_ran.to_excel(writer, 'transaction_info')
        writer.save()

retrieve_data_from_db(enterprise_id_list, cus_number, selection_type, out_dir)







