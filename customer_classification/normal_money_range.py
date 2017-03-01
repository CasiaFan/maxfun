# -*- coding:utf-8 -*-
import pandas as pd
from connectDB.connect_db import extractDataFromDB
import numpy as np
# from dateutil.parser import parse

enterprise_file = "/home/fanzong/Downloads/kedanjia.xlsx"
enter_df = pd.read_excel(enterprise_file, sheetname='Sheet2')
enter_df.time = enter_df.time % 100

connectDB = extractDataFromDB(localhost="112.74.30.59", username="fanzong", password="maxfun", dbname="maxfun_tp", tbname="transaction")
db_cursor = connectDB.connect_db()
ave_price = []
df_total = pd.DataFrame()
for j, id in enumerate(enter_df.id):
# for j, id in enumerate([6580]):
    """
    sql_cmd = "SELECT * FROM transaction WHERE enterprise_id=%d and MONTH(create_time)=%d" %(id, enter_df.ix[j, 'time'])
    df = connectDB.get_data_by_sql_cmd(db_cursor, sql_cmd)
    df_total = pd.concat([df_total, df], axis=0)
    """
    sql_cmd = "SELECT price FROM transaction WHERE enterprise_id=%d and MONTH(create_time)=%d" %(id, enter_df.ix[j, 'time'])
    df = connectDB.get_data_by_sql_cmd(db_cursor, sql_cmd, ['price'])
    sort_df = np.sort(np.asarray(df.price))
    sort_df = sort_df[sort_df > 1]
    q3_index = int(len(sort_df) * 0.75)
    q3 = sort_df[q3_index]
    # print q3
    derivatives = np.diff(sort_df, n=1)/(sort_df[:-1] + 1)
    # print derivatives
    der_fil = [index for index in range(len(derivatives)) if derivatives[index] > 4]
    found = False
    for i in der_fil:
        if sort_df[i+1] > q3:
            print "enterprise id: %s" % id
            print sort_df[i]
            found = True
            break
    if not found:
        i = len(sort_df) - 1
    ave_price.append(sum(sort_df[:i+1])/(i+1))
ave_price_df = pd.DataFrame(ave_price, index=enter_df.index)
enter_df['new_ave'] = ave_price_df
enter_df.to_excel("/home/fanzong/Downloads/average_price_fil.xlsx", encoding='utf-8')
connectDB.disconnect_db(db_cursor)
"""
df_total.to_excel("/home/fanzong/Downloads/transaction.xlsx", encoding='utf-8')
connectDB.disconnect_db(db_cursor)
"""
