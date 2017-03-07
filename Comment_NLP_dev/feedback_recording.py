#! -*- coding: utf-8 -*-

import logging
import pymysql
import sqlalchemy
import ConfigParser
import os
import pandas as pd
import preprocessing as pp
from logging.config import fileConfig
from sqlalchemy import create_engine


class DatabaseFeedback():
    """
    Retrieve bedug error information of sentiment in feedback table then import to results table for rewriting and training set table for retraining
    """
    def __init__(self, localhost, username, password, dbname):
        self.localhost = localhost
        self.username = username
        self.password = password
        self.dbname = dbname


    def rewrite_feedback_results(self, feedback_df, feedback_fields, results_tb, results_fields):
        # connect database with results table
        try:
            assert len(feedback_fields) == 3 and len(results_fields) == 3
        except:
            logging.error("fields from feedback table and results tablemust consist of 3 items: id, comment and sentiment label!")
            exit(-1)
        db = pymysql.connect(self.localhost, self.username, self.password, self.dbname)
        cursor = db.cursor()
        for index in feedback_df.index:
            feedback_id = feedback_df[feedback_fields[0]][index]
            feedback_comment = feedback_df[feedback_fields[1]][index]
            feedback_sentiment = feedback_df[feedback_fields[2]][index]
            sql_cmd = "UPDATE " + results_tb + " SET " + results_fields[2] + " = " + feedback_sentiment + " WHERE " + results_fields[0] + " = " + str(feedback_id) + " AND " + results_fields[1] + " = " + feedback_comment
            try:
                cursor.execute(sql_cmd)
                # commit changes
                db.commit()
            except:
                logging.warn("Error encountered! Rollback to previous state")
                db.rollback()
        db.close()


    def save_feedback_to_training_set(self, feedback_df, training_set_tb, training_set_fields):
        dtype_dict = {header: sqlalchemy.types.NVARCHAR(1024) for header in training_set_fields}
        con = "mysql+pymysql://{}:{}@{}/{}?charset=utf8".format(self.username, self.password, self.localhost, self.dbname)
        engine = create_engine(con, encoding='utf-8')
        feedback_df.columns = training_set_fields
        feedback_df.to_sql(name=training_set_tb, con=engine, dtype=dtype_dict, if_exists='append', index=False)


    def feedback(self, feedback_tb, feedback_fields, results_tb, results_fields, training_set_tb, training_set_fields, mode='all', previous_feedback_count_file=None):
        """
        :param results_tb: sentiment prediction results table
        :param training_set_tb: training set table for training lstm classifier
        :param feedback_fields: feedback columns header in feedback table for rewriting. Its order should be id, comment, sentiment label
        :param results_fields: results columns header whose content corresponds to the feedback_fields
        :param training_set_fields: training set tb header
        :param mode: all or new. If all, all records in feedback table will be used; otherwise, only new records are used
        :param previous_feedback_count_file: file recording feedback count last time to filter out new records
        """
        if not os.path.exists(previous_feedback_count_file):
            mode = 'all'
        if mode == 'all':
            feedback_df = next(pp.get_df_from_db(self.localhost, self.username, self.password, self.dbname, feedback_tb, fields=feedback_fields))
            with open(previous_feedback_count_file, "wb") as fo:
                fo.write("{}".format(len(feedback_df.index)))
            fo.close()
        elif mode == 'new':
            with open(previous_feedback_count_file, 'rb') as fi:
                previous_count = int(fi.readline().strip())
            fi.close()
            feedback_df = next(pp.get_df_from_db(self.localhost, self.username, self.password, self.dbname, feedback_tb, feedback_fields)).iloc[previous_count:]
            with open(previous_feedback_count_file, 'wb') as fo:
                fo.write("{}".format(previous_count + len(feedback_df.index)))
            fo.close()
        logging.info("Rewrite wrong prediction in results repository database")
        self.rewrite_feedback_results(feedback_df, feedback_fields, results_tb, results_fields)
        logging.info("Save feedback to training set to improve classifier property")
        training_df = feedback_df[feedback_fields[1:]]
        self.save_feedback_to_training_set(training_df, training_set_tb, training_set_fields)

if __name__ == "__main__":
    config = ConfigParser.ConfigParser()
    config.read("sentiment_config.ini")
    fileConfig("logging_conf.ini")
    db_feedback = DatabaseFeedback(localhost=config.get("database", "localhost"),
                                   username=config.get("database", "username"),
                                   password=config.get("database", "password"),
                                   dbname=config.get("database", "dbname"))
    logging.info("Whole comment feedback ...")
    db_feedback.feedback(feedback_tb=config.get("feedback", "comment_feedback_tb"),
                         feedback_fields=eval(config.get("feedback", "comment_feedback_fields")),
                         results_tb=config.get("database", "comment_output_tbname"),
                         results_fields=eval(config.get("feedback", "results_fields")),
                         training_set_tb=config.get("database", "comment_train_tb"),
                         training_set_fields=["comment", "label"],
                         mode='new',
                         previous_feedback_count_file=config.get("model_save", "feedback_count_file"))
    logging.info("Phrase feedback ...")
    db_feedback.feedback(feedback_tb=config.get("feedback", "phrase_feedback_tb"),
                         feedback_fields=eval(config.get("feedback", "phrase_feedback_fields")),
                         results_tb=config.get("database", "phrase_output_tbname"),
                         results_fields=eval(config.get("feedback", "results_fields")),
                         training_set_tb=config.get("database", "comment_train_tb"),
                         training_set_fields=["comment", "label"],
                         mode='new',
                         previous_feedback_count_file=config.get("model_save", "feedback_count_file"))

