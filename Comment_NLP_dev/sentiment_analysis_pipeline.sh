#!/usr/bin/env bash
model_path=sentiment/model/phrase_lstm_model.h5
if [ ! -f $model_path ]; then
    nohup python sentiment.py --model_override > sentiment_pipeline_nohup.log
else
    nohup python sentiment.py >> sentiment_pipeline_nohup.log
fi