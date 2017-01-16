#!/usr/bin/env bash
# acitavte tensorflow environment first
source activate tensorflow
model_path=sentiment/model/phrase_lstm_model.h5
if [ ! -f $model_path ]; then
    python sentiment.py --model_override > sentiment_analysis.log
else
    python sentiment.py >> sentiment_analysis.log
fi
# deactivate tensorflow
source deactivate tensorflow
