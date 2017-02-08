#!/bin/sh
# acitavte tensorflow environment first
source activate tensorflow
dir=/home/maxfun/maxfun_zong/yunxiu_python_data/Comment_NLP_dev
model_path=$dir/sentiment/model/phrase_lstm_model.h5
if [ ! -f $model_path ]; then
    python $dir/sentiment.py --model_override > $dir/sentiment_analysis.log
else
    python $dir/sentiment.py >> $dir/sentiment_analysis.log
fi
# deactivate tensorflow
source deactivate tensorflow
