#!/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin:/home/maxfun/anaconda2/bin
# acitavte tensorflow environment first
source activate tensorflow
DATE=`date +%Y-%m-%d`
dir=/home/maxfun/maxfun_zong/yunxiu_python_data/Comment_NLP_dev
model_path=$dir/sentiment/model/phrase_lstm_model.h5
echo "Analysis of comment data in $DATE start!" >> $dir/sentiment_analysis.log
if [ ! -f $model_path ]; then
    python $dir/sentiment.py --config sentiment_config.ini --model_override --database_override > $dir/sentiment_analysis.log
else
    python $dir/sentiment.py --config sentiment_config.ini >> $dir/sentiment_analysis.log
fi
# rewrite feedback
python feedback_recording.py
# deactivate tensorflow
source deactivate tensorflow
echo "Analysis of comment data in $DATE done!" >> $dir/sentiment_analysis.log
