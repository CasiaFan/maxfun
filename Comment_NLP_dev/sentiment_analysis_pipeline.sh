#!/usr/bin/env bash
model_path=sentiment/model/phrase_lstm_model.h5
if [ ! -f $model_path ]; then
    python sentiment.py --model_override
else
    python sentiment.py
fi