# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import codecs
from Preprocessing import *

file = "/home/fanzong/Downloads/comments2tokens-labels.xlsx"
df = pd.read_excel(file)
initial_comments = np.asarray(df['initial_comment'])
labels = np.asarray(df['labels'])
format_comment = np.asarray(df['filtered_comment'])

