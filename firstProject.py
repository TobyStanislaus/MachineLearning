from __future__ import absolute_import,division,print_function,unicode_literals

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import clear_output
#from six.moves import urllib

import tensorflow as tf
from tensorflow.keras import layers

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
#print(dftrain.head())

y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')


#print(dftrain.head())

CATEGORICAL_COLUMNS = ['sex','n_siblings_spouses','parch','class','deck',
                       'embark_town','alone']

NUMERIC_COLUMNS = ['age','fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocab = dftrain[feature_name].unique()

  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name,vocab))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name,dtype=tf.float32))


print(feature_columns)