from __future__ import absolute_import,division,print_function,unicode_literals

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import clear_output
#from six.moves import urllib

import tensorflow as tf

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


def make_input_fn(data_df,label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df),label_df))

    if shuffle:
      ds=ds.shuffle(1000)
    ds = ds.batch(batch_size).repeat(num_epochs)
    return ds
  return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

