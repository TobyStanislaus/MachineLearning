import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras
import keras


data=pd.read_csv('train.csv')
y=data.Survived

def makeData(data):
  data.Age = data.Age.astype(float)
  data.Fare = data.Fare.astype(float)
  data.fillna(0)

  
  data['Fare'] = data['Fare'].fillna(data['Fare'].median())
  data['Age'] = data['Age'].fillna(data['Age'].median())



  '''
  plt.figure(figsize=(10,10))
  ax = plt.subplot()
  ax.scatter(data[data['Survived'] == 1]['Age'],data[data['Survived'] == 1]['Fare'], c='green', s=data[data['Survived'] == 1]['Fare'] )

  ax.scatter(data[data['Survived'] == 0]['Age'],data[data['Survived'] == 0]['Fare'], c='red', s=data[data['Survived'] == 0]['Fare'] )

  plt.show()
  '''

  sex1={'male':1, 'female':2}
  data.Sex=data.Sex.map(sex1)

  data.Age = data.Age.astype(int)
  data.Fare = data.Fare.astype(int)


  features = ['Pclass','Sex','Age','Parch','Fare']

  X=data[features]


  X = ((X-X.min()) / (X.max() - X.min()))
  return X

X=makeData(data)

model = keras.Sequential([
  keras.layers.Dense(128,activation=tf.nn.relu, input_shape=(5,)),
  keras.layers.Dense(256,activation=tf.nn.relu),
  keras.layers.Dense(256,activation=tf.nn.relu),
  keras.layers.Dense(1,activation=tf.nn.sigmoid)
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X,y,epochs=30,batch_size=1)



model.save('Titanic Model.keras')