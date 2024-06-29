import pandas as pd
import tensorflow as tf


testing=pd.read_csv('test.csv')
model=tf.keras.models.load_model('Titanic Model.keras')

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

testX=makeData(testing)

pred = model.predict(testX)

for i,row in testing.iterrows():
  if pred[i]>=0.5:
    print(row)
    print('Alive')
  else:
    print(row)
    print('Dead')
  print()


