import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

'''
rank2_tensor = tf.Variable([['test','toby'],['stanislaus','interview']])

print(rank2_tensor.shape)
'''

'''
tensor1 = tf.ones([1,2,3])
print(tensor1)
tensor2=tf.reshape(tensor1,[2,3,1])
print(tensor2)
tensor3=tf.reshape(tensor2,[3,1,2])
print(tensor3)

tensor4=tf.reshape(tensor3,[-1])
print(tensor4)
'''
  

'''
t1=tf.zeros([5,5])
print(t1)
t2=tf.reshape(t1,[-1])
print(t2)
'''
