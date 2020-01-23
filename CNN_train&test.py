# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 17:37:35 2018

@author: Miranda
"""

import tensorflow as tf
from numpy.random import RandomState
import xlrd
import matplotlib.pyplot as plt
import numpy as np
import os
import xlsxwriter 


data_1 = xlrd.open_workbook(r'F:\大学\科研立项\18年3月\3-30\词向量_准备.xlsx')
# data_2 = xlrd.open_workbook(r'F:\大学\科研立项\18年3月\词向量10\Dataset2.xlsx')
table_1 = data_1.sheets()[0]
# table_2 = data_2.sheets()[0]
# Y = table.col_values(0)
X = []
Y = []
# for i in range(table_1.nrows):
for i in range(20000):
    xx = table_1.row_values(i+1)
    xx = xx[3:table_1.ncols]
    X.append(xx)
    yy = table_1.cell_value(i+1,2)
    Y.append([yy,1-yy])

dataset_size = 20000 #table_1.nrows


X_test = []
Y_test = []
for i in range(20000):
    xx = table_1.row_values(i+20001)
    xx = xx[3:table_1.ncols]
    X_test.append(xx)
    yy = table_1.cell_value(i+20001,2)
    Y_test.append([yy,1-yy])
   

'''
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,1000)
# Y = [[int(x1 < 2)] for x1 in X]
Y = [[int(x1<0.5),1-int(x1<0.5)]for x1 in X[:,1]]
'''
# ---------------------------------------------------------------------
x = tf.placeholder(tf.float32,shape = (None,1000), name ='x-input')
y_ = tf.placeholder(tf.float32,shape = (None,2), name = 'y-input')
W = tf.Variable(tf.zeros([1000, 2]))
keep_prob = tf.placeholder(tf.float32)


batch_size = 128

''' Define funstions '''
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W,m,n):
    return tf.nn.conv2d(x,W,strides=[1,m,n,1],padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')




''' Main function '''

x_1 = tf.reshape(x, [-1,10,100,1])
      
# Convolution Layer #1
W_conv1 = weight_variable([2,20,1,25])
b_conv1 = bias_variable([25])
h_conv1 = tf.nn.relu(conv2d(x_1,W_conv1,1,1)+b_conv1)
    
# Pooling Layer #1
h_pool1 = tf.nn.max_pool(h_conv1,ksize=[1,2,20,1],strides=[1,1,1,1],padding='VALID')
    
# Convolution Layer #2
W_conv2 = weight_variable([3,9,25,100])
b_conv2 = bias_variable([100])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2,3,9)+b_conv2)
    
# Pooling Layer #2
h_pool2 = tf.nn.max_pool(h_conv2,ksize=[1,3,3,1],strides=[1,3,3,1],padding='VALID')
    
    
# Dense Layer 
pool_shape = h_pool2.get_shape().as_list()
nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    
W_fc1 = weight_variable([nodes,200])
b_fc1 = bias_variable([200])
    
h_pool2_flat = tf.reshape(h_pool2,[-1,nodes])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
    
# Dropout regulation
# keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
    
# Logits Layer
W_fc2 = weight_variable([200,2])
b_fc2 = bias_variable([2])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

cross_entropy = -tf.reduce_mean(
    y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# ----------------------------------------------------------------------------

# Create a file to save word embedding.
# workbook = xlsxwriter.Workbook('Dataset1.xlsx')
# sheet1 = workbook.add_worksheet()





saver = tf.train.Saver()
# with tf.Graph().as_default() as g:
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    steps = 10000
    entropy = []
    accu = []
    for i in range(steps):
        start = (i*batch_size) % dataset_size
        end = min(start+batch_size,dataset_size) 
        sess.run(train_step,feed_dict = {x:X[start:end], y_ :Y[start:end],keep_prob:0.5})
        
        if i % 100 == 0:
            total_cross_entropy = sess.run(cross_entropy,feed_dict = {x:X, y_:Y,keep_prob:0.5})
            entropy.append(total_cross_entropy)
            accura = sess.run(accuracy,feed_dict = {x:X, y_ :Y,keep_prob:0.5})
            accu.append(accura)
            
    #save_path = saver.save(sess , "F:\mdoel\ex1mode2.ckpt")
    # result = sess.run(y_conv,feed_dict = {x:X_test,keep_prob : 1})
    sen = sess.run(h_fc1_drop,feed_dict = {x:X, y_ :Y,keep_prob:0.5})   
    


     
accux = range(0, 100)
plt.plot(accux,accu,label="accuracy",color='r') 
plt.xlabel('train step (unit:*100)') 
plt.legend() 
plt.show()  

plt.plot(accux,entropy,label="Entropy",color='r') 
plt.xlabel('train step (unit:*100)') 
plt.legend() 
plt.show()  

'''
workbook = xlsxwriter.Workbook('Dataset1.xlsx')
sheet1 = workbook.add_worksheet()

sen = np.array(sen)
for i in range(dataset_size):
    for j in range(200):
        sheet1.write(i,j,sen[i,j])
workbook.close()
'''
