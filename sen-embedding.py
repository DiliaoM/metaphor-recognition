# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 15:05:44 2018

@author: Miranda
"""






import tensorflow as tf
from numpy.random import RandomState
import xlrd
import matplotlib.pyplot as plt
import numpy as np
import os
import xlsxwriter 
from tensorflow.contrib import rnn



# 重置
tf.reset_default_graph()


data_1 = xlrd.open_workbook(r'F:\大学\科研立项\18年3月\3-30\词向量_最终准备.xlsx')
# data_2 = xlrd.open_workbook(r'F:\大学\科研立项\18年3月\词向量10\Dataset2.xlsx')
table_1 = data_1.sheets()[0]
# table_2 = data_2.sheets()[0]
# Y = table.col_values(0)
X = []
Y = []
for i in range(table_1.nrows-1):
# for i in range(20000):
    xx = table_1.row_values(i+1)
    xx = xx[4:table_1.ncols]
    X.append(xx)
    yy = table_1.cell_value(i+1,2)
    Y.append([yy,1-yy])

dataset_size = table_1.nrows



'''
X_test = []
Y_test = []
for i in range(20000):
    xx = table_1.row_values(i+20001)
    xx = xx[3:table_1.ncols]
    X_test.append(xx)
    yy = table_1.cell_value(i+20001,2)
    Y_test.append([yy,1-yy])
    
    

rdm = RandomState(1)
dataset_size = 129
X = rdm.rand(dataset_size,1000)

# Y = [[int(x1 < 2)] for x1 in X]
Y = [[int(x1<0.5),1-int(x1<0.5)]for x1 in X[:,1]]
'''
# ---------------------------------------------------------------------

# Hyperparameters
lr = 0.001
training_iters = 1000
batch_size = 100


n_inputs = 100
n_steps = 10
n_hidden_units = 200
n_classes = 2

# tf Graph input
x = tf.placeholder(tf.float32,shape = (None,1000), name ='x-input')
y_ = tf.placeholder(tf.float32,shape = (None,2), name = 'y-input')

# Define weights
weights = {
        # (10,200)
        'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
        # (200,2)
        'out': tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
        }
biases = {
        'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),
        'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))
        }

# def RNN(X,weights,biases):
# hidden layer for input to cell
# X (8 batch, 10 steps, 100 inputs)
# -->(8*10, 100 inputs)
X1 = tf.reshape(x,[-1,n_inputs])
# X_in ==> (8 batch* 10 steps, 200 hidden)
X_in = tf.matmul(X1,weights['in'])+biases['in']
# X_in ==> (8 batch, 10 steps, 200 hidden)
X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_units])

# cell
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=1.0,state_is_tuple=True,reuse=tf.AUTO_REUSE)
# lstm cell is divided into two parts (c_state,m_state)
_init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)

outputs,states = tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state = _init_state,time_major=False)

# hidden layer for outputs as the final results
results = tf.matmul(states[1],weights['out'])+biases['out']
   
    # return results




# pred = RNN(x,weights,biases)
pred = results
cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y_))
# tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    ac = []
    for i in range(training_iters):
        start = (i*batch_size) % dataset_size
        end = min(start+batch_size,dataset_size) 
        if end == dataset_size:
            start = -batch_size
        accu = sess.run(accuracy,feed_dict = {x:X[start:end], y_ :Y[start:end]})
        if i % 10 == 0:
            ac.append(accu)
    
    # Create the sen2vec
    workbook = xlsxwriter.Workbook('sentence1.xlsx') 
    sheet1 = workbook.add_worksheet()
    
    for i in range(dataset_size//batch_size +1 ):
        start = (i*batch_size) % dataset_size
        q = dataset_size - start
        end = min(start+batch_size,dataset_size) 
        if end == dataset_size:
            start = -batch_size
        sen1 = sess.run(states,feed_dict = {x:X[start:end], y_ :Y[start:end]})
        sen1 = np.array(sen1)
        sen2 = sen1[1]
        if q>= batch_size:
            for j in range(batch_size):
                for k in range(200):
                    sheet1.write(start+j,k,sen2[j,k])
        else:
            q = batch_size - q
            for j in range(batch_size):
                if q+j+1 >=batch_size:
                    break
                for k in range(200):
                    sheet1.write(dataset_size-batch_size+q+j,k,sen2[q+j+1,k])
            
                        
    workbook.close()   

accux = range(0, 100)
plt.plot(accux,ac,label="accuracy",color='r') 
plt.xlabel('train step (unit:*100)') 
plt.legend() 
plt.show()          