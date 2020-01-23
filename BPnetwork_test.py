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
import random


data_1 = xlrd.open_workbook(r'F:\大学\科研立项\18年3月\3-30\词向量_最终准备.xlsx')
# data_2 = xlrd.open_workbook(r'F:\大学\科研立项\18年3月\词向量10\Dataset2.xlsx')
table_1 = data_1.sheets()[0]
# table_2 = data_2.sheets()[0]
# Y = table.col_values(0)
# 将比喻句和非比喻句的数据分开。
X_a = []
X_b = []
Y_a = []
Y_b = []
for i in range(table_1.nrows-1):
    xx = table_1.row_values(i+1) 
    xx = xx[4:table_1.ncols]
    yy = table_1.cell_value(i+1,2)
    if yy == 1:  
        X_a.append(xx)
        Y_a.append([yy,1-yy])
    else:
        X_b.append(xx)
        Y_b.append([yy,1-yy])
        
dataset_size = 181*54



# 初次抽取样本
X_1 = random.sample(X_a,1810)
Y_1 = []
for i in range(1810):
    a = X_a.index(X_1[i])
    Y_1.append(Y_a[a])
    
X_2 = random.sample(X_b,1810*5)
Y_2 = []
for i in range(1810*5):
    b = X_b.index(X_2[i])
    Y_2.append(Y_b[b])

X_test = random.sample(X_1,181)
Y_test = []
for i in range(181):
    c = X_1.index(X_test[i])
    Y_test.append(Y_1[c])
    del X_1[c],Y_1[c]
    
X_3 = random.sample(X_2,181*5)
Y_3 = []
for i in range(181*5):
    d = X_2.index(X_3[i]) 
    X_test.append(X_3[i])
    Y_test.append(Y_2[d])
    del X_2[d],Y_2[d]
    
X_1 = X_1 + X_2
Y_1 = Y_1 + Y_2


# 打乱顺序
X_train = X_1
Y_train = Y_1
Z = list(zip(X_train, Y_train))
random.shuffle(Z)
X_train[:], Y_train[:] = zip(*Z)


# ---------------------------------------------------------------------


batch_size = 8


w1 = tf.Variable(tf.random_normal([1000,500],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([500,100],stddev=1,seed=1))
w3 = tf.Variable(tf.random_normal([100,2],stddev=1,seed=1))

b1 = tf.Variable(tf.constant(0.1,shape=[500]))
b2 = tf.Variable(tf.constant(0.1,shape=[100]))
b3 = tf.Variable(tf.constant(0.1,shape=[2]))

x = tf.placeholder(tf.float32,shape = (None,1000),name='x-input')
y_ = tf.placeholder(tf.float32,shape = (None,2),name = 'y-input')

a_1 = tf.nn.sigmoid(tf.matmul(x,w1)+b1)
a_2 = tf.nn.sigmoid(tf.matmul(a_1,w2)+b2)
y = tf.nn.softmax(tf.matmul(a_2,w3)+b3)

cross_entropy = -tf.reduce_mean(
        y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    steps = 10000
    entropy = []
    accu = []
    for i in range(steps):
        start = (i*batch_size) % dataset_size
        end = min(start+batch_size,dataset_size) 
        sess.run(train_step,feed_dict = {x :X_train[start:end], y_ :Y_train[start:end]})
        if i % 100 == 0:
            total_cross_entropy = sess.run(cross_entropy,feed_dict = {x :X_train, y_ :Y_train})
            entropy.append(total_cross_entropy)
            accura = sess.run(accuracy,feed_dict = {x :X_train, y_ :Y_train})
            accu.append(accura)
    
    result_train = sess.run(y,feed_dict = {x:X_train})
    
    result = sess.run(y,feed_dict = {x:X_test})
    # writer = tf.summary.FileWriter("./example1_log",sess.graph)
# writer.close()


accux = range(0, 100)
plt.plot(accux,accu,label="accuracy",color='r') 
plt.xlabel('train step (unit:*100)') 
plt.legend() 
plt.show()  

plt.plot(accux,entropy,label="Entropy",color='r') 
plt.xlabel('train step (unit:*100)') 
plt.legend() 
plt.show()  

# ---------------------------------------
Y_train = np.array(Y_train)
count = 0
biyu = 0
fei = 0
for i in range(181*6*9):
    if result_train[i,0] >= 0.5:
        c = 1
    else:
        c = 0
    if c==Y_train[i,0]:
        count +=1
        if c ==1:
            biyu += 1
        else:
            fei += 1
print(count/(181*6*9))
print(biyu/(181*9))
print(fei/(181*5*9))





Y_test = np.array(Y_test)
count = 0
biyu = 0
fei = 0
for i in range(181*6):
    if result[i,0] >= 0.5:
        c = 1
    else:
        c = 0
    if c==Y_test[i,0]:
        count +=1
        if c ==1:
            biyu += 1
        else:
            fei += 1
print(count/(181*6))
print(biyu/181)
print(fei/(181*5))




