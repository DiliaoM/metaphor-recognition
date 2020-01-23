# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 15:08:04 2018

@author: Miranda
"""

import numpy as np
import jieba.posseg as pseg
from gensim.models import word2vec
import xlrd
import xlsxwriter 

# import model.
model = word2vec.Word2Vec.load("sohu.model")

# Open the file.
data = xlrd.open_workbook('dataset1&2整合.xlsx')
 
# Get the worksheet
sheet1 = data.sheets()[0]
data_1 = sheet1.col_values(5)
nrow_1 = sheet1.nrows
data_1 = data_1[2:nrow_1]

# Create a file to save word embedding.
workbook = xlsxwriter.Workbook('Dataset1.xlsx') 
sheet3 = workbook.add_worksheet()
sheet1 = workbook.add_worksheet()

wordvec = 200
wordset = 10

# Counting.
l = 0
list3 = [] #index
for i in range(nrow_1-2):
    list1 = []
    words = pseg.cut(data_1[i])
    j = 0
    for word,flag in words:
        if flag not in ['x','w','m','un']:
            j = j+1
            if j<=wordset:     
                try:
                    x = model[word]
                    x = x.tolist()
                    list1 = list1 + x
                except:
                    j = wordset+2
            if j >wordset:
                break
    if j < wordset:
        list2 = [0 for k in range(wordvec*(wordset-j))] 
        list1 = list1 + list2
        
    if j != 12: 
        for k in range(wordset*wordvec):
            sheet3.write(l,k,list1[k])
        l = l+1
        
workbook.close()        
    
    
    
    
    
    
