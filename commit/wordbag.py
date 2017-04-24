# coding=utf-8  
'''
Created on 2016年11月27日

@author: admin
'''
import codecs
from fileinput import filename
import os 
import time          

from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.metrics import precision_recall_fscore_support as score

import numpy as np


path='data/'

def tain_cluster(filename):
    #如果是excel另存为csv 则需要修改读取方式r为wU
#     datafile = file(path+filename, 'rU')
#     lines = datafile.readlines()
#     datafile.close()
#     cv = CountVectorizer(binary = False, decode_error = 'ignore',stop_words = None)
#     cv_fit = cv.fit_transform(lines)
#     feature = cv_fit.toarray()
#     clf = KMeans(n_clusters=11)
#     s = clf.fit(feature)
#     print s
#     print clf.inertia_
#     print clf.predict(feature)
#      
     
     
    datafile = file(path+filename, 'rU')
    lines = datafile.readlines()
    datafile.close()
    cv = CountVectorizer(binary = False, decode_error = 'ignore',stop_words = None)
#     transformer = TfidfTransformer()
#     tfidf_data = transformer.fit_transform(vectorizer.fit_transform(data))
    #第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
#     feature = transformer.fit_transform(cv.fit_transform(lines))
    cv_fit = cv.fit_transform(lines)
    feature = cv.get_feature_names()
#     count = cv.vocabulary_
    print feature
    tf = cv_fit.toarray()
    row = len(tf)
    col = len(feature)
    total_tf = [0]*col
    
    
    for c in range(col):
        for r in range(row):
            total_tf[c]+=tf[r][c]
            
    print total_tf       
#     steps = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}
#     for i in steps:
#         clf = DBSCAN(eps=i,min_samples=5)
#         s = clf.fit_predict(feature)
# #         print s
#         print i,clf.labels_
#     print clf.components_
#     print clf.(feature)
      
#     for i in range(5,51,1):
#         clf = KMeans(n_clusters=i)
#         s = clf.fit(feature)
#         print i , clf.inertia_
#     outputf = codecs.open(path+'outcount.txt','w','utf-8')#open for 'w'riting
#     outputf.write(str(cv.get_feature_names()))
#     outputf.write('\n')
#     for mtx in cv_fit.toarray():
#         outputf.write(str(mtx).replace("\n", ""))
#         outputf.write('\n')
#     outputf.close()
#     
if __name__ == "__main__":
    tain_cluster("simdata-no-arg.csv")
