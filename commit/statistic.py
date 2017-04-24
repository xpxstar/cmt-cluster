# coding=utf-8  
'''
Created on 2016年11月27日

@author: admin
'''
import codecs
from fileinput import filename
import os 
import time          

import hdbscan
from sklearn.cluster import DBSCAN
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.metrics import precision_recall_fscore_support as score

import numpy as np


path='data/'

def count(filename):
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
     
     
    outputf = codecs.open(path+filename,'rU','utf-8')# open for 'w'riting
    contexts = outputf.readlines()
    map = [0]*200
    for dd in contexts:
        map[int(dd)]+=1
    outputf.close()
    for dd in map:
        print dd
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
    count("top-10-label.txt")
