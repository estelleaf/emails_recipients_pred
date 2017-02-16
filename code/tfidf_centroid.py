#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 14:38:49 2017

@author: estelleaflalo
"""
import random
import operator
import pandas as pd
import numpy as np
from collections import Counter

path_to_data= "/Users/estelleaflalo/Desktop/M2_Data_Science/Second_Period/Text_and_Graph/Project/text_and_graph/Data/"
#path_to_data = 'C:/Nicolas/M2 MVA/ALTEGRAD/Kaggle/text_and_graph/Data/'
#path_to_data = '/Users/domitillecoulomb/Documents/DATA_SCIENCE/Semester2/Text_Graph/text_and_graph/Data'


training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)

training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)

test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)

test_info= pd.read_csv(path_to_data + 'test_info.csv', sep=',', header=0)

emails_ids_per_sender = {}
for index, series in training.iterrows():
    row = series.tolist()
    sender = row[0]
    ids = row[1:][0].split(' ')
    emails_ids_per_sender[sender] = ids

# save all unique sender names
all_senders = emails_ids_per_sender.keys()

# create address book with frequency information for each user
address_books = {}
i = 0

for sender, ids in emails_ids_per_sender.iteritems():
    recs_temp = []
    for my_id in ids:
        recipients = training_info[training_info['mid']==int(my_id)]['recipients'].tolist()
        recipients = recipients[0].split(' ')
        # keep only legitimate email addresses
        recipients = [rec for rec in recipients if '@' in rec]
        recs_temp.append(recipients)
    # flatten    
    recs_temp = [elt for sublist in recs_temp for elt in sublist]
    # compute recipient counts
    rec_occ = dict(Counter(recs_temp))
    # order by frequency
    sorted_rec_occ = sorted(rec_occ.items(), key=operator.itemgetter(1), reverse = True)
    # save
    address_books[sender] = sorted_rec_occ
    
    if i % 10 == 0:
        print i
    i += 1
  
# save all unique recipient names    
all_recs = list(set([elt[0] for sublist in address_books.values() for elt in sublist]))

# save all unique user names 
all_users = []
all_users.extend(all_senders)
all_users.extend(all_recs)
all_users = list(set(all_users))

from sklearn.feature_extraction.text import TfidfVectorizer

training_info_mat=training_info.as_matrix()
content_train=training_info_mat[:,2]
vectorizer_train = TfidfVectorizer( max_features=None,stop_words='english',use_idf=True)
vec_train = vectorizer_train.fit_transform(content_train)
bow_train=vec_train.toarray()

test_info_mat=test_info.as_matrix()
content_test=test_info_mat[:,2]
vectorizer_test = TfidfVectorizer( max_features=None,stop_words='english',use_idf=True)
vec_test = vectorizer_test.fit_transform(content_test)
bow_test=vec_test.toarray()

from numpy.linalg import norm
#norma=norm(bow_train, axis=1, ord=2) 
#bow_train_normed=norm(bow_train.astype(np.float) / norma[:,None],axis=1)

def centroid(r,dataset,bow):
    info_recip_index=dataset[dataset['recipients'].str.contains(r)].index.tolist() #"rick.dietz@enron.com"
    bow_recip=bow[info_recip_index]  
    norma=norm(bow_recip, axis=1, ord=2) 
    bow_recip_normzd=bow_recip.astype(np.float) / (norma[:,None]+10**(-7))
    return np.sum(bow_recip_normzd,axis=0)    

def all_centroids(dataset,bow):
    centroid_d={}
    for r in all_users:
        centroid_d[r]=centroid(r,dataset,bow)
    return centroid_d
    
centroid("rick.dietz@enron.com",training_info,bow_train)         

        
test=all_centroids(training_info,bow_train)  

