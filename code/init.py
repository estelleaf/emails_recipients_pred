# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:08:16 2017

@author: pleklol
"""

# -*- coding: utf-8 -*-

#%%
import random
import operator
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity

from paths import path

##########################
# load some of the files #                           
##########################




def init_dic(training,training_info):
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
    return all_users, all_senders,all_recs,address_books ,emails_ids_per_sender


def split(training,training_info):
    
    emails_ids_per_sender = {}
    for index, series in training.iterrows():
        row = series.tolist()
        sender = row[0]
        ids = row[1:][0].split(' ')
        emails_ids_per_sender[sender] = ids
    
    recs = {}
    for sender, ids in emails_ids_per_sender.iteritems():
        recs_temp=[]
        for my_id in ids:
            recipients = training_info[training_info['mid']==int(my_id)]['recipients'].tolist()
            recipients = recipients[0].split(' ')
            # keep only legitimate email addresses
            recipients = [rec for rec in recipients if '@' in rec]
            recs_temp.append(recipients)
        recs[sender]=recs_temp
            
            
    X_train={} 
    X_dev={}
    Y_train={}
    Y_dev={}
            
    for sender, ids in emails_ids_per_sender.iteritems():   
        X = ids
        Y = recs[sender]
        X_train[sender], X_dev[sender], Y_train[sender], Y_dev[sender] = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_train[sender]=np.array(X_train[sender]).astype('int')
        X_dev[sender]=np.array(X_dev[sender]).astype('int')
        
    return X_train, X_dev, Y_train,Y_dev
    
def csv_to_sub(training,training_info,test,test_info):
    emails_ids_per_sender = {}
    for index, series in training.iterrows():
        row = series.tolist()
        sender = row[0]
        ids = row[1:][0].split(' ')
        emails_ids_per_sender[sender] = ids
    
    recs = {}
    for sender, ids in emails_ids_per_sender.iteritems():
        recs_temp=[]
        for my_id in ids:
            recipients = training_info[training_info['mid']==int(my_id)]['recipients'].tolist()
            recipients = recipients[0].split(' ')
            # keep only legitimate email addresses
            recipients = [rec for rec in recipients if '@' in rec]
            recs_temp.append(recipients)
        recs[sender]=recs_temp
    
    X_train={}
    Y_train={}
    for sender, ids in emails_ids_per_sender.iteritems():   
        X_train[sender] = ids
        Y_train[sender] = recs[sender]
        X_train[sender]=np.array(X_train[sender]).astype('int')


    emails_ids_per_sender = {}
    for index, series in test.iterrows():
        row = series.tolist()
        sender = row[0]
        ids = row[1:][0].split(' ')
        emails_ids_per_sender[sender] = ids
    
    recs = {}
    X_test={}

    for sender, ids in emails_ids_per_sender.iteritems():   
        X_test[sender]= ids
        X_test[sender]=np.array(X_test[sender]).astype('int')
    
    
   
        
    return X_train,Y_train,X_test
   
    
        