# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:08:16 2017

@author: pleklol
"""

# -*- coding: utf-8 -*-

#%%
import random
import operator
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

path_to_data =  'C:/Users/vpell_000/Documents/MVA/ALTEGRAD/Kaggle/text_and_graph/Data/'

##########################
# load some of the files #                           
##########################

training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)

training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)

test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)

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

    

    