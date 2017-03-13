#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:49:42 2017

@author: domitillecoulomb
"""

import numpy as np
import pandas as pd
from loss_function import score
from init import split, init_dic

from paths import path # on a besoin de path_to_code pour pouvoir importer paths.py, le serpent se mort la queue :D
path_to_code, path_to_data, path_to_results = path("domitille")

training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)

training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)

test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)

test_info= pd.read_csv(path_to_data + 'test_info.csv', sep=',', header=0)

print "Splitting data"
X_train, X_dev, Y_train,Y_dev=split(training,training_info,42)

all_users, all_senders,all_recs,address_books ,emails_ids_per_sender=init_dic(training,training_info)


#%%
index=3
sender=all_senders[index]

X_train_S = X_train[sender]
training_info_S=training_info.loc[training_info['mid'].isin(X_train_S)]

X_test_S = X_train[sender]
testing_info_S=training_info.loc[training_info['mid'].isin(X_test_S)]

#%%
from keras.preprocessing.text import Tokenizer

MAX_NB_WORDS = 20000

# get the raw text data
mails_train = training_info_S['body'].tolist()
mails_test = testing_info_S['body'].tolist()

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, char_level=False)
tokenizer.fit_on_texts(mails_train)
sequences = tokenizer.texts_to_sequences(mails_train)
sequences_test = tokenizer.texts_to_sequences(mails_test)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))












