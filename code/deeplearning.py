#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 15:18:12 2017

@author: estelleaflalo
"""


import numpy as np
import pandas as pd
from loss_function import score
from init import split, init_dic

from keras.preprocessing.sequence import pad_sequences

from paths import path # on a besoin de path_to_code pour pouvoir importer paths.py, le serpent se mort la queue :D
path_to_code, path_to_data, path_to_results = path("estelle")

training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)

training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)

test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)

test_info= pd.read_csv(path_to_data + 'test_info.csv', sep=',', header=0)

print "Splitting data"
X_train, X_dev, Y_train,Y_dev=split(training,training_info,42)

index=3
all_users, all_senders,all_recs,address_books ,emails_ids_per_sender=init_dic(training,training_info)

sender=all_senders[index]

X_train_S = X_train[sender]
training_info_S=training_info.loc[training_info['mid'].isin(X_train_S)]

X_test_S = X_dev[sender]
testing_info_S=training_info.loc[training_info['mid'].isin(X_test_S)]

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

seq_lens = [len(s) for s in sequences]
seq_test_lens = [len(s) for s in sequences_test]

MAX_SEQUENCE_LENGTH = max( max(seq_lens),max(seq_test_lens))

# pad sequences with 0s
x_train_S = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
x_test_S = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', x_train_S.shape)
print('Shape of data test tensor:', x_test_S.shape)



y_train_new=np.zeros((len(Y_train[sender]),len(address_books[sender])))
y_test_new=np.zeros((len(Y_dev[sender]),len(address_books[sender])))

rec_index={}
i=0
for r in [elt[0] for elt in address_books[sender]]:
    rec_index[r]=i
    i+=1

index=[]
for i,mail in enumerate(Y_dev[sender]):
    for rec in Y_dev[sender][i]:
        index.append(rec_index[rec])
    y_test_new[i][index]=1

index=[]
for i,mail in enumerate(Y_train[sender]):
    for rec in Y_train[sender][i]:
        index.append(rec_index[rec])
    y_train_new[i][index]=1
      



from keras.layers import Dense, Input, Flatten
from keras.layers import GlobalAveragePooling1D, Embedding
from keras.models import Model

EMBEDDING_DIM = 50  #petit dataset, donc si ca avait été 300 ca aurait surement overfitte
N_CLASSES = len(address_books[sender])

# input: a sequence of MAX_SEQUENCE_LENGTH integers
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)   #taille vocabulaire=MAX_SEQUENCE_LENGTH (200000 les plus freq) x EMB x SEQ

embedded_sequences = embedding_layer(sequence_input)

average = GlobalAveragePooling1D()(embedded_sequences)
predictions = Dense(N_CLASSES, activation='softmax')(average)
#20 classes en sorties, 
model = Model(sequence_input, predictions)
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['mse'])


model.fit(x_train_S, y_train_new,
          nb_epoch=500, batch_size=128)

pred=model.predict(x_test_S)
pred=np.argsort(pred,axis=1)[:,::-1][:,:10]

index_rec={v: k for k, v in rec_index.iteritems()}

y_pred_S=[]           
for i in range(pred.shape[0]):
    temp=[]
    for j in range(pred.shape[1]):
        temp.append(index_rec[pred[i,j]])
    y_pred_S.append(temp)
