#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 15:18:12 2017

@author: estelleaflalo
"""


import numpy as np
import pandas as pd    
from keras.layers import Dense, Input
from keras.layers import GlobalAveragePooling1D, Embedding
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from init import init_dic, csv_to_sub

from keras.preprocessing.sequence import pad_sequences

from paths import path # on a besoin de path_to_code pour pouvoir importer paths.py, le serpent se mort la queue :D
path_to_code, path_to_data, path_to_results = path("estelle")

training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)
training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)
test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)
test_info= pd.read_csv(path_to_data + 'test_info.csv', sep=',', header=0)

all_users, all_senders,all_recs,address_books ,emails_ids_per_sender=init_dic(training,training_info)

X_train, Y_train, X_dev = csv_to_sub(training, training_info, test, test_info)

predictions_per_sender = {}

for p in range(len(all_senders)):
    
    index=p  
    sender=all_senders[index]
    EMBEDDING_DIM = 50  
    N_CLASSES = len(address_books[sender])
    MAX_NB_WORDS = 20000
          
    X_train_S = X_train[sender]
    training_info_S=training_info.loc[training_info['mid'].isin(X_train_S)]
    
    X_test_S = X_dev[sender]
    testing_info_S=test_info.loc[test_info['mid'].isin(X_test_S)]
    

    
    
    
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
       
    y_train_new=np.zeros((len(Y_train[sender]),len(address_books[sender])))
    
    
    rec_index={}
    i=0
    for r in [elt[0] for elt in address_books[sender]]:
        rec_index[r]=i
        i+=1
    
    
    index=[]
    for i,mail in enumerate(Y_train[sender]):
        for rec in Y_train[sender][i]:
            index.append(rec_index[rec])
        y_train_new[i][index]=1
          

    
    # Neural Net
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
              nb_epoch=10, batch_size=128)
    
    pred=model.predict(x_test_S)
    pred=np.argsort(pred,axis=1)[:,::-1][:,:10]
    
    index_rec={v: k for k, v in rec_index.iteritems()}
    
    y_pred_S=[]           
    for i in range(pred.shape[0]):
        temp=[]
        for j in range(pred.shape[1]):
            temp.append(index_rec[pred[i,j]])
        y_pred_S.append(temp)
        
        
    predictions_per_sender[sender] = []
    for (mid, pred) in zip(X_dev[sender],y_pred_S):
        predictions_per_sender[sender].append([mid, pred])

    print "Sender Number : " + str(p)
    
c=0
with open(path_to_results + 'predictions_deeplearning.txt', 'wb') as my_file:

    my_file.write('mid,recipients' + '\n')
    for sender, preds_for_sender in predictions_per_sender.iteritems():

        for (mid, pred) in  preds_for_sender:
            c += 1
            print 'mid',  mid
            print 'pred', pred
            my_file.write(str(mid) + ',' + ' '.join(pred) + '\n')


if c !=2362:
    print 'Il y a un pb ! Le doc devrait avoir 2362 lignes et il en a {}'.format(c)
else:
    print 'Ok'
