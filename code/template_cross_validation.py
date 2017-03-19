#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:41:18 2017

@author: domitillecoulomb
"""

from paths import path # on a besoin de path_to_code pour pouvoir importer paths.py, le serpent se mort la queue :D
path_to_code, path_to_data, path_to_results = path("domitille")

import numpy as np
import pandas as pd
from loss_function import score
from init import split, init_dic
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

##########################
# some function #                           
##########################

def Knn(bow_train, bow_test, training_info_S, test_info_S, K=20):
    df_knn = pd.DataFrame(columns=('mid', 'recipients'))

    for i, mid in enumerate(test_info_S['mid']):
        # get K-nearest neighbors in term of cosine(tfidf)


        cosine_similarities = cosine_similarity(bow_test[i][np.newaxis, :], bow_train).flatten()
        temp = np.concatenate((training_info_S['recipients'].values[:, np.newaxis], cosine_similarities[:, np.newaxis]),
                              axis=1)
        temp = temp[temp[:, 1].argsort()[::-1].tolist()]
        knn_liste = temp[:K]

        # get all the recipients in the K-nns
        all_recipients_in_Knn = []
        for j in range(K):
            all_recipients_in_Knn.extend(knn_liste[:, 0][j].split(' '))

        all_recipients_in_Knn = list(set(all_recipients_in_Knn))

        # compute the score for each recipients
        recipients_score = {}
        for recipient in all_recipients_in_Knn:
            idx = [ind for ind in range(K) if recipient in knn_liste[ind, 0]]
            recipients_score[recipient] = np.sum(knn_liste[idx, 1])
            # recipients_score[recipient] = np.sum(knn_liste[recipient in knn_liste[:,0]][:, 1])
        sorted_recipients_by_score = sorted(recipients_score, key=recipients_score.get, reverse=True)[:10]

        df_knn.loc[i] = [int(mid), sorted_recipients_by_score]

    return df_knn

####################################
# load the files and preprocessing #                           
####################################

training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)
training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)

print "Splitting data"
X_train, X_dev, Y_train,Y_dev=split(training,training_info,42)

print "Building dictionnaries"
all_users, all_senders,all_recs,address_books ,emails_ids_per_sender=init_dic(training,training_info)

####################
# CROSS_VALIDATION #                           
####################

# Fixing parameters
use_idf = True; print 'Parameter use_idf is set to {}'.format(use_idf)
K=23; print 'parameter K is set to {}'.format(K)
max_df = 0.95; print 'To build the vocabulary, the tfidfVectorizer will use max_df={}'.format(max_df)
sublinear_tf  = True # default is False in sklearn

#Storing cross_Val results                                         
scores_per_sender_knn={}
predictions_per_sender_knn = {}

for p in range(len(all_senders)):
    
    index=p #Select a sender S
    print 'sender', index
    sender=all_senders[index]
    X_train_S=X_train[sender]
    X_dev_S=X_dev[sender]
    Y_dev_S=Y_dev[sender]
    Y_train_S=Y_train[sender]

    # vectorize mails sent by a unique sender
    vectorizer_sender = TfidfVectorizer(max_df=0.95, stop_words='english', use_idf=use_idf, sublinear_tf=sublinear_tf)

    #train    
    training_info_S=training_info.loc[training_info['mid'].isin(X_train_S)]
    training_info_S=training_info_S.set_index(np.arange(len(training_info_S)))                                                                 
    training_info_S_mat=training_info_S.as_matrix()
    content_train=training_info_S_mat[:,2]
    
    vec_train = vectorizer_sender.fit_transform(content_train)
    bow_train=vec_train.toarray()
        
    #test
    test_info_S=training_info.loc[training_info['mid'].isin(X_dev_S)] 
    test_info_S=test_info_S.set_index(np.arange(len(test_info_S)))                                                                 
    test_info_S_mat=test_info_S.as_matrix()
    content_test=test_info_S_mat[:,2]
       
    vec_test = vectorizer_sender.transform(content_test)
    bow_test=vec_test.toarray() 

    """Algorithm : KNN"""
    # compute K-nn for each message m in the test set
    test_knn = Knn(bow_train, bow_test, training_info_S, test_info_S, K=K)
    test_knn['mid'] = test_knn['mid'].astype(int)
    
    predictions_per_sender_knn[sender] = {}
    for (mid, pred) in zip(test_knn['mid'].values,test_knn['recipients'].values):
        predictions_per_sender_knn[sender][mid] = pred
        
    #Score
    sc_glob_knn=0 
    for m,pr in zip(X_dev_S,Y_dev_S):
        sc_glob_knn += score(pr,predictions_per_sender_knn[sender][m])
        
    sc_temp_knn = float(sc_glob_knn)/len(Y_dev_S)
   
    scores_per_sender_knn[sender] = sc_temp_knn
                             
    print 'score for sender:', sender, sc_temp_knn

print 'Global Score with KNN:', np.mean(scores_per_sender_knn.values())
