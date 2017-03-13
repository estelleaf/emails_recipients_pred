#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:41:18 2017

@author: domitillecoulomb
"""

from paths import path # on a besoin de path_to_code pour pouvoir importer paths.py, le serpent se mort la queue :D
path_to_code, path_to_data, path_to_results = path("domitille")

# import sys pour ajouter le path_to_code pour que import init fonctionne
import sys
#sys.path.append(path_to_code)

import numpy as np
import pandas as pd
from numpy.linalg import norm
import operator
from loss_function import score
from init import split, init_dic
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from paths import path # on a besoin de path_to_code pour pouvoir importer paths.py, le serpent se mort la queue :D
path_to_code, path_to_data, path_to_results = path("domitille")

#Creation of centroids for each recipient
def centroid(sender,dataset_info,bow):
    df_tfidf = pd.DataFrame(columns=('recipient','tf_idf'))
    i=0
    for r in [elt[0] for elt in address_books[sender]]:
        info_recip_index=dataset_info[dataset_info['recipients'].str.contains(r)].index.tolist() #"rick.dietz@enron.com"
        bow_recip=bow[info_recip_index]  
        norma=norm(bow_recip, axis=1, ord=2) 
        bow_recip_normzd=bow_recip.astype(np.float) / (norma[:,None]+10**(-7))
        centroid_s_r= np.sum(bow_recip_normzd,axis=0) 
        df_tfidf.loc[i]  = [r, centroid_s_r]
        i+=1
    return df_tfidf

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



##########################
# load some of the files #                           
##########################

training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)

training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)

test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)

test_info= pd.read_csv(path_to_data + 'test_info.csv', sep=',', header=0)


print "Splitting data"
X_train, X_dev, Y_train,Y_dev=split(training,training_info,42)

print "Building dictionnaries"
all_users, all_senders,all_recs,address_books ,emails_ids_per_sender=init_dic(training,training_info)

use_idf = True
print 'Parameter use_idf is set to {}'.format(use_idf)
K=20
print 'parameter K is set to {}'.format(K)
max_df = 0.95
print 'To build the vocabulary, the tfidfVectorizer will use max_df={}'.format(max_df)
sublinear_tf  = True # default is False in sklearn
if not sublinear_tf:
    print 'The tf is replaced by (1 + log(tf))'
                                           
""" TFIDF CENTROID """

scores_per_sender_knn={}
scores_per_sender_centroid={}
scores_per_sender_frequency={}
predictions_per_sender_knn = {}
predictions_per_sender_centroid = {}
predictions_per_sender_frequency = {}

for p in range(len(all_senders)):
    
    #Select a sender S
    index=p
    print 'sender', index
    sender=all_senders[index]
    X_train_S=X_train[sender]
    X_dev_S=X_dev[sender]
    Y_dev_S=Y_dev[sender]
    Y_train_S=Y_train[sender]
    

    ##############Create TF IDF vector from mails sent by sender S


    # vectorize mails sent by a unique sender
    vectorizer_sender = TfidfVectorizer(max_df=0.95, stop_words='english', use_idf=use_idf, sublinear_tf=sublinear_tf)

    
    #vectorize mails sent by a unique sender
    #vectorizer_sender = TfidfVectorizer(max_df=0.95,stop_words='english',use_idf=True)
    
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

    """Method 1: KNN"""
    # compute K-nn for each message m in the test set
    test_knn = Knn(bow_train, bow_test, training_info_S, test_info_S, K=K)
    test_knn['mid'] = test_knn['mid'].astype(int)
    
    predictions_per_sender_knn[sender] = {}
    for (mid, pred) in zip(test_knn['mid'].values,test_knn['recipients'].values):
        predictions_per_sender_knn[sender][mid] = pred
    
    """Method 2: Centroid"""
    #Creation of centroids for each recipient       
    centroid_S_df=centroid(sender,training_info_S,bow_train)   
    centroid_S_arr=np.vstack(centroid_S_df['tf_idf'].as_matrix())
    
    #Similiarity
    
    predictions_per_sender_centroid[sender]={}
    for mid,k in zip(X_dev_S,range(bow_test.shape[0])):
        mail_test=bow_test[k]
        cosine_similarities = linear_kernel(mail_test[np.newaxis,:], centroid_S_arr).flatten()
        similar_centroids = [i for i in cosine_similarities.argsort()[::-1]]
        predictions_per_sender_centroid[sender][mid] = centroid_S_df.ix[similar_centroids[:10]]['recipient'].tolist()
    
    """Method 3: Frequency"""
    
    predictions_per_sender_frequency[sender]={}
    k_most = [elt[0] for elt in address_books[sender][:10]]
    for mid in X_dev_S:
        predictions_per_sender_frequency[sender][mid] = k_most

    
    #Score
    
    sc_glob_knn=0 
    sc_glob_centroid=0
    sc_glob_frequency=0
    for m,pr in zip(X_dev_S,Y_dev_S):
        sc_glob_knn += score(pr,predictions_per_sender_knn[sender][m])
        sc_glob_centroid += score(pr,predictions_per_sender_centroid[sender][m])
        sc_glob_frequency += score(pr,predictions_per_sender_frequency[sender][m])
        
    sc_temp_knn = float(sc_glob_knn)/len(Y_dev_S)
    sc_temp_centroid = float(sc_glob_centroid)/len(Y_dev_S)
    sc_temp_frequency = float(sc_glob_frequency)/len(Y_dev_S)
    
    scores_per_sender_knn[sender] = sc_temp_knn
    scores_per_sender_centroid[sender] = sc_temp_centroid
    scores_per_sender_frequency[sender] = sc_temp_frequency
                              
    print 'score knn:', sc_temp_knn
    print 'score centroid:', sc_temp_centroid
    print 'score frequency:', sc_temp_frequency
    
senders_centroid = []
senders_knn = []
senders_frequency = []
for s in scores_per_sender_knn.keys():
    if (scores_per_sender_knn[s] < scores_per_sender_centroid[s]) & (scores_per_sender_frequency[s] < scores_per_sender_centroid[s]) :
        senders_centroid.append(s)
    elif (scores_per_sender_knn[s] > scores_per_sender_centroid[s]) & (scores_per_sender_frequency[s] < scores_per_sender_knn[s]) :
        senders_knn.append(s)
    else: senders_frequency.append(s)
        

print 'Global Score with TFIDF CENTROID:', np.mean(scores_per_sender_centroid.values())
print 'Global Score with KNN:', np.mean(scores_per_sender_knn.values())
print 'Global Score with KNN:', np.mean(scores_per_sender_frequency.values())

##senders_frequency = senders_frequency + senders_centroid

""" FREQUENCE FOR BAD RESULTS """

#predictions_per_sender = {}
"""
k = 10
    
for sdr in [elt[0] for elt in sorted(scores_per_sender.items(), key=operator.itemgetter(1))[:60]]:
    X_dev_S = X_dev[sdr]
    Y_dev_S = Y_dev[sdr]
    k_most = [elt[0] for elt in address_books[sdr][:k]]
    #for id_predict in X_dev[sdr]:
    #    Y_dev[sdr]
    sc_glob=0  
    for l in Y_dev[sdr]:
        sc_glob += score(l,k_most)
        
    scores_per_sender[sdr] = float(sc_glob)/len(Y_dev_S)
        
        
        """
        
