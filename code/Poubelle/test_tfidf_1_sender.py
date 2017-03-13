#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:50:30 2017

@author: estelleaflalo
"""

# import sys pour ajouter le path_to_code pour que import init fonctionne
import sys
from paths import path

import numpy as np
from init import split, init_dic
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy.linalg import norm
from sklearn.metrics.pairwise import linear_kernel
from loss_function import score
#from tfidf_centroid import centroid

########################
# load some of the files #
##########################





path_to_code, path_to_data, path_to_results = path('estelle')








sys.path.append(path_to_code)


training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)

training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)

test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)

test_info= pd.read_csv(path_to_data + 'test_info.csv', sep=',', header=0)



print "Splitting data"
X_train, X_dev, Y_train,Y_dev=split(training,training_info)

print "Building dictionnaries"
all_users, all_senders,all_recs,address_books ,emails_ids_per_sender=init_dic(training,training_info)


#Select a sender S
index=3
sender=all_senders[index]
X_train_S=X_train[sender]
X_dev_S=X_dev[sender]
Y_dev_S=Y_dev[sender]
Y_train_S=Y_train[sender]


##############Create TF IDF vector from mails sent by sender S


#vectorize mails sent by a unique sender
vectorizer_sender = TfidfVectorizer(max_df=0.95,stop_words='english',use_idf=True)

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
    
    

    
centroid_S_df=centroid(sender,training_info_S,bow_train)

centroid_S_arr=np.vstack(centroid_S_df['tf_idf'].as_matrix())

#Similiarity
rec_pred_S=[]
for k in range(bow_test.shape[0]):
    mail_test=bow_test[k]
    cosine_similarities = linear_kernel(mail_test, centroid_S_arr).flatten()
    similar_centroids = [i for i in cosine_similarities.argsort()[::-1]]
    rec_pred_S.append(centroid_S_df.ix[similar_centroids[:10]]['recipient'].tolist())
    

                 
sc_glob=0   
for i in range(len(Y_dev_S)):
    sc_glob+=score(Y_dev_S[i],rec_pred_S[i])
    
final_score=float(sc_glob)/len(Y_dev_S)
