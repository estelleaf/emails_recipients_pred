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
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from init import split, init_dic
from sklearn.metrics.pairwise import cosine_similarity

path_to_data= "/Users/estelleaflalo/Desktop/M2_Data_Science/Second_Period/Text_and_Graph/Project/text_and_graph/Data/"
#path_to_data = 'C:/Nicolas/M2 MVA/ALTEGRAD/Kaggle/text_and_graph/Data/'
#path_to_data = '/Users/domitillecoulomb/Documents/DATA_SCIENCE/Semester2/Text_Graph/text_and_graph/Data'


training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)

training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)

test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)

test_info= pd.read_csv(path_to_data + 'test_info.csv', sep=',', header=0)



###################################################
print "Building dictionnaries"
all_users, all_senders,all_recs,address_books =init_dic(training,training_info)


training_info_mat=training_info.as_matrix()
content_train=training_info_mat[:,2]
vectorizer_train = TfidfVectorizer(max_df=0.95, min_df=2,stop_words='english',use_idf=True)
vec_train = vectorizer_train.fit_transform(content_train)
bow_train=vec_train.toarray()

test_info_mat=test_info.as_matrix()
content_test=test_info_mat[:,2]
vectorizer_test = TfidfVectorizer( max_df=0.95, min_df=2,stop_words='english',use_idf=True)
vec_test = vectorizer_test.fit_transform(content_test)
bow_test=vec_test.toarray()


#norma=norm(bow_train, axis=1, ord=2) 
#bow_train_normed=norm(bow_train.astype(np.float)# / norma[:,None],axis=1)


def centroid(r,dataset_info,bow):
    info_recip_index=dataset[dataset['recipients'].str.contains(r)].index.tolist() #"rick.dietz@enron.com"
    bow_recip=bow[info_recip_index]  
    norma=norm(bow_recip, axis=1, ord=2) 
    bow_recip_normzd=bow_recip.astype(np.float) / (norma[:,None]+10**(-7))
    return np.sum(bow_recip_normzd,axis=0)    
#centroid("rick.dietz@enron.com",training_info,bow_train)    

#CREATE 1 tf_idf DATAFRAME FOR EACH SENDER
def create_tfidf_df(u,):   
    """Create Dataframe"""
    df_tfidf = pd.DataFrame(columns=('sender', 'recipient','tf_idf'))
    """Get email ids of that sender and select the dataset"""
    mid = np.array(emails_ids_per_sender[u]).astype('int')
    index = training_info.loc[training_info['mid'].isin(mid)].index.tolist()
    dataset = training_info.loc[training_info['mid'].isin(mid)].set_index(np.arange(len(index)))
    bow = bow_train[index]
    """Fill the dataframe with the centroids of the recipients"""
    i=0
    for r in [elt[0] for elt in address_books[u]]:
        df_tfidf.loc[i] = [u, r, ','.join(['%f' % num for num in centroid(r,dataset,bow)])]
        i+=1  
    return df_tfidf


 # LOOPING OVER THE SENDERS TO CREATE ALL THE DATAFRAMES AND SAVE IT TO CSV

for sdr in all_senders:
    
    df_test = create_tfidf_df(u = sdr)
    
    idx = all_senders.index(sdr)
    
    df_test.to_csv(path_to_data + 'tf_idf/' + 'tf_idf_%s.csv' %idx, index=None)
    
    del df_test # saving memory ! """
    
# READING A DATAFRAME 

# Function to read and store the tf_idf scores related to one user in a given dictionary
def read_tfidf_dict(path, sdr, dictionary_to_be_filled):
    idx = all_senders.index(sdr)
    df = pd.read_csv(path + 'tf_idf_%s.csv' %idx)
    for index, series in df.iterrows():
        row = series.tolist()
        sender = row[0]
        recipient = row[1]
        centroid = np.array(row[2].split(',')).astype(float)
        dictionary_to_be_filled[recipient] = centroid
#d={}
#path = '/Users/domitillecoulomb/Documents/DATA_SCIENCE/Semester2/Text_Graph/Projet/tf_idf/'
#test = read_tfidf_dict(path, 'andrea.ring@enron.com', d)

def read_tfidf_array(path, sdr, nb_tokens = 185041):
    idx = all_senders.index(sdr)
    df = pd.read_csv(path + 'tf_idf_%s.csv' %idx)
    vectors_array = np.empty((len(address_books[sdr]),nb_tokens))
    recipients_index = np.array(['init'])
    for index, series in df.iterrows():
        row = series.tolist()
        sender = row[0]
        recipient = row[1]
        centroid = np.array(row[2].split(',')).astype(float)
        vectors_array[index] = centroid
        recipients_index = np.append(recipients_index, recipient)
    return vectors_array, recipients_index[1:]
    
#path = '/Users/domitillecoulomb/Documents/DATA_SCIENCE/Semester2/Text_Graph/Projet/tf_idf/'
#v,r = read_tfidf_array(path, 'andrea.ring@enron.com', 185041)


