#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:50:30 2017

@author: estelleaflalo
"""


import sys


#path_to_code = 'C:/Nicolas/M2 MVA/ALTEGRAD/Kaggle/text_and_graph/code'
#sys.path.append(path_to_code)


from paths import path # on a besoin de path_to_code pour pouvoir importer paths.py, le serpent se mort la queue :D

path_to_code, path_to_data, path_to_results = path("nicolas")
#path_to_code, path_to_data, path_to_results = path("domitille")
#path_to_code, path_to_data, path_to_results = path("victor")
sys.path.append(path_to_code)



import numpy as np
from init import split, init_dic, csv_to_sub
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import euclidean_distances

# from loss_function import score
# from tfidf_centroid import centroid


##########################
# load some of the files #
##########################
def Knn(bow_train, bow_test, training_info_S, test_info_S, K=30):
    df_knn = pd.DataFrame(columns=('mid', 'recipients'))

    for i, mid in enumerate(test_info_S['mid']):
        # get K-nearest neighbors in term of cosine(tfidf)


        cosine_similarities = euclidean_distances(bow_test[i][np.newaxis, :], bow_train).flatten()
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



training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)

training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)

test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)

test_info = pd.read_csv(path_to_data + 'test_info.csv', sep=',', header=0)

print "Building dictionnaries"

_, all_senders, _, address_books, _ = init_dic(training, training_info)

X_train, Y_train, X_test = csv_to_sub(training, training_info, test, test_info)

predictions_per_sender = {}

# set the hyper-parameters like : use_id, etc...
use_idf = True
print 'Parameter use_idf is set to {}'.format(use_idf)


K=20


print 'parameter K is set to {}'.format(K)
max_df = 0.95
min_df = 1
print 'To build the vocabulary, the tfidfVectorizer will use max_df={} and min_df={}'.format(max_df, min_df)
sublinear_tf  = True # default is False in sklearn
if sublinear_tf:
    print 'The tf is replaced by (1 + log(tf))'

for p in range(len(all_senders)):

    # Select a sender S
    index = p
    sender = all_senders[index]
    X_train_S = X_train[sender]
    X_dev_S = X_test[sender]
    Y_train_S = Y_train[sender]

    ##############Create TF IDF vector from mails sent by sender S


    # vectorize mails sent by a unique sender
    vectorizer_sender = TfidfVectorizer(max_df=0.95, stop_words='english', use_idf=use_idf, sublinear_tf=sublinear_tf)

    # train
    training_info_S = training_info.loc[training_info['mid'].isin(X_train_S)]
    training_info_S = training_info_S.set_index(np.arange(len(training_info_S)))
    training_info_S_mat = training_info_S.as_matrix()
    content_train = training_info_S_mat[:, 2]

    vec_train = vectorizer_sender.fit_transform(content_train)
    bow_train = vec_train.toarray()
    # test
    test_info_S = test_info.loc[test_info['mid'].isin(X_dev_S)]
    test_info_S = test_info_S.set_index(np.arange(len(test_info_S)))
    test_info_S_mat = test_info_S.as_matrix()
    content_test = test_info_S_mat[:, 2]

    vec_test = vectorizer_sender.transform(content_test)
    bow_test = vec_test.toarray()

    # compute K-nn for each message m in the test set


    # training_info_S['recipients'][training_info_S['mid']==392289].tolist()[0].split(' ')

    test_knn = Knn(bow_train, bow_test, training_info_S, test_info_S, K=K)
    test_knn['mid'] = test_knn['mid'].astype(int)


    # add a entry corresponding to the sendr in the dictionnary
    predictions_per_sender[sender] = []
    for (mid, pred) in zip(test_knn['mid'].values,test_knn['recipients'].values):
        predictions_per_sender[sender].append([mid, pred])

    print "Sender Number : " + str(p)


c=0
with open(path_to_results + 'predictions_knn20_euclideandist.txt', 'wb') as my_file:

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
    print 'everything went smoooothly (trust me, I do maths)'