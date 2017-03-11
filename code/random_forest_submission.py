#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# on a besoin de path_to_code pour pouvoir importer paths.py, le serpent se mort la queue :D
path_to_code = 'C:/Nicolas/M2 MVA/ALTEGRAD/Kaggle/text_and_graph/code'
# path_to_code = "/Users/estelleaflalo/Desktop/M2_Data_Science/Second_Period/Text_and_Graph/Project/text_and_graph/code/"

import sys

sys.path.append(path_to_code)

from paths import path

path_to_code, path_to_data, path_to_results = path("nicolas")

import numpy as np
from init import split, init_dic, csv_to_sub
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm
from loss_function import score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from knn import knn_predictor

training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)

training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)

test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)

test_info = pd.read_csv(path_to_data + 'test_info.csv', sep=',', header=0)

print "Building dictionnaries"

_, all_senders, _, address_books, _ = init_dic(training, training_info)
X_train, Y_train, X_dev = csv_to_sub(training, training_info, test, test_info)
#X_train, X_dev, Y_train, Y_dev = split(training, training_info,42)
predictions_per_sender_RF = {}







n_estimators=10
max_depth=15
max_features='log2'


for p in range(len(all_senders)):

    # Select a sender S
    index = p
    sender = all_senders[index]
    X_train_S = X_train[sender]
    X_dev_S = X_dev[sender]
    Y_train_S = Y_train[sender]

    ##############Create TF IDF vector from mails sent by sender S


    # vectorize mails sent by a unique sender
    vectorizer_sender = CountVectorizer(stop_words='english')

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

    all_recs_S = get_all_recs_per_sender(training_info_S)
    n_class = len(all_recs_S)

    n_pred = 10
    n_mail = content_train.shape[0]

    #create dummy matrix
    y_train = np.zeros((n_mail, n_class), dtype=int)
    for i in range(n_mail):
        for j in range(n_class):
            if  all_recs_S[j] in training_info_S['recipients'].values[i]:
                y_train[i, j] = 1


    n_test = bow_test.shape[0]
    RF = RandomForestRegressor(n_estimators=10, max_depth=15, max_features='log2', n_jobs=-1)
    RF.fit(bow_train, y_train)
    y_test = RF.predict(bow_test).reshape((n_test, n_class))

    #sanity check : if n_recipient is <10 from thte begining (e.g. p=37 has only one correspondant)
    if n_class < 10:
        if n_class == 1:
            print n_class
            predictions_per_sender_RF[sender] = []
            ind = np.argsort(y_test, axis=1)[:, -1][:, None]
            print ind
            for i, mid in enumerate(test_info_S['mid'].astype(int)):
                predictions_per_sender_RF[sender].append([ mid, [ all_recs_S[j] for j in ind[i, :] ]])
        else:
            predictions_per_sender_RF[sender] = []
            ind = np.argsort(y_test, axis=1)[:, -(n_class + 1):-1].reshape((n_test, n_class))
            for i, mid in enumerate(test_info_S['mid'].astype(int)):
                predictions_per_sender_RF[sender].append([mid, [all_recs_S[j] for j in ind[i, :]]])
    else:
        ind = np.argsort(y_test, axis=1)[:, -11:-1]

        predictions_per_sender_RF[sender] = []
        for i, mid in enumerate(test_info_S['mid'].astype(int)):
            predictions_per_sender_RF[sender].append([ mid, [ all_recs_S[j] for j in ind[i, :] ]])


    ########" KNN ###################
    knn_predictor_ = knn_predictor(sender=sender)
    test_knn = knn_predictor_.fit_predict(bow_train, bow_test, training_info_S, test_info_S, K=K)
    knn_predictor_.build_prediction_dictionnary(predictions_per_sender_knn, test_knn)




    print "Sender Number : " + str(p)


c=0 # compteur : a priori faut que ce soit 2362

with open(path_to_results + 'predictions_RF_count_vectrorizer_.txt', 'wb') as my_file:

    my_file.write('mid,recipients' + '\n')
    for sender, preds_for_sender in predictions_per_sender_RF.iteritems():

        for (mid, pred) in  preds_for_sender:
            c += 1
            print 'mid',  mid
            print 'pred', pred
            my_file.write(str(mid) + ',' + ' '.join(pred) + '\n')


if c !=2362:
    print 'Il y a un pb ! Le doc devrait avoir 2362 lignes et il en a {}'.format(c)
else:
    print 'everything went smoooothly (trust me, I do maths)'

