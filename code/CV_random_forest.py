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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm
from loss_function import score,score_en_mode_numpy
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)

training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)

test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)

test_info = pd.read_csv(path_to_data + 'test_info.csv', sep=',', header=0)

print "Building dictionnaries"

_, all_senders, _, address_books, _ = init_dic(training, training_info)
X_train, X_dev, Y_train, Y_dev = split(training, training_info, 1)
predictions_per_sender_RF = {}




def get_all_recs_per_sender(train_info_S):
    all_recs = []
    for recs in train_info_S['recipients'].values:
        all_recs.extend(recs.split(' '))

    all_recs = list(set(all_recs))
    return all_recs


n_estimators=100
max_depth = 100
max_features='log2'

score_per_sender = {}
for p in range(len(all_senders)):

    # Select a sender S
    index = p
    sender = all_senders[index]
    X_train_S = X_train[sender]
    X_dev_S = X_dev[sender]
    Y_train_S = Y_train[sender]
    Y_dev_S = Y_dev[sender]
    ##############Create TF IDF vector from mails sent by sender S


    # vectorize mails sent by a unique sender
    vectorizer_sender = CountVectorizer(stop_words='english')
    vectorizer_sender  = TfidfVectorizer(stop_words='english', sublinear_tf=True)

    # /!\ en ross_val on ne travaille pas avec le test_info.csv, mais on split le train info
    # train
    training_info_S = training_info.loc[training_info['mid'].isin(X_train_S)]
    training_info_S = training_info_S.set_index(np.arange(len(training_info_S)))
    training_info_S_mat = training_info_S.as_matrix()
    content_train = training_info_S_mat[:, 2]

    vec_train = vectorizer_sender.fit_transform(content_train)
    bow_train = vec_train.toarray()
    # validation
    test_info_S = training_info.loc[training_info['mid'].isin(X_dev_S)]
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
    RF = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, n_jobs=-1)
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
            print n_class
            predictions_per_sender_RF[sender] = []
            ind = np.argsort(y_test, axis=1)[:, -n_class:].reshape((n_test, n_class))
            for i, mid in enumerate(test_info_S['mid'].astype(int)):
                predictions_per_sender_RF[sender].append([mid, [all_recs_S[j] for j in ind[i, :]]])
    else:
        ind = np.argsort(y_test, axis=1)[:, -10:]

        predictions_per_sender_RF[sender] = []
        for i, mid in enumerate(test_info_S['mid'].astype(int)):
            predictions_per_sender_RF[sender].append([ mid, [ all_recs_S[j] for j in ind[i, :] ]])


    print "Sender Number : " + str(p)

    ############ Calcul du score ##############

    sender_score = 0
    for prediction, truth in zip(predictions_per_sender_RF[sender], Y_dev_S):
        sender_score += score(truth, prediction[1])

    score_per_sender[sender] = sender_score / len(Y_dev_S)

    print 'score RF for {} is : {}:'.format(sender, score_per_sender[sender])


total_score = np.mean(score_per_sender.values())
print total_score



for pred, truth in zip(predictions_per_sender_RF[sender], Y_dev_S):
    print 'pred', pred[1]
    print 'and truth :', truth

    print score(truth, pred[1])