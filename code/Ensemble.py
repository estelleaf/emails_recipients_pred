
#!/usr/bin/env python2
# -*- coding: utf-8 -*-


# import sys pour ajouter le path_to_code pour que import init fonctionne
path_to_code = 'C:/Nicolas/M2 MVA/ALTEGRAD/Kaggle/text_and_graph/code'
#path_to_code = "/Users/estelleaflalo/Desktop/M2_Data_Science/Second_Period/Text_and_Graph/Project/text_and_graph/code/"

import sys
sys.path.append(path_to_code)

from paths import path # on a besoin de path_to_code pour pouvoir importer paths.py, le serpent se mort la queue :D
path_to_code, path_to_data, path_to_results = path("nicolas")

import numpy as np
from init import split, init_dic, csv_to_sub
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm
from loss_function import score
from knn import knn_predictor
from Random_forest import Random_forest_predictor

training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)

training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)

test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)

test_info = pd.read_csv(path_to_data + 'test_info.csv', sep=',', header=0)

print "Building dictionnaries"

_, all_senders, _, address_books, _ = init_dic(training, training_info)
X_train, X_dev, Y_train, Y_dev = split(training, training_info, 42)
#X_train2, Y_train2, X_test = csv_to_sub(training, training_info, test, test_info)

new_index_train = []
for array in X_train.values():
    new_index_train.extend(array.tolist())

train_info = training_info.loc[training_info['mid'].isin(new_index_train)]



# set the hyper-parameters like : use_id, etc...
use_idf = True
print 'Parameter use_idf is set to {}'.format(use_idf)
K = 30
print 'parameter K is set to {}'.format(K)
max_df = 0.95
min_df = 1
print 'To build the vocabulary, the tfidfVectorizer will use max_df={} and min_df={}'.format(max_df, min_df)
sublinear_tf = True  # default is False in sklearn
if sublinear_tf:
    print 'The tf is replaced by (1 + log(tf))'


# Parametre de la random forest

n_estimators, max_depth, n_jobs = 100, 100, -2 #mettez pas -1 ou alors faites rien pendant



predictions_per_sender_knn = {} #initialise le dictionnary des prediction final pour knn
predictions_per_sender_freq = {}#initialise le dictionnary des prediction final pour freq
predictions_per_sender_RF = {}


score_per_sender_RF = {} # dict to store score on the validation set for each sender
score_per_sender_knn = {}

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
    #vectorizer_sender = CountVectorizer(stop_words='english')
    vectorizer_sender = TfidfVectorizer(stop_words='english', sublinear_tf=True)

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

#   ##############################################
    # on fait nos differend modeles : knn, RF
    ##############################################

    # knn
    knn_predictor_ = knn_predictor(sender=sender)
    test_knn = knn_predictor_.fit_predict(bow_train, bow_test, training_info_S, test_info_S,  K=K)
    knn_predictor_.build_prediction_dictionnary(predictions_per_sender_knn, test_knn)


    # random forest on tfidf
    RF_predictor = Random_forest_predictor(sender = sender)
    RF_predictor.fit_predict_build_pred_dictionnary(training_info_S, content_train, test_info_S, bow_train, bow_test,
                                                    n_estimators, max_depth, n_jobs, predictions_per_sender_RF )

    sender_score_RF = 0
    sender_score_knn = 0
    for prediction_RF, prediction_knn, truth in zip(predictions_per_sender_RF[sender], predictions_per_sender_knn[sender], Y_dev_S):
        sender_score_RF += score(truth, prediction_RF[1])
        sender_score_knn += score(truth, prediction_knn[1])

    score_per_sender_RF[sender] = sender_score_RF / len(Y_dev_S)
    score_per_sender_knn[sender] = sender_score_knn / len(Y_dev_S)

    print 'score RF for {} (p={}) is : {}:'.format(sender, p, score_per_sender_RF[sender])
    print 'score knn for {} (p={}) is : {}:'.format(sender, p, score_per_sender_knn[sender])


print 'Total score RF :', np.mean(score_per_sender_RF.values())
print 'Total score knn : ', np.mean(score_per_sender_knn.values())

#########################################
# On entraine l'aggragateur d'expert
########################################

senders_knn = []
senders_RF = []
for sender in score_per_sender_knn.keys():
    if score_per_sender_knn[sender] < score_per_sender_RF[sender]:
        senders_RF.append(sender)
    else:
        senders_knn.append(sender)










