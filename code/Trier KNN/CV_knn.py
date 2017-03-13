# -*- coding: utf-8 -*-
"""
Created on Wed Mar 08 01:21:32 2017

@author: Victor
"""

from paths import path


path_to_code, path_to_data, path_to_results = path('victor')





import sys

sys.path.append(path_to_code)

import numpy as np
from init import split, init_dic, csv_to_sub
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm
from sklearn.metrics.pairwise import linear_kernel


# from loss_function import score
# from tfidf_centroid import centroid


##########################
# load some of the files #
##########################




training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)

training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)

test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)

test_info = pd.read_csv(path_to_data + 'test_info.csv', sep=',', header=0)

print "Building dictionnaries"

#_, all_senders, _, address_books, _ = init_dic(training, training_info)

_, all_senders, _, address_books, _ = init_dic(training, training_info)

X_train, X_dev, Y_train, Y_dev = split(training, training_info,42)
X_train2, Y_train2, X_test = csv_to_sub(training, training_info, test, test_info)
predictions_per_sender = {}

# set the hyper-parameters like : use_id, etc...
use_idf = True
print 'Parameter use_idf is set to {}'.format(use_idf)
K=30
print 'parameter K is set to {}'.format(K)
max_df = 0.95
min_df = 1
print 'To build the vocabulary, the tfidfVectorizer will use max_df={} and min_df={}'.format(max_df, min_df)

def Knn(bow_train, bow_test, training_info_S, test_info_S, K=30):
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


for p in range(len(all_senders)):

    # Select a sender S
    index = p
    sender = all_senders[index]
    X_train_S = X_train[sender]
    X_dev_S = X_dev[sender]
    Y_train_S = Y_train[sender]

    ##############Create TF IDF vector from mails sent by sender S


    # vectorize mails sent by a unique sender
    vectorizer_sender = TfidfVectorizer(max_df=0.95, stop_words='english', use_idf=use_idf)

    # train
    training_info_S = training_info.loc[training_info['mid'].isin(X_train_S)]
    training_info_S = training_info_S.set_index(np.arange(len(training_info_S)))
    training_info_S_mat = training_info_S.as_matrix()
    content_train = training_info_S_mat[:, 2]

    vec_train = vectorizer_sender.fit_transform(content_train)
    bow_train = vec_train.toarray()
    # test
    test_info_S = training_info.loc[training_info['mid'].isin(X_dev_S)]
    test_info_S = test_info_S.set_index(np.arange(len(test_info_S)))
    test_info_S_mat = test_info_S.as_matrix()
    content_test = test_info_S_mat[:, 2]

    vec_test = vectorizer_sender.transform(content_test)
    bow_test = vec_test.toarray()



    # compute K-nn for each message m in the test set
    
    test_knn = Knn(bow_train, bow_test, training_info_S, test_info_S, K=K)
    
        # Similiarity
    rec_pred_S = []
    
    
    predictions_per_sender[sender] = []
    for (mid, pred) in zip(test_knn['mid'],test_knn['recipients']):
        predictions_per_sender[sender].append([mid, pred])
        # alternative
        # predictions_per_sender[sender].append(mid)
        # predictions_per_sender[sender].append([pred])
    print "Sender Number : " + str(p)

c=0 # compteur : a priori faut que ce soit 2362
with open(path_to_results + 'test.txt', 'wb') as my_file:
    my_file.write('mid,recipients' + '\n')
    for sender, preds_for_sender in predictions_per_sender.iteritems():

        for (mid, pred) in  preds_for_sender:
            c += 1
            print 'mid',  mid
            print 'pred', pred
            my_file.write(str(mid) + ',' + ' '.join(pred) + '\n')


def score_en_mode_numpy(y_true,y_pred):
    c = 0
    temp=np.zeros(y_pred.shape[0])
    for i in range(y_pred.shape[0]):
        if y_pred[i] in y_true.tolist():
            c += 1
            temp[i] = float(c)/(i+1)
        else :
            temp[i] = 0
    return sum(temp)/min(y_true.shape[0],10)


def score(y_true,y_pred):
    c = 0
    n_pred = len(y_pred) # should be 10
    n_true = len(y_true)
    temp=np.zeros(n_pred)

    for i in range(n_pred):
        if y_pred[i] in y_true:
            c += 1
            temp[i] = float(c)/(i+1)
        else :
            temp[i] = 0

    return sum(temp)/min(n_true,10)


def test_score_on_train(path_to_results, ground_truth, txt='random'):
    """ option = 'random' ou  'frequency' """

    if txt == 'random':
        pred = pd.read_csv(path_to_results + 'train_predictions_random.txt', sep=',', header=0)
    elif txt == 'frequency':
        pred = pd.read_csv(path_to_results + 'train_predictions_frequency.txt', sep=',', header=0)
    else:
        pred = pd.read_csv(path_to_results + txt, sep=',', header=0)


    number_of_mail = len(ground_truth.keys())
    mail_score = np.zeros(shape=(number_of_mail, 1))

    for i, email_id in enumerate(ground_truth.keys()):
        y_pred = pred[pred['mid'] == int(email_id)]['recipients'].tolist()[0].split(' ')
        mail_score[i] = score(ground_truth[email_id], y_pred)

    return mail_score.mean()




ground_truth = {}
for sender, ids in X_dev.iteritems():
    recs_temp = []
    for my_id in ids:
        recipients = training_info[training_info['mid'] == int(my_id)]['recipients'].tolist()
        recipients = recipients[0].split(' ')
        # keep only legitimate email addresses
        recipients = [rec for rec in recipients if '@' in rec]

        # store the ground truth for the loss function in a dict { 'id_of_the_mail': list(recipients) }
        ground_truth[my_id] = recipients
