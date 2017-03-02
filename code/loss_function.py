#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 16:59:46 2017

@author: estelleaflalo
"""

from __future__ import division
import numpy as np
import pandas as pd
import operator
from collections import Counter

from paths import path
path_to_code, path_to_data, path_to_results = path('estelle')

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


def test_score_on_train(path_to_results, ground_truth, option='random'):
    """ option = 'random' ou  'frequency' """

    if option == 'random':
        pred = pd.read_csv(path_to_results + 'train_predictions_random.txt', sep=',', header=0)
    elif option == 'frequency':
        pred = pd.read_csv(path_to_results + 'train_predictions_frequency.txt', sep=',', header=0)
    else:
        raise ValueError('wrong option typ, should be random or frequency ')


    number_of_mail = len(ground_truth.keys())
    mail_score = np.zeros(shape=(number_of_mail, 1))

    for i, email_id in enumerate(ground_truth.keys()):
        y_pred = pred[pred['mid'] == int(email_id)]['recipients'].tolist()[0].split(' ')
        mail_score[i] = score(ground_truth[email_id], y_pred)

    return mail_score.mean()





# petit copier coller de baseline.py pour créer le dico groun_truth

##########################
# load some of the files #
##########################

training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)

training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)

test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)

################################
# create some handy structures #
################################

# convert training set to dictionary
emails_ids_per_sender = {}
for index, series in training.iterrows():
    row = series.tolist()
    sender = row[0]
    ids = row[1:][0].split(' ')
    emails_ids_per_sender[sender] = ids

# save all unique sender names
all_senders = emails_ids_per_sender.keys()

# create address book with frequency information for each user
address_books = {}
i = 0
ground_truth = {}
for sender, ids in emails_ids_per_sender.iteritems():
    recs_temp = []
    for my_id in ids:
        recipients = training_info[training_info['mid'] == int(my_id)]['recipients'].tolist()
        recipients = recipients[0].split(' ')
        # keep only legitimate email addresses
        recipients = [rec for rec in recipients if '@' in rec]
        recs_temp.append(recipients)

        # store the ground truth for the loss function in a dict { 'id_of_the_mail': list(recipients) }
        ground_truth[my_id] = recipients

    # flatten
    recs_temp = [elt for sublist in recs_temp for elt in sublist]
    # compute recipient counts
    rec_occ = dict(Counter(recs_temp))
    # order by frequency
    sorted_rec_occ = sorted(rec_occ.items(), key=operator.itemgetter(1), reverse=True)
    # save
    address_books[sender] = sorted_rec_occ

    if i % 10 == 0:
        print i
    i += 1


# Fin copié-collé
# Test du score sur e train en mode brutasse

total_score = test_score_on_train(path_to_results, ground_truth, option='frequency')
print 'Total score =', total_score



