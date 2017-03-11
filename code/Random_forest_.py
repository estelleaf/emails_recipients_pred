# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor


def get_all_recs_per_sender(train_info_S):
    """ For a fixed sender, return the list of all the recipients of this sender in the training set"""
    all_recs = []
    for recs in train_info_S['recipients'].values:
        all_recs.extend(recs.split(' '))

    all_recs = list(set(all_recs))
    return all_recs



class Random_forest_predictor:
    """ Classe pour predire à l'aide de rendom forest. Basé sur le randomforestregressor de scikit """

    def __init__(self, sender):
        self.sender= sender



    def fit_predict_build_pred_dictionnary(self, training_info_S, content_train, test_info_S, bow_train, bow_test,
                                           n_estimators, max_depth, n_jobs, predictions_per_sender_RF ):
        """
        Method 'all in one' that fit predict and append the prediction dico
        :param training_info_S:
        :param content_train:
        :param bow_train:
        :param bow_test:
        :param n_estimators: param
        :param max_depth:
        :param n_jobs:
        :return:
        """
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
        RF = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features='log2', n_jobs=n_jobs)
        RF.fit(bow_train, y_train)
        y_test = RF.predict(bow_test).reshape((n_test, n_class))

        #sanity check : if n_recipient is <10 from the begining (e.g. p=37 has only one correspondant)
        if n_class < 10:
            # we can't tke the 10 bets cause there are less than 10 recs
            if n_class == 1:
                print n_class
                predictions_per_sender_RF[self.sender] = []
                ind = np.argsort(y_test, axis=1)[:, -1][:, None]
                print ind
                for i, mid in enumerate(test_info_S['mid'].astype(int)):
                    predictions_per_sender_RF[self.sender].append([ mid, [ all_recs_S[j] for j in ind[i, :] ]])
            else:
                predictions_per_sender_RF[self.sender] = []
                ind = np.argsort(y_test, axis=1)[:, n_class + 1:].reshape((n_test, n_class))
                for i, mid in enumerate(test_info_S['mid'].astype(int)):
                    predictions_per_sender_RF[self.sender].append([mid, [all_recs_S[j] for j in ind[i, :]]])
        else:
            #do the normal way, take te 10 best
            ind = np.argsort(y_test, axis=1)[:, -10:]

            predictions_per_sender_RF[self.sender] = []
            for i, mid in enumerate(test_info_S['mid'].astype(int)):
                predictions_per_sender_RF[self.sender].append([ mid, [ all_recs_S[j] for j in ind[i, :] ]])