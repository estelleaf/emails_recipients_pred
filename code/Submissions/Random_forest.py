# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def get_all_recs_per_sender(train_info_S):
    """
    For a fixed sender, return the list of all the recipients of this sender in the training set.
    :param train_info_S: A structure like train_info, but for one sender only, since we work sender by sender
    :return: a list of all the recipients for the current sender
    """
    all_recs = []
    for recs in train_info_S['recipients'].values:
        all_recs.extend(recs.split(' '))

    all_recs = list(set(all_recs))
    return all_recs



class Random_forest_predictor:
    """ Class to predict recipients for one sender.
    Based on scikit RandomForestRegressor (we tried classifier but regressor gives
     better performance on a validation set.
     """

    def __init__(self, sender):
        self.sender = sender



    def fit_predict_build_pred_dictionnary(self, training_info_S, content_train, test_info_S, bow_train, bow_test,
                                           n_estimators, max_depth, n_jobs, predictions_per_sender_RF ):
        """
        Method 'all-in-one', we fit/predict/create a new entry in the dictionnary : 'prediction_per_sender_RF'
        :param training_info_S: A dataframe similar to train_info, but for one sender only, since we work sender by sender
        :param content_train: A np array of shape (n_mail,) containing the content of each mail in one single string
        :param test_info_S: A dataFrame similar to test_info, but for one sender only, since we work sender by sender
        :param bow_train: Dense numpy matrix which contain, for each mail in the train, it's TfIdf vector
        :param bow_test: Dense numpy matrix which contain, for each mail in the test, it's TfIdf vector
        :param n_estimators: scikit argument of random forest
        :param max_depth: scikit argument of random forest
        :param n_jobs: scikit argument for parralelism
        :param predictions_per_sender_RF: A dictionnary for which we will create a new entry with key : self.sender

        :return: Nothing, the dict is modified inside the method
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
        #RF = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features='log2', n_jobs=n_jobs)

        RF.fit(bow_train, y_train)
        y_test = RF.predict(bow_test).reshape((n_test, n_class))

        #sanity check : if n_recipient is <10 from the begining (e.g. p=37 has only one correspondant)
        if n_class < 10:
            # we can't take the 10 best cause there are less than 10 recs in the train
            if n_class == 1:
                print n_class
                predictions_per_sender_RF[self.sender] = []
                ind = np.argsort(y_test, axis=1)[:, -1][:, None]
                print ind
                for i, mid in enumerate(test_info_S['mid'].astype(int)):
                    predictions_per_sender_RF[self.sender].append([ mid, [ all_recs_S[j] for j in ind[i, :] ]])
            else:
                predictions_per_sender_RF[self.sender] = []
                ind = np.argsort(y_test, axis=1)[:, -n_class:].reshape((n_test, n_class))
                for i, mid in enumerate(test_info_S['mid'].astype(int)):
                    predictions_per_sender_RF[self.sender].append([mid, [all_recs_S[j] for j in ind[i, :]]])
        else:
            #do the normal way, take te 10 best
            ind = np.argsort(y_test, axis=1)[:, -10:]

            predictions_per_sender_RF[self.sender] = []
            for i, mid in enumerate(test_info_S['mid'].astype(int)):
                predictions_per_sender_RF[self.sender].append([ mid, [ all_recs_S[j] for j in ind[i, :] ]])