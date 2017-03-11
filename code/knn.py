# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity



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




class knn_predictor():
    """ A class with a scikit -like API .fit() et .predict() method

    """
    def __init__(self, sender):
        self.sender = sender

    def fit_predict(self, bow_train, bow_test, training_info_S, test_info_S,  K=30):
        """
        Calculate for each dev message, it's K-nns in the train. Then compute similarity score for each user involved in the K-nns
         and take the 10 first user with the best score. A lot of the work is done in the function Knn().
        :param X_train_S:
        :param X_dev_S:
        :param training_info:
        :param test_info:
        :param K:
        :param use_idf:
        :param sublinear_tf:
        :return:
        """
        # compute K-nn for each message m in the test set

        test_knn = Knn(bow_train, bow_test, training_info_S, test_info_S, K=K)
        test_knn['mid'] = test_knn['mid'].astype(int) # convert the mid column into int type for submission

        return test_knn

    def build_prediction_dictionnary(self, predictions_per_sender, test_knn):
        """
        Take the dictionnary of the prediction and fill it with the new prediction in the right way for the submission function
        :param predictions_per_sender: a dictionnary that will be filled with a new key (self.sender) that contain predictions on the test_set for self.sender
        :param test_knn: the data frame ('mid', 'recipients') returned by the method fit_predict()
        :return: nothing ---> predictions_per_sender : the input dictionnary with the new key and the preiction for each message is modified within th func
        """

        predictions_per_sender[self.sender] = []
        for (mid, pred) in zip(test_knn['mid'].values, test_knn['recipients'].values):
            predictions_per_sender[self.sender].append([mid, pred])

