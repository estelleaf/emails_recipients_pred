
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
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm



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

        return predictions_per_sender



if __name__ == '__main__':

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
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from numpy.linalg import norm
    from loss_function import score

    training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)

    training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)

    test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)

    test_info = pd.read_csv(path_to_data + 'test_info.csv', sep=',', header=0)

    print "Building dictionnaries"

    _, all_senders, _, address_books, _ = init_dic(training, training_info)

    X_train, Y_train, X_test = csv_to_sub(training, training_info, test, test_info)


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


    predictions_per_sender_knn = {} #initialise le dictionnary des prediction final pour knn
    predictions_per_sender_freq = {}#initialise le dictionnary des prediction final pour freq

    for p in range(len(all_senders)):
        # Select a sender S
        index = p
        sender = all_senders[index]
        X_train_S = X_train[sender]
        X_dev_S = X_test[sender]
        Y_train_S = Y_train[sender]


        vectorizer_sender = TfidfVectorizer(max_df=max_df, min_df = min_df, stop_words='english', use_idf=use_idf,
                                            sublinear_tf=sublinear_tf)

        # create BoW for train
        training_info_S = training_info.loc[training_info['mid'].isin(X_train_S)]
        training_info_S = training_info_S.set_index(np.arange(len(training_info_S)))
        training_info_S_mat = training_info_S.as_matrix()
        content_train = training_info_S_mat[:, 2]

        vec_train = vectorizer_sender.fit_transform(content_train)
        bow_train = vec_train.toarray()

        # Create BoW for test
        test_info_S = test_info.loc[test_info['mid'].isin(X_dev_S)]
        test_info_S = test_info_S.set_index(np.arange(len(test_info_S)))
        test_info_S_mat = test_info_S.as_matrix()
        content_test = test_info_S_mat[:, 2]

        vec_test = vectorizer_sender.transform(content_test)
        bow_test = vec_test.toarray()


        # on fait nos differend modeles : knn, frequency, centroids...

        # knn
        knn_predictor_ = knn_predictor(sender=sender)
        test_knn = knn_predictor_.fit_predict(bow_train, bow_test, training_info_S, test_info_S,  K=K)
        knn_predictor_.build_prediction_dictionnary(predictions_per_sender_knn, test_knn)


        # freq
        #TODO: adapter le code de baseline a nos nvelles stuctures de données (on peut aps copier coller)

        # centroid (domitille s'en charge ?)

        print "Sender Number : " + str(p)






