#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:50:30 2017

@author: estelleaflalo
"""
import re
import string
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cosine
from init import split, init_dic, csv_to_sub
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import sys
from paths import path # on a besoin de path_to_code pour pouvoir importer paths.py, le serpent se mort la queue :D
from gensim.models.word2vec import Word2Vec

path_to_code, path_to_data, path_to_results = path("estelle")

#path_to_code, path_to_data, path_to_results = path("victor")


sys.path.append(path_to_code)

path_to_wv ='/Users/estelleaflalo/Desktop/M2_Data_Science/Second_Period/Text_and_Graph/Cours5/Lab/for_moodle/code/'
path_to_stopwords = '/Users/estelleaflalo/Desktop/M2_Data_Science/Second_Period/Text_and_Graph/Cours5/Lab/for_moodle/data/'

##########################
# load some of the files #
##########################

training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)

training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)

test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)

test_info = pd.read_csv(path_to_data + 'test_info.csv', sep=',', header=0)

print "Building dictionnaries"

_, all_senders, _, address_books, _ = init_dic(training, training_info)

X_train, Y_train, X_test = csv_to_sub(training, training_info, test, test_info)




# returns the vector of a word

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
    


# remove dashes and apostrophes from punctuation marks 
punct = string.punctuation.replace('-', '').replace("'",'')
# regex to match intra-word dashes and intra-word apostrophes
my_regex = re.compile(r"(\b[-']\b)|[\W_]")


#################
### functions ###
#################
with open(path_to_stopwords + 'smart_stopwords.txt', 'r') as my_file: 
    stpwds = my_file.read().splitlines()
def my_vector_getter(word, wv):
    try:
		# we use reshape because cosine similarity in sklearn now works only for multidimensional arrays
        word_array = wv[word].reshape(1,-1)
        return (word_array)
    except KeyError:
        print 'word: <', word, '> not in vocabulary!'

# returns cosine similarity between two word vectors
def my_cos_similarity(word1, word2, wv):
    sim = cosine(my_vector_getter(word1, wv),my_vector_getter(word2, wv)) 
    return (round(sim, 4))



# performs basic pre-processing
# note: we do not lowercase for consistency with Google News embeddings
def clean_string(string, punct=punct, my_regex=my_regex):
    # remove formatting
    str = re.sub('\s+', ' ', string)
	# remove punctuation (preserving dashes)
    str = ''.join(l for l in str if l not in punct)
    # remove dashes that are not intra-word
    str = my_regex.sub(lambda x: (x.group(1) if x.group(1) else ' '), str)
    # strip extra white space
    str = re.sub(' +',' ',str)
    # strip leading and trailing white space
    str = str.strip()
    return str
    
    
#newsgroups = fetch_20newsgroups()

content_train = training_info.as_matrix()[:, 2]
content_test=test_info.as_matrix()[:, 2]
content=np.hstack((content_train, content_test))
docs= content.tolist()
lists_of_tokens = []
for i, doc in enumerate(docs):
        # clean document with the clean_string() function
    clean_string(doc)
        #### your code here ####
        # tokenize (split based on whitespace)
    tokens = doc.split(' ')
        # remove stopwords
    tokens = [token for token in tokens if token not in stpwds]
        # remove tokens less than 2 characters in size
    tokens = [token for token in tokens if len(token)>=2]
        # save result
    lists_of_tokens.append(tokens)
    if i%10 == 0:
        print i, 'emails processed'
        



    # create empty word vectors for the words in vocabulary	
    # we set size=300 to match dim of GNews word vectors
mcount = 5 #contrainte pr que le mot apparaisse dans le vocab
emb=300 #taille embedding
vector = Word2Vec(size=emb, min_count=mcount)   
vector.build_vocab(lists_of_tokens)  
vocab = [elt[0] for elt in vector.wv.vocab.items()] 
    # we load only the Google word vectors corresponding to our vocabulary
vector.intersect_word2vec_format(path_to_wv + 'GoogleNews-vectors-negative300.bin', binary=True)
       



predictions_per_sender = {}

# set the hyper-parameters like : use_id, etc...
use_idf = True
print 'Parameter use_idf is set to {}'.format(use_idf)
K=30

print 'parameter K is set to {}'.format(K)


for p in range(len(all_senders)):

    # Select a sender S
    index = p
    sender = all_senders[index]
    X_train_S = X_train[sender]
    X_dev_S = X_test[sender]
    Y_train_S = Y_train[sender]


    



    ##############Create TF IDF vector from mails sent by sender S


    # vectorize mails sent by a unique sender
    
    # train
    training_info_S = training_info.loc[training_info['mid'].isin(X_train_S)]
    training_info_S = training_info_S.set_index(np.arange(len(training_info_S)))
    training_info_S_mat = training_info_S.as_matrix()
    content_train = training_info_S_mat[:, 2]
    lists_of_tokens_S = []
    docs= content_train.tolist()
    for i, doc in enumerate(docs):
            # clean document with the clean_string() function
        clean_string(doc)
            #### your code here ####
            # tokenize (split based on whitespace)
        tokens = doc.split(' ')
            # remove stopwords
        tokens = [token for token in tokens if token not in stpwds]
            # remove tokens less than 2 characters in size
        tokens = [token for token in tokens if len(token)>=2]
            # save result
        lists_of_tokens_S.append(tokens)
        if i%10 == 0:
            print i, 'emails processed'
            

    bow_train=np.zeros((X_train_S.shape[0],emb))
    for k in range(len(lists_of_tokens_S)):
        temp=np.zeros((1,emb))
        count=0
        for word in lists_of_tokens_S[k]:
            if word in vocab:
                temp+=my_vector_getter(word, vector)
            #print my_vector_getter(word, vec_train)
                count+=1
            else:pass
        if count==0:
            bow_train[k]=np.zeros(emb)
        else:bow_train[k]=temp/count
    print "Word2Vec train done"

    # test
    test_info_S = test_info.loc[test_info['mid'].isin(X_dev_S)]
    test_info_S = test_info_S.set_index(np.arange(len(test_info_S)))
    test_info_S_mat = test_info_S.as_matrix()
    content_test = test_info_S_mat[:, 2]

    docs= content_test.tolist()

    lists_of_tokens_S = []
    docs= content_test.tolist()
    for i, doc in enumerate(docs):
            # clean document with the clean_string() function
        clean_string(doc)
            #### your code here ####
            # tokenize (split based on whitespace)
        tokens = doc.split(' ')
            # remove stopwords
        tokens = [token for token in tokens if token not in stpwds]
            # remove tokens less than 2 characters in size
        tokens = [token for token in tokens if len(token)>=2]
            # save result
        lists_of_tokens_S.append(tokens)
        if i%10 == 0:
            print i, 'emails processed'
            

    bow_test=np.zeros((X_dev_S.shape[0],emb))
    for k in range(len(lists_of_tokens_S)):
        temp=np.zeros((1,emb))
        count=0
        for word in lists_of_tokens_S[k]:
            if word in vocab:
                temp+=my_vector_getter(word, vector)
            #print my_vector_getter(word, vec_train)
                count+=1
            else:pass
        if count==0:
            bow_test[k]=np.zeros(emb)
        else:bow_test[k]=temp/count
    print "Word2Vec test done"
            
    # compute K-nn for each message m in the test set


    # training_info_S['recipients'][training_info_S['mid']==392289].tolist()[0].split(' ')

    test_knn = Knn(bow_train, bow_test, training_info_S, test_info_S, K=K)
    test_knn['mid'] = test_knn['mid'].astype(int)


    # add a entry corresponding to the sendr in the dictionnary
    predictions_per_sender[sender] = []
    for (mid, pred) in zip(test_knn['mid'].values,test_knn['recipients'].values):
        predictions_per_sender[sender].append([mid, pred])

    print "Sender Number : " + str(p)


c=0 # compteur : a priori faut que ce soit 2362

with open(path_to_results + 'predictions_knn_with_word2vec.txt', 'wb') as my_file:

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
