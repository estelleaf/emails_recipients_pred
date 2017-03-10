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


def f(string_liste):
    res = ''
    for string in string_liste:
        res += string.split(' ')[0] +  '  '

    return res

print '1998', '1998' in f(test_info['date'])
print '1999', '1999' in f(test_info['date'])
print '2000', '2000' in f(test_info['date'])
print '2001', '2001' in f(test_info['date'])
print '2002', '2002' in f(test_info['date'])
print '2003', '2003' in f(test_info['date'])
