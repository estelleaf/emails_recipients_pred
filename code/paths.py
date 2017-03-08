#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:08:42 2017

@author: estelleaflalo
"""


def path(user):
    if user=="domitille":
        path_to_code = '/Users/domitillecoulomb/Documents/DATA_SCIENCE/Semester2/Text_Graph/text_and_graph/code'
        path_to_data = '/Users/domitillecoulomb/Documents/DATA_SCIENCE/Semester2/Text_Graph/text_and_graph/Data'
        path_to_results = '/Users/domitillecoulomb/Documents/DATA_SCIENCE/Semester2/Text_Graph/text_and_graph/Predictions/'
    
    
    if user=="nicolas":
        path_to_code = 'C:/Nicolas/M2 MVA/ALTEGRAD/Kaggle/text_and_graph/code'
        path_to_data = 'C:/Nicolas/M2 MVA/ALTEGRAD/Kaggle/text_and_graph/Data/'
        path_to_results = 'C:/Nicolas/M2 MVA/ALTEGRAD/Kaggle/text_and_graph/Predictions/'
    
    if user=="victor":
        pass
    
    if user=="estelle":
        path_to_code = "/Users/estelleaflalo/Desktop/M2_Data_Science/Second_Period/Text_and_Graph/Project/text_and_graph/code/"
        path_to_data= "/Users/estelleaflalo/Desktop/M2_Data_Science/Second_Period/Text_and_Graph/Project/text_and_graph/Data/"
        path_to_results= "/Users/estelleaflalo/Desktop/M2_Data_Science/Second_Period/Text_and_Graph/Project/text_and_graph/Predictions/"

    return path_to_code, path_to_data, path_to_results


