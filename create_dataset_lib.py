# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:24:38 2024

@author: slienard
"""

from read_csv import read_csv
import os



def create_data(path_file, list_feature, name_target):
    a = 0
    for path in os.listdir(path_file):
        # check if current path is a file
        if os.path.isfile(os.path.join(path_file, path)):
            a += 1
            print('File count:', a)
    
    
        
if __name__ == "__main__":
    create_data('1D_MUA/', None, None)