# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 08:32:10 2024

@author: slienard
"""

import csv


def read_csv(path_folder, path_file, list_name):
    out = [None] * len(list_name)
    path_file = path_folder + '/' + path_file + ".csv"
    with open(path_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            row = row[0]
            row = row.split(',')    
            if row[0] in list_name:
                out[list_name.index(row[0])] = row[1:]
    return out
            
            
        
if __name__ == "__main__":
    print(read_csv("2D_MUA","0_MUA",['X','Title']))