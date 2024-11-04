# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 08:32:10 2024

@author: slienard
"""

import csv


def read_csv(path_folder, path_file, list_name):
    out = []
    path_file = path_folder + '/' + path_file + ".csv"
    with open(path_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            row = row[0]
            row = row.split(',')
            if row[0] in list_name:
                print(row)
                out.append(row[1:])
                print(out)
    return out
            
            
        
if __name__ == "__main__":
    read_csv("2D_MUA","0_MUA",['X'])