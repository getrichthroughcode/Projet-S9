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
            
def operation_csv_folder(path_folder, name_file, nb_file, func, list_name):
    for i in range(nb_file):
        path_file = path_folder + '/' + path_file + "_{i}.csv"
        return
        
        
if __name__ == "__main__":
    print(read_csv("Singer","0_Singer",['aX','Title']))
    ans = read_csv("2D_MUA","0_MUA",['aX','Title'])
    mean = 0
    for i in range(len(ans[0])):
        print(float(ans[0][i]))
        float(ans[0][i])
    print(mean/len(ans[0]))
    
    
    