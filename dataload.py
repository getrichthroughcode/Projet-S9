# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 10:23:56 2024

@author: slienard
"""


def read_csv(path_folder, path_file, list_name):
    csv.field_size_limit(100000000)
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

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import PIL.Image as Image
import glob
from torch.utils.data import DataLoader

import zipfile
import os

import csv

print(read_csv("test/test","0_MRU",['X']))


class CustomDataset():
    def __init__(self):
        self.imgs_path = "test/test/"
        file_list = glob.glob(self.imgs_path + "*")
        self.data = []
        for class_path in file_list:
            data_list = read_csv("test/test/", class_path[10:-4], ['type_movement','X']) #TODO ameliorer les fonctions
            class_name = data_list[0][0]
            self.data.append([data_list[1], class_name])
        self.class_map = {"MRU" : 0, "MUA": 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        position = self.data[idx][0]
        class_name = self.data[idx][1]
        class_id = self.class_map[class_name]  # Utilisation correcte du mapping
        print(f"Item at {idx}:, {class_name} -> Class ID: {class_id}")
        return position, class_id




dataset = CustomDataset()

print(f"Length of dataset: {len(dataset)}")

dataset.__getitem__(0)  