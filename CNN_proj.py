import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import PIL.Image as Image
import csv

def read_csv_file(path_folder, path_file, list_name):
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


#100
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),#100
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),#100
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2),#100
            nn.ReLU())
        self.fc = nn.Linear(100 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

import torch as t
import PIL.Image as Image
import torchvision.transforms as T
import os


class MNISTDataset(t.utils.data.Dataset):
    def __init__(self, dat_dir,nb_class):

        self.dat_dir = dat_dir
        self.num_classes = nb_class

        self.point_list = []
        self.label_list = []
        for i in range(self.num_classes):
            # path_cur = os.path.join(self.MNIST_dir,'{}'.format(i))
            # img_list_cur = os.listdir(path_cur)
            #
            # img_list_cur = [os.path.join('{}'.format(i), file) for file in img_list_cur]
            #
            # self.img_list += img_list_cur
            #
            # label_list_cur = [i] * len(img_list_cur)
            # self.label_list += label_list_cur

            path_cur = os.path.join(self.MNIST_dir, '{}'.format(i))
            img_list_cur = os.listdir(path_cur)

            img_list_cur = [os.path.join('{}'.format(i), file) for file in img_list_cur]

            self.img_list += img_list_cur

            label_list_cur = [i] * len(img_list_cur)
            self.label_list += label_list_cur

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):

        img_path = os.path.join(self.MNIST_dir, self.img_list[idx])

        I_PIL = Image.open(img_path)

        I = T.ToTensor()(I_PIL)

        return I, t.tensor(self.label_list[idx]), img_path