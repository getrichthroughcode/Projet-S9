# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:25:46 2024

@author: simon
"""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import PIL.Image as Image

#100
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),#50
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),#25
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2),#12
            nn.ReLU())
        self.fc = nn.Linear(12 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


#%% Réaliser un dataset


















