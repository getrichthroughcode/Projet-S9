# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 10:17:45 2024

@author: slienard
"""
import numpy as np

def calc_autocorr(input):
    a = np.array(input, dtype = 'float64')
    print(a)
    
calc_autocorr([1.01,1.00003])