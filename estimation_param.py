# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:54:39 2024

@author: slienard
"""

import numpy as np
from MRU_gen import *
from MUA_gen import MUA_gen

def estimate(X_pos):
    X_vit=[]
    X_acc=[]
    # inte = L.T@np.random.randn(3, 1).tolist()
    X_pos = X_pos
    for i in range(0,len(X_pos)-2):
        # inte = np.sqrt(9.81) * L.T @ np.random.randn(3, 1).tolist()
        vit = float((X_pos[i+1] - X_pos[i]) /1 )
        X_vit.append(vit)
        acc = float((X_pos[i+2] - 2*X_pos[i+1]+X_pos[i]) /1 )
        X_acc.append(acc)
    X_vit.insert(0,(X_pos[1] - X_pos[0]) / 1)
    X_vit.append((X_pos[-1] - X_pos[-2]) / 1)
    X_acc.insert(0,(X_pos[2] - 2 * X_pos[1] + X_pos[0]) / (1**2))
    X_acc.append((X_pos[-1] - 2 * X_pos[-2] + X_pos[-3]) / (1**2))
    return (X_vit,X_acc) #S.median_filter()

def MRU_param_estimation(X_pos,T):
    vit_est,acc_est = estimate(X_pos)
    corr = np.correlate(acc_est, acc_est, "full") / len(acc_est)
    mat_det_sig = np.array([[46/(3*T**9),133/(6*T**9),-10/(3*T**9)], [11/(18*T**6),-22/(9*T**6),17/(36*T**6)]])
    sigma = 36*T**10/501 * mat_det_sig@(corr[len(corr)//2:len(corr)//2 +3 ]).T
    return sigma

def MUA_param_estimation(X_pos,T):
    vit_est,acc_est = estimate(X_pos)
    jerk_est,_ = estimate(acc_est)
    corr = np.correlate(jerk_est, jerk_est, "full") / len(jerk_est)
    A = np.array([[33/60*1/(T), 20/(T**9)],[13/60*1/T, -15/(T**9)],[1/120*1/T, 6/(T**9)],[0, 1/(T**9)]])
    sigma = np.linalg.inv(A.T@A)@A.T@(corr[len(corr)//2:len(corr)//2 +4]).T
    return sigma



