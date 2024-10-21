import numpy as np
import os

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

def estimate(X_pos):
    X_vit=[0]
    X_acc=[0,0]
    vit = (X_pos[1] - X_pos[0]) / 2
    X_vit.append(vit)
    for i in range(2,len(X_pos)):
        vit = (X_pos[i] - X_pos[i-1]) /2
        X_vit.append(vit)
        acc = (X_pos[i] - 2*X_pos[i-1] +X_pos[i-2]) /4
        X_acc.append(acc)
    return (X_vit,X_acc)

def MUA_gen(length, T, x_0):
    L=[]
    L.append(x_0)  # Ensure x_0 is a column vector
    Q = np.array([
        [T**5 / 20, T**4 / 8, T**3 / 6],
        [T**4 / 8, T**3 / 3, T**2 / 2],
        [T**3 / 6, T**2 / 2, T]
    ])
    print("The white-noise jerk model is used")

    phi = np.array([[1, T, T ** 2 / 2],
                    [0, 1, T],
                    [0, 0, 1]])
    for i in range(length):
        U = np.random.randn(3, 1)  # Generate a random vector
        R = np.linalg.cholesky(Q)  # Cholesky decomposition
        B = R.T @ U         # Generate the noise vector
        # Update x with the new state
        x_new  =phi @ L[-1] + B
        L.append(x_new)

    return L
