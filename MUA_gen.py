import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import scipy.ndimage as S


def estimate(X_pos):
    X_vit=[0]
    X_acc=[0,0]
    vit = (X_pos[1] - X_pos[0]) / 2
    X_vit.append(vit)
    for i in range(2,len(X_pos)):
        vit = (X_pos[i] - X_pos[i-1]) /1
        X_vit.append(vit)
        acc = (X_pos[i] - 2*X_pos[i-1] +X_pos[i-2]) /1
        X_acc.append(acc)
    return (X_vit,S.median_filter(X_acc,size = 3))

def MUA_gen(length, T, x_0,n):
    L=[]
    # n=0.0005
    q=n*9.81*T
    if (np.shape(x_0) != (3, 1)):
        return []
    L.append(x_0)  # Ensure x_0 is a column vector
    Q = q* np.array([
        [T**5 / 20, T**4 / 8, T**3 / 6],
        [T**4 / 8, T**3 / 3, T**2 / 2],
        [T**3 / 6, T**2 / 2, T]
    ])


    phi = np.array([[1, T, T ** 2 / 2],
                    [0, 1, T],
                    [0, 0, 1]])
    for i in range(length):
        U = np.random.randn(3, 1)  # Generate a random vector
        R = np.linalg.cholesky(Q)  # Cholesky decomposition
        B = R @ U         # Generate the noise vector
        # Update x with the new state
        x_new  =phi @ L[-1] + B
        L.append(x_new)

    return L