
import numpy as np

def MRU_gen(length, T, x_0,n):
    L=[]
    q=n*9.81*T
    if(np.shape(x_0)!=(2,1)):
        return []
    L.append(x_0)  # Ensure x_0 is a column vector
    Q = q* np.array([
        [T**3 / 3, T**2 / 2],
        [T**2 / 2, T]
    ])
    phi = np.array([[1, T],
                    [0, 1]
                    ])
    for i in range(length):
        U = np.random.randn(2, 1)  # Generate a random vector
        R = np.linalg.cholesky(Q)  # Cholesky decomposition
        B = R.T @ U         # Generate the noise vector
        # Update x with the new state
        x_new  =phi @ L[-1] + B
        L.append(x_new)

    return L

