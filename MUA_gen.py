import numpy as np

def MUA_gen(length, T, x_0, WNJ_WSA):
    L=[]
    L.append(x_0)  # Ensure x_0 is a column vector

    if WNJ_WSA:
        Q = np.array([
            [T**5 / 20, T**4 / 8, T**3 / 6],
            [T**4 / 8, T**3 / 3, T**2 / 2],
            [T**3 / 6, T**2 / 2, T]
        ])
        print("The white-noise jerk model is used")
    else:
        Q = np.array([
            [T**4 / 4, T**3 / 2, T**2 / 2],
            [T**3 / 2, T**2 / 2, T],
            [T**2 / 2, T, 1]
        ])
        print("The Wiener-sequence acceleration model is used")

    for i in range(length):
        U = np.random.randn(3, 1)  # Generate a random vector
        R = np.linalg.cholesky(Q)  # Cholesky decomposition
        B = R.T@U         # Generate the noise vector
        print(B)
        # Update x with the new state
        phi = np.array([[1, T, T**2 / 2],
                                 [0, 1, T],
                                 [0, 0, 1]])
        x_new  =phi @ L[-1] + B
        L.append(x_new)

    return L


