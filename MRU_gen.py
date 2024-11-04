import matplotlib.pyplot as plt
import numpy as np
from MUA_gen import estimate
from whiteness_test import test_1
from whiteness_test import test_2

def MRU_gen(length, T, x_0,n):
    L=[]
    q=n*9.81*T
    if(np.shape(x_0)!=(2,1)):
        print("x_0 has not the shape (2,1)")
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

if __name__ == "__main__":
    length = 999  # Nb d'echantillons
    T = 1  # période d'échantillonage
    x_0 = np.array([[0], [0]])  # Vecteur initial
    n = 0.005  # Pour fixer q = n*9.81*T
    X_MRU = MRU_gen(length, T, x_0, n)
    Y_MRU = MRU_gen(length, T, x_0, n)
    x_coords_MRU = [xi[0, 0] for xi in X_MRU]
    y_coords_MRU = [yi[0, 0] for yi in Y_MRU]

    vit_est_x_MRU, acc_est_x_MRU = estimate(x_coords_MRU)
    vit_est_y_MRU, acc_est_y_MRU = estimate(y_coords_MRU)
    correlation_x_MRU = np.correlate(acc_est_x_MRU, acc_est_x_MRU,'same')
    correlation_y_MRU = np.correlate(acc_est_y_MRU, acc_est_y_MRU,'same')
    lags_x_MRU = np.arange(-len(acc_est_x_MRU)/2 , len(acc_est_x_MRU)/2)
    lags_y_MRU = np.arange(-len(acc_est_y_MRU)/2, len(acc_est_y_MRU)/2)
    correlation_x_MRU_dec = np.roll(correlation_x_MRU, int(len(correlation_x_MRU) / 2 ))
    correlation_y_MRU_dec = np.roll(correlation_y_MRU, int(len(correlation_y_MRU) / 2 ))
    print(test_2(correlation_x_MRU_dec.tolist(),10))
    print(test_1(correlation_y_MRU_dec.tolist()))

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(lags_x_MRU, correlation_x_MRU)
    plt.title("Fonction de corrélation en x")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(lags_y_MRU, correlation_y_MRU)
    plt.title("Fonction de corrélation en y")
    plt.grid()

    # Display the combined plot
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()

