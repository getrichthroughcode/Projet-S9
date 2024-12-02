import matplotlib.pyplot as plt
import numpy as np
from estimation_param import estimate
from whiteness_test import *


import numpy as np

def MRU_gen(length, T, x_0, n):
    # Vérification de la forme de x_0
    if x_0.shape != (2,):
        print("x_0 has not the shape (2,)")
        return np.empty((2,))  # Retourne un tableau vide si la condition échoue

    # Préallocation pour un tableau contenant toutes les étapes
    L = np.zeros((length + 1, 2))  # Préallouer un tableau pour tous les états
    L[0] = x_0  # Initialisation avec l'état initial

    # Paramètres de la simulation
    q = n * 9.81 * T
    Q = q * np.array([
        [T**3 / 3, T**2 / 2],
        [T**2 / 2, T]
    ])
    phi = np.array([
        [1, T],
        [0, 1]
    ])

    # Boucle principale
    R = np.linalg.cholesky(Q)  # Cholesky decomposition (invariant dans la boucle)
    for i in range(length):
        U = np.random.randn(2,)  # Génère un vecteur aléatoire
        B = R @ U  # Génère le bruit
        L[i + 1] = phi @ L[i] + B  # Calcule le nouvel état

    return L



if __name__ == "__main__":
    length = 10000  # Nb d'echantillons
    T = 1  # période d'échantillonage
    x_0 = np.array([0, 0])  # Vecteur initial
    n = 0.05  # Pour fixer q = n*9.81*T
    X_MRU = MRU_gen(length, T, x_0, n)
    Y_MRU = MRU_gen(length, T, x_0, n)
    x_coords_MRU = X_MRU[:, 0]
    # y_coords_MRU =
    #
    # x_vits_MRU = [xi[1, 0] for xi in X_MRU]
    # y_vits_MRU = [yi[1, 0] for yi in X_MRU]
    #
    vit_est_x_MRU, acc_est_x_MRU = estimate(x_coords_MRU)
    # vit_est_y_MRU, acc_est_y_MRU = estimate(y_coords_MRU)
    #
    # correlation_x_MRU = np.correlate(acc_est_x_MRU, acc_est_x_MRU,'full')[len(acc_est_x_MRU)-1 :]
    # correlation_y_MRU = np.correlate(acc_est_y_MRU, acc_est_y_MRU,'full')[len(acc_est_y_MRU)-1 :]
    # lags_x_MRU = np.arange(-len(acc_est_x_MRU) +1 , len(acc_est_x_MRU))
    # lags_y_MRU = np.arange(0, len(acc_est_y_MRU))
    # print(test_1(correlation_x_MRU.tolist()))
    # print(test_2(correlation_y_MRU.tolist(),5))

    # plt.figure(figsize=(14, 6))
    #
    # plt.plot(vit_est_x_MRU, label='Estimation')
    # plt.plot(x_vits_MRU, label='Réelle')
    # plt.legend()
    # plt.title("Vitesse réelle et vitesse estimée")
    # plt.grid()
    # plt.show()
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(x_coords_MRU, y_coords_MRU, label='Trajectoire (x, y)')
    # plt.quiver(x_coords_MRU, y_coords_MRU, x_vits_MRU, y_vits_MRU, angles='xy', scale_units='xy', scale=0.5, color='r',
    #            label='Vitesse instantanée réel')
    # plt.title('Trajectoire synthétique avec vitesse instantanée réel')
    # plt.xlabel('Position x')
    # plt.ylabel('Position y')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    test_corr = auto_coord(acc_est_x_MRU)
    R_th = np.zeros_like(test_corr)
    R_th[len(test_corr)//2]=2/3 * n * 9.81/T
    R_th[len(test_corr) // 2 -1] = 1/6 * n * 9.81/T
    R_th[len(test_corr) // 2 +1] = 1/6 * n * 9.81/T
    plt.figure(figsize=(10, 6))
    #utilisé ma correlation et revoir tous ca !!!
    plt.plot(test_corr[len(test_corr)//2 -10 : len(test_corr)//2 +10],label = "Générer")
    plt.plot(R_th[len(R_th)//2 -10 : len(R_th)//2 +10],'x',label = "Théorique")
    plt.legend()
    plt.title("Fonction de corrélation de l'accélération")

    plt.grid()
    plt.show()




    #test model ar ??

