import matplotlib.pyplot as plt
import numpy as np
from estimation_param import *
from whiteness_test import *
import time
from scipy.signal import correlate

import numpy as np
# intercorr en 2d
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
    length = 20000  # Nb d'echantillons
    T = 1  # période d'échantillonage
    x_0 = np.array([0, 0])  # Vecteur initial
    n = 0.05  # Pour fixer q = n*9.81*T
    q= n*9.81*T
    start = time.perf_counter()
    X_MRU = MRU_gen(length, T, x_0, n)
    end = time.perf_counter()
    print(end-start)
    Y_MRU = MRU_gen(length, T, x_0, n)
    x_coords_MRU = X_MRU[:, 0]
    x_vits_MRU = X_MRU[:,1]
    
    
    vit_est_x_MRU, acc_est_x_MRU = estimate(x_coords_MRU)
    end = time.perf_counter()
    print(end-start)
    
    
    correlation_x_MRU = np.correlate(acc_est_x_MRU, acc_est_x_MRU,'full')[len(acc_est_x_MRU)-1 :]
    lags_x_MRU = np.arange(-len(acc_est_x_MRU) +1 , len(acc_est_x_MRU))
    
    
    plt.figure(figsize=(14, 6))
    
    plt.plot(vit_est_x_MRU,"x", label='Estimation')
    plt.plot(x_vits_MRU, label='Réelle')
    plt.legend()
    plt.title("Vitesse réelle et vitesse estimée")
    plt.grid()
    plt.show()
    plt.figure(figsize=(14, 6))
    plt.plot(vit_est_x_MRU[0:50],"x", label='Estimation')
    plt.plot(x_vits_MRU[0:50], label='Réelle')
    plt.legend()
    plt.title("Vitesse réelle et vitesse estimée (zoom)")
    plt.grid()
    plt.show()
    
    test_corr= correlate(acc_est_x_MRU, acc_est_x_MRU, mode="full", method="fft") / len(acc_est_x_MRU)
    R_th = np.zeros_like(test_corr)
    R_th[len(test_corr)//2]=2/3 * n * 9.81/T
    R_th[len(test_corr) // 2 -1] = 1/6 * n * 9.81/T
    R_th[len(test_corr) // 2 +1] = 1/6 * n * 9.81/T
    plt.figure(figsize=(10, 6))
    
    #utilisé ma correlation et revoir tous ca !!!
    plt.plot(np.linspace(-10,10,num=21),test_corr[len(test_corr)//2 -10 : len(test_corr)//2 +11],label = "Générer")
    plt.plot(np.linspace(-10,10,num=21),R_th[len(R_th)//2 -10 : len(R_th)//2 +11],'x',label = "Théorique")
    plt.legend()
    plt.title("Fonction de corrélation de l'accélération avec length = {} et q={}".format(length,q))
    
    plt.grid()
    plt.show()
    
    length = len(X_MRU[:,0])
    sigma_n = 10
    noise = np.sqrt(sigma_n)*np.random.randn(length,)
    x_cood_bruit= X_MRU[:,0]+ noise
    vit_est_x_MRU_b, acc_est_x_MRU_b = estimate(x_cood_bruit)
    test_corr= correlate(acc_est_x_MRU_b, acc_est_x_MRU_b, mode="full", method="fft") / len(acc_est_x_MRU_b)
    R_th = np.zeros_like(test_corr)
    R_th[len(test_corr)//2]=2/3 * n * 9.81/T + 6/T**4 * sigma_n
    R_th[len(test_corr) // 2 -1] = 1/6 * n * 9.81/T -4/T**4 * sigma_n
    R_th[len(test_corr) // 2 +1] = 1/6 * n * 9.81/T -4/T**4 * sigma_n
    R_th[len(test_corr) // 2 -2] = 1/T**4 * sigma_n
    R_th[len(test_corr) // 2 +2] = 1/T**4 * sigma_n
    plt.figure(figsize=(10, 6))
    
    #utilisé ma correlation et revoir tous ca !!!
    plt.plot(np.linspace(-10,10,num=21),test_corr[len(test_corr)//2 -10 : len(test_corr)//2 +11],label = "Générer")
    plt.plot(np.linspace(-10,10,num=21),R_th[len(R_th)//2 -10 : len(R_th)//2 +11],'x',label = "Théorique")
    plt.legend()
    plt.title("Fonction de corrélation de l'accélération bruitée avec \n length = {:.4f},sigma_n = {:.4f} et q={:.4f} ".format(length,sigma_n,q))
    
    




    #test model ar ??

