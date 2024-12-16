import numpy as np
from estimation_param import *
import matplotlib.pyplot as plt
from whiteness_test import *
import mpl_toolkits.mplot3d
from scipy.signal import correlate
import scipy.signal as ss
# T=1
# n=1
# length = 100
# q = n*9.81*T
# Q = q* np.array([
#         [T**5 / 20, T**4 / 8, T**3 / 6],
#         [T**4 / 8, T**3 / 3, T**2 / 2],
#         [T**3 / 6, T**2 / 2, T]
#     ])
# L = np.linalg.cholesky(Q)
#
# # Extraire les coefficients de L
# l11 = L[0, 0]
# l21 = L[1, 0]
# l22 = L[1, 1]
# l31 = L[2, 0]
# l32 = L[2, 1]
# l33 = L[2, 2]


def MUA_gen(length, T, x_0, n):
    # Vérification de la forme de x_0
    if x_0.shape != (3,):
        print("x_0 has not the shape (3,)")
        return np.empty((3,))  # Retourne un tableau vide si la condition échoue

    # Préallocation pour un tableau contenant toutes les étapes
    L = np.zeros((length + 1, 3))  # Préallouer un tableau pour tous les états
    L[0] = x_0  # Initialisation avec l'état initial

    # Paramètres de la simulation
    q = n * 9.81 * T
    Q = q * np.array([
        [T**5 / 20, T**4 / 8, T**3 / 6],
        [T**4 / 8, T**3 / 3, T**2 / 2],
        [T**3 / 6, T**2 / 2, T]
    ])
    phi = np.array([
        [1, T, T**2 / 2],
        [0, 1, T],
        [0, 0, 1]
    ])

    # Boucle principale
    R = np.linalg.cholesky(Q)  # Cholesky decomposition (invariant dans la boucle)
    for i in range(length):
        U = np.random.randn(3,)  # Génère un vecteur aléatoire
        B = R @ U  # Génère le bruit
        L[i + 1] = phi @ L[i] + B  # Calcule le nouvel état

    return L
#afficher vitesse instantanée (fleches)
if __name__ == "__main__":
    n=1
    T=1
    q=n*T*9.81
    length = 30000
    x_0=np.array([0,0,0])
    x=MUA_gen(length, T, x_0,n)

    x_coords = x[:,0]
    y_coords = y[:,0]
    x_accs = x[:,2]
    x_vits = x[:,1]

    plt.plot(x_coords)
    plt.title("Trajectoire uniformément accélérrer avec\n length = {:.4f} et q={:.4f}".format(length,q))

    vit_est_x,acc_est_x = estimate(x_coords)

    jerk_est_x,_ = estimate(acc_est_x)
    test_corr = correlate(jerk_est_x, jerk_est_x, mode="full", method="fft") / len(jerk_est_x)
    R_th = np.zeros_like(test_corr)
    R_th[len(test_corr) // 2] = 33/60 * q/T
    R_th[len(test_corr) // 2 - 1] = 13/60 * q/T
    R_th[len(test_corr) // 2 + 1] = R_th[len(test_corr) // 2 - 1]
    R_th[len(test_corr) // 2 - 2] = 1/120 * q/T
    R_th[len(test_corr) // 2 + 2] = R_th[len(test_corr) // 2 - 2]

    plt.figure(figsize=(10, 6))
    # utilisé ma correlation et revoir tous ca !!!
    plt.plot(np.arange(-10,11),test_corr[len(test_corr)//2 -10 : len(test_corr)//2 +11], label="Générer")
    plt.plot(np.arange(-10,11),R_th[len(R_th)//2 -10 : len(R_th)//2 +11], 'x', label="Théorique")
    plt.legend()
    plt.title("Fonction de corrélation de l'accélération avec length = {} et q={}".format(length,q))

    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(acc_est_x,"x",label='Estimation')
    plt.plot(x_accs,label='Réelle')
    plt.legend()
    plt.title("Accélération réelle et accélération estimée")
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(acc_est_x[0:50],"x",label='Estimation')
    plt.plot(x_accs[0:50],label='Réelle')
    plt.legend()
    plt.title("Accélération réelle et accélération estimée (zoom)")
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(vit_est_x,"x", label='Estimation')
    plt.plot(x_vits, label='Réelle')
    plt.legend()
    plt.title("Vitesse réelle et vitesse estimée")
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(vit_est_x[0:50],"x", label='Estimation')
    plt.plot(x_vits[0:50], label='Réelle')
    plt.legend()
    plt.title("Vitesse réelle et vitesse estimée (zoom)")
    plt.grid()
    plt.show()
    
    length = len(X_MUA[:,0])
    sigma_n = 10
    noise = np.sqrt(sigma_n)*np.random.randn(length,)
    x_cood_bruit= X_MUA[:,0]+ noise
    vit_est_x_RUA_b, acc_est_x_MUA_b = estimate(x_cood_bruit)
    test_corr= correlate(acc_est_x_MUA_b, acc_est_x_MUA_b, mode="full", method="fft") / len(acc_est_x_MRU_b)
    R_th = np.zeros_like(test_corr)
    R_th[len(test_corr) // 2] = 33/60 * q/T + 20/T**9 * sigma_n
    R_th[len(test_corr) // 2 - 1] = 13/60 * q/T - 15/T**9 * sigma_n 
    R_th[len(test_corr) // 2 + 1] = R_th[len(test_corr) // 2 - 1]
    R_th[len(test_corr) // 2 - 2] = 1/120 * q/T + 6/T**9 * sigma_n 
    R_th[len(test_corr) // 2 + 2] = R_th[len(test_corr) // 2 - 2]
    R_th[len(test_corr) // 2 - 3] = 1/T**9 * sigma_n 
    R_th[len(test_corr) // 2 + 3] = R_th[len(test_corr) // 2 - 3]
    plt.plot(np.linspace(-10,10,num=21),test_corr[len(test_corr)//2 -10 : len(test_corr)//2 +11],label = "Générer")
    plt.plot(np.linspace(-10,10,num=21),R_th[len(R_th)//2 -10 : len(R_th)//2 +11],'x',label = "Théorique")
    plt.legend()
    plt.title("Fonction de corrélation de l'accélération bruitée avec \n length = {:.4f},sigma_n = {:.4f} et q={:.4f} ".format(length,sigma_n,q))
    
  
