from cmath import log10
import numpy as np
from MRU_gen import *
import time
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from estimation_param import *
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
def MRU_param_estimation(X_pos,T):
    vit_est,acc_est = estimate(X_pos)
    corr = np.correlate(acc_est, acc_est, "full") / len(acc_est)
    mat_det_sig = np.array([[46/(3*T**9),133/(6*T**9),-10/(3*T**9)], [11/(18*T**6),-22/(9*T**6),17/(36*T**6)]])
    sigma = 36*T**10/501 * mat_det_sig@(corr[len(corr)//2:len(corr)//2 +3 ]).T
    return sigma
def bruitage(S,SNR):

    length = len(S)
    noise = np.random.randn(length,)

    sigma_c = 10**(-SNR/10) * np.mean(S**2)/np.mean(noise**2)

    bruit = np.sqrt(sigma_c)*noise

    return sigma_c,bruit

if __name__ == '__main__':
    # length = 10000  # Nb d'echantillons
    T = 1  # période d'échantillonage
    x_0 = np.array([0, 0])  # Vecteur initial
    n = 0.05  # Pour fixer q = n*9.81*T
    SNR_list = np.arange(0, 1001, 10)
    length_list = np.arange(0,20000,500)
    N_moy = 100
    accu = np.zeros((len(length_list),len(SNR_list), N_moy,2))
    l = 0
    for length in length_list:

        for SNR_id in range(len(SNR_list)):
            print("Parametres")
            print(SNR_list[SNR_id])
            print(length)
            start = time.perf_counter()
            for i in range(N_moy):
                X_MRU = MRU_gen(length, T, x_0, n)
                sigma_c,bruit = bruitage(X_MRU[:,0], SNR_list[SNR_id])
                erreur_SNR = SNR_list[SNR_id] - 10*np.log10(np.mean(X_MRU[:,0]**2)/np.mean(bruit**2))
                sigma_th = np.array([n*9.81*T,sigma_c])
                sigma = MRU_param_estimation(X_MRU[:,0] + bruit,T)


                erreur_est= (np.abs(sigma_th - sigma)/sigma_th)

                accu[l,SNR_id,i,:]= erreur_est

            end = time.perf_counter()
            # print(end-start)

        l+=1


#%%

sig_traj = accu[:,:,:,0]
sig_bruit = accu[:,:,:,1]
mean_sig_traj= np.mean(sig_traj,axis=2)
mean_sig_bruit= np.mean(sig_bruit,axis=2)
mean_all_v1 = np.mean(accu,axis=3)
mean_all_fin = np.mean(mean_all_v1,axis=2)
#%%
test_accu =np.load("4_accu_l_SNR.npy")
test_mean_all_v1 = np.mean(test_accu,axis=3)
test_mean_all_fin = np.mean(test_mean_all_v1,axis=2)
#%%
from mpl_toolkits.mplot3d import Axes3D
# SNR, Length = np.meshgrid(SNR_list, length_list)

mean_all_fin_log = np.log10(mean_all_fin)<1

plt.figure(figsize=(10, 8))
plt.imshow(mean_all_fin_log, aspect='auto', origin='lower',
           extent=[min(length_list), max(length_list), min(SNR_list), max(SNR_list)],
           cmap='gist_ncar')

# Appliquer une échelle logarithmique sur les axes
plt.xscale('log')  # Échelle logarithmique pour l'axe SNR
plt.yscale('log')

# Ajouter des labels et une barre de couleur
plt.colorbar(label='Valeurs (z) de mean_all_fin')
plt.title('Heatmap de mean_all_fin (Échelle logarithmique) inf 10%')
plt.xlabel('lengt (log)')
plt.ylabel('SNR ')

# Personnalisation des ticks pour l'échelle logarithmique
plt.xticks(length_list, rotation=45)
plt.yticks(SNR_list)
plt.grid(False)
plt.tight_layout()
plt.show()