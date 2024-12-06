from cmath import log10
import numpy as np
from MRU_gen import *
import time
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from estimation_param import *

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
    SNR_list = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5]
    length_list = [50000, 10000, 5000, 1000, 500, 100]
    N_moy = 100
    accu = np.zeros((len(length_list),len(SNR_list), N_moy,2))
    l = 0
    for length in length_list:

        for SNR_id in range(len(SNR_list)):

            start = time.perf_counter()
            for i in range(N_moy):
                if i%10==0:
                    print(i)
                X_MRU = MRU_gen(length, T, x_0, n)
                sigma_c,bruit = bruitage(X_MRU[:,0], SNR_list[SNR_id])
                erreur_SNR = SNR_list[SNR_id] - 10*np.log10(np.mean(X_MRU[:,0]**2)/np.mean(bruit**2))
                sigma_th = np.array([n*9.81*T,sigma_c])
                sigma = MRU_param_estimation(X_MRU[:,0] + bruit,T)


                erreur_est= (np.abs(sigma_th - sigma)/sigma_th)

                accu[l,SNR_id,i,:]= erreur_est

            end = time.perf_counter()
            print(end-start)

        l+=1


#%%
SNR_list = [1000,900,800,700,600,500,400,300,200,100,90,80,70,60,50,40,30,20,10,5]
length_list = [50000,10000,5000,1000,500,100]
sig_traj = accu[:,:,:,0]
sig_bruit = accu[:,:,:,1]
mean_sig_traj= np.mean(sig_traj,axis=2)
mean_sig_bruit= np.mean(sig_bruit,axis=2)
#%%
from mpl_toolkits.mplot3d import Axes3D
SNR, length = np.meshgrid(SNR_list, length_list)

SNR_flat = SNR.flatten()
length_flat = length.flatten()
mean_sig_traj_flat = mean_sig_traj.flatten()

# Création du graphique 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Tracé des points 3D
scatter = ax.scatter(SNR_flat, length_flat, mean_sig_traj_flat, c=mean_sig_traj_flat, cmap='viridis', s=50)

# Ajout des labels
ax.set_xlabel('SNR')
ax.set_ylabel('Length')
ax.set_zlabel('Mean Signal Trajectory')
ax.set_title('3D Scatter Plot of Mean Signal Trajectory')

# Ajout d'une barre de couleur
fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)

plt.show()
