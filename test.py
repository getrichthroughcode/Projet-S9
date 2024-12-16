from cmath import log10
import numpy as np
from MUA_gen import *
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
    x_0 = np.array([0,0])  # Vecteur initial
    n = 0.05  # Pour fixer q = n*9.81*T
    # SNR_list = [500, 400, 300, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 9, 8, 7, 6, 5]
    SNR_list = np.linspace(500,400,10)
    length_list = np.linspace(10000,20000,10,dtype='int')
    # length_list = [20000, 15000, 10000, 5000, 1000, 500, 100, 50, 10]
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
                print(i)
                X_MUA = MRU_gen(length, T, x_0, n)
                sigma_c,bruit = bruitage(X_MUA[:,0], SNR_list[SNR_id])
                erreur_SNR = SNR_list[SNR_id] - 10*np.log10(np.mean(X_MUA[:,0]**2)/np.mean(bruit**2))
                sigma_th = np.array([n*9.81*T,sigma_c])
                sigma = MRU_param_estimation(X_MUA[:,0] + bruit,T)


                erreur_est= (np.abs(sigma_th - sigma)/sigma_th)*100

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
#Load
test_accu =np.load("accu_plot_fin.npy")
test_mean_all_v1 = np.mean(test_accu,axis=3)
test_mean_all_fin = np.mean(test_mean_all_v1,axis=2)

#%%
# SNR_list = [500,400,300,200,100,90,80,70,60,50,40,30,20,10,9,8,7,6,5]
# length_list = [20000,15000,10000,5000,1000,500,100,50,10]
from mpl_toolkits.mplot3d import Axes3D
SNR, Length = np.meshgrid(SNR_list, length_list)

mean_all_fin_log = np.log10(mean_all_fin)#5%
# mean_all_fin_log = np.log10(mean_sig_bruit)<(1)


SNR_list_log = np.log10(SNR_list)
length_list_log = np.log10(length_list)

# Création de la heatmap
plt.figure(figsize=(10, 8))
plt.imshow(np.flip(np.flip(mean_all_fin_log,axis=1),axis=0), aspect='auto', cmap='seismic', origin='lower',
           extent=[SNR_list_log.min(), SNR_list_log.max(), length_list_log.min(), length_list_log.max()])#gist_ncar

# Ajuster les ticks des axes pour afficher les valeurs originales
plt.colorbar(label='log10(mean_all_fin)')
plt.xticks(SNR_list_log, labels=SNR_list, rotation=45)
plt.yticks(length_list_log, labels=length_list)
plt.xlabel('SNR (log10)')
plt.ylabel('Length (log10)')
plt.title('Heatmap: log10(mean_sig_bruit) with log-scaled axes')

# Affichage
plt.show()