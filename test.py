from cmath import log10
import numpy as np
from MUA_gen import *
from MRU_gen import *
import time
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from estimation_param import *
from whiteness_test import *
from scipy.signal import correlate
from Singer_gen import Singer_gen

def traj_choice(X,T):
    sigma_MRU_est = MRU_param_estimation(X,T)
    sigma_MUA_est = MUA_param_estimation(X,T)
    vit_est_x,acc_est_x = estimate(X)
    jerk_est_x,_ = estimate(acc_est_x)
    corr = correlate(acc_est_x, acc_est_x, mode="full", method="fft") / len(acc_est_x)
    corr_jerk = correlate(jerk_est_x, jerk_est_x, mode="full", method="fft") / len(jerk_est_x)
    R_th_MRU = np.zeros(7)
    q,sigma_n = sigma_MRU_est
    R_th_MRU[len(R_th_MRU)//2]=2/3 * q/T + 6/T**4 * sigma_n
    R_th_MRU[len(R_th_MRU) // 2 -1] = 1/6 * q/T -4/T**4 * sigma_n
    R_th_MRU[len(R_th_MRU) // 2 +1] = 1/6 * q/T -4/T**4 * sigma_n
    R_th_MRU[len(R_th_MRU) // 2 -2] = 1/T**4 * sigma_n
    R_th_MRU[len(R_th_MRU) // 2 +2] = 1/T**4 * sigma_n
    R_th_MUA = np.zeros(7)
    q,sigma_n = sigma_MUA_est
    R_th_MUA[len(R_th_MUA) // 2] = 33/60 * q/T + 20/T**9 * sigma_n
    R_th_MUA[len(R_th_MUA) // 2 - 1] = 13/60 * q/T - 15/T**9 * sigma_n 
    R_th_MUA[len(R_th_MUA) // 2 + 1] = R_th_MUA[len(R_th_MUA) // 2 - 1]
    R_th_MUA[len(R_th_MUA) // 2 - 2] = 1/120 * q/T + 6/T**9 * sigma_n 
    R_th_MUA[len(R_th_MUA) // 2 + 2] = R_th_MUA[len(R_th_MUA) // 2 - 2]
    R_th_MUA[len(R_th_MUA) // 2 - 3] = 1/T**9 * sigma_n 
    R_th_MUA[len(R_th_MUA) // 2 + 3] = R_th_MUA[len(R_th_MUA) // 2 - 3]
    corr_comp = corr[len(corr) // 2 -3 : len(corr) // 2+4]
    error_MRU = np.sum(np.abs(corr_comp-R_th_MRU))
    
    corr_norm = (corr_comp - min(corr_comp)) / (max(corr_comp) - min(corr_comp))
    
    rho_estime = Singer_param_estimation(X, T)
    alpha_estime = -np.log(rho_estime)/T
    E_estime =np.exp(-np.abs(np.arange(-3, 4)) * alpha_estime)
    normalized_E_est = (E_estime - min(E_estime)) / (max(E_estime) - min(E_estime))
    error_Singer = np.sum(np.abs(corr_norm-normalized_E_est))
    
    corr_comp_j = corr_jerk[len(corr_jerk) // 2 -3 : len(corr_jerk) // 2+4]
    error_MUA_j = np.sum(np.abs(corr_comp_j-R_th_MUA))
    print("MRU error {:.4f}".format(error_MRU))
    print("MUA error {:.4f}".format(error_MUA_j))
    print("Singer error {:.4f}".format(error_Singer))
    plt.figure(figsize=(10, 6))
    plt.plot(corr_norm,'o',label = "Générée")
    plt.plot(normalized_E_est,'x',label = "Th_sing")
    plt.legend()
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(corr_comp_j,'o',label = "Générée")
    plt.plot(R_th_MUA,'x',label = "Th_sing")
    plt.legend()
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(corr_comp,'o',label = "Générée")
    plt.plot(R_th_MRU,'x',label = "Th_sing")
    plt.legend()
    plt.show()
    if(min(error_MRU,error_MUA_j)==error_MRU):
        print("MRU avec les paramètres sigma = {:.4f} et sigma_n = {:.4f}".format(sigma_MRU_est[0],sigma_MRU_est[1]))
        # print(min(error_MRU,error_MUA_j))
        return 0,sigma_MRU_est[0],sigma_MRU_est[1]
    else:
        print("MUA avec les paramètres sigma = {:.4f} et sigma_n = {:.4f}".format(sigma_MUA_est[0],sigma_MUA_est[1]))
        # print(min(error_MRU,error_MUA_j))
        return 1,sigma_MUA_est[0],sigma_MUA_est[1]
    
    return
 
# length = 10000
# T = 1  # période d'échantillonage
# x_0 = np.array([0,0])  # Vecteur initial
# n = 5  # 
length = 30000  # Nb d'echantillons
T = 1  # période d'échantillonage
x_0 = np.array([[0], [0], [0]])  # Vecteur initial
alpha = 0.01
n = 0.05
sigma_m = n * 9.81 * T
sigma_n = 1
q11 = (1 / (2 * alpha ** 5)) * (
            2 * alpha * T - 2 * alpha ** 2 * T ** 2 + 2 * alpha ** 3 * T ** 3 / 3 - 4 * alpha * T * np.exp(
        -alpha * T) - np.exp(-2 * alpha * T) + 1)
q12 = (1 / (2 * alpha ** 4)) * (alpha ** 2 * T ** 2 + 1 + np.exp(-2 * alpha * T) + np.exp(-alpha * T) * (
        -2 + 2 * alpha * T) - 2 * alpha * T)
q13 = (1 / (2 * alpha ** 3)) * (1 - 2 * alpha * T * np.exp(-alpha * T) - np.exp(-2 * alpha * T))
q22 = (1 / (2 * alpha ** 3)) * (2 * alpha * T - 3 + 4 * np.exp(-alpha * T) - np.exp(-2 * alpha * T))
q23 = (1 / (2 * alpha ** 2)) * (1 - np.exp(-alpha * T)) ** 2
q33 = -(1 / (2 * alpha)) * (np.exp(-2 * alpha * T) - 1)

Q = 2 * alpha * sigma_m ** 2 * np.array([
    [q11, q12, q13],
    [q12, q22, q23],
    [q13, q23, q33]
])    
x =Singer_gen(length, T, x_0, alpha, sigma_m, Q)
traj_choice(x[:,0],T)
    
def bruitage(S,SNR):

    length = len(S)
    noise = np.random.randn(length,)

    sigma_c = 10**(-SNR/10) * np.mean(S**2)/np.mean(noise**2)

    bruit = np.sqrt(sigma_c)*noise
    
    
    return sigma_c,bruit


# if __name__ == '__main__':
#     # length = 10000  # Nb d'echantillons
#     T = 1  # période d'échantillonage
#     x_0 = np.array([0,0])  # Vecteur initial
#     n = 5  # Pour fixer q = n*9.81*T
#     SNR_list = [500, 400, 300, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 9, 8, 7, 6, 5]
#     # SNR_list = np.linspace(500,400,10)
#     # length_list = np.linspace(10000,20000,10,dtype='int')
#     length_list = [20000, 15000, 10000, 5000, 1000, 500, 100, 50, 10]
#     N_moy = 10
#     accu = np.zeros((len(length_list),len(SNR_list), N_moy,2))
#     accu_choice = -1 *np.ones((len(length_list),len(SNR_list), N_moy))
#     l = 0
#     for length in length_list:

#         for SNR_id in range(len(SNR_list)):
#             print("Parametres")
#             print(SNR_list[SNR_id])
#             print(length)
#             start = time.perf_counter()
#             for i in range(N_moy):
#                 print(i)
                
#                 X_MUA = MRU_gen(length, T, x_0, n)
#                 sigma_c,bruit = bruitage(X_MUA[:,0], SNR_list[SNR_id])
#                 erreur_SNR = SNR_list[SNR_id] - 10*np.log10(np.mean(X_MUA[:,0]**2)/np.mean(bruit**2))
#                 sigma_th = np.array([n*9.81*T,sigma_c])
                
#                 sigma = MRU_param_estimation(X_MUA[:,0] + bruit,T)
                

#                 erreur_est= (np.abs(sigma_th - sigma)/sigma_th)*100

#                 accu[l,SNR_id,i,:]= erreur_est
#                 inter_choice = traj_choice(X_MUA[:,0],T)
#                 accu_choice[l,SNR_id,i] = inter_choice[0]
#                 print(10*np.log10(np.mean(X_MUA**2)/np.mean(bruit**2)))
#                 print(n*9.81*T)
#             end = time.perf_counter()
#             # print(end-start)
#         l+=1


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
SNR_list = [500,400,300,200,100,90,80,70,60,50,40,30,20,10,9,8,7,6,5]
length_list = [20000,15000,10000,5000,1000,500,100,50,10]
from mpl_toolkits.mplot3d import Axes3D
# SNR, Length = np.meshgrid(SNR_list, length_list)

mean_all_fin_log = np.log10(mean_sig_traj)#5%
# mean_all_fin_log = np.log10(mean_sig_bruit)<(1)


SNR_list_log = np.array(SNR_list)
length_list_log = np.array(length_list)

# Création de la heatmap
plt.figure(figsize=(10, 8))
plt.imshow(np.flip(mean_all_fin_log), aspect='auto', cmap='seismic', origin='lower',
           extent=[SNR_list_log.min(), SNR_list_log.max(), length_list_log.min(), length_list_log.max()],interpolation='bilinear',interpolation_stage='data')#gist_ncar

# Ajuster les ticks des axes pour afficher les valeurs originales
plt.colorbar(label='log10(mean_all_fin)')
plt.xticks(SNR_list_log, labels=SNR_list, rotation=45)
plt.yticks(length_list_log, labels=length_list)
plt.xlabel('SNR ')
plt.ylabel('Length ')
plt.title('Heatmap: log10(mean_sig_bruit) with log-scaled axes')

# Affichage
plt.show()