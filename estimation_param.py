import numpy as np
from MRU_gen import *
from scipy.signal import correlate

def estimate(X_pos):
    X_vit=[]
    X_acc=[]
    # inte = L.T@np.random.randn(3, 1).tolist()
    X_pos = X_pos
    for i in range(0,len(X_pos)-2):
        # inte = np.sqrt(9.81) * L.T @ np.random.randn(3, 1).tolist()
        vit = float((X_pos[i+1] - X_pos[i]) /1 )
        X_vit.append(vit)
        acc = float((X_pos[i+2] - 2*X_pos[i+1]+X_pos[i]) /1 )
        X_acc.append(acc)
    X_vit.insert(0,(X_pos[1] - X_pos[0]) / 1)
    X_vit.append((X_pos[-1] - X_pos[-2]) / 1)
    X_acc.insert(0,(X_pos[2] - 2 * X_pos[1] + X_pos[0]) / (1**2))
    X_acc.append((X_pos[-1] - 2 * X_pos[-2] + X_pos[-3]) / (1**2))
    return (X_vit,X_acc) #S.median_filter()

def MRU_param_estimation(X_pos,T):
    vit_est,acc_est = estimate(X_pos)
    corr = np.correlate(acc_est, acc_est, "full") / len(acc_est)
    mat_det_sig = np.array([[46/(3*T**9),133/(6*T**9),-10/(3*T**9)], [11/(18*T**6),-22/(9*T**6),17/(36*T**6)]])
    sigma = 36*T**10/501 * mat_det_sig@(corr[len(corr)//2:len(corr)//2 +3 ]).T
    return sigma

def Singer_param_estimation(X_pos,T):
    vit_est,acc_est = estimate(X_pos)
    corr = correlate(acc_est, acc_est, "full", method='fft')
    len_dyn_pt_0 = np.linspace(0, len(acc_est), len(acc_est))
    len_dyn_pt_1 = np.linspace(len(acc_est) - 1, 0, len(acc_est) - 1)
    len_dyn = np.concatenate((len_dyn_pt_0, len_dyn_pt_1))
    len_dyn[0] = 1e-8
    len_dyn[len(len_dyn) - 1] = 1e-8
    corr = corr / len_dyn
    rho = corr[len(corr)//2 + 3:len(corr)//2 + 9]
    ratios = rho[1:]/rho[:-1]
    return ratios[0]

def MUA_param_estimation(X_pos):
    vit_est,acc_est = estimate(X_pos)
    jerk_est,_ = estimate(X_pos)
    corr = np.correlate(jerk_est, jerk_est, "full") / len(jerk_est)
    return 0











if __name__ == "__main__":
    n =1
    T= 1
    q = 9.81*n*T
    Nb_moy = 100
    length = 30000
    accu = np.zeros((Nb_moy,2))
    for i in range(Nb_moy):
        if(i%10==0):
            print(i)
        x_0 = np.array([0, 0])
        X = MRU_gen(length, T, x_0, n)
        x_coords = X[:,0]
        x_coords_bruit = x_coords + np.random.randn(len(x_coords),)
        accu[i,:]= MRU_param_estimation(x_coords_bruit,T)
    accu = np.sum(accu,axis=0)/accu.shape[0]
    error = np.abs(accu - np.array([q,1]))/100
    print("Erreur d'estimation pour un moyennage de {} avec sigma = {} et sigma_n = {} ,\n sigma_est = {}% et sigma_n_est = {}% ".format(Nb_moy,q,1,error[0],error[1]))
