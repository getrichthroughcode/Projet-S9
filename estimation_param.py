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
    rho_estime = ratios[0]
    alpha = -np.log(rho_estime) / T
    q11 = (1 / (2 * alpha ** 5)) * (
            2 * alpha * T - 2 * alpha ** 2 * T ** 2 + 2 * alpha ** 3 * T ** 3 / 3 - 4 * alpha * T * np.exp(
        -alpha * T) - np.exp(-2 * alpha * T) + 1)
    q12 = (1 / (2 * alpha ** 4)) * (alpha ** 2 * T ** 2 + 1 + np.exp(-2 * alpha * T) + np.exp(-alpha * T) * (
            -2 + 2 * alpha * T) - 2 * alpha * T)
    q13 = (1 / (2 * alpha ** 3)) * (1 - 2 * alpha * T * np.exp(-alpha * T) - np.exp(-2 * alpha * T))
    q22 = (1 / (2 * alpha ** 3)) * (2 * alpha * T - 3 + 4 * np.exp(-alpha * T) - np.exp(-2 * alpha * T))
    q23 = (1 / (2 * alpha ** 2)) * (1 - np.exp(-alpha * T)) ** 2
    q33 = -(1 / (2 * alpha)) * (np.exp(-2 * alpha * T) - 1)
    delta = -1 / (T ** 2 * alpha ** 2) * (-1 - alpha * T + 1 / rho_estime)
    A = 4 / (alpha ** 2 * T ** 2) * np.sinh(alpha * T / 2) ** 2
    terme_1 = 2 * alpha * q12 * q13 / T - 2 * alpha * q13 / T ** 2 + 2 * alpha * q13 ** 2 * delta / q11
    terme_2 = 2 * alpha * (q23 - q12 * q13 / q11) / T + 2 * alpha * (q23 - q12 * q13 / q11) ** 2 * delta / (
            q22 - q12 ** 2 / q11)
    terme_3 = 2 * alpha * (
            q33 - q13 ** 2 / q11 - ((q23 - q12 * q13 / q11) / np.sqrt(q22 - q12 ** 2 / q11)) ** 2) * delta
    terme_4 = 2 * alpha * q11 / T ** 4
    terme_5 = (np.sqrt(2 * alpha) * q12 / T * np.sqrt(q11) - np.sqrt(2 * alpha * q11) / T ** 2 + delta * (
            np.sqrt(2 * alpha / q11) * q13)) ** 2
    terme_6 = (np.sqrt(2 * alpha * (q22 - q12 ** 2 / q11)) / T + delta * (
            np.sqrt(2 * alpha) * (q23 - q12 * q13 / q11) / np.sqrt(q22 - q12 ** 2 / q11))) ** 2
    terme_7 = delta ** 2 * 2 * alpha * (
            q33 - q13 ** 2 / q11 - ((q23 - q12 * q13 / q11) / np.sqrt(q22 - q12 ** 2 / q11)) ** 2)

    K = A * (A + 2 * (terme_1 + terme_2 + terme_3)) + terme_4 + terme_5 + terme_6 + terme_7
    sigma_m = np.sqrt(corr[len(corr) // 2] / K)

    alpha_11 = np.sqrt(2 * alpha) * sigma_m * np.sqrt(q11)
    alpha_21 = np.sqrt(2 * alpha) * sigma_m * q12 / np.sqrt(q11)
    alpha_22 = np.sqrt(2 * alpha) * sigma_m * np.sqrt(q22 - q12 ** 2 / q11)
    alpha_31 = np.sqrt(2 * alpha) * sigma_m * q13 / np.sqrt(q11)
    alpha_32 = np.sqrt(2 * alpha) * sigma_m * (q23 - q12 * q13 / q11) / np.sqrt(q22 - q12 ** 2 / q11)
    alpha_33 = np.sqrt(2 * alpha) * sigma_m * np.sqrt(
        q33 - q13 ** 2 / q11 - ((q23 - q12 * q13 / q11) / np.sqrt(q22 - q12 ** 2 / q11)) ** 2)

    B = alpha_11 / T ** 2
    delta = -1 / (T ** 2 * alpha ** 2) * (-1 - alpha * T + 1 / rho_estime)
    C = alpha_21 / T - alpha_11 / T ** 2 + alpha_31 * delta
    D = alpha_22 / T + alpha_32 * delta
    F = alpha_33 * delta

    R_0 = A * (A * sigma_m ** 2 + 2 * (C * alpha_31 + D * alpha_32 + F * alpha_33)) + B ** 2 + C ** 2 + D ** 2 + F ** 2
    sigma_n = (corr[len(corr) // 2] - R_0)*T**4/6

    return ratios[0], sigma_m, sigma_n

