"""
Created on 2025-01-10 23:28:14

@author: Abdoulaye Diallo <abdoulayediallo338@gmail.com>
"""
import os 
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from EntropyHub import SampEn
import tqdm

# Set src directory as the working directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.Criterium import *
from utils.motiongen import *
from utils.estimate import *
from utils.plot_simu import *
from utils.plot_sampen_insights import *
from utils.plot_2d_heatmap_or_contour import *


# Test des fonctions Sample Entropy et DCCA

# Génération de données temporelles synthétiques pour valider les fonctions

def generate_custom_signals():
    """
    Generate custom signals for testing the SampEn function.
    Returns a dictionary of signals keyed by descriptive names.
    """
    np.random.seed(42)
    signals = {}
    length = 1000
    
    # Signal 1: Constant 
    signals['constant'] = np.ones(length)

    # Signal 2: white noise
    signals['white noise'] = np.random.randn(length)

    # Signal 3: Random walk
    signals['random walk'] = np.cumsum(np.random.randn(length))

    # Signal 4: Sine wave 10 Hz
    signals['sine wave'] = np.sin(10*np.linspace(0, 2*np.pi, length))

    # Signal 5: Sine wave with noise 
    signals['noisy sine wave'] = np.sin(10*np.linspace(0, 2*np.pi, length)) + 3*np.random.randn(length)
    # Signal 6: Sine wave with outliers 
    t = np.linspace(0, 2*np.pi, length)
    sine_wave = np.sin(10 * t)  # 10 Hz sine
    # Insert outliers in about 1% of samples
    outlier_indices = np.random.choice(range(length), size=int(0.01 * length), replace=False)
    sine_wave[outlier_indices] = 50
    signals['sine wave with outliers'] = sine_wave
    return signals


def test_SampEn():
    """
    Test the SampEn function with custom signals.
    """
    signals = generate_custom_signals()
    

    results = {}
    print("===== Custom SampEn Tests =====")
    # fix parameter 
    m = 2
    r = 0.2

    for name, signal in signals.items():
        print(f"Signal: {name}")
        print(f"Length: {len(signal)}")
        sampen_val = sampen(signal, m, r)
        print(f"Sample Entropy: {sampen_val}")
        results[name] = sampen_val
    results_frame = pd.DataFrame(results, index=['Sample Entropy'])    
    print(results_frame)

def plot_custom_signals():
    """
    Plot the custom signals generated for testing.
    """
    signals = generate_custom_signals()
    fig, axs = plt.subplots(len(signals), 1, figsize=(10, 10))
    for i, (name, signal) in enumerate(signals.items()):
        axs[i].plot(signal)
        axs[i].set_title(name)
    plt.tight_layout()
    plt.show()


# Simulation pour déterminer l'influence des paramètres des mouvements sur la sample entropy

def simulate_MRU():
    """
    Simulation de Monte-Carlo pour voir l'impact des paramètres du mouvement rectiligne uniforme sur la sample entropy.
    """
    print("<===== Simulation MRU =====>")
    print("---------------------------------------------------------------------------------")
    print("1.1 Influence de la période d'échantillonnage et du facteur de bruit sur la sample entropy de la position")
    print("Simulation effectuée sur les positions exactes du modèle")
    print("---------------------------------------------------------------------------------")
    # Paramètres de simulation fixe
    N = 2000  # Nombre d'échantillons
    nrealisations = 100  # Nombre de réalisations
    x_0 = np.array([0,0])
    print(f"Nombre d'échantillons à tester: {N}")

    # Paramètres à tester
    Ts = [0.01, 0.1, 0.5, 1,10,15,20,60]  # Période d'échantillonnage en secondes
    n = [1000, 2000, 5000, 10000, 15000, 20000, 30000, 50000]  # Facteur de bruit
    results1_1 = []
    for T in Ts:
        for noise in n:
            samp_pos_all = []
            for i in tqdm.tqdm(range(nrealisations), desc=f"RSimulation pour T={T} s et n={noise}"):
                L = MRU_gen(2000, T, x_0, noise)
                x = L[:,0]
                samp = sampen(x, 2, 0.2)
                samp_pos_all.append(samp)
            results1_1.append({
                'T': T, 
                'n': noise,
                'Sample Entropy': np.mean(samp_pos_all),
                'Std': np.std(samp_pos_all)})    
    results_frame1_1 = pd.DataFrame(results1_1)
    results_frame1_1.to_csv("results/simulation_1_1.csv", index=False)
    print("Résultats de simu 1.1 sauvegardés avec succès.")
    print("Fin de la simulation 1.1")



    print("---------------------------------------------------------------------------------")
    print("1.2 Influence du nombre d'échantillons et du facteur de bruit sur la sample entropy de la position")
    print("Simulation effectuée sur les positions exactes du modèle")
    print("---------------------------------------------------------------------------------")
    # Paramètres fixes
    T = 1.0  # Période d'échantillonnage
    print(f"Période d'échantillonnage: {T} s")
    print(f"Nombre de réalisations: {nrealisations}")
    print(f"Conditions Initiales: {x_0}")

    # Paramètres à tester
    Ns = [100, 500, 1000, 2000, 5000, 10000]  # Nombre d'échantillons
    results1_2 = []
    for N in Ns:
        for noise in n:
            samp_pos_all = []
            for i in tqdm.tqdm(range(nrealisations), desc=f"Simulation pour N={N} et n={noise}"):
                L = MRU_gen(N, 1.0, x_0, noise)
                x = L[:,0]
                samp = sampen(x, 2, 0.2)
                samp_pos_all.append(samp)
            results1_2.append({
                'N': N, 
                'n': noise,
                'Sample Entropy': np.mean(samp_pos_all),
                'Std': np.std(samp_pos_all)})
    results_frame1_2 = pd.DataFrame(results1_2)
    results_frame1_2.to_csv("results/simulation_1_2.csv", index=False)
    print("Résultats de simu 1.2 sauvegardés avec succès.")
    print("Fin de la simulation 1.2")




    print("---------------------------------------------------------------------------------")
    print("1.3 Influence de la période d'échantillonnage et du facteur de bruit sur la sample entropy de la vitesse estimée")
    print("Simulation effectuée sur les vitesses estimées du modèle")
    print("---------------------------------------------------------------------------------")
    # Paramètres fixes
    N_1_3 = 2000
    print(f"Nombre d'échantillons: {N_1_3}")
    print(f"Nombre de réalisations: {nrealisations}")
    print(f"Conditions Initiales: {x_0}")
    results1_3 = []
    for T in Ts:
        for noise in n:
            samp_pos_all = []
            for i in tqdm.tqdm(range(nrealisations), desc=f"Simulation pour T={T} et n={noise}"):
                L = MRU_gen(2000, T, x_0, noise)
                x = L[:,0]
                v = estimate_v_retrograde(x, T)
                samp = sampen(v, 2, 0.2)
                samp_pos_all.append(samp)
            results1_3.append({
                'T': T, 
                'n': noise,
                'Sample Entropy v_est': np.mean(samp_pos_all),
                'Std': np.std(samp_pos_all)})
    results_frame1_3 = pd.DataFrame(results1_3)
    results_frame1_3.to_csv("results/simulation_1_3.csv", index=False)
    print("Résultats de simu 1.3 sauvegardés avec succès.")
    print("Fin de la simulation 1.3")


    print("---------------------------------------------------------------------------------")
    print("1.4 Influence du nombre d'échantillons et du facteur de bruit sur la sample entropy de la vitesse estimée")
    print("Simulation effectuée sur les vitesses estimées du modèle")
    print("---------------------------------------------------------------------------------")
    # Paramètres fixes
    print(f"Période d'échantillonnage: {T} s")
    print(f"Nombre de réalisations: {nrealisations}")
    print(f"Conditions Initiales: {x_0}")

    results1_4 = []

    for N in Ns:
        for noise in n:
            samp_pos_all = []
            for i in tqdm.tqdm(range(nrealisations), desc=f"Simulation pour N={N} et n={noise}"):
                L = MRU_gen(N, 1.0, x_0, noise)
                x = L[:,0]
                v = estimate_v_retrograde(x, T)
                samp = sampen(v, 2, 0.2)
                samp_pos_all.append(samp)
            results1_4.append({
                'N': N, 
                'n': noise,
                'Sample Entropy': np.mean(samp_pos_all),
                'Std': np.std(samp_pos_all)})
    results_frame1_4 = pd.DataFrame(results1_4)
    results_frame1_4.to_csv("results/simulation_1_4.csv", index=False)
    print("Résultats de simu 1.4 sauvegardés avec succès.")
    print("Fin de la simulation 1.4")

    




    print("---------------------------------------------------------------------------------")
    print("1.5 Influence de la période d'échantillonnage et du facteur de bruit sur la sample entropy de l'accélération estimée")
    print("Simulation effectuée sur les accélérations estimées du modèle")
    print("---------------------------------------------------------------------------------")
    # Paramètres fixes
    N_1_5 = 2000
    print(f"Nombre d'échantillons: {N_1_5}")
    print(f"Nombre de réalisations: {nrealisations}")
    print(f"Conditions Initiales: {x_0}")

    results1_5 = []
    for T in Ts:
        for noise in n:
            samp_pos_all = []
            for i in tqdm.tqdm(range(nrealisations), desc=f"Simulation pour T={T} et n={noise}"):
                L = MRU_gen(2000, T, x_0, noise)
                x = L[:,0]
                a = estimate_a_retrograde(x, T)
                samp = sampen(a, 2, 0.2)
                samp_pos_all.append(samp)
            results1_5.append({
                'T': T, 
                'n': noise,
                'Sample Entropy': np.mean(samp_pos_all),
                'Std': np.std(samp_pos_all)})
    results_frame1_5 = pd.DataFrame(results1_5)
    results_frame1_5.to_csv("results/simulation_1_5.csv", index=False)
    print("Résultats de simu 1.5 sauvegardés avec succès.")
    print("Fin de la simulation 1.5")


    print("---------------------------------------------------------------------------------")
    print("1.6 Influence du nombre d'échantillons et du facteur de bruit sur la sample entropy de l'accélération estimée")
    print("Simulation effectuée sur les accélérations estimées du modèle")
    print("---------------------------------------------------------------------------------")
    # Paramètres fixes
    print(f"Période d'échantillonnage: T = 1.0 s")
    print(f"Nombre de réalisations: {nrealisations}")
    print(f"Conditions Initiales: {x_0}")

    results1_6 = []
    for N in Ns:
        for noise in n:
            samp_pos_all = []
            for i in tqdm.tqdm(range(nrealisations), desc=f"Simulation pour N={N} et n={noise}"):
                L = MRU_gen(N, 1.0, x_0, noise)
                x = L[:,0]
                a = estimate_a_retrograde(x, T)
                samp = sampen(a, 2, 0.2)
                samp_pos_all.append(samp)
            results1_6.append({
                'N': N, 
                'n': noise,
                'Sample Entropy': np.mean(samp_pos_all),
                'Std': np.std(samp_pos_all)})
    results_frame1_6 = pd.DataFrame(results1_6)
    results_frame1_6.to_csv("results/simulation_1_6.csv", index=False)
    print("Résultats de simu 1.6 sauvegardés avec succès.")
    print("Fin de la simulation 1.6")


     


    return


def simulate_MUA():
    """
    Simulation de Monte-Carlo pour voir l'impact des paramètres du mouvement rectiligne uniformément accéléré sur la sample entropy.
    """
    print("<===== Simulation MUA =====>")
    print("---------------------------------------------------------------------------------")
    print("2.1 Influence de la période d'échantillonnage et du facteur de bruit sur la sample entropy de la position")
    print("Simulation effectuée sur les positions exactes du modèle")
    print("---------------------------------------------------------------------------------")
    # Paramètres de simulation fixe
    N = 2000  # Nombre d'échantillons
    nrealisations = 100  # Nombre de réalisations
    x_0 = np.array([0, 0, 0])  # Position, vitesse, accélération initiales
    print(f"Nombre d'échantillons à tester: {N}")

    # Paramètres à tester
    Ts = [0.01, 0.1, 0.5, 1, 10, 15, 20, 60]  # Période d'échantillonnage en secondes
    n = [1000, 2000, 5000, 10000, 15000, 20000, 30000, 50000]  # Facteur de bruit
    results2_1 = []
    for T in Ts:
        for noise in n:
            samp_pos_all = []
            for i in tqdm.tqdm(range(nrealisations), desc=f"Simulation pour T={T} s et n={noise}"):
                L = MUA_gen(N, 1.0, x_0, noise)  # Génération du MUA
                x = L[:, 0]  # Extraire la position
                samp = sampen(x, 2, 0.2)  # Calcul de la Sample Entropy
                samp_pos_all.append(samp)
            results2_1.append({
                'T': T,
                'n': noise,
                'Sample Entropy': np.mean(samp_pos_all),
                'Std': np.std(samp_pos_all)})
    results_frame2_1 = pd.DataFrame(results2_1)
    results_frame2_1.to_csv("results/simulation_2_1.csv", index=False)
    print("Résultats de simu 2.1 sauvegardés avec succès.")
    print("Fin de la simulation 2.1")

    # ---------------------------------------------------------------------------------
    print("2.2 Influence du nombre d'échantillons et du facteur de bruit sur la sample entropy de la position")
    print("Simulation effectuée sur les positions exactes du modèle")
    print("---------------------------------------------------------------------------------")
    T = 1.0  # Période d'échantillonnage fixée
    Ns = [100, 500, 1000, 2000, 5000, 10000]  # Nombre d'échantillons à tester
    results2_2 = []
    for N in Ns:
        for noise in n:
            samp_pos_all = []
            for i in tqdm.tqdm(range(nrealisations), desc=f"Simulation pour N={N} et n={noise}"):
                L = MUA_gen(N, 1.0, x_0, noise)
                x = L[:, 0]
                samp = sampen(x, 2, 0.2)
                samp_pos_all.append(samp)
            results2_2.append({
                'N': N,
                'n': noise,
                'Sample Entropy': np.mean(samp_pos_all),
                'Std': np.std(samp_pos_all)})
    results_frame2_2 = pd.DataFrame(results2_2)
    results_frame2_2.to_csv("results/simulation_2_2.csv", index=False)
    print("Résultats de simu 2.2 sauvegardés avec succès.")
    print("Fin de la simulation 2.2")

    # ---------------------------------------------------------------------------------
    print("2.3 Influence de la période d'échantillonnage et du facteur de bruit sur la sample entropy de la vitesse estimée")
    print("Simulation effectuée sur les vitesses estimées du modèle")
    print("---------------------------------------------------------------------------------")
    N_2_2 = 2000
    results2_3 = []
    for T in Ts:
        for noise in n:
            samp_vel_all = []
            for i in tqdm.tqdm(range(nrealisations), desc=f"Simulation pour T={T} s et n={noise}"):
                L = MUA_gen(2000, T, x_0, noise)
                v = estimate_v_retrograde(L[:, 0], T)
                samp = sampen(v, 2, 0.2)
                samp_vel_all.append(samp)
            results2_3.append({
                'T': T,
                'n': noise,
                'Sample Entropy v_est': np.mean(samp_vel_all),
                'Std': np.std(samp_vel_all)})
    results_frame2_3 = pd.DataFrame(results2_3)
    results_frame2_3.to_csv("results/simulation_2_3.csv", index=False)
    print("Résultats de simu 2.3 sauvegardés avec succès.")
    print("Fin de la simulation 2.3")

    # ---------------------------------------------------------------------------------
    print("2.4 Influence du nombre d'échantillons et du facteur de bruit sur la sample entropy de la vitesse estimée")
    print("Simulation effectuée sur les vitesses estimées du modèle")
    print("---------------------------------------------------------------------------------")
    results2_4 = []
    for N in Ns:
        for noise in n:
            samp_vel_all = []
            for i in tqdm.tqdm(range(nrealisations), desc=f"Simulation pour N={N} et n={noise}"):
                L = MUA_gen(N, 1.0, x_0, noise)
                v = estimate_v_retrograde(L[:, 0], T)
                samp = sampen(v, 2, 0.2)
                samp_vel_all.append(samp)
            results2_4.append({
                'N': N,
                'n': noise,
                'Sample Entropy': np.mean(samp_vel_all),
                'Std': np.std(samp_vel_all)})
    results_frame2_4 = pd.DataFrame(results2_4)
    results_frame2_4.to_csv("results/simulation_2_4.csv", index=False)
    print("Résultats de simu 2.4 sauvegardés avec succès.")
    print("Fin de la simulation 2.4")

    # ---------------------------------------------------------------------------------
    print("2.5 Influence de la période d'échantillonnage et du facteur de bruit sur la sample entropy de l'accélération estimée")
    print("Simulation effectuée sur les accélérations estimées du modèle")
    print("---------------------------------------------------------------------------------")
    results2_5 = []
    N_2_5 = 2000
    for T in Ts:
        for noise in n:
            samp_acc_all = []
            for i in tqdm.tqdm(range(nrealisations), desc=f"Simulation pour T={T} s et n={noise}"):
                L = MUA_gen(2000, T, x_0, noise)
                a = estimate_a_retrograde(L[:, 0], T)
                samp = sampen(a, 2, 0.2)
                samp_acc_all.append(samp)
            results2_5.append({
                'T': T,
                'n': noise,
                'Sample Entropy': np.mean(samp_acc_all),
                'Std': np.std(samp_acc_all)})
    results_frame2_5 = pd.DataFrame(results2_5)
    results_frame2_5.to_csv("results/simulation_2_5.csv", index=False)
    print("Résultats de simu 2.5 sauvegardés avec succès.")
    print("Fin de la simulation 2.5")

    # ---------------------------------------------------------------------------------
    print("2.6 Influence du nombre d'échantillons et du facteur de bruit sur la sample entropy de l'accélération estimée")
    print("Simulation effectuée sur les accélérations estimées du modèle")
    print("---------------------------------------------------------------------------------")
    results2_6 = []
    for N in Ns:
        for noise in n:
            samp_acc_all = []
            for i in tqdm.tqdm(range(nrealisations), desc=f"Simulation pour N={N} et n={noise}"):
                L = MUA_gen(N, 1.0, x_0, noise)
                a = estimate_a_retrograde(L[:, 0], T)
                samp = sampen(a, 2, 0.2)
                samp_acc_all.append(samp)
            results2_6.append({
                'N': N,
                'n': noise,
                'Sample Entropy': np.mean(samp_acc_all),
                'Std': np.std(samp_acc_all)})
    results_frame2_6 = pd.DataFrame(results2_6)
    results_frame2_6.to_csv("results/simulation_2_6.csv", index=False)
    print("Résultats de simu 2.6 sauvegardés avec succès.")
    print("Fin de la simulation 2.6")

    return



def simulate_Singer():
    """
    Simulation de Monte-Carlo pour voir l'impact des paramètres du modèle de Singer sur la sample entropy.
    """
    print("<===== Simulation Singer =====>")

    # Paramètres fixes pour toutes les simulations
    nrealisations = 100  # Nombre de réalisations
    x_0 = np.array([0, 0, 0])  # Conditions initiales
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0]  # Valeurs de alpha
    sigma_ms = [1000, 2000, 5000, 10000, 15000, 20000, 30000, 50000]  # Valeurs de sigma_m
    Ts = [0.01, 0.1, 0.5, 1, 10, 15, 20, 60]  # Périodes d'échantillonnage
    Ns = [100, 500, 1000, 2000, 5000, 10000]  # Nombres d'échantillons

    # ---------------------------------------------------------------------------------
    print("3.1 Influence de alpha et sigma_m sur la sample entropy de la position")
    results3_1 = []
    for alpha in alphas:
        for sigma_m in sigma_ms:
            samp_pos_all = []
            for i in tqdm.tqdm(range(nrealisations), desc=f"Alpha={alpha}, Sigma_m={sigma_m}"):
                L = Singer_gen(2000, 1.0, x_0, alpha, sigma_m)
                x = L[:, 0]  # Extraire la position
                samp = sampen(x, 2, 0.2)
                samp_pos_all.append(samp)
            results3_1.append({
                'alpha': alpha,
                'sigma_m': sigma_m,
                'Sample Entropy': np.mean(samp_pos_all),
                'Std': np.std(samp_pos_all)})
    pd.DataFrame(results3_1).to_csv("results/simulation_3_1.csv", index=False)
    print("Simulation 3.1 terminée et sauvegardée.")

    # ---------------------------------------------------------------------------------
    print("3.2 Influence de N et sigma_m sur la sample entropy de la position")
    results3_2 = []
    for N in Ns:
        for sigma_m in sigma_ms:
            samp_pos_all = []
            for i in tqdm.tqdm(range(nrealisations), desc=f"N={N}, Sigma_m={sigma_m}"):
                L = Singer_gen(N, 1.0, x_0, 1.0, sigma_m)
                x = L[:, 0]  # Extraire la position
                samp = sampen(x, 2, 0.2)
                samp_pos_all.append(samp)
            results3_2.append({
                'N': N,
                'sigma_m': sigma_m,
                'Sample Entropy': np.mean(samp_pos_all),
                'Std': np.std(samp_pos_all)})
    pd.DataFrame(results3_2).to_csv("results/simulation_3_2.csv", index=False)
    print("Simulation 3.2 terminée et sauvegardée.")

    # ---------------------------------------------------------------------------------
    print("3.3 Influence de T et sigma_m sur la sample entropy de la position")
    results3_3 = []
    for T in Ts:
        for sigma_m in sigma_ms:
            samp_pos_all = []
            for i in tqdm.tqdm(range(nrealisations), desc=f"T={T}, Sigma_m={sigma_m}"):
                L = Singer_gen(2000, T, x_0, 1.0, sigma_m)
                x = L[:, 0]  # Extraire la position
                samp = sampen(x, 2, 0.2)
                samp_pos_all.append(samp)
            results3_3.append({
                'T': T,
                'sigma_m': sigma_m,
                'Sample Entropy': np.mean(samp_pos_all),
                'Std': np.std(samp_pos_all)})
    pd.DataFrame(results3_3).to_csv("results/simulation_3_3.csv", index=False)
    print("Simulation 3.3 terminée et sauvegardée.")

    # ---------------------------------------------------------------------------------
    print("3.4 Influence de alpha et sigma_m sur la sample entropy de la vitesse estimée")
    results3_4 = []
    for alpha in alphas:
        for sigma_m in sigma_ms:
            samp_vel_all = []
            for i in tqdm.tqdm(range(nrealisations), desc=f"Alpha={alpha}, Sigma_m={sigma_m}"):
                L = Singer_gen(2000, 1.0, x_0, alpha, sigma_m)
                v = estimate_v_retrograde(L[:, 0], 1.0)  # Calcul de la vitesse estimée
                samp = sampen(v, 2, 0.2)
                samp_vel_all.append(samp)
            results3_4.append({
                'alpha': alpha,
                'sigma_m': sigma_m,
                'Sample Entropy': np.mean(samp_vel_all),
                'Std': np.std(samp_vel_all)})
    pd.DataFrame(results3_4).to_csv("results/simulation_3_4.csv", index=False)
    print("Simulation 3.4 terminée et sauvegardée.")

    # ---------------------------------------------------------------------------------
    print("3.5 Influence de N et sigma_m sur la sample entropy de la vitesse estimée")
    results3_5 = []
    for N in Ns:
        for sigma_m in sigma_ms:
            samp_vel_all = []
            for i in tqdm.tqdm(range(nrealisations), desc=f"N={N}, Sigma_m={sigma_m}"):
                L = Singer_gen(N, 1.0, x_0, 1.0, sigma_m)
                v = estimate_v_retrograde(L[:, 0], 1.0)
                samp = sampen(v, 2, 0.2)
                samp_vel_all.append(samp)
            results3_5.append({
                'N': N,
                'sigma_m': sigma_m,
                'Sample Entropy': np.mean(samp_vel_all),
                'Std': np.std(samp_vel_all)})
    pd.DataFrame(results3_5).to_csv("results/simulation_3_5.csv", index=False)
    print("Simulation 3.5 terminée et sauvegardée.")

    # ---------------------------------------------------------------------------------
    print("3.6 Influence de T et sigma_m sur la sample entropy de la vitesse estimée")
    results3_6 = []
    for T in Ts:
        for sigma_m in sigma_ms:
            samp_vel_all = []
            for i in tqdm.tqdm(range(nrealisations), desc=f"T={T}, Sigma_m={sigma_m}"):
                L = Singer_gen(2000, T, x_0, 1.0, sigma_m)
                v = estimate_v_retrograde(L[:, 0], T)
                samp = sampen(v, 2, 0.2)
                samp_vel_all.append(samp)
            results3_6.append({
                'T': T,
                'sigma_m': sigma_m,
                'Sample Entropy': np.mean(samp_vel_all),
                'Std': np.std(samp_vel_all)})
    pd.DataFrame(results3_6).to_csv("results/simulation_3_6.csv", index=False)
    print("Simulation 3.6 terminée et sauvegardée.")

    return

def simulate_sampen():
    """
    Simulation of motion models (MRU, MUA, Singer) to assess the influence of 
    Sample Entropy parameters (pattern size m, threshold r, noise, and extra Singer params)
    on position, velocity, and acceleration.
    """
    print("<===== Simulation of Motion Models with Noise and Singer Parameters =====>")
    motion_models = ['MRU', 'MUA', 'Singer']
    nrealisations = 50  # Number of realizations
    Ts = [0.01, 0.1, 0.5, 1, 10]  # Sampling periods in seconds
    Ns = [100, 500, 1000, 2000]  # Number of samples
    ms = [2, 3, 10,15,20]  # Pattern sizes for Sample Entropy
    noise_factors = [1000, 2000, 5000, 10000]  # Noise factors for MRU and MUA
    x_0_mru = np.array([0, 0])  # Initial conditions for MRU
    x_0_mua = np.array([0, 0, 0])  # Initial conditions for MUA
    x_0_singer = np.array([0, 0, 0])  # Initial conditions for Singer
    
    # Parameters specific to each model
    model_params = {
        'MRU': {'gen_func': MRU_gen, 'x_0': x_0_mru, 'noise_factors': noise_factors},
        'MUA': {'gen_func': MUA_gen, 'x_0': x_0_mua, 'noise_factors': noise_factors},
        'Singer': {
            'gen_func': Singer_gen, 
            'x_0': x_0_singer,
            'alphas': [0.1, 0.5, 1.0, 2.0, 5.0],  # Varying alpha
            'sigma_ms': [1000, 2000, 5000, 10000]  # Varying sigma_m
        }
    }
    
    for model in motion_models:
        print(f"Simulating for {model}...")
        params = model_params[model]
        gen_func = params['gen_func']
        x_0 = params['x_0']
        
        results = []
        
        for m in ms:
            for T in Ts:
                for N in Ns:
                    if model in ['MRU', 'MUA']:
                        for noise in params['noise_factors']:
                            samp_pos_all = []
                            samp_vel_all = []
                            samp_acc_all = []
                            
                            for i in tqdm.tqdm(range(nrealisations), desc=f"{model} m={m}, T={T}, N={N}, noise={noise}"):
                                L = gen_func(N, T, x_0, noise)
                                
                                # Position
                                x = L[:, 0]
                                samp_pos_all.append(sampen(x, m, 0.2))
                                
                                # Velocity (retrogressively estimated)
                                v = estimate_v_retrograde(x, T)
                                samp_vel_all.append(sampen(v, m, 0.2))
                                
                                # Acceleration (retrogressively estimated)
                                a = estimate_a_retrograde(x, T)
                                samp_acc_all.append(sampen(a, m, 0.2))
                            
                            results.append({
                                'm': m,
                                'T': T,
                                'N': N,
                                'noise': noise,
                                'Sample Entropy Position': np.mean(samp_pos_all),
                                'Sample Entropy Velocity': np.mean(samp_vel_all),
                                'Sample Entropy Acceleration': np.mean(samp_acc_all),
                                'Std Position': np.std(samp_pos_all),
                                'Std Velocity': np.std(samp_vel_all),
                                'Std Acceleration': np.std(samp_acc_all)
                            })
                    elif model == 'Singer':
                        for alpha in params['alphas']:
                            for sigma_m in params['sigma_ms']:
                                samp_pos_all = []
                                samp_vel_all = []
                                samp_acc_all = []
                                
                                for i in tqdm.tqdm(range(nrealisations), desc=f"Singer m={m}, T={T}, N={N}, alpha={alpha}, sigma_m={sigma_m}"):
                                    L = gen_func(N, T, x_0, alpha, sigma_m)
                                    
                                    # Position
                                    x = L[:, 0]
                                    samp_pos_all.append(sampen(x, m, 0.2))
                                    
                                    # Velocity (retrogressively estimated)
                                    v = estimate_v_retrograde(x, T)
                                    samp_vel_all.append(sampen(v, m, 0.2))
                                    
                                    # Acceleration (retrogressively estimated)
                                    a = estimate_a_retrograde(x, T)
                                    samp_acc_all.append(sampen(a, m, 0.2))
                                
                                results.append({
                                    'm': m,
                                    'T': T,
                                    'N': N,
                                    'alpha': alpha,
                                    'sigma_m': sigma_m,
                                    'Sample Entropy Position': np.mean(samp_pos_all),
                                    'Sample Entropy Velocity': np.mean(samp_vel_all),
                                    'Sample Entropy Acceleration': np.mean(samp_acc_all),
                                    'Std Position': np.std(samp_pos_all),
                                    'Std Velocity': np.std(samp_vel_all),
                                    'Std Acceleration': np.std(samp_acc_all)
                                })
        
        # Save results for the motion model
        results_df = pd.DataFrame(results)
        if model == 'Singer':
            results_df.to_csv(f"results/simulation_{model}_varying_m_noise_and_params.csv", index=False)
        else:
            results_df.to_csv(f"results/simulation_{model}_varying_m_and_noise.csv", index=False)
        print(f"Results for {model} saved successfully.")
    
    print("All simulations completed successfully.")



def normalize_trajectory(x, y):
    # Centrer les trajectoires autour de l'origine
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    
    # Normaliser par la norme maximale
    max_norm = max(np.max(np.abs(x_centered)), np.max(np.abs(y_centered)))
    x_normalized = x_centered / max_norm
    y_normalized = y_centered / max_norm
    
    return x_normalized, y_normalized


if __name__ == "__main__":
    
    """
    t1 = time.time()
    simulate_MRU()
    print(f"Fin de la simulation MRU en {time.time() - t1} secondes.")

    t2 = time.time()
    simulate_MUA()
    print(f"Fin de la simulation MUA en {time.time() - t2} secondes.")

    t3 = time.time()
    simulate_Singer()
    print(f"Fin de la simulation Singer en {time.time() - t3} secondes.")
    print("Fin de toutes les simulations sur SampEn.")
    
    """
    """
    # Paramètres MRU :
    N = 50
    T = 5.0
    x_0_mru = np.array([0, 1])
    y_0_mru = np.array([0, 1])
    noise = 10
    L_x = MRU_gen(N, T, x_0_mru, noise)
    x_mru = L_x[:,0]
    L_y = MRU_gen(N, T, y_0_mru, noise)
    y_mru = L_y[:,0]

    # Paramètres MUA
    x_0_mua = np.array([0, 1, 0])
    y_0_mua = np.array([0, 1, 0])
    L_x_mua = MUA_gen(N, T, x_0_mua, noise)
    x_mua = L_x_mua[:,0]
    L_y_mua = MUA_gen(N, T, y_0_mua, noise)
    y_mua = L_y_mua[:,0]

    # Paramètres Singer
    alpha = 0.03
    sigma_m = 100
    x_0_singer = np.array([0, 1, 0])
    y_0_singer = np.array([0, 1, 0])
    L_x_singer = Singer_gen(N, T, x_0_singer, alpha, sigma_m)
    x_singer = L_x_singer[:,0]
    L_y_singer = Singer_gen(N, T, y_0_singer, alpha, sigma_m)
    y_singer = L_y_singer[:,0]

    # Normalisation des trajectoires

    # Normalisation des trajectoires MRU
    x_mru_norm, y_mru_norm = normalize_trajectory(x_mru, y_mru)

    # Normalisation des trajectoires MUA
    x_mua_norm, y_mua_norm = normalize_trajectory(x_mua, y_mua)

    # Normalisation des trajectoires Singer
    x_singer_norm, y_singer_norm = normalize_trajectory(x_singer, y_singer)
    
    # Plot des trajectoires normalisées
    plt.figure(figsize=(10, 10))
    plt.plot(x_mru_norm, y_mru_norm, label=f"Uniform rectilinear motion (N={N}, T={T}, noise={noise})")
    plt.plot(x_mua_norm, y_mua_norm, label=f"Uniformly accelerated motion (N={N}, T={T}, noise={noise})")
    plt.plot(x_singer_norm, y_singer_norm, label=f"Singer (N={N}, T={T}, alpha={alpha}, sigma_m={sigma_m})")
    plt.legend(loc="best")
    plt.title("Let's compare the shape of the 3 classes of trajectories")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()
    
    
    axis_t = np.linspace(0, N, N-1)
    # Calcul de la Sample Entropy pour les trajectoires MRU
    sampen_mru,m,m_1 = sampen(x_mru_norm, 2, 0.2) #matches of size 2  and m_1 matches of size 3 
    print(f"{m.shape}")
    plt.figure(figsize=(12, 6))
    plt.plot(x_mru_norm, label="Time Series", color="blue")

    # Tracer les motifs de taille 2 (m)
    step_m = 25 # Pas d'échantillonnage pour m
    for i, pattern in enumerate(m[::step_m]):
        start_idx = i * step_m  # Index de départ réel
        end_idx = start_idx + len(pattern)  # Index de fin réel
        if end_idx <= len(x_mru_norm):  # Vérifier les limites
            plt.plot(axis_t[start_idx:end_idx], x_mru_norm[start_idx:end_idx],
                    'o-', label=f"Pattern m={len(pattern)} (Index {start_idx})", color="orange")

    # Tracer les motifs de taille 3 (m+1)
    step_m1 = 10  # Pas d'échantillonnage pour m+1
    for i, pattern in enumerate(m_1[::step_m1]):
        start_idx = i * step_m1  # Index de départ réel
        end_idx = start_idx + len(pattern)  # Index de fin réel
        if end_idx <= len(x_mru_norm):  # Vérifier les limites
            plt.plot(axis_t[start_idx:end_idx], x_mru_norm[start_idx:end_idx],
                    's-', label=f"Pattern m+1={len(pattern)}", color="red")

    # Ajouter des titres et une légende
    plt.title("Sample Entropy: Highlighting Matched Patterns of m and m+1", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Time serie Value", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()
    """
        
    
    





    
    """
    # Example data: Sample Entropy values for three trajectory classes
    np.random.seed(42)
    n_points = 30
    sample_entropy_singer = np.random.normal(0.8, 0.05, n_points)
    sample_entropy_accelerated = np.random.normal(1.2, 0.05, n_points)
    sample_entropy_other = np.random.normal(1.6, 0.05, n_points)

    # Combine data
    sample_entropy = np.concatenate([sample_entropy_singer, sample_entropy_accelerated, sample_entropy_other])
    labels = np.array(["Singer"] * n_points + ["Accelerated"] * n_points + ["Other"] * n_points)

    # Thresholds for decision boundary
    threshold_1 = 1.0  # Boundary between Singer and Accelerated
    threshold_2 = 1.4  # Boundary between Accelerated and Other

    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.scatter(sample_entropy[:n_points], np.zeros(n_points), label="Singer Motion", color="orange")
    plt.scatter(sample_entropy[n_points:2*n_points], np.zeros(n_points), label="Uniformly Accelerated Motion ", color="blue")
    plt.scatter(sample_entropy[2*n_points:], np.zeros(n_points), label="Uniform Rectilinear Motion", color="green")

    # Add decision boundaries
    plt.axvline(threshold_1, color='red', linestyle='--', label="Threshold 1")
    plt.axvline(threshold_2, color='purple', linestyle='--', label="Threshold 2")

    # Add labels and legend
    plt.title("Decision Boundary Based on Sample Entropy", fontsize=14)
    plt.xlabel("Sample Entropy", fontsize=12)
    plt.yticks([])  # Remove y-axis ticks as they are not needed
    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()
    """
    """
    t = time.time()
    simulate_sampen()
    print(f"Fin de la simulation en {time.time() - t} secondes.")
    """
    # Paths to simulation result CSV files


# Load the simulation results
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
