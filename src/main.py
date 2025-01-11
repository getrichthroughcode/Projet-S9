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
    print("1.1 Influence de la période d'échantillonnage et du facteur de bruit sur la sample entropy")
    print("Simulation effectuée sur les positions exactes du modèle")
    print("---------------------------------------------------------------------------------")
    # Paramètres de simulation fixe
    N = 2000  # Nombre d'échantillons
    nrealisations = 100  # Nombre de réalisations
    x_0 = np.array([0,0])
    print(f"Nombre d'échantillons à tester: {N}")

    # Paramètres à tester
    Ts = [0.01, 0.1, 0.5, 1,10,15,20,60]  # Période d'échantillonnage en secondes
    n = [1, 2, 5, 10, 15, 20, 30, 50]  # Facteur de bruit
    results1_1 = []
    for T in Ts:
        for noise in n:
            samp_pos_all = []
            for i in tqdm.tqdm(range(nrealisations), desc=f"RSimulation pour T={T} s et n={noise}"):
                L = MRU_gen(N, T, x_0, noise)
                x = L[:,0]
                samp = sampen(x, 2, 0.2)
                samp_pos_all.append(samp)
            results1_1.append({
                'T': T, 
                'n': noise,
                'Sample Entropy': np.mean(samp_pos_all),
                'Std': np.std(samp_pos_all)})    
    results_frame1_1 = pd.DataFrame(results1_1)
    print("Fin de la simulation 1.1")

    print("---------------------------------------------------------------------------------")
    return results_frame1_1




if __name__ == "__main__":
    """
    t1 = time.time() 
    test_SampEn()
    print(f"Execution time: {time.time() - t1} s")
    plot_custom_signals()
    """
    t2 = time.time()
    results1_1 = simulate_MRU()
    print(results1_1)
    print(f"Execution time: {time.time() - t2} s")
    
    # Save results
    results1_1.to_csv("results/simulation_MRU.csv", index=False)
    print("Results saved successfully.")
    
    # Plot results
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for T in results1_1['T'].unique():
        df = results1_1[results1_1['T'] == T]
        ax.errorbar(df['n'], df['Sample Entropy'], label=f"T={T} s")
    ax.set_xlabel("Facteur de bruit n")
    ax.set_ylabel("Sample Entropy")
    ax.set_title("Influence de la période d'échantillonnage et du facteur de bruit sur la sample entropy")
    ax.legend()
    plt.show()
    
    #Plot surface rsults
    fig3 = plt.figure(figsize=(12, 8))
    ax_b = fig3.add_subplot(111, projection='3d')

    # Create a meshgrid from the unique values of 'N' and 'n'
    T_unique_T = results1_1['T'].unique()
    n_unique_T = results1_1['n'].unique()
    N_T, n_T = np.meshgrid(T_unique_T, n_unique_T)


    sampen_pos_2d_T = results1_1['Sample Entropy'].values.reshape(len(n_unique_T), len(T_unique_T))


    surface_b = ax_b.plot_surface(N_T, n_T, sampen_pos_2d_T, cmap='viridis', edgecolor='k')

    # Customize the view angle for better visualization
    ax_b.view_init(65, 112)

    # Add titles and labels
    ax_b.set_title('3D Visualization of Sample Entropy for MRU: True Positions', fontsize=16, fontweight='bold', pad=20)
    ax_b.set_xlabel('T (Sampling)', fontsize=12, labelpad=10)
    ax_b.set_ylabel('n (Noise Factor)', fontsize=12, labelpad=10)
    ax_b.set_zlabel('Mean(Sample Entropy)', fontsize=12, labelpad=10)

    # Add a color bar to indicate the scale of Sample Entropy
    cbar_b = fig3.colorbar(surface_b, shrink=0.5, aspect=10)
    cbar_b.set_label('Mean(Sample Entropy)', fontsize=12)

    # Show the plot
    plt.show()
