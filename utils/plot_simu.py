import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def plot_simu_MRU_aligned(file_path: str):
    """
    Improved visualization of the MRU motion simulation with aligned axis origins.
    
    Parameters:
    file_path: str, the path to the file containing the simulation data
    
    Returns:
    None
    """

    # Read the data
    data = pd.read_csv(file_path)
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create a meshgrid from the unique values of 'N' and 'n'
    N_unique = data['T'].unique()
    n_unique = data['sigma_m'].unique()
    N, n = np.meshgrid(N_unique, n_unique)
    
    sampen_pos_2d = data['Sample Entropy'].values.reshape(len(n_unique), len(N_unique))

    # Plot the surface
    surface = ax.plot_surface(N, n, sampen_pos_2d, cmap='viridis', edgecolor='none', alpha=0.9)
    
    # Add a wireframe for better surface visibility
    ax.plot_wireframe(N, n, sampen_pos_2d, color='k', linewidth=0.5, alpha=0.5)

    # Add contour lines at the base
    ax.contour(N, n, sampen_pos_2d, zdir='z', offset=0, cmap='viridis', linestyles="dotted")

    # Customize the view angle for better visualization
    ax.view_init(30, -60)

    # Add titles and labels
    ax.set_title('Improved 3D Visualization of Sample Entropy', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('N (Number of samples)', fontsize=12, labelpad=10)
    ax.set_ylabel('n (Noise Factor)', fontsize=12, labelpad=10)
    ax.set_zlabel('Mean(Sample Entropy)', fontsize=12, labelpad=10)

    # Align axes to the same origin (0, 0, 0)
    ax.set_xlim([0, max(N_unique)])
    ax.set_ylim([0, max(n_unique)])
    ax.set_zlim([0, np.max(sampen_pos_2d)])
    ax.set_box_aspect([1, 1, 0.5])  # Maintain aspect ratio for better visualization

    # Add a color bar to indicate the scale of Sample Entropy
    cbar = fig.colorbar(surface, shrink=0.6, aspect=12, pad=0.1)
    cbar.set_label('Mean(Sample Entropy)', fontsize=12)

    # Show the plot
    plt.tight_layout()
    plt.show()