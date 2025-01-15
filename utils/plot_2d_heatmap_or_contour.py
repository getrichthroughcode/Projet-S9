import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_2d_heatmap_or_contour(data_path, metric='Sample Entropy Position', x_param='T', y_param='N', plot_type='heatmap'):
    """
    Plots a 2D heatmap or contour plot of the Sample Entropy metrics.

    Args:
        data_path (str): Path to the CSV file containing simulation results.
        metric (str): The metric to visualize (e.g., 'Sample Entropy Position', 'Sample Entropy Velocity').
        x_param (str): Parameter to plot on the x-axis (e.g., 'T' for sampling period).
        y_param (str): Parameter to plot on the y-axis (e.g., 'N' for number of samples).
        plot_type (str): Type of plot ('heatmap' or 'contour').
    """
    # Load the data
    data = pd.read_csv(data_path)
    
    # Ensure parameters exist in the data
    if x_param not in data.columns or y_param not in data.columns or metric not in data.columns:
        raise ValueError(f"One of the specified parameters ({x_param}, {y_param}, {metric}) is not in the dataset.")
    
    # Pivot the data for 2D visualization
    data_pivot = data.pivot_table(index=y_param, columns=x_param, values=metric)
    
    # Generate the plot
    plt.figure(figsize=(12, 8))
    if plot_type == 'heatmap':
        sns.heatmap(data_pivot, cmap='viridis', annot=True, fmt=".2f", cbar_kws={'label': metric})
        plt.title(f"Heatmap of {metric} ({y_param} vs {x_param})", fontsize=16)
    elif plot_type == 'contour':
        X, Y = np.meshgrid(data_pivot.columns, data_pivot.index)
        Z = data_pivot.values
        plt.contourf(X, Y, Z, cmap='viridis')
        plt.colorbar(label=metric)
        plt.title(f"Contour Plot of {metric} ({y_param} vs {x_param})", fontsize=16)
    else:
        raise ValueError("Invalid plot_type. Choose 'heatmap' or 'contour'.")
    
    # Customize the plot
    plt.xlabel(x_param, fontsize=14)
    plt.ylabel(y_param, fontsize=14)
    plt.tight_layout()
    plt.show()