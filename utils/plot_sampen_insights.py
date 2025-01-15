import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sampen_insights(data_path, metric='Sample Entropy Position', x_param='T', y_param='N', hue_param=None):
    """
    Plots insights from the Sample Entropy simulation results.

    Args:
        data_path (str): Path to the CSV file containing simulation results.
        metric (str): The metric to plot (e.g., 'Sample Entropy Position', 'Sample Entropy Velocity').
        x_param (str): Parameter to plot on the x-axis (e.g., 'T' for sampling period).
        y_param (str): Parameter to plot on the y-axis (e.g., 'N' for number of samples).
        hue_param (str): Parameter to use for color coding (e.g., 'noise', 'm', or 'alpha').
    """
    # Load the data
    data = pd.read_csv(data_path)
    
    # Ensure parameters exist in the data
    if x_param not in data.columns or y_param not in data.columns or metric not in data.columns:
        raise ValueError(f"One of the specified parameters ({x_param}, {y_param}, {metric}) is not in the dataset.")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    if hue_param:
        sns.scatterplot(
            data=data,
            x=x_param,
            y=y_param,
            hue=hue_param,
            size=metric,
            sizes=(50, 200),
            palette='viridis',
            alpha=0.8
        )
        plt.legend(title=hue_param)
    else:
        sns.scatterplot(
            data=data,
            x=x_param,
            y=y_param,
            size=metric,
            sizes=(50, 200),
            color='blue',
            alpha=0.8
        )

    # Customize the plot
    plt.title(f"{metric} vs {x_param} and {y_param}", fontsize=16)
    plt.xlabel(x_param, fontsize=14)
    plt.ylabel(y_param, fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()