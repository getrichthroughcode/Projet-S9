import numpy as np





def estimate_v_retrograde(x,T):
    """
    Estimate the velocity of a time series using a retrograde difference scheme.
    
    Parameters:
    x: numpy array of time series data
    T: float, sampling period
    
    Returns:
    v: numpy array of estimated velocity
    """
    # Ensure x is a 1D array
    x = np.ravel(x)
    
    # Initialize velocity array
    v = np.zeros_like(x)
    
    # Calculate velocity
    v[0] = 0
    v[1:] = (x[1:] - x[:-1]) / T
    
    return v

def estimate_a_retrograde(x, T):
    """
    Estimate the acceleration of a time series using a retrograde difference scheme.
    
    Parameters:
    x: numpy array of time series data
    T: float, sampling period
    
    Returns:
    a: numpy array of estimated acceleration
    """
    # Ensure x is a 1D array
    x = np.ravel(x)
    
    # Initialize acceleration array
    a = np.zeros_like(x)
    
    # Calculate acceleration for the valid range
    a[1:-1] = (x[2:] - 2 * x[1:-1] + x[:-2]) / T**2
    
    return a


