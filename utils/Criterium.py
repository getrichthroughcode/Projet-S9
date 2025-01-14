import numpy as np
from math import log, isinf
from scipy.spatial.distance import pdist
import time


def sampen(signal, m, r, dist_type=None):
    """
    -----------------------------------------------------------------------
    #                           H    Y    D    R    A
    -----------------------------------------------------------------------
    # Function 'sampen' computes the Sample Entropy of a given signal.
    #
    #   Input parameters:
    #       - signal:       Signal vector with dims. [1xN]
    #       - m:            Embedding dimension (m < N).
    #       - r:            Tolerance (percentage applied to the SD).
    #       - dist_type:    (Optional) Distance type, specified by a string.
    #                       Default value: 'chebychev' (type help pdist for
    #                       further information).
    #
    #   Output variables:
    #       - value:        SampEn value. Since SampEn is not defined whenever
    #                       B = 0, the output value in that case is NaN in
    #                       the original reference. Here, the script sets it
    #                       to Inf initially, then bounds it at the end.
    #
    #   Versions:
    #       - 1.0:          (21/09/2018) Original script (MATLAB).
    #       - 1.1:          (09/11/2018) Upper bound is added. Now, SampEn is
    #                       not able to return Inf values.
    #
    #   Python translation: 1.0
    #       - Author:       (Adapted from V. Martínez-Cagigal)
    #       - Date:         (Python translation: 11/01/2025)
    #
    #   References:
    #       [1] Richman, J. S., & Moorman, J. R. (2000). Physiological
    #           time-series analysis using approximate entropy and sample
    #           entropy. American Journal of Physiology-Heart and
    #           Circulatory Physiology, 278(6), H2039-H2049.
    #
    -----------------------------------------------------------------------
    """

    # Error detection and defaults
    if signal is None or m is None or r is None:
        raise ValueError("Not enough parameters. You must specify signal, m, and r.")

    # If dist_type is not provided, default to 'chebychev' (like the MATLAB code).
    if dist_type is None:
        dist_type = 'chebychev'
        #print("[WARNING] Using default distance method: 'chebychev'.")

    # Check that signal is a vector
    if not isinstance(signal, (list, np.ndarray)):
        raise ValueError("The 'signal' parameter must be a list or numpy array.")

    # Check that dist_type is a string
    if not isinstance(dist_type, str):
        raise ValueError("Distance type must be a string.")

    signal = np.array(signal, dtype=float).flatten()
    if m > len(signal):
        raise ValueError("Embedding dimension must be smaller than the signal length (m < N).")

    # Useful parameters
    N = len(signal)         # Signal length
    sigma = np.std(signal)  # Standard deviation

    # ----------------------------------------------------------------------
    # Create the matrix of matches (similar to the MATLAB approach):
    #
    # In MATLAB (1-based indexing), we did:
    #
    #   matches = NaN(m+1, N);
    #   for i = 1:1:m+1
    #       matches(i,1:N+1-i) = signal(i:end);
    #   end
    #   matches = matches';
    #
    # In Python (0-based indexing), replicate exactly:
    # ----------------------------------------------------------------------
    matches = np.full((m+1, N), np.nan)
  
    for i in range(m+1):
        matches[i, 0:N - i] = signal[i:]
    matches = matches.T  # Transpose
  

    # ----------------------------------------------------------------------
    # Check the matches for dimension 'm'
    #   d_m = pdist(matches(:,1:m), dist_type)
    #   If empty => value=Inf (no pairs => B=0)
    # ----------------------------------------------------------------------
    d_m = pdist(matches[:, :m], dist_type)
    

    # If there are no pairwise distances (i.e., array is empty), then B=0
    if d_m.size == 0:
        # If B = 0, SampEn is not defined: no regularity detected
        # The MATLAB code sets value=Inf, then bounds it if isinf(value).
        value = float('inf')
    else:
        # Check the matches for m+1
        d_m1 = pdist(matches[:, :m+1], dist_type)
        
        
        # Compute A and B
        B = np.sum(d_m <= r * sigma)
        A = np.sum(d_m1 <= r * sigma)
        

        # If B=0 or A=0, the log term will blow up => infinite.
        if B == 0 or A == 0:
            value = float('inf')
        else:
            # Sample Entropy value
            # norm. factor: [nchoosek(N-m+1,2)/nchoosek(N-m,2)] = ((N-m+1)/(N-m-1))
            value = -log((A / B) * ((N - m + 1) / (N - m - 1)))

    # ----------------------------------------------------------------------
    # If A=0 or B=0 => we got Inf above.
    # The lowest non-zero conditional probability that SampEn should report
    # is A/B = 2 / [(N-m-1)*(N-m)]
    # So we bound the infinite value here:
    #
    #   if isinf(value)
    #       value = -log(2/((N-m-1)*(N-m)));
    # ----------------------------------------------------------------------
    if isinf(value):
        # Lower bound: 0
        # Upper bound in code: log(N-m)+log(N-m-1)-log(2)
        # => We set it with -log(2 / ((N - m - 1) * (N - m)))
        value = -log(2 / ((N - m - 1) * (N - m)))

    return value



def DCCA(x, y, s):
    """
    Detrended cross-correlation coefficient as described in:
    Podobnik, B. & Stanley, H. Detrended cross-correlation analysis:
    a new method for analyzing two non-stationary time series.
    Phys. Rev. Lett. 100, 084102, DOI: 10.1103/PhysRevLett.100.084102 (2008).

    Parameters:
    x, y: numpy arrays of time series data
    s: int, window size

    Returns:
    rho_DCCA: DCCA coefficient
    F_DCCA, F_DFA_X, F_DFA_Y: Covariance and variances used in the calculation
    """
    # Ensure time series have the same length
    assert len(x) == len(y), "Time series must be the same length"

    # Ensure x and y are 1D arrays
    x = np.ravel(x)
    y = np.ravel(y)

    # Integrated time series
    X = np.cumsum(x - np.mean(x))
    Y = np.cumsum(y - np.mean(y))

    # Number of overlapping windows
    N = len(x) - s + 1

    # Initialize variables
    F_DCCA = 0
    F_DFA_X = 0
    F_DFA_Y = 0

    # Loop over all windows
    for k in range(N):
        # Extract window
        X_win = X[k:k + s]
        Y_win = Y[k:k + s]

        # Fit linear trends
        p_X = np.polyfit(np.arange(s), X_win, 1)
        p_Y = np.polyfit(np.arange(s), Y_win, 1)

        # Detrend
        X_detrend = X_win - np.polyval(p_X, np.arange(s))
        Y_detrend = Y_win - np.polyval(p_Y, np.arange(s))

        # Update covariance and variances
        F_DCCA += np.sum(X_detrend * Y_detrend) / (s - 1)
        F_DFA_X += np.sum(X_detrend**2) / (s - 1)
        F_DFA_Y += np.sum(Y_detrend**2) / (s - 1)

    # Normalize by the number of windows
    F_DCCA /= N
    F_DFA_X = np.sqrt(F_DFA_X / N)
    F_DFA_Y = np.sqrt(F_DFA_Y / N)

    # Compute DCCA coefficient
    rho_DCCA = F_DCCA / (F_DFA_X * F_DFA_Y)

    return rho_DCCA, F_DCCA, F_DFA_X, F_DFA_Y



def sampen_multivariate(signal, m, r, dist_type=None):
    """
    Computes the Sample Entropy for a multivariate signal (e.g., 2D or higher).
    
    Parameters:
        signal: np.ndarray
            Multivariate signal array of shape (N, d), where N is the number of samples
            and d is the number of dimensions.
        m: int
            Embedding dimension (m < N).
        r: float
            Tolerance (percentage applied to the SD of the signal in each dimension).
        dist_type: str, optional
            Distance type, specified as a string. Default is 'euclidean'.
    
    Returns:
        float: Sample Entropy value for the multivariate signal.
    """
    # Error checking
    if signal is None or m is None or r is None:
        raise ValueError("Not enough parameters. You must specify signal, m, and r.")
    
    if not isinstance(signal, (list, np.ndarray)):
        raise ValueError("The 'signal' parameter must be a list or numpy array.")
    
    signal = np.array(signal, dtype=float)
    if len(signal.shape) == 1:  # Convert 1D signal to 2D for consistency
        signal = signal[:, np.newaxis]
    
    N, d = signal.shape  # N: number of samples, d: dimensions
    
    if m >= N:
        raise ValueError("Embedding dimension must be smaller than the number of samples (m < N).")
    
    if dist_type is None:
        dist_type = 'euclidean'  # Default distance metric

    # Calculate standard deviation for each dimension
    sigma = np.std(signal, axis=0)

    # Create the matrix of matches (time-delayed vectors) for dimension m and m+1
    matches_m = np.array([signal[i:i + m] for i in range(N - m)])
    matches_m1 = np.array([signal[i:i + m + 1] for i in range(N - m - 1)])

    # Calculate pairwise distances for matches of size m and m+1
    d_m = pdist(matches_m.reshape(-1, d * m), metric=dist_type)
    d_m1 = pdist(matches_m1.reshape(-1, d * (m + 1)), metric=dist_type)

    # Apply the tolerance threshold
    threshold_m = r * np.linalg.norm(sigma)  # Multivariate threshold
    B = np.sum(d_m <= threshold_m)
    A = np.sum(d_m1 <= threshold_m)

    # Compute SampEn
    if B == 0 or A == 0:
        value = float('inf')  # No regularity detected
    else:
        value = -np.log((A / B) * ((N - m + 1) / (N - m - 1)))

    # Bound the result to avoid infinite values
    if np.isinf(value):
        value = -np.log(2 / ((N - m - 1) * (N - m)))

    return value




