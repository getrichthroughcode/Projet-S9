



import numpy as np       
""" Base Sample Entropy function."""

def SampEn(Sig, m=2, tau=1, r=None, Logx=np.exp(1), Vcp=False):

    Sig = np.squeeze(Sig)
    N = Sig.shape[0]  
    if r is None:
        r = 0.2*np.std(Sig)
    
    assert N>10 and Sig.ndim == 1,  "Sig:   must be a numpy vector"
    assert isinstance(m,int) and (m > 0), "m:     must be an integer > 0"
    assert isinstance(tau,int) and (tau > 0), "tau:   must be an integer > 0"
    assert isinstance(r,(int,float)) and (r>=0), "r:     must be a positive value"
    assert isinstance(Logx,(int,float)) and (Logx>0), "Logx:     must be a positive value"
    assert isinstance(Vcp,bool), "Vcp:     must be a Boolean"    

    Counter = (abs(np.expand_dims(Sig,axis=1)-np.expand_dims(Sig,axis=0))<= r)*np.triu(np.ones((N,N)),1)  
    M = np.hstack((m*np.ones(N-m*tau), np.repeat(np.arange(m-1,0,-1),tau)))
    A = np.zeros(m + 1)
    B = np.zeros(m + 1)
    A[0] = np.sum(Counter)
    B[0] = N*(N-1)/2
        
    for n in range(M.shape[0]):
        ix = np.where(Counter[n, :] == 1)[0]
            
        for k in range(1,int(M[n]+1)):              
            ix = ix[ix + (k*tau) < N]
            p1 = np.tile(Sig[n: n+1+(tau*k):tau], (ix.shape[0], 1))                       
            p2 = Sig[np.expand_dims(ix,axis=1) + np.arange(0,(k*tau)+1,tau)]
            ix = ix[np.amax(abs(p1 - p2), axis=1) <= r] 
            if ix.shape[0]:
                Counter[n, ix] += 1
            else:
                break
            
        for k in range(1, m+1):
            A[k] = np.sum(Counter > k)
            B[k] = np.sum(Counter[:,:-(k*tau)] >= k)
            
        with np.errstate(divide='ignore', invalid='ignore'):
            Samp = -np.log(A/B)/np.log(Logx) 
    
        
        if Vcp:
            Temp = np.vstack(np.where(Counter>m)).T
            if len(Temp)>1:
                Ka = np.zeros(len(Temp)-1)             
                for k in range(len(Temp)-1):
                    TF = (abs(Temp[k+1:,:] - Temp[k,0]) <= m*tau) + (abs(Temp[k+1:,:] - Temp[k,1]) <= m*tau)
                    Ka[k] = TF.any(axis=1).sum()
            else:
                Ka = 0
            
            Temp = np.vstack(np.where(Counter[:,:-m*tau] >=m)).T        
            if len(Temp)>1:
                Kb = np.zeros(len(Temp)-1)   
                for k in range(len(Temp)-1):
                    TF = (abs(Temp[k+1:,:] - Temp[k,0]) <= (m-1)*tau) + (abs(Temp[k+1:,:] - Temp[k,1]) <= (m-1)*tau)
                    Kb[k] = TF.any(axis=1).sum()      
            else:
                Kb = 0
                
            Ka = np.sum(Ka)
            Kb = np.sum(Kb)
            CP = A[-1]/B[-1]
            Vcp = (CP*(1-CP)/B[-1]) + (Ka - Kb*(CP**2))/(B[-1]**2)

            return Samp, A, B, (Vcp, Ka, Kb)
        
        else:
            return Samp, A, B    

def Distx(Vex, r):
    Nt = Vex.shape[0]
    Counter = np.zeros((Nt-1,Nt-1),dtype=bool)
    for x in range(Nt-1):
         Counter[x,x:] = np.all(abs(Vex[x+1:,:] - Vex[x,:]) <= r, axis=1)
     
    return Counter

       

def MvSampEn(Data, m=None, tau=None, r=0.2, Norm=False, Logx=np.exp(1)):
    Data = np.squeeze(Data)
    assert Data.shape[0]>10 and Data.ndim==2 and Data.shape[1]>1,  "Data:   must be an NxM numpy matrix where N>10 and M>1"
    N, Dn = Data.shape 
    if m is None:    m = 2*np.ones(Dn, dtype=int)
    if tau is None:  tau = np.ones(Dn, dtype=int)
    m = m.astype(int)
    tau = tau.astype(int) 
  
    # and np.issubdtype(m.dtype, np.integer)
    # and np.issubdtype(tau.dtype, np.integer)
    assert isinstance(m,np.ndarray) and all(m>0) and m.size==Dn and m.ndim==1, "m:     must be numpy vector of M positive integers"
    assert isinstance(tau,np.ndarray) and all(tau>0) and tau.size==Dn and tau.ndim==1, "tau:   must be numpy vector of M positive integers"
    assert isinstance(r,(int,float)) and (r>=0), "r:     must be a positive value"
    assert isinstance(Logx,(int,float)) and (Logx>0), "Logx:     must be a positive value"
    assert isinstance(Norm,bool), "Norm:     must be a Boolean"    

    if Norm: Data = Data/np.std(Data,axis=0)
    
    Nx = N - max((m-1)*tau)
    Ny = N - max(m*tau)       
    Vex = np.zeros((Nx,sum(m)))
    q = 0
    for k in range(Dn):
        for p in range(m[k]):
            Vex[:,q] = Data[p*tau[k]:Nx+p*tau[k],  k]
            q += 1
            
    Count0 = Distx(Vex,r)
    B0 = np.sum(Count0)/(Nx*(Nx-1)/2)
            
    B1 = np.zeros(Dn)
    Vez = np.inf*np.ones((1,sum(m)+1));
    Temp = np.cumsum(m)
    for k in range(Dn):
        Sig = np.expand_dims(Data[m[k]*tau[k]:Ny+m[k]*tau[k], k],1)
        Vey = np.hstack((Vex[:Ny, :Temp[k]], Sig, Vex[:Ny, Temp[k]:]))
        Vez = np.vstack((Vez, Vey))
        Count1 = Distx(Vey, r)
        B1[k] = np.sum(Count1)/(Ny*(Ny-1)/2)
    Vez = Vez[1:,:]
    Count1 = Distx(Vez, r)
    Bt = np.sum(Count1)/(Dn*Ny*((Dn*Ny)-1)/2)
       
    with np.errstate(divide='ignore', invalid='ignore'):
        Samp = -np.log(Bt/B0)/np.log(Logx) 
   
    return Samp, B0, Bt, B1 



def XSampEn(*Sig, m=2, tau=1, r=None, Logx=np.exp(1), Vcp=False):
    assert len(Sig)<=2,  """Input arguments to be passed as data sequences:
        - A single Nx2 numpy matrix with each column representing Sig1 and Sig2 respectively.       \n or \n
        - Two individual numpy vectors representing Sig1 and Sig2 respectively."""
    if len(Sig)==1:
        Sig = np.squeeze(Sig)
        assert max(Sig.shape)>10 and min(Sig.shape)==2,  """Input arguments to be passed as data sequences:
            - A single Nx2 numpy matrix with each column representing Sig1 and Sig2 respectively.       \n or \n
            - Two individual numpy vectors representing Sig1 and Sig2 respectively."""
        if Sig.shape[0] == 2:
            Sig = Sig.transpose()  
        S1 = Sig[:,0]; S2 = Sig[:,1]     
        
    elif len(Sig)==2:
        S1 = np.squeeze(Sig[0])
        S2 = np.squeeze(Sig[1])
        
    N  = S1.shape[0]
    N2 = S2.shape[0]
    if r is None:
        r = 0.2*np.sqrt((np.var(S1)*(N-1) + np.var(S2)*(N2-1))/(N+N2-1))
     
    assert N>10 and N2>10,  "Sig1/Sig2:   Each sequence must be a numpy vector (N>10)"
    assert isinstance(m,int) and (m > 0), "m:     must be an integer > 0"
    assert isinstance(tau,int) and (tau > 0), "tau:   must be an integer > 0"
    assert isinstance(r,(int,float)) and r>=0, "r:     must be a positive value"
    assert isinstance(Logx,(int,float)) and (Logx>0), "Logx:     must be a positive value"
    assert isinstance(Vcp,bool), "Vcp:     must be a Boolean"    

    M = np.hstack((m*np.ones(N-m*tau), np.repeat(np.arange(m-1,0,-1),tau)))   
    Counter = 1*(abs(np.expand_dims(S1,axis=1) - np.expand_dims(S2,axis=0))<= r)
    A = np.zeros(m+1)
    B = np.zeros(m+1)
    A[0] = np.sum(Counter)
    B[0] = N*N2
    
    for n in range(M.shape[0]):
        ix = np.where(Counter[n, :] == 1)[0]        
        for k in range(1,int(M[n]+1)):              
            ix = ix[ix + (k*tau) < N2]
            if not len(ix):    break  
            p1 = np.tile(S1[n: n+1+(tau*k):tau], (ix.shape[0], 1))                       
            p2 = S2[np.expand_dims(ix,axis=1) + np.arange(0,(k*tau)+1,tau)]
            ix = ix[np.amax(abs(p1 - p2), axis=1) <= r] 
            Counter[n, ix] += 1
    
    for k in range(1, m+1):
        A[k] = np.sum(Counter > k)
        B[k] = np.sum(Counter >= k)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        XSamp = -np.log(A/B)/np.log(Logx)
 
    
    if Vcp:
        T1,T2 = np.expand_dims(np.where(Counter>m),axis=1)
        Ka = np.triu(abs(T1-T1.T)<=m*tau,1) + np.triu(abs(T2-T2.T)<=m*tau,1) 

        T1,T2 = np.expand_dims(np.where(Counter[:,:-m*tau]>=m),axis=1)
        Kb = np.triu(abs(T1-T1.T)<=(m-1)*tau,1) + np.triu(abs(T2-T2.T)<=(m-1)*tau,1) 
                    
        Ka = np.sum(Ka)
        Kb = np.sum(Kb)
        CP = A[-1]/B[-1]
        Vcp = (CP*(1-CP)/B[-1]) + (Ka - Kb*(CP**2))/(B[-1]**2)

        return XSamp, A, B, (Vcp, Ka, Kb)
    
    else:
        return XSamp, A, B  
    

def dcca(ts1, ts2, scales, m=1):
    """
    Perform Detrended Cross-Correlation Analysis (DCCA) between two time series.

    Parameters:
    - ts1, ts2: Input time series (1D numpy arrays of the same length).
    - scales: List or array of window sizes (integers).
    - m: Order of the polynomial for detrending (default is 1, linear detrending).

    Returns:
    - F: Array of fluctuation functions for each scale.
    
    """
    ts1 = np.asarray(ts1, dtype=np.float64)
    ts2 = np.asarray(ts2, dtype=np.float64)

    if ts1.shape != ts2.shape:
        raise ValueError("Time series must have the same length.")

    N = len(ts1)
    X = np.cumsum(ts1 - np.mean(ts1))
    Y = np.cumsum(ts2 - np.mean(ts2))

    F = np.zeros(len(scales))

    for idx, s in enumerate(scales):
        if s < m + 2:
            raise ValueError(f"Scale {s} is too small for polynomial order {m}.")

        segments = N // s
        cov = []

        for v in range(segments):
            idx_start = v * s
            idx_end = idx_start + s

            # Indices for the segment
            indices = np.arange(idx_start, idx_end)

            # Fit polynomials to the segments
            coeffs_X = np.polyfit(indices, X[idx_start:idx_end], m)
            trend_X = np.polyval(coeffs_X, indices)
            F_X = X[idx_start:idx_end] - trend_X

            coeffs_Y = np.polyfit(indices, Y[idx_start:idx_end], m)
            trend_Y = np.polyval(coeffs_Y, indices)
            F_Y = Y[idx_start:idx_end] - trend_Y

            # Compute covariance of residuals
            cov_seg = np.mean(F_X * F_Y)
            cov.append(cov_seg)

        # Average covariance over all segments
        F[idx] = np.sqrt(np.mean(cov))

    return F
