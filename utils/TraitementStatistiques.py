



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
