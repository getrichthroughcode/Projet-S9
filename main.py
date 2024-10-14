import numpy as np
import matplotlib.pyplot as plt

g =9.81
n = 2
T = 1
N = 1028
w = 2
q = n*g*T

# circulaire


Q = q * np.array([[T**3/3,T**2/2,0,0],
                 [T**2/2,T,0,0],
                 [0,0,T**3/3,T**2/2],
                 [0,0,T**2/2,T]])

Phi = np.array([[1,np.sin(w*T)/w,(1-np.cos(w*T))/w**2,0,0,0],
                [0,np.cos(w*T),np.sin(w*T)/w,0,0,0],
                [0,-w*np.sin(w*T),np.cos(w*T),0,0,0],
                [0,0,0,1,np.sin(w*T)/w,(1-np.cos(w*T))/w**2],
                [0,0,0,0,np.cos(w*T),np.sin(w*T)/w],
                [0,0,0,0,-w*np.sin(w*T)/w**2,np.cos(w*T)]])

x= np.zeros((6,N))

x[:,0]=np.array([0,1,0,1,0,1])

for k in range(1,N):
    U = np.random.randn(6,1)
    R = np.linalg.cholesky(Q)
    B = R.T @ U
    x[:,k] = Phi@x[:,k-1] + B
