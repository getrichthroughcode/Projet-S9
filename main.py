import numpy as np
import matplotlib.pyplot as plt

g =9.81
n = 2
T = 1
N = 100
omega = 1
q = n*g*T
sigma = 1

# circulaire


Phi = np.array([
    [1, np.sin(omega * T) / omega, 0, -(1 - np.cos(omega * T)) / omega],
    [0, np.cos(omega * T), 0, -np.sin(omega * T)],
    [0, (1 - np.cos(omega * T)) / omega, 1, np.sin(omega * T) / omega],
    [0, np.sin(omega * T), 0, np.cos(omega * T)]
])

# Define matrix Q
Q = sigma * np.array([
    [2 * (omega * T - np.sin(omega * T)) / omega**3, (1 - np.cos(omega * T)) / omega**2, 0, (omega*T - np.sin(omega * T)) / omega**2],
    [(1 - np.cos(omega * T)) / omega**2, T, - (omega*T - np.sin(omega * T)) / omega**2, 0],
    [0, - (omega*T - np.sin(omega * T)) / omega**2, 2 * (omega * T - np.sin(omega * T)) / omega**3, (1 - np.cos(omega * T)) / omega**2],
    [(omega*T - np.sin(omega * T) / omega**2), 0, (1 - np.cos(omega * T)) / omega**2, T]
])

x = []
x.append(np.array([[0],[0],[0],[0]]))
for k in range(1,N):
    U = np.random.randn(4,1)
    R = np.linalg.cholesky(Q)
    B = R.T @ U
    x.append(np.reshape(Phi @ x[-1],(4,1)) + B)

x_coord = [xi[0,0] for xi in x]
y_coord = [yi[2,0] for yi in x]

plt.figure(figsize = (8,8))
plt.plot(x_coord,y_coord)
plt.show()


