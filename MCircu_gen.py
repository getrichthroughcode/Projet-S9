import numpy as np
import matplotlib.pyplot as plt

# circulaire

def Mcircu_gen(length, T, x_0,omega,sigma):
    if (np.shape(x_0) != (4, 1)):
        print("x_0 has not the shape (4,1)")
        return []
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
    x.append(x_0)
    for k in range(1,length):
        U = np.random.randn(4,1)
        R = np.linalg.cholesky(Q)
        B = R @ U
        x.append(np.reshape(Phi @ x[-1],(4,1)) + B)
    return x

if __name__ == "__main__":
    length = 100  # Nb d'echantillons
    T = 1  # période d'échantillonage
    x_0 = np.array([[0], [0], [0], [0]])  # Vecteur initial
    sigma = 1
    omega = 1
    cood_circu = Mcircu_gen(length, T, x_0, omega, sigma)
    x_coord = [xi[0,0] for xi in cood_circu]
    y_coord = [yi[2,0] for yi in cood_circu]

    plt.figure(figsize = (8,8))
    plt.plot(x_coord,y_coord)
    plt.show()

