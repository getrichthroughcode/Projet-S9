import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from estimation_param import estimate
from whiteness_test import *

#valeur de T => trop gd attention
def Singer_gen(length, T, x_0,alpha,sigma_m ):
    L=[]
    L.append(x_0)  # Ensure x_0 is a column vector
    q11 = (1 / (2 * alpha ** 5)) * (
                2 * alpha * T - 2 * alpha ** 2 * T ** 2 + 2 * alpha ** 3 * T ** 3 / 3 - 4 * alpha * T * np.exp(
            -alpha * T) - np.exp(-2 * alpha * T) + 1)
    q12 = (1 / (2 * alpha ** 4)) * (alpha ** 2 * T ** 2 + 1 + np.exp(-2 * alpha * T) + np.exp(-alpha * T) * (-2 + 2 * alpha * T)- 2 * alpha * T)
    q13 = (1 / (2 * alpha ** 3)) * (1 - 2 * alpha * T * np.exp(-alpha * T) - np.exp(-2 * alpha * T))
    q22 = (1 / (2 * alpha ** 3)) * (2 * alpha * T - 3 + 4 * np.exp(-alpha * T) - np.exp(-2 * alpha * T))
    q23 = (1 / (2 * alpha ** 2)) * (1 - np.exp(-alpha * T)) ** 2
    q33 = -(1 / (2 * alpha)) * (np.exp(-2 * alpha * T) - 1)

    Q = 2*alpha*sigma_m**2 * np.array([
        [q11, q12, q13],
        [q12, q22, q23],
        [q13,  q23, q33]
    ])
    phi = np.array([[1, T, (1 / alpha ** 2) * (-1 + alpha * T + np.exp(-alpha * T))],
                    [0, 1, (1 / alpha) * (1 - np.exp(-alpha * T))],
                    [0, 0, np.exp(-alpha * T)]])
    for i in range(length):
        U = np.random.randn(3, 1)  # Generate a random vector
        R = np.linalg.cholesky(Q)  # Cholesky decomposition
        B = R @ U         # Generate the noise vector
        # Update x with the new state

        x_new  =phi @ L[-1] + B
        L.append(x_new)

    return L
if __name__ == "__main__":
    length = 100
    alpha=0.1
    sigma_m = 1
    T=1
    x_0=np.array([[0],[0],[0]])

    x=Singer_gen(length, T, x_0,alpha,sigma_m)
    y=Singer_gen(length, T, x_0,alpha,sigma_m)
    z = Singer_gen(length, T, x_0,alpha,sigma_m)
    x_coords = [xi[0, 0] for xi in x]
    y_coords = [yi[0,0] for yi in y]
    z_coords = [yi[0,0] for yi in z]
    x_accs = [xi[2, 0] for xi in x]
    y_accs = [yi[2,0] for yi in y]
    x_vits = [xi[1, 0] for xi in x]
    y_vits = [yi[1, 0] for yi in y]
    x_vits_est, x_accs_est = estimate(x_coords)
    y_vits_est, y_accs_est = estimate(y_coords)
    # plt.figure(figsize=(10, 6))
    # plt.plot(x_coords, y_coords, label='Trajectoire (x, y)')
    # plt.quiver(x_coords, y_coords, x_vits, y_vits, angles='xy', scale_units='xy', scale=0.5, color='r',
    #            label='Vitesse instantanée réel')
    # plt.title('Trajectoire synthétique avec vitesse instantanée réel')
    # plt.xlabel('Position x')
    # plt.ylabel('Position y')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot3D(x_coords, y_coords, z_coords)
    # ax.set_title('Trajectoire 3D')
    # ax.set_xlabel('Position x')
    # ax.set_ylabel('Position y')
    # ax.set_zlabel('Position z')
    # plt.show()

    correlation_x = np.correlate(x_accs, x_accs, mode='full')
    correlation_y= np.correlate(y_accs, y_accs, mode='full')
    lags_x = np.arange(-len(x_accs) + 1, len(x_accs))
    lags_y = np.arange(-len(y_accs) + 1, len(y_accs))

    # plt.figure(figsize=(10, 6))
    # plt.plot( x_vits_est,label='Estimation')
    # plt.plot(x_vits,label='Reelle')
    # plt.title("Comparaison vitesse x")
    # plt.legend()
    # plt.grid()
    # plt.show()
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot( x_accs_est, label='Estimation')
    # plt.plot(x_accs, label='Reelle')
    # plt.title("Comparaison acceleration x")
    # plt.grid()
    # plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(lags_x, correlation_x)
    # plt.plot(lags_x,np.correlate([10*elm for elm in x_accs_est], [10*elm for elm in x_accs_est], mode='full'))
    plt.title("Fonction de corrélation en x")
    plt.grid()
    plt.show()
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(lags_y, correlation_y)
    # # plt.plot(lags_y,np.correlate([10*elm for elm in y_accs_est], [10*elm for elm in y_accs_est], mode='full'))
    # plt.title("Fonction de corrélation en y")
    # plt.grid()
    # plt.show()