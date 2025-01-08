import numpy as np
import matplotlib.pyplot as plt
import time

import estimation_param


def estimate(X_pos, T):
    X_vit = []
    X_acc = []
    # inte = L.T@np.random.randn(3, 1).tolist()
    X_pos = X_pos
    for i in range(0, len(X_pos) - 2):
        # inte = np.sqrt(9.81) * L.T @ np.random.randn(3, 1).tolist()
        vit = float((X_pos[i + 1] - X_pos[i]) / T)
        X_vit.append(vit)
        acc = float((X_pos[i + 2] - 2 * X_pos[i + 1] + X_pos[i]) / T ** 2)
        X_acc.append(acc)
    X_vit.insert(0, (X_pos[1] - X_pos[0]) / T)
    X_vit.append((X_pos[-1] - X_pos[-2]) / T)
    X_acc.insert(0, (X_pos[2] - 2 * X_pos[1] + X_pos[0]) / (T ** 2))
    X_acc.append((X_pos[-1] - 2 * X_pos[-2] + X_pos[-3]) / (T ** 2))
    return (X_vit, X_acc)  #S.median_filter()


def Singer_gen(length, T, x_0, alpha, sigma_m, Q):
    L = np.zeros((length + 1, 3))  # Préallouer un tableau pour tous les états
    L[0] = np.resize(x_0,(3,))  # Initialisation avec l'état initial
    phi = np.array([[1, T, (1 / alpha ** 2) * (-1 + alpha * T + np.exp(-alpha * T))],
                    [0, 1, (1 / alpha) * (1 - np.exp(-alpha * T))],
                    [0, 0, np.exp(-alpha * T)]])
    R = np.linalg.cholesky(Q)  # Cholesky decomposition
    for i in range(length):
        U = np.random.randn(3,)  # Generate a random vector
        B = R @ U  # Generate the noise vector
        # Update x with the new state

        L[i + 1] = phi @ L[i] + B  # Calcule le nouvel état

    return L


if __name__ == "__main__":
    length = 30000  # Nb d'echantillons
    T = 1  # période d'échantillonage
    x_0 = np.array([[0], [0], [0]])  # Vecteur initial
    alpha = 0.1
    n = 0.05
    sigma_m = n * 9.81 * T
    sigma_n = 1
    q11 = (1 / (2 * alpha ** 5)) * (
                2 * alpha * T - 2 * alpha ** 2 * T ** 2 + 2 * alpha ** 3 * T ** 3 / 3 - 4 * alpha * T * np.exp(
            -alpha * T) - np.exp(-2 * alpha * T) + 1)
    q12 = (1 / (2 * alpha ** 4)) * (alpha ** 2 * T ** 2 + 1 + np.exp(-2 * alpha * T) + np.exp(-alpha * T) * (
            -2 + 2 * alpha * T) - 2 * alpha * T)
    q13 = (1 / (2 * alpha ** 3)) * (1 - 2 * alpha * T * np.exp(-alpha * T) - np.exp(-2 * alpha * T))
    q22 = (1 / (2 * alpha ** 3)) * (2 * alpha * T - 3 + 4 * np.exp(-alpha * T) - np.exp(-2 * alpha * T))
    q23 = (1 / (2 * alpha ** 2)) * (1 - np.exp(-alpha * T)) ** 2
    q33 = -(1 / (2 * alpha)) * (np.exp(-2 * alpha * T) - 1)

    Q = 2 * alpha * sigma_m ** 2 * np.array([
        [q11, q12, q13],
        [q12, q22, q23],
        [q13, q23, q33]
    ])

    alpha_11 = np.sqrt(2 * alpha) * sigma_m * np.sqrt(q11)
    alpha_21 = np.sqrt(2 * alpha) * sigma_m * q12 / np.sqrt(q11)
    alpha_22 = np.sqrt(2 * alpha) * sigma_m * np.sqrt(q22 - q12 ** 2 / q11)
    alpha_31 = np.sqrt(2 * alpha) * sigma_m * q13 / np.sqrt(q11)
    alpha_32 = np.sqrt(2 * alpha) * sigma_m * (q23 - q12 * q13 / q11) / np.sqrt(q22 - q12 ** 2 / q11)
    alpha_33 = np.sqrt(2 * alpha) * sigma_m * np.sqrt(
        q33 - q13 ** 2 / q11 - ((q23 - q12 * q13 / q11) / np.sqrt(q22 - q12 ** 2 / q11)) ** 2)

    X_Sin = Singer_gen(length, T, x_0, alpha, sigma_m, Q)

    x_coords_Sin = np.array([xi[0] for xi in X_Sin])

    _, x_accs_Sin = estimate(x_coords_Sin, T)

    rho = np.exp(-alpha * T)
    A = 4 / (alpha ** 2 * T ** 2) * np.sinh(alpha * T / 2) ** 2
    B = alpha_11 / T ** 2
    delta = -1 / (T ** 2 * alpha ** 2) * (-1 - alpha * T + 1 / rho)
    C = alpha_21 / T - alpha_11 / T ** 2 + alpha_31 * delta
    D = alpha_22 / T + alpha_32 * delta
    F = alpha_33 * delta

    R_0 = A * (A * sigma_m ** 2 + 2 * (C * alpha_31 + D * alpha_32 + F * alpha_33)) + B ** 2 + C ** 2 + D ** 2 + F ** 2
    R_1 = rho * A * (A * sigma_m ** 2 + C * alpha_31 + D * alpha_32 + F * alpha_33) + B * (A * alpha_31 + C)
    R_2 = rho ** 2 * A * (A * sigma_m ** 2 + C * alpha_31 + D * alpha_32 + F * alpha_33) + A * B * rho * alpha_31

    R_0_bruit = 6 * sigma_n ** 2 / T ** 4
    R_1_bruit = -4 * sigma_n ** 2 / T ** 4
    R_2_bruit = sigma_n ** 2 / T ** 4

    R_h = np.array([[R_0 + R_0_bruit], [R_1 + R_1_bruit], [R_2 + R_2_bruit]])
    # _, x_accs_bruit = estimate(np.sqrt(Sigma[1,0])*np.random.randn(len(x_coords_Sin)), T)
    # x_accs_Sin = np.array(x_accs_Sin) - np.array(x_accs_bruit)

    print("autocorrelation théorique en 0 :", R_0)

    correlation_x_Sin = np.correlate(x_accs_Sin, x_accs_Sin, mode='full') / len(x_accs_Sin)

    plt.figure(figsize=(14, 6))

    print("autocorrelation réelle en 0 :", correlation_x_Sin[len(correlation_x_Sin) // 2])

    R_th = np.zeros_like(correlation_x_Sin)
    R_th[len(correlation_x_Sin) // 2 - 1] = R_1
    R_th[len(correlation_x_Sin) // 2 + 1] = R_th[len(correlation_x_Sin) // 2 - 1]
    R_th[len(correlation_x_Sin) // 2] = R_0
    R_th[len(correlation_x_Sin) // 2 - 2] = R_2
    R_th[len(correlation_x_Sin) // 2 + 2] = R_th[len(correlation_x_Sin) // 2 - 2]

    test = np.array([[2/3/T,1/6/T,0],[6/T**4,-4/T**4,1/T**4]])
    test = test @ np.array([[2/3/T,6/T**4],[1/6/T,-4/T**4],[0,1/T**4]])
    inv_test = np.linalg.inv(test)
    inv = 36*T**10/501*np.array([[53/T**8,-10/3/T**5],[-10/3/T**5,17/36/T**2]])
    # plt.plot(acc_est_x,label='Estimation')
    # plt.plot(x_accs,label='Réelle')

    plt.plot(np.arange(-10, 11), correlation_x_Sin[len(correlation_x_Sin) // 2 - 10: len(correlation_x_Sin) // 2 + 11],
             label='Estimation Calculée')
    plt.plot(np.arange(-10, 11), R_th[len(R_th) // 2 - 10: len(R_th) // 2 + 11], 'x', label='Estimation théorique')
    E = sigma_m ** 2 * np.exp(-np.abs(np.arange(-10, 11)) * alpha)
    plt.plot(np.arange(-10, 11), E,'.' , label='Valeur théorique')
    plt.title('Comparaison entre la corrélation théorique, la correlation théorique estimée et la corrélation calculée')
    plt.legend()
    plt.grid(True)

    plt.show()
    X_Sin_h = Singer_gen(length, T, x_0, 50, sigma_m, Q)
    Y_Sin_h = Singer_gen(length, T, x_0, 50, sigma_m, Q)
    X_Sin_b = Singer_gen(length, T, x_0, 0.01, sigma_m, Q)
    Y_Sin_b = Singer_gen(length, T, x_0, 0.01, sigma_m, Q)

    x_coords_Sin_h = [xi[0] for xi in X_Sin_h]
    y_coords_Sin_h = [yi[0] for yi in Y_Sin_h]
    x_coords_Sin_b = [xi[0] for xi in X_Sin_b]
    y_coords_Sin_b = [yi[0] for yi in Y_Sin_b]

    plt.figure(figsize=(10, 6))

    plt.show()
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(x_coords_Sin_h, y_coords_Sin_h, label='Trajectoire (x, y)')
    plt.title('Trajectoire synthétique dans le plan (x, y) avec alpha grand (alpha = 50)')
    plt.xlabel('Position x')
    plt.ylabel('Position y')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(x_coords_Sin_b, y_coords_Sin_b, label='Trajectoire (x, y)')
    plt.title('Trajectoire synthétique dans le plan (x, y) avec alpha petit (alpha = 0.1)')
    plt.xlabel('Position x')
    plt.ylabel('Position y')
    plt.legend()
    plt.grid(True)

    # Display the combined plot
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()
    print(' rho estimé : ', estimation_param.Singer_param_estimation(x_coords_Sin,T))