import numpy as np
import matplotlib.pyplot as plt


def simulate_curvilinear_motion(T, length, x_0):
    states = []
    states.append(x_0)

    Q = np.diag([T ** 3 / 3, T ** 2 / 2, T ** 3 / 3, T ** 2 / 2, 0.1, 0.1])

    for k in range(1, length):
        x, v_x, y, v_y, a_t, a_n = states[k - 1]

        theta_k = np.arctan2(v_y, v_x)

        # Définir la matrice de transition d'état F
        F = np.array([
            [1, T, 0, 0, 0, 0],
            [0, 1, 0, 0, T * np.cos(theta_k), -T * np.sin(theta_k)],
            [0, 0, 1, T, 0, 0],
            [0, 0, 0, 1, T * np.sin(theta_k), T * np.cos(theta_k)],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Bruit de processus
        w_k = np.random.multivariate_normal(mean=[0, 0, 0, 0, 0, 0], cov=Q)

        states.append( F @ states[k - 1] + w_k)

    return states

if __name__ == "__main__":
    T = 1.0
    length = 500
    x_0 = np.array([0, 1, 0, 1, 0.5, 0.2])  # positions, vitesses, accélérations initiales

    states = simulate_curvilinear_motion(T, length, x_0)
    x_coords_curvi = [xi[0] for xi in states]
    y_coords_curvi = [yi[2] for yi in states]
    plt.figure(figsize=(8, 8))
    plt.plot(x_coords_curvi, y_coords_curvi)
    plt.show()