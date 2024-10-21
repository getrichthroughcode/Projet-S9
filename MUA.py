import numpy as np
import matplotlib.pyplot as plt


def MUA_gen(length, T, x_0):
    L = []
    L.append(x_0)  # Ensure x_0 is a column vector
    Q = np.array([
        [T ** 5 / 20, T ** 4 / 8, T ** 3 / 6],
        [T ** 4 / 8, T ** 3 / 3, T ** 2 / 2],
        [T ** 3 / 6, T ** 2 / 2, T]
    ])
    print("The white-noise jerk model is used")

    for i in range(length):
        U = np.random.randn(3, 1)  # Generate a random vector
        R = np.linalg.cholesky(Q)  # Cholesky decomposition
        B = R.T @ U  # Generate the noise vector
        print(B)
        # Update x with the new state
        phi = np.array([[1, T, T ** 2 / 2],
                        [0, 1, T],
                        [0, 0, 1]])
        x_new = phi @ L[-1] + B
        L.append(x_new)

    return L


length = 100
T = 1
x_0 = np.array([[0], [0], [0]])
WNJ_WSA = 1
x = MUA_gen(length, T, x_0)
y = MUA_gen(length, T, x_0)

x_coords = [xi[0, 0] for xi in x]
y_coords = [yi[0, 0] for yi in y]
u_speeds = [xi[1, 0] for xi in x]  # Speed in the x-direction (index 1)
v_speeds = [yi[1, 0] for yi in y]  # Speed in the y-direction (index 3)


plt.figure(figsize=(10, 6))
plt.plot(x_coords, y_coords, label='Trajectoire (x, y)', alpha=0.5)
plt.quiver(x_coords[:-1], y_coords[:-1], u_speeds[:-1], v_speeds[:-1],angles='xy',scale_units='xy', color='r', scale=1, label='Vitesse (u, v)')
plt.title('Trajectoire synthétique et vecteurs de vitesse dans le plan (x, y)')
plt.xlabel('Position x')
plt.ylabel('Position y')
plt.legend()
plt.grid(True)
plt.show()