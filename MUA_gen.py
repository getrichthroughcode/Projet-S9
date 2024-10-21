import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d



def MUA_gen(length, T, x_0):
    L=[]
    L.append(x_0)  # Ensure x_0 is a column vector
    Q = np.array([
        [T**5 / 20, T**4 / 8, T**3 / 6],
        [T**4 / 8, T**3 / 3, T**2 / 2],
        [T**3 / 6, T**2 / 2, T]
    ])
    print("The white-noise jerk model is used")

    phi = np.array([[1, T, T ** 2 / 2],
                    [0, 1, T],
                    [0, 0, 1]])
    for i in range(length):
        U = np.random.randn(3, 1)  # Generate a random vector
        R = np.linalg.cholesky(Q)  # Cholesky decomposition
        B = R.T @ U         # Generate the noise vector
        print(B)
        # Update x with the new state

        x_new  =phi @ L[-1] + B
        L.append(x_new)

    return L

length = 100
T=1
x_0=np.array([[0],[0],[0]])

x=MUA_gen(length, T, x_0)
y=MUA_gen(length, T, x_0)
z = MUA_gen(length, T, x_0)
x_coords = [xi[0, 0] for xi in x]
y_coords = [yi[0,0] for yi in y]
z_coords = [yi[0,0] for yi in z]

# plt.figure(figsize=(10, 6))
# plt.plot(x_coords, y_coords, label='Trajectoire (x, y)')
# plt.title('Trajectoire synthétique dans le plan (x, y)')
# plt.xlabel('Position x')
# plt.ylabel('Position y')
# plt.legend()
# plt.grid(True)
# plt.show()
#
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(x_coords, y_coords, z_coords)
ax.set_title('Trajectoire 3D')
ax.set_xlabel('Position x')
ax.set_ylabel('Position y')
ax.set_zlabel('Position z')

# Affichage de la figure
plt.show()
