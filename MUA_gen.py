import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

def estimate(X_pos):
    X_vit=[0]
    X_acc=[0,0]
    vit = (X_pos[1] - X_pos[0]) / 2
    X_vit.append(vit)
    for i in range(2,len(X_pos)):
        vit = (X_pos[i] - X_pos[i-1]) /2
        X_vit.append(vit)
        acc = (X_pos[i] - 2*X_pos[i-1] +X_pos[i-2]) /4
        X_acc.append(acc)
    return (X_vit,X_acc)

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
        # Update x with the new state
        x_new  =phi @ L[-1] + B
        L.append(x_new)

    return L
#afficher vitesse instantanée (fleches)

length = 100
T=1
x_0=np.array([[0],[0],[0]])

x=MUA_gen(length, T, x_0)
y=MUA_gen(length, T, x_0)
z = MUA_gen(length, T, x_0)
x_coords = [xi[0, 0] for xi in x]
y_coords = [yi[0,0] for yi in y]
z_coords = [yi[0,0] for yi in z]
x_accs = [xi[2, 0] for xi in x]
y_accs = [yi[2,0] for yi in y]
plt.figure(figsize=(10, 6))
plt.plot(x_coords, y_coords, label='Trajectoire (x, y)')
plt.title('Trajectoire synthétique dans le plan (x, y)')
plt.xlabel('Position x')
plt.ylabel('Position y')
plt.legend()
plt.grid(True)
plt.show()
#
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

plt.figure(figsize=(10, 6))
plt.plot(lags_x, correlation_x)
plt.title("Fonction de corrélation en x")
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(lags_y, correlation_y)
plt.title("Fonction de corrélation en y")
plt.grid()
plt.show()

vit_est,acc_est = estimate(x_coords)
plt.figure(figsize=(10, 6))
plt.plot(acc_est,label='Estimation')
plt.plot(x_accs,label='Réelle')
plt.legend()
plt.title("Accélération réelle et accélération estimée")
plt.grid()
plt.show()
