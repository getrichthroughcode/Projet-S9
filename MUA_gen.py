import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import scipy.ndimage as S

def estimate(X_pos):
    X_vit=[0]
    X_acc=[0,0]
    vit = (X_pos[1] - X_pos[0]) / 2
    X_vit.append(vit)
    for i in range(2,len(X_pos)):
        vit = (X_pos[i] - X_pos[i-1]) /1
        X_vit.append(vit)
        acc = (X_pos[i] - 2*X_pos[i-1] +X_pos[i-2]) /1
        X_acc.append(acc)
    return (X_vit,S.median_filter(X_acc,size = 3))

def MUA_gen(length, T, x_0,n):
    L=[]
    # n=0.0005
    q=n*9.81*T
    if (np.shape(x_0) != (3, 1)):
        print("x_0 has not the shape (3,1)")
        return []
    L.append(x_0)  # Ensure x_0 is a column vector
    Q = q* np.array([
        [T**5 / 20, T**4 / 8, T**3 / 6],
        [T**4 / 8, T**3 / 3, T**2 / 2],
        [T**3 / 6, T**2 / 2, T]
    ])


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
if __name__ == "__main__":
    length = 100
    T=1
    x_0=np.array([[0],[0],[0]])

    x=MUA_gen(length, T, x_0,1)
    y=MUA_gen(length, T, x_0,1)
    z = MUA_gen(length, T, x_0,1)
    x_coords = [xi[0, 0] for xi in x]
    y_coords = [yi[0,0] for yi in y]
    z_coords = [yi[0,0] for yi in z]
    x_accs = [xi[2, 0] for xi in x]
    y_accs = [yi[2,0] for yi in y]
    x_vits = [xi[1, 0] for xi in x]
    y_vits = [yi[1,0] for yi in y]
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(x_coords, y_coords, label='Trajectoire (x, y)')
    # plt.title('Trajectoire synthétique dans le plan (x, y)')
    # plt.xlabel('Position x')
    # plt.ylabel('Position y')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
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

    # plt.figure(figsize=(10, 6))
    # plt.plot(lags_x, correlation_x)
    # plt.title("Fonction de corrélation en x")
    # plt.grid()
    # plt.show()
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(lags_y, correlation_y)
    # plt.title("Fonction de corrélation en y")
    # plt.grid()
    # plt.show()

    vit_est_x,acc_est_x = estimate(x_coords)
    vit_est_y,acc_est_y = estimate(y_coords)

    plt.figure(figsize=(10, 6))
    plt.plot(acc_est_x,label='Estimation')
    plt.plot(x_accs,label='Réelle')
    plt.legend()
    plt.title("Accélération réelle et accélération estimée")
    plt.grid()
    plt.show()
    #
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(x_coords, y_coords, label='Trajectoire (x, y)')
    # plt.quiver(x_coords, y_coords, x_vits, y_vits, angles='xy',scale_units='xy', scale=0.5, color='r', label='Vitesse instantanée réel')
    # plt.title('Trajectoire synthétique avec vitesse instantanée réel')
    # plt.xlabel('Position x')
    # plt.ylabel('Position y')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    #
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, y_coords, label='Trajectoire (x, y)')
    plt.quiver(x_coords[1:], y_coords[1:], vit_est_x[1:] ,vit_est_y[1:],angles='xy', scale_units='xy', scale=10, color='r', label='Vitesse instantanée estimmée')
    plt.title('Trajectoire synthétique avec vitesses instantanées estimmée')
    plt.xlabel('Position x')
    plt.ylabel('Position y')
    plt.legend()
    plt.grid(True)
    plt.show()
