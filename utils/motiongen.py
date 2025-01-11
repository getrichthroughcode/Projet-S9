import numpy as np 



def MRU_gen(length, T, x_0, n):
    # Vérification de la forme de x_0
    if x_0.shape != (2,):
        print("x_0 has not the shape (2,)")
        return np.empty((2,))  # Retourne un tableau vide si la condition échoue

    # Préallocation pour un tableau contenant toutes les étapes
    L = np.zeros((length + 1, 2))  # Préallouer un tableau pour tous les états
    L[0] = x_0  # Initialisation avec l'état initial

    # Paramètres de la simulation
    q = n * 9.81 * T
    Q = q * np.array([
        [T**3 / 3, T**2 / 2],
        [T**2 / 2, T]
    ])
    phi = np.array([
        [1, T],
        [0, 1]
    ])

    # Boucle principale
    R = np.linalg.cholesky(Q)  # Cholesky decomposition (invariant dans la boucle)
    #print(f"{R=}")
    for i in range(length):
        U = np.random.randn(2,)  # Génère un vecteur aléatoire
        B = R @ U  # Génère le bruit
        L[i + 1] = phi @ L[i] + B  # Calcule le nouvel état

    return L




def MUA_gen(length, T, x_0, n):
    # Vérification de la forme de x_0
    if x_0.shape != (3,):
        print("x_0 has not the shape (3,)")
        return np.empty((3,))  # Retourne un tableau vide si la condition échoue

    # Préallocation pour un tableau contenant toutes les étapes
    L = np.zeros((length + 1, 3))  # Préallouer un tableau pour tous les états
    L[0] = x_0  # Initialisation avec l'état initial

    # Paramètres de la simulation
    q = n * 9.81 * T
    Q = q * np.array([
        [T**5 / 20, T**4 / 8, T**3 / 6],
        [T**4 / 8, T**3 / 3, T**2 / 2],
        [T**3 / 6, T**2 / 2, T]
    ])
    phi = np.array([
        [1, T, T**2 / 2],
        [0, 1, T],
        [0, 0, 1]
    ])

    # Boucle principale
    R = np.linalg.cholesky(Q)  # Cholesky decomposition (invariant dans la boucle)
    #print(f"{R=}")
    for i in range(length):
        U = np.random.randn(3,)  # Génère un vecteur aléatoire
        B = R @ U  # Génère le bruit
        L[i + 1] = phi @ L[i] + B  # Calcule le nouvel état

    return L


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
    R = np.linalg.cholesky(Q)  # Cholesky decomposition
    #print(f"{R=}")
    for i in range(length):
        U = np.random.randn(3, 1)  # Generate a random vector

        B = R @ U         # Generate the noise vector
        # Update x with the new state

        x_new  =phi @ L[-1] + B
        L.append(x_new)

    return L