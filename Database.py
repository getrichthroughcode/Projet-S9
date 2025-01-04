# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:54:55 2024

@author: Simon
"""

import numpy as np
from MUA_gen import MUA_gen
from MRU_gen import MRU_gen
from Singer_gen import Singer_gen
from create_csv import create
import random

######### LES VARIABLES ########## 

nb_files = 2000
nb_points = 1000
n = 1

def generate_MUA_1D_database():
    for idx_file in range(nb_files):
        if idx_file % 100 == 0:
            print(idx_file)
        a = random.uniform(-10, 10)
        b = random.uniform(-10, 10)
        c = random.uniform(-10, 10)

        X = MUA_gen(nb_points, 1, np.array([[0], [0], [0]]), n)
        pX = [xi[0,0] for xi in X]  # Position X
        vX = [xi[1,0] for xi in X]  # Vitesse X
        aX = [xi[2,0] for xi in X]  # Accélération Z (ou autre variable)
        
        create(folder_name = "test_1D_MUA", 
               number = idx_file,
               type_movement = "MUA",
               Classe = 0,
               Nb_sample = nb_points,
               Period = 1,
               n = n,
               X_Sigma = None, X_Alpha = None,
               Y_Sigma = None, Y_Alpha = None,
               Z_Sigma = None, Z_Alpha = None,
               X = pX, Y = None, Z = None,
               vX = vX, vY = None, vZ = None, 
               aX = aX, aY = None, aZ = None)


def generate_MUA_2D_database():
    for idx_file in range(nb_files):
        X = MUA_gen(nb_points, 1, np.array([[0], [0], [0]]), n)
        pX = [xi[0,0] for xi in X]  # Position X
        vX = [xi[1,0] for xi in X]  # Vitesse X
        aX = [xi[2,0] for xi in X]  # Accélération Z (ou autre variable)
        
        Y = MUA_gen(nb_points, 1, np.array([[0], [0], [0]]))
        pY = [yi[0,0] for yi in Y]  # Position X
        vY = [yi[1,0] for yi in Y]  # Vitesse X
        aY = [yi[2,0] for yi in Y]  # Accélération Z (ou autre variable)

        
        create(folder_name = "2D_MUA", 
               number = idx_file,
               type_movement = "MUA",
               Nb_sample = nb_points,
               Period = 1,
               n = n,
               X_Sigma = None, X_Alpha = None,
               Y_Sigma = None, Y_Alpha = None,
               Z_Sigma = None, Z_Alpha = None,
               X = pX, Y = pY, Z = None,
               vX = vX, vY = vY, vZ = None, 
               aX = aX, aY = aY, aZ = None)


def generate_MUA_3D_database():
    for idx_file in range(nb_files):
        X = MUA_gen(nb_points, 1, np.array([[0], [0], [0]]), n)
        pX = [xi[0,0] for xi in X]  # Position X
        vX = [xi[1,0] for xi in X]  # Vitesse X
        aX = [xi[2,0] for xi in X]  # Accélération Z (ou autre variable)
        
        Y = MUA_gen(nb_points, 1, np.array([[0], [0], [0]]))
        pY = [yi[0,0] for yi in Y]  # Position X
        vY = [yi[1,0] for yi in Y]  # Vitesse X
        aY = [yi[2,0] for yi in Y]  # Accélération Z (ou autre variable)

        Z = MUA_gen(nb_points, 1, np.array([[0], [0], [0]]))
        pZ = [yi[0,0] for yi in Y]  # Position X
        vZ = [yi[1,0] for yi in Y]  # Vitesse X
        aZ = [yi[2,0] for yi in Y]  # Accélération Z (ou autre variable)
        
        create(folder_name = "3D_MUA", 
               number = idx_file,
               type_movement = "MUA",
               Nb_sample = nb_points,
               Period = 1,
               n = n,
               X_Sigma = None, X_Alpha = None,
               Y_Sigma = None, Y_Alpha = None,
               Z_Sigma = None, Z_Alpha = None,
               X = pX, Y = pY, Z = pZ,
               vX = vX, vY = vY, vZ = vZ, 
               aX = aX, aY = aY, aZ = aZ)


def generate_Singer_1D_database():
    for idx_file in range(nb_files):
        a = random.uniform(-10, 10)
        b = random.uniform(-10, 10)
        c = random.uniform(-10, 10)
        
        X = Singer_gen(nb_points, 10, np.array([[0], [0], [0]]), 1, 1)
        pX = [xi[0,0] for xi in X]  # Position X
        vX = [xi[1,0] for xi in X]  # Vitesse X
        aX = [xi[2,0] for xi in X]  # Accélération Z (ou autre variable)

        Sigma_rand = 1
        Alpha_rand = random.uniform(0.0001, 0.1)

        create(folder_name = "1D_Singer", 
               number = idx_file,
               type_movement = "Singer",
               Classe= 3,
               Nb_sample = nb_points,
               Period = 1,
               n = None,
               X_Sigma = Sigma_rand, X_Alpha = Alpha_rand,
               Y_Sigma = None, Y_Alpha = None,
               Z_Sigma = None, Z_Alpha = None,
               X = pX, Y = None, Z = None,
               vX = vX, vY = None, vZ = None, 
               aX = aX, aY = None, aZ = None)


def generate_Singer_2D_database():
    for idx_file in range(nb_files):
        X = Singer_gen(nb_points, 10, np.array([[0], [0], [0]]), 1, 1)
        pX = [xi[0,0] for xi in X]  # Position X
        vX = [xi[1,0] for xi in X]  # Vitesse X
        aX = [xi[2,0] for xi in X]  # Accélération Z (ou autre variable)

        Y = Singer_gen(nb_points, 10, np.array([[0], [0], [0]]), 1, 1)
        pY = [yi[0,0] for yi in Y]  # Position X
        vY = [yi[1,0] for yi in Y]  # Vitesse X
        aY = [yi[2,0] for yi in Y]  # Accélération Z (ou autre variable)

        
        create(folder_name = "2D_Singer", 
               number = idx_file,
               type_movement = "Singer",
               Classe= 3,
               Nb_sample = nb_points,
               Period = 1,
               n = None,
               X_Sigma = 1, X_Alpha = 1,
               Y_Sigma = 1, Y_Alpha = 1,
               Z_Sigma = None, Z_Alpha = None,
               X = pX, Y = pY, Z = None,
               vX = vX, vY = vY, vZ = None, 
               aX = aX, aY = aY, aZ = None)


def generate_Singer_3D_database():
    for idx_file in range(nb_files):
        X = Singer_gen(nb_points, 10, np.array([[0], [0], [0]]), 1, 1)
        pX = [xi[0,0] for xi in X]  # Position X
        vX = [xi[1,0] for xi in X]  # Vitesse X
        aX = [xi[2,0] for xi in X]  # Accélération Z (ou autre variable)

        Y = Singer_gen(nb_points, 10, np.array([[0], [0], [0]]), 1, 1)
        pY = [yi[0,0] for yi in Y]  # Position X
        vY = [yi[1,0] for yi in Y]  # Vitesse X
        aY = [yi[2,0] for yi in Y]  # Accélération Z (ou autre variable)

        Z = Singer_gen(nb_points, 10, np.array([[0], [0], [0]]), 1, 1)
        pZ = [zi[0,0] for zi in Z]  # Position X
        vZ = [zi[1,0] for zi in Z]  # Vitesse X
        aZ = [zi[2,0] for zi in Z]  # Accélération Z (ou autre variable)
                
        create(folder_name = "3D_Singer", 
               number = idx_file,
               type_movement = "Singer",
               Nb_sample = nb_points,
               Period = 1,
               n = None,
               X_Sigma = 1, X_Alpha = 1,
               Y_Sigma = 1, Y_Alpha = 1,
               Z_Sigma = 1, Z_Alpha = 1,
               X = pX, Y = pY, Z = pZ,
               vX = vX, vY = vY, vZ = vZ, 
               aX = aX, aY = aY, aZ = aZ)


def generate_MRU_1D_database():
    for idx_file in range(nb_files):
        a = random.uniform(-10, 10)
        b = random.uniform(-10, 10)
        c = random.uniform(-10, 10)

        X = MRU_gen(nb_points, 1, np.array([[0], [0]]),1)
        pX = [xi[0,0] for xi in X]  # Position X
        vX = [xi[1,0] for xi in X]  # Vitesse X

        create(folder_name = "test_1D_MRU", 
               number = idx_file,
               type_movement = "MRU",
               Classe= 3,
               Nb_sample = nb_points,
               Period = 1,
               n = n,
               X_Sigma = None, X_Alpha = None,
               Y_Sigma = None, Y_Alpha = None,
               Z_Sigma = None, Z_Alpha = None,
               X = pX, Y = None, Z = None,
               vX = vX, vY = None, vZ = None, 
               aX = None, aY = None, aZ = None)
        

def generate_MRU_2D_database():
    for idx_file in range(nb_files):
        X = MRU_gen(nb_points, 1, np.array([[0], [0]]), n)
        pX = [xi[0,0] for xi in X]  # Position X
        vX = [xi[1,0] for xi in X]  # Vitesse X
        
        Y = MRU_gen(nb_points, 1, np.array([[0], [0]]), n)
        pY = [yi[0,0] for yi in Y]  # Position X
        vY = [yi[1,0] for yi in Y]  # Vitesse X

        create(folder_name = "2D_MRU", 
               number = idx_file,
               type_movement = "MRU",
               Nb_sample = nb_points,
               Period = 1,
               n = n,
               X_Sigma = None, X_Alpha = None,
               Y_Sigma = None, Y_Alpha = None,
               Z_Sigma = None, Z_Alpha = None,
               X = pX, Y = pY, Z = None,
               vX = vX, vY = vY , vZ = None, 
               aX = None, aY = None, aZ = None)


def generate_MRU_3D_database():
    for idx_file in range(nb_files):
        X = MRU_gen(nb_points, 1, np.array([[0], [0]]), n)
        pX = [xi[0,0] for xi in X]  # Position X
        vX = [xi[1,0] for xi in X]  # Vitesse X
        
        Y = MRU_gen(nb_points, 1, np.array([[0], [0]]), n)
        pY = [yi[0,0] for yi in Y]  # Position X
        vY = [yi[1,0] for yi in Y]  # Vitesse X
        
        Z = MRU_gen(nb_points, 1, np.array([[0], [0]]), n)
        pZ = [zi[0,0] for zi in Z]  # Position X
        vZ = [zi[1,0] for zi in Z]  # Vitesse X

        create(folder_name = "3D_MRU", 
               number = idx_file,
               type_movement = "MRU",
               Nb_sample = nb_points,
               Period = 1,
               n = n,
               X_Sigma = None, X_Alpha = None,
               Y_Sigma = None, Y_Alpha = None,
               Z_Sigma = None, Z_Alpha = None,
               X = pX, Y = pY, Z = pZ,
               vX = vX, vY = vY , vZ = vZ, 
               aX = None, aY = None, aZ = None)


def generate_Singer_1D_haut_database():
    for idx_file in range(nb_files):
        
        mu, sigma = 1000, 1
        alpha = np.random.normal(mu, sigma)
        
        X = Singer_gen(nb_points, 10, np.array([[0], [0], [0]]), 1, 1)
        pX = [xi[0,0] for xi in X]  # Position X
        vX = [xi[1,0] for xi in X]  # Vitesse X
        aX = [xi[2,0] for xi in X]  # Accélération Z (ou autre variable)
        
        create(folder_name = "1D_Singer_haut", 
                number = idx_file,
                type_movement = "Singer",
                Classe= 2,
                Nb_sample = nb_points,
                Period = 1,
                n = n,
                X_Sigma = 1, X_Alpha = alpha,
                Y_Sigma = None, Y_Alpha = None,
                Z_Sigma = None, Z_Alpha = None,
                X = pX, Y = None, Z = None,
                vX = vX, vY = None , vZ = None, 
                aX = None, aY = None, aZ = None)
        
def generate_Singer_1D_bas_database():
    for idx_file in range(nb_files):
        
        # pour définir Alpha
        mu, sigma = 0.001, 1
        
        alpha = np.random.normal(mu, sigma)
        
        X = Singer_gen(nb_points, 10, np.array([[0], [0], [0]]), 1, 1)
        pX = [xi[0,0] for xi in X]  # Position X
        vX = [xi[1,0] for xi in X]  # Vitesse X
        aX = [xi[2,0] for xi in X]  # Accélération Z (ou autre variable)
        
        create(folder_name = "1D_Singer_bas", 
                number = idx_file,
                type_movement = "Singer",
                Classe= 3,
                Nb_sample = nb_points,
                Period = 1,
                n = n,
                X_Sigma = 1, X_Alpha = alpha,
                Y_Sigma = None, Y_Alpha = None,
                Z_Sigma = None, Z_Alpha = None,
                X = pX, Y = None, Z = None,
                vX = vX, vY = None , vZ = None, 
                aX = None, aY = None, aZ = None)
        
        
def generate_Singer_1D_moyenhaut_database():
    for idx_file in range(nb_files):
        
        mu, sigma = 0.1, 1
        alpha = np.random.normal(mu, sigma)
        
        X = Singer_gen(nb_points, 10, np.array([[0], [0], [0]]), 1, 1)
        pX = [xi[0,0] for xi in X]  # Position X
        vX = [xi[1,0] for xi in X]  # Vitesse X
        aX = [xi[2,0] for xi in X]  # Accélération Z (ou autre variable)
        
        create(folder_name = "1D_Singer_moyenhaut", 
                number = idx_file,
                type_movement = "Singer",
                Classe= 4,
                Nb_sample = nb_points,
                Period = 1,
                n = n,
                X_Sigma = 1, X_Alpha = alpha,
                Y_Sigma = None, Y_Alpha = None,
                Z_Sigma = None, Z_Alpha = None,
                X = pX, Y = None, Z = None,
                vX = vX, vY = None , vZ = None, 
                aX = None, aY = None, aZ = None)
        
        
def generate_Singer_1D_moyenbas_database():
    for idx_file in range(nb_files):
        
        mu, sigma = 10, 1
        alpha = np.random.normal(mu, sigma)
        
        X = Singer_gen(nb_points, 10, np.array([[0], [0], [0]]), 1, 1)
        pX = [xi[0,0] for xi in X]  # Position X
        vX = [xi[1,0] for xi in X]  # Vitesse X
        aX = [xi[2,0] for xi in X]  # Accélération Z (ou autre variable)
        
        create(folder_name = "1D_Singer_moyenbas", 
                number = idx_file,
                type_movement = "Singer",
                Classe= 5,
                Nb_sample = nb_points,
                Period = 1,
                n = n,
                X_Sigma = 1, X_Alpha = alpha,
                Y_Sigma = None, Y_Alpha = None,
                Z_Sigma = None, Z_Alpha = None,
                X = pX, Y = None, Z = None,
                vX = vX, vY = None , vZ = None, 
                aX = None, aY = None, aZ = None)
        


if __name__ == "__main__":
    #generate_Singer_1D_haut_database()
    generate_MRU_1D_database()
    generate_MUA_1D_database()

    
    