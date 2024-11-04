# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:54:55 2024

@author: Simon
"""

import numpy as np
from MUA_gen import MUA_gen
from Singer_gen import Singer_gen
from create_csv import create

######### LES VARIABLES ########## 

nb_files = 50
nb_points = 100

def generate_MUA_1D_database():
    for idx_file in range(nb_files):
        X = MUA_gen(nb_points, 1, np.array([[0], [0], [0]]))
        pX = [xi[0,0] for xi in X]  # Position X
        vX = [xi[1,0] for xi in X]  # Vitesse X
        aX = [xi[2,0] for xi in X]  # Accélération Z (ou autre variable)

        print(pX)
        
        create(folder_name = "1D_MUA", 
               number = idx_file,
               type_movement = "MUA",
               Period = 1,
               X_Sigma = 1, X_Alpha = 0,
               Y_Sigma = None, Y_Alpha = None,
               Z_Sigma = None, Z_Alpha = None,
               X = pX, Y = None, Z = None,
               vX = vX, vY = None, vZ = None, 
               aX = aX, aY = None, aZ = None)
        
def generate_MUA_2D_database():
    for idx_file in range(nb_files):
        X = MUA_gen(nb_points, 1, np.array([[0], [0], [0]]))
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
               Period = 1,
               X_Sigma = 1, X_Alpha = 0,
               Y_Sigma = 1, Y_Alpha = 0,
               Z_Sigma = None, Z_Alpha = None,
               X = pX, Y = pY, Z = None,
               vX = vX, vY = vY, vZ = None, 
               aX = aX, aY = aY, aZ = None)
        
def generate_MUA_3D_database():
    for idx_file in range(nb_files):
        X = MUA_gen(nb_points, 1, np.array([[0], [0], [0]]))
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
               Period = 1,
               X_Sigma = 1, X_Alpha = 0,
               Y_Sigma = 1, Y_Alpha = 0,
               Z_Sigma = 1, Z_Alpha = 0,
               X = pX, Y = pY, Z = pZ,
               vX = vX, vY = vY, vZ = vZ, 
               aX = aX, aY = aY, aZ = aZ)

def generate_Singer_1D_database():
    for idx_file in range(nb_files):
        X = Singer_gen(nb_points, 10, np.array([[0], [0], [0]]), 1, 1)
        pX = [xi[0,0] for xi in X]  # Position X
        vX = [xi[1,0] for xi in X]  # Vitesse X
        aX = [xi[2,0] for xi in X]  # Accélération Z (ou autre variable)

        
        create(folder_name = "1D_Singer", 
               number = idx_file,
               type_movement = "Singer",
               Period = 1,
               X_Sigma = 1, X_Alpha = 0,
               Y_Sigma = None, Y_Alpha = None,
               Z_Sigma = None, Z_Alpha = None,
               X = pX, Y = None, Z = None,
               vX = vX, vY = None, vZ = None, 
               aX = aX, aY = None, aZ = None)
        

if __name__ == "__main__":
    generate_MUA_3D_database()