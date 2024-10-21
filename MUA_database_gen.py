# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:54:55 2024

@author: Simon
"""

import numpy as np
from MUA_gen import MUA_gen
from create_csv import create

######### LES VARIABLES ########## 

nb_files = 50
nb_points = 100

def generate_MUA_database():
    for idx_file in range(nb_files):
        X = MUA_gen(nb_points, 1, np.array([[0], [0], [0]]))
        pX = [xi[0,0] for xi in X]  # Position X
        vX = [xi[1,0] for xi in X]  # Vitesse X
        aX = [xi[2,0] for xi in X]  # Accélération Z (ou autre variable)

        print(pX)
        
        create(folder_name = "Training", 
               number = idx_file,
               type_movement = "MUA",
               Period = 1,
               Sigma = 1, Alpha = 0,
               X = pX, Y = None, Z = None,
               vX = vX, vY = None, vZ = None, 
               aX = aX, aY = None, aZ = None)


def generate_MUA_database():
    for idx_file in range(nb_files):
        X = MUA_gen(nb_points, 1, np.array([[0], [0], [0]]))
        pX = [xi[0,0] for xi in X]  # Position X
        vX = [xi[1,0] for xi in X]  # Vitesse X
        aX = [xi[2,0] for xi in X]  # Accélération Z (ou autre variable)

        print(pX)
        
        create(folder_name = "Training", 
               number = idx_file,
               type_movement = "MUA",
               Period = 1,
               Sigma = 1, Alpha = 0,
               X = pX, Y = None, Z = None,
               vX = vX, vY = None, vZ = None, 
               aX = aX, aY = None, aZ = None)

def main():
    for idx_file in range(nb_files):
        X = Singer_gen(nb_points, 10, np.array([[0], [0], [0]]), 1, 1)
        pX = [xi[0,0] for xi in X]  # Position X
        vX = [xi[1,0] for xi in X]  # Vitesse X
        aX = [xi[2,0] for xi in X]  # Accélération Z (ou autre variable)

        
        create(folder_name = "Singer", 
               number = idx_file,
               type_movement = "Singer",
               Period = 1,
               Sigma = 1, Alpha = 1,
               X = pX, Y = None, Z = None,
               vX = vX, vY = None, vZ = None, 
               aX = aX, aY = None, aZ = None)
        

if __name__ == "__main__":
    main()