# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:29:32 2024

@author: slienard
"""
import os
import csv


def create(folder_name, number, type_movement, Period, Sigma, Alpha = 0,
           X=None, Y=None, Z=None, vX=None, vY=None, vZ=None, 
           aX=None, aY=None, aZ=None):
    
    # Chemin du dossier à créer (dans le même répertoire que le fichier .py)
    folder = os.path.join(os.path.dirname(__file__), folder_name)
    print(os.path.dirname(__file__))
    print(folder)
    
    title = f"{number}_{type_movement}.csv"

    # Créer le dossier s'il n'existe pas
    if not os.path.exists(folder):
        print("Création du dossier:", folder)
        os.makedirs(folder)

    # Chemin complet du fichier CSV
    csv_path = os.path.join(folder, title)
    
    # Ouverture du fichier CSV pour l'écriture
    with open(csv_path, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)

        # Écriture de l'entête
        spamwriter.writerow(['Title'] + [title])
        spamwriter.writerow(['type_movement'] + [type_movement])
        spamwriter.writerow(['Period'] + [Period])
        spamwriter.writerow(['Sigma'] + [Sigma])
        spamwriter.writerow(['Alpha'] + [Alpha])
        
        # Écriture des données s'il y en a
        if X is not None:
            spamwriter.writerow(['X'] + X)
        if Y is not None:
            spamwriter.writerow(['Y'] + Y)   
        if Z is not None:
            spamwriter.writerow(['Z'] + Z) 
        
        if vX is not None:
            spamwriter.writerow(['vX'] + vX)
        if vY is not None:
            spamwriter.writerow(['vY'] + vY)        
        if vZ is not None:
            spamwriter.writerow(['vZ'] + vZ)
        
        if aX is not None: 
            spamwriter.writerow(['aX'] + aX)
        if aY is not None:
            spamwriter.writerow(['aY'] + aY)        
        if aZ is not None:
            spamwriter.writerow(['aZ'] + aZ)

