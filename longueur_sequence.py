import os
import numpy as np

"""
Ce script parcourt un dossier de données prétraitées (contenant des fichiers .npy )
et calcule les statistiques suivantes pour chaque séquence :
- Nombre total de vidéos
- Longueur moyenne des séquences 
- Longueurs minimale et maximale des séquences 

"""

output_base_path = 'corpus_pretraite'
sequence_lengths = []

for corpus_lettre in os.listdir(output_base_path):
    lettre_path = os.path.join(output_base_path, corpus_lettre)
    for fichier in os.listdir(lettre_path):
        if fichier.endswith('.npy'):
            fichier_path = os.path.join(lettre_path, fichier)
            data = np.load(fichier_path)
            sequence_lengths.append(data.shape[0])

print(f"Nombre total de vidéos: {len(sequence_lengths)}")
print(f"Longueur moyenne: {np.mean(sequence_lengths):.1f} frames")
print(f"Longueur min/max: {np.min(sequence_lengths)}/{np.max(sequence_lengths)} frames")
