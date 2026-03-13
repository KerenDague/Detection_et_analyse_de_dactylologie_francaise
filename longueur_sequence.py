import os
import numpy as np
import matplotlib.pyplot as plt

"""
Ce script parcourt un dossier de données prétraitées (contenant des fichiers .npy)
et calcule les statistiques suivantes pour chaque séquence :
- Nombre total de vidéos
- Longueur moyenne des séquences 
- Longueurs minimale et maximale des séquences 
- Liste des fichiers problématiques (shape[0] == 0, aucune main détectée)
"""

output_base_path = 'corpus_pretraite'
sequence_lengths = []
fichiers = [] 

for corpus_lettre in os.listdir(output_base_path):
    lettre_path = os.path.join(output_base_path, corpus_lettre)
    if not os.path.isdir(lettre_path):
        continue

    for fichier in os.listdir(lettre_path):
        if fichier.endswith('.npy'):
            fichier_path = os.path.join(lettre_path, fichier)
            data = np.load(fichier_path, allow_pickle=True)
            sequence_lengths.append(data.shape[0])
            fichiers.append(fichier_path)

sequence_lengths = np.array(sequence_lengths)

# Détection des abérrants par méthode IQR (inspiration : https://medium.com/@morepravin1989/outlier-detection-with-the-iqr-method-a-complete-guide-c0199bbc10bd)
Q1 = np.percentile(sequence_lengths, 25)
Q3 = np.percentile(sequence_lengths, 75)
IQR = Q3 - Q1
borne_basse = Q1 - 1.5 * IQR
borne_haute = Q3 + 1.5 * IQR

print(f"Q1={Q1:.0f}, Q3={Q3:.0f}, IQR={IQR:.0f}")
print(f"Bornes : [{borne_basse:.0f} frames, {borne_haute:.0f} frames]")

fichiers_aberrants = [
    (fichiers[i], sequence_lengths[i])
    for i in range(len(fichiers))
    if sequence_lengths[i] < borne_basse or sequence_lengths[i] > borne_haute
]

plt.figure(figsize=(10, 6))
plt.hist(sequence_lengths, bins=30, edgecolor='black')
plt.axvline(np.mean(sequence_lengths), color='red', linestyle='dashed', linewidth=1, label=f'Moyenne: {np.mean(sequence_lengths):.1f}')
plt.axvline(np.median(sequence_lengths), color='green', linestyle='dashed', linewidth=1, label=f'Médiane: {np.median(sequence_lengths):.1f}')
plt.xlabel('Nombre de frames par vidéo')
plt.ylabel('Nombre de videos')
plt.title('Distribution des longueurs des séquences')
plt.legend()
plt.grid(True)
plt.show()

print(f"Nombre total de fichiers: {len(sequence_lengths)}")
print(f"Longueur moyenne: {np.mean(sequence_lengths):.1f} frames")
print(f"Longueur médiane: {np.median(sequence_lengths):.1f} frames")
print(f"Longueur min/max: {np.min(sequence_lengths)}/{np.max(sequence_lengths)} frames")

#Fichiers aberrants
print(f"Nombre de fichiers aberrants : {len(fichiers_aberrants)}")
if fichiers_aberrants:
    print("Fichiers concernés :")
    for chemin, longueur in sorted(fichiers_aberrants, key=lambda x: x[1]):
        print(f"  - {chemin}  ({longueur} frames)")
else:
    print("Aucun fichier aberrant trouvé.")
