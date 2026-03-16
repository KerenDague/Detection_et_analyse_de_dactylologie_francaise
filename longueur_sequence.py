import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split

"""
Ce script :
1. Parcourt un dossier de données prétraitées (contenant des fichiers .npy)
   et calcule les statistiques suivantes pour chaque séquence :
   - Nombre total de vidéos
   - Longueur moyenne des séquences
   - Longueurs minimale et maximale des séquences
   - Liste des fichiers problématiques (shape[0] == 0, aucune main détectée)

2. Crée un Dataset PyTorch (LSFDataset) en paddant/tronquant les séquences
   à la longueur maximale calculée via la méthode IQR, et enregistre :
   - les fichiers .npy paddés dans un dossier par lettre (corpus_pretraite_padded/)
   - l'ensemble du dataset (tenseurs + labels + métadonnées) dans un fichier .pt (dataset_lsf.pt)
"""

# PARTIE 1 : Analyse statistique des séquences

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

# La longueur max est dérivée directement de l'IQR (borne haute)
longueur_max = int(borne_haute)
print(f"\nlongueur_max retenue : {longueur_max} frames")


# PARTIE 2 : Création du Dataset PyTorch

class LSFDataset(Dataset):
    def __init__(self, base_path, letters, longueur_max, output_path=None):
        self.samples = []
        self.letter_map = {letter: num for num, letter in enumerate(letters)}
        self.output_path = output_path
        self.longueur_max = longueur_max

        for letter in letters:
            npy_path = os.path.join(base_path, letter)
            if output_path:
                os.makedirs(os.path.join(output_path, letter), exist_ok=True)

            for file in os.listdir(npy_path):
                if file.endswith('.npy'):
                    self.samples.append((os.path.join(npy_path, file), self.letter_map[letter]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = np.load(path)

        if len(data) > self.longueur_max:
            data = data[:self.longueur_max]
        elif len(data) < self.longueur_max:
            pad_width = self.longueur_max - len(data)
            padding = np.zeros((pad_width, 63))
            data = np.vstack((data, padding))

        if self.output_path:
            new_path = os.path.join(self.output_path, os.path.basename(os.path.dirname(path)), os.path.basename(path))
            np.save(new_path, data)

        return torch.from_numpy(data).float(), torch.tensor(label)


base_path = "corpus_pretraite"
output_path = "corpus_pretraite_padded"   # dossier des .npy paddés (un sous-dossier par lettre)
dataset_save_path = "dataset_lsf.pt"      # fichier PyTorch contenant le dataset complet
letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

dataset = LSFDataset(base_path, letters, longueur_max=longueur_max, output_path=output_path)
print(f"\nDataset créé : {len(dataset)} échantillons")


#Sauvegarde du dataset 
#sauvegarde des tenseurs (données + labels) dans un fichier .pt
all_data = []
all_labels = []
for i in range(len(dataset)):
    x, y = dataset[i]
    all_data.append(x)
    all_labels.append(y)

all_data   = torch.stack(all_data)    # shape : (N, longueur_max, 63)
all_labels = torch.stack(all_labels)  # shape : (N,)

torch.save({"data": all_data, "labels": all_labels, "letter_map": dataset.letter_map, "longueur_max": longueur_max}, dataset_save_path)

print(f"Dataset sauvegardé → {dataset_save_path}  ({all_data.shape[0]} échantillons, shape tenseur : {tuple(all_data.shape)})")

#si les fichiers .npy paddé ont déjà été écrit dans output_path par LSFDataset.__getitem__
print(f"Fichiers .npy paddés enregistrés dans → {os.path.abspath(output_path)}/")
